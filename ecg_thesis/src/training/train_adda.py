from __future__ import annotations

import csv
import hashlib
import os
import shutil
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.adda import ADDAClassifier
from src.training.evaluate import predict_model
from src.training.train import load_model_from_checkpoint
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb, should_log_artifacts


def train_adda(
    source_train_dataset,
    source_val_dataset,
    target_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    checkpoint_prefix = train_cfg.get("checkpoint_prefix", "clef_adda")
    backup_dir = _checkpoint_backup_dir(config)
    if backup_dir is not None:
        ensure_dir(backup_dir)
        print(f"Checkpoint backup enabled: {backup_dir}")

    model, source_ckpt = build_adda_from_config(config, device)
    model = model.to(device)
    print(
        "Initialized ADDA from source checkpoint:",
        {
            "source_init_checkpoint": config["adda"]["source_init_checkpoint"],
            "source_epoch": source_ckpt.get("epoch"),
            "source_best_epoch": source_ckpt.get("best_epoch"),
            "source_encoder_frozen": _all_frozen(model.source_encoder),
            "classifier_frozen": _all_frozen(model.classifier),
            "target_encoder_trainable": any(param.requires_grad for param in model.target_encoder.parameters()),
        },
    )
    wandb_run = init_wandb(
        config,
        job_type="train",
        default_name=checkpoint_prefix,
        extra_config={"method": "clef_adda", "source_checkpoint_epoch": source_ckpt.get("epoch")},
    )

    source_loader = DataLoader(
        source_train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        source_val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer_d = torch.optim.AdamW(
        model.domain_discriminator.parameters(),
        lr=float(train_cfg["discriminator_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    optimizer_m = torch.optim.AdamW(
        [param for param in model.target_encoder.parameters() if param.requires_grad],
        lr=float(train_cfg["target_encoder_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler_m = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_m, mode="max", factor=0.5, patience=3)

    total_epochs = int(train_cfg["epochs"])
    steps_per_epoch = max(len(source_loader), len(target_loader))
    best_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    patience = int(train_cfg["early_stopping_patience"])
    history = []
    best_path = ckpt_dir / f"{checkpoint_prefix}_best.pt"
    latest_path = ckpt_dir / f"{checkpoint_prefix}_latest.pt"

    for epoch in range(1, total_epochs + 1):
        should_stop = False
        model.train()
        losses_d = []
        losses_m = []
        domain_true = []
        domain_pred = []
        target_as_source = []
        source_domain_probs = []
        target_domain_probs = []
        target_entropies = []
        target_pseudo_counts = np.zeros(int(config["data"]["num_classes"]), dtype=np.int64)
        source_iter = cycle(source_loader) if len(source_loader) < steps_per_epoch else iter(source_loader)
        target_iter = cycle(target_loader) if len(target_loader) < steps_per_epoch else iter(target_loader)

        progress = tqdm(range(steps_per_epoch), desc=f"adda epoch {epoch}/{total_epochs}", leave=True, dynamic_ncols=True, mininterval=1.0)
        for _ in progress:
            x_s = _batch_x_to_device(next(source_iter), device)
            x_t = _batch_x_to_device(next(target_iter), device)

            _set_requires_grad(model.domain_discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                f_s = model.forward_source_features(x_s)
                f_t = model.forward_target_features(x_t).detach()
            logits_s = model.forward_domain_from_features(f_s)
            logits_t = model.forward_domain_from_features(f_t)
            y_s = torch.ones_like(logits_s)
            y_t = torch.zeros_like(logits_t)
            loss_d = loss_fn(logits_s, y_s) + loss_fn(logits_t, y_t)
            loss_d.backward()
            optimizer_d.step()

            with torch.no_grad():
                prob_s = torch.sigmoid(logits_s)
                prob_t = torch.sigmoid(logits_t)
                pred_s = (prob_s >= 0.5).long()
                pred_t = (prob_t >= 0.5).long()
                domain_true.append(torch.cat([torch.ones_like(pred_s), torch.zeros_like(pred_t)]).cpu().numpy())
                domain_pred.append(torch.cat([pred_s, pred_t]).cpu().numpy())
                target_as_source.append(float(pred_t.float().mean().cpu()))
                source_domain_probs.append(float(prob_s.mean().cpu()))
                target_domain_probs.append(float(prob_t.mean().cpu()))

            _set_requires_grad(model.domain_discriminator, False)
            model.domain_discriminator.eval()
            optimizer_m.zero_grad(set_to_none=True)
            f_t = model.forward_target_features(x_t)
            logits_t_for_m = model.forward_domain_from_features(f_t)
            loss_m = loss_fn(logits_t_for_m, torch.ones_like(logits_t_for_m))
            loss_m.backward()
            optimizer_m.step()
            _set_requires_grad(model.domain_discriminator, True)
            model.domain_discriminator.train()

            with torch.no_grad():
                target_logits = model.classifier(f_t.detach())
                target_probs = torch.softmax(target_logits, dim=1)
                entropy = -(target_probs * torch.log(target_probs.clamp_min(1e-8))).sum(dim=1)
                target_entropies.append(float(entropy.mean().cpu()))
                pseudo = target_probs.argmax(dim=1).detach().cpu().numpy()
                target_pseudo_counts += np.bincount(pseudo, minlength=len(target_pseudo_counts))

            losses_d.append(float(loss_d.detach().cpu()))
            losses_m.append(float(loss_m.detach().cpu()))
            progress.set_postfix(loss_d=f"{losses_d[-1]:.4f}", loss_m=f"{losses_m[-1]:.4f}", refresh=False)

        domain_accuracy = float((np.concatenate(domain_true) == np.concatenate(domain_pred)).mean())
        val_result = predict_model(model, val_loader, device, desc=f"adda source-val epoch {epoch}")
        val_metrics = val_result["metrics"]
        scheduler_m.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "loss_d": float(np.mean(losses_d)),
            "loss_m": float(np.mean(losses_m)),
            "domain_accuracy": domain_accuracy,
            "target_as_source_rate": float(np.mean(target_as_source)),
            "source_domain_prob_mean": float(np.mean(source_domain_probs)),
            "target_domain_prob_mean": float(np.mean(target_domain_probs)),
            "target_prediction_entropy": float(np.mean(target_entropies)),
            "source_val_accuracy": val_metrics["accuracy"],
            "source_val_macro_f1": val_metrics["macro_f1"],
            "target_encoder_lr": optimizer_m.param_groups[0]["lr"],
            "discriminator_lr": optimizer_d.param_groups[0]["lr"],
        }
        for class_idx, class_name in enumerate(config["data"]["class_names"]):
            row[f"target_pseudo_count_{class_name}"] = int(target_pseudo_counts[class_idx])
            row[f"target_pseudo_rate_{class_name}"] = float(target_pseudo_counts[class_idx] / max(1, int(target_pseudo_counts.sum())))
        history.append(row)
        print(
            f"adda epoch {epoch}/{total_epochs}: loss_d={row['loss_d']:.4f}, "
            f"loss_m={row['loss_m']:.4f}, val_f1={row['source_val_macro_f1']:.4f}, "
            f"domain_acc={row['domain_accuracy']:.4f}, target_as_source={row['target_as_source_rate']:.4f}, "
            f"Dsrc={row['source_domain_prob_mean']:.4f}, Dtgt={row['target_domain_prob_mean']:.4f}, "
            f"target_entropy={row['target_prediction_entropy']:.4f}",
            flush=True,
        )
        wandb_run.log({f"train/{key}": value for key, value in row.items() if key != "epoch"}, step=epoch)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(_payload(model, optimizer_d, optimizer_m, scheduler_m, config, epoch, best_f1, best_epoch, stale_epochs, history), best_path, backup_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                should_stop = True

        _save_checkpoint(_payload(model, optimizer_d, optimizer_m, scheduler_m, config, epoch, best_f1, best_epoch, stale_epochs, history), latest_path, backup_dir)
        if should_stop:
            break

    train_log_name = f"{checkpoint_prefix}_train_log.csv"
    _write_history_csv(history, log_dir / train_log_name)
    if backup_dir is not None:
        _copy_to_backup(log_dir / train_log_name, backup_dir)
    wandb_run.summary_update({"best_epoch": best_epoch, "best_source_val_macro_f1": best_f1})
    if should_log_artifacts(config):
        wandb_run.log_artifact(best_path, name=f"{checkpoint_prefix}_best", artifact_type="model")
        wandb_run.log_artifact(log_dir / train_log_name, name=f"{checkpoint_prefix}_train_log", artifact_type="train_log")
    wandb_run.finish()
    return {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "checkpoint_backup_dir": str(backup_dir) if backup_dir is not None else None,
        "best_epoch": best_epoch,
        "best_source_val_macro_f1": best_f1,
        "history": history,
    }


def build_adda_from_config(config: dict[str, Any], device: torch.device) -> tuple[ADDAClassifier, dict[str, Any]]:
    model_cfg = config["model"]
    checkpoint_path = _resolve_checkpoint(config["adda"]["source_init_checkpoint"], config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ADDA source initialization checkpoint not found: {checkpoint_path}")
    source_model, source_ckpt = load_model_from_checkpoint(
        checkpoint_path,
        device,
        model_kwargs_override={"clef_checkpoint_path": model_cfg.get("clef_checkpoint_path")},
    )
    expected_model = config.get("adda", {}).get("source_init_expected_model", "clef_pretrained")
    if source_ckpt.get("model_name") != expected_model:
        raise ValueError(
            f"ADDA source checkpoint model mismatch: expected {expected_model!r}, got {source_ckpt.get('model_name')!r}"
        )
    if not hasattr(source_model, "encoder") or not hasattr(source_model, "classifier"):
        raise ValueError("ADDA source model must expose encoder and classifier modules")
    model = ADDAClassifier(
        source_encoder=source_model.encoder,
        classifier=source_model.classifier,
        embedding_dim=int(source_model.embedding_dim),
        discriminator_hidden_dim=int(model_cfg.get("discriminator_hidden_dim", 256)),
        discriminator_hidden_dims=model_cfg.get("discriminator_hidden_dims"),
        dropout=float(model_cfg.get("discriminator_dropout", model_cfg.get("dropout", 0.1))),
    )
    return model, source_ckpt


def load_adda_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    model_kwargs_override: dict[str, Any] | None = None,
) -> tuple[ADDAClassifier, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    if model_kwargs_override:
        config = _copy_config_with_model_overrides(config, model_kwargs_override)
    model, _ = build_adda_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(
        "Loaded ADDA checkpoint:",
        {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_metric": checkpoint.get("best_metric"),
            "fingerprint": checkpoint.get("model_state_fingerprint"),
        },
    )
    return model, checkpoint


def _batch_x_to_device(batch, device: torch.device) -> torch.Tensor:
    if len(batch) == 2:
        x, _y = batch
        return x.to(device, non_blocking=True)
    if len(batch) == 3 and isinstance(batch[2], dict):
        x, _y, _meta = batch
        return x.to(device, non_blocking=True)
    if len(batch) == 3:
        x, _rr, _y = batch
        return x.to(device, non_blocking=True)
    if len(batch) == 4:
        x, _rr, _y, _meta = batch
        return x.to(device, non_blocking=True)
    raise ValueError(f"Unsupported ADDA batch length: {len(batch)}")


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(value)


def _all_frozen(module: torch.nn.Module) -> bool:
    return all(not param.requires_grad for param in module.parameters())


def _resolve_checkpoint(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config.get("_base_dir", ".")) / path).resolve()


def _payload(model, optimizer_d, optimizer_m, scheduler_m, config, epoch, best_metric, best_epoch, stale_epochs, history):
    state_dict = model.state_dict()
    return {
        "model_state_dict": state_dict,
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "optimizer_m_state_dict": optimizer_m.state_dict(),
        "scheduler_m_state_dict": scheduler_m.state_dict(),
        "model_name": "adda",
        "backbone": config["model"]["backbone"],
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "config": config,
        "class_names": config["data"]["class_names"],
        "model_state_fingerprint": _state_dict_fingerprint(state_dict),
    }


def _save_checkpoint(payload, path: str | Path, backup_dir: Path | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)
    print(
        f"Saved checkpoint {path} "
        f"(epoch={payload.get('epoch')}, best_epoch={payload.get('best_epoch')}, "
        f"best_metric={payload.get('best_metric'):.6f}, fingerprint={payload.get('model_state_fingerprint')})"
    )
    if backup_dir is not None:
        _copy_to_backup(path, backup_dir)


def _write_history_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _copy_to_backup(path: str | Path, backup_dir: Path) -> None:
    path = Path(path)
    if path.exists():
        ensure_dir(backup_dir)
        shutil.copy2(path, backup_dir / path.name)


def _checkpoint_backup_dir(config: dict[str, Any]) -> Path | None:
    value = os.environ.get("ECG_CHECKPOINT_BACKUP_DIR") or config.get("paths", {}).get("checkpoint_backup_dir")
    if value in (None, "", "null", "None"):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config.get("_base_dir", ".")) / path).resolve()


def _copy_config_with_model_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    copied = dict(config)
    copied["model"] = dict(config["model"])
    copied["model"].update({key: value for key, value in overrides.items() if value is not None})
    return copied


def _torch_load_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]
