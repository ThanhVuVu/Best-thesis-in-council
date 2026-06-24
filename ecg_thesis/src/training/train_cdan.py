from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cdan import CDANModel
from src.training.evaluate import predict_model
from src.training.metrics import classification_metrics
from src.training.train import DynamicWeightedFocalLoss, FocalLoss, compute_class_weights
from src.training.train_dann import (
    _count_matching_state_keys,
    _dataset_labels,
    _group_lr,
    _model_kwargs,
    _source_batch_to_device,
    _target_batch_to_device,
    _torch_load_checkpoint,
    dann_lambda,
)
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb, should_log_artifacts


def train_cdan(
    source_train_dataset,
    source_val_dataset,
    target_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    model_cfg = config["model"]
    train_cfg = config["training"]
    cdan_cfg = config["cdan"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    checkpoint_prefix = train_cfg.get("checkpoint_prefix", "cdan")
    wandb_run = init_wandb(
        config,
        job_type="train_cdan",
        default_name=checkpoint_prefix,
        extra_config={"output_dir": str(output_dir), "device": str(device)},
    )
    backup_dir = _checkpoint_backup_dir(config)
    if backup_dir is not None:
        ensure_dir(backup_dir)
        print(f"Checkpoint backup enabled: {backup_dir}")

    model = CDANModel(
        backbone=model_cfg["backbone"],
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
        backbone_kwargs=_model_kwargs(model_cfg),
        reuse_backbone_classifier=bool(model_cfg.get("reuse_backbone_classifier", False)),
        conditioning=str(cdan_cfg.get("conditioning", "auto")),
        randomized_threshold=int(cdan_cfg.get("randomized_threshold", 4096)),
        random_dim=int(cdan_cfg.get("random_dim", 1024)),
        domain_hidden_dim=cdan_cfg.get("domain_hidden_dim"),
    ).to(device)
    _load_source_initialization(model, config, device)

    source_loader = DataLoader(
        source_train_dataset,
        batch_size=int(train_cfg["source_batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=int(train_cfg["target_batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        source_val_dataset,
        batch_size=int(train_cfg["source_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    source_labels = _dataset_labels(source_train_dataset)
    class_weights = compute_class_weights(source_labels).to(device) if train_cfg.get("use_class_weights", True) else None
    source_loss = str(train_cfg.get("source_loss", "weighted_ce")).lower()
    if source_loss == "focal":
        cls_loss_fn = FocalLoss(weight=class_weights, gamma=float(train_cfg.get("focal_gamma", 2.0)))
    elif source_loss in {"dynamic_focal", "dynamic_weighted_focal"}:
        cls_loss_fn = DynamicWeightedFocalLoss(
            num_classes=int(model_cfg["num_classes"]),
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            eps=float(train_cfg.get("dynamic_focal_eps", 0.05)),
        )
    else:
        cls_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    domain_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    optimizer_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_cls(_cdan_param_groups(model, model_cfg, train_cfg))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    total_epochs = int(train_cfg["epochs"])
    steps_per_epoch = max(len(source_loader), len(target_loader))
    total_steps = total_epochs * steps_per_epoch
    best_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    patience = int(train_cfg["early_stopping_patience"])
    history = []
    global_step = 0
    best_path = ckpt_dir / f"{checkpoint_prefix}_best.pt"
    latest_path = ckpt_dir / f"{checkpoint_prefix}_latest.pt"

    for epoch in range(1, total_epochs + 1):
        model.train()
        losses_total = []
        losses_cls = []
        losses_domain = []
        source_entropies = []
        target_entropies = []
        source_true = []
        source_pred = []
        domain_true = []
        domain_pred = []
        source_iter = cycle(source_loader) if len(source_loader) < steps_per_epoch else iter(source_loader)
        target_iter = cycle(target_loader) if len(target_loader) < steps_per_epoch else iter(target_loader)

        progress = tqdm(range(steps_per_epoch), desc=f"cdan epoch {epoch}/{total_epochs}", leave=True, dynamic_ncols=True, mininterval=1.0)
        for _ in progress:
            global_step += 1
            grl_lambda = dann_lambda(global_step, total_steps, cdan_cfg)
            lambda_base = float(cdan_cfg.get("lambda_base", cdan_cfg.get("alpha", 1.0)))
            if epoch <= int(cdan_cfg.get("warmup_epochs", 0)):
                grl_lambda = 0.0
                lambda_base = 0.0

            source_inputs, y_s = _source_batch_to_device(next(source_iter), device)
            target_inputs = _target_batch_to_device(next(target_iter), device)

            optimizer.zero_grad(set_to_none=True)
            f_s = model.extract_features(*source_inputs)
            logits_s = model.label_classifier(f_s)
            f_t = model.extract_features(*target_inputs)
            logits_t = model.label_classifier(f_t)
            loss_cls = cls_loss_fn(logits_s, y_s)

            features_all = torch.cat([f_s, f_t], dim=0)
            logits_all = torch.cat([logits_s, logits_t], dim=0)
            domain_logits = model.forward_domain_from_features(
                features_all,
                logits_all,
                lambd=grl_lambda,
                detach_softmax=bool(cdan_cfg.get("detach_softmax_in_T", True)),
            )
            y_domain = torch.cat(
                [
                    torch.ones(f_s.shape[0], 1, dtype=torch.float32),
                    torch.zeros(f_t.shape[0], 1, dtype=torch.float32),
                ],
                dim=0,
            ).to(device)
            probabilities_all = torch.softmax(logits_all, dim=1)
            entropy_all = _entropy(probabilities_all)
            loss_domain = _domain_loss(
                domain_loss_fn=domain_loss_fn,
                domain_logits=domain_logits,
                domain_labels=y_domain,
                entropy=entropy_all,
                source_batch_size=f_s.shape[0],
                cdan_cfg=cdan_cfg,
            )
            loss = loss_cls + lambda_base * loss_domain
            loss.backward()
            optimizer.step()

            losses_total.append(float(loss.detach().cpu()))
            losses_cls.append(float(loss_cls.detach().cpu()))
            losses_domain.append(float(loss_domain.detach().cpu()))
            source_true.append(y_s.detach().cpu().numpy())
            source_pred.append(logits_s.argmax(dim=1).detach().cpu().numpy())
            domain_true.append(y_domain.detach().cpu().numpy().reshape(-1))
            domain_pred.append((torch.sigmoid(domain_logits) >= 0.5).long().detach().cpu().numpy().reshape(-1))
            source_entropies.append(float(_entropy(torch.softmax(logits_s, dim=1)).mean().detach().cpu()))
            target_entropies.append(float(_entropy(torch.softmax(logits_t, dim=1)).mean().detach().cpu()))
            progress.set_postfix(
                loss=f"{losses_total[-1]:.4f}",
                cls=f"{losses_cls[-1]:.4f}",
                dom=f"{losses_domain[-1]:.4f}",
                lam=f"{grl_lambda:.3f}",
                base=f"{lambda_base:.2f}",
                refresh=False,
            )

        train_metrics = classification_metrics(np.concatenate(source_true), np.concatenate(source_pred))
        domain_acc = float((np.concatenate(domain_true) == np.concatenate(domain_pred)).mean())
        val_result = predict_model(model, val_loader, device, desc=f"cdan val epoch {epoch}")
        val_metrics = val_result["metrics"]
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses_total)),
            "loss_cls": float(np.mean(losses_cls)),
            "loss_domain": float(np.mean(losses_domain)),
            "source_train_accuracy": train_metrics["accuracy"],
            "source_train_macro_f1": train_metrics["macro_f1"],
            "source_val_accuracy": val_metrics["accuracy"],
            "source_val_macro_f1": val_metrics["macro_f1"],
            "domain_accuracy": domain_acc,
            "source_entropy": float(np.mean(source_entropies)),
            "target_entropy": float(np.mean(target_entropies)),
            "lr": _group_lr(optimizer, "classifier", default_index=0),
            "lambda": dann_lambda(global_step, total_steps, cdan_cfg),
            "lambda_base": 0.0
            if epoch <= int(cdan_cfg.get("warmup_epochs", 0))
            else float(cdan_cfg.get("lambda_base", cdan_cfg.get("alpha", 1.0))),
            "method": str(cdan_cfg.get("method", "cdan_e")),
        }
        encoder_lr = _group_lr(optimizer, "encoder", default_index=None)
        domain_lr = _group_lr(optimizer, "domain", default_index=None)
        if encoder_lr is not None:
            row["encoder_lr"] = encoder_lr
        if domain_lr is not None:
            row["domain_lr"] = domain_lr
        history.append(row)
        wandb_run.log({f"train/{key}": value for key, value in row.items() if key != "epoch"}, step=epoch)
        print(
            f"cdan epoch {epoch}/{total_epochs}: loss={row['loss']:.4f}, "
            f"cls={row['loss_cls']:.4f}, dom={row['loss_domain']:.4f}, "
            f"val_f1={row['source_val_macro_f1']:.4f}, domain_acc={row['domain_accuracy']:.4f}, "
            f"target_entropy={row['target_entropy']:.4f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(_payload(model, optimizer, scheduler, config, epoch, best_f1, best_epoch, stale_epochs, history), best_path, backup_dir)
        else:
            stale_epochs += 1

        _save_checkpoint(_payload(model, optimizer, scheduler, config, epoch, best_f1, best_epoch, stale_epochs, history), latest_path, backup_dir)
        if stale_epochs >= patience:
            break

    train_log_name = f"{checkpoint_prefix}_train_log.csv"
    _write_history_csv(history, log_dir / train_log_name)
    if backup_dir is not None:
        _copy_to_backup(log_dir / train_log_name, backup_dir)
    wandb_run.summary_update({"best_epoch": best_epoch, "best_source_val_macro_f1": best_f1})
    if should_log_artifacts(config):
        wandb_run.log_artifact(best_path, name=f"{checkpoint_prefix}_best", artifact_type="model")
    wandb_run.finish()
    return {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "checkpoint_backup_dir": str(backup_dir) if backup_dir is not None else None,
        "best_epoch": best_epoch,
        "best_source_val_macro_f1": best_f1,
        "history": history,
    }


def load_cdan_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    model_kwargs_override: dict[str, Any] | None = None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load_checkpoint(checkpoint_path, device)
    model_cfg = dict(checkpoint["config"]["model"])
    if model_kwargs_override:
        model_cfg.update({key: value for key, value in model_kwargs_override.items() if value is not None})
    cdan_cfg = dict(checkpoint["config"].get("cdan", {}))
    model = CDANModel(
        backbone=model_cfg["backbone"],
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
        backbone_kwargs=_model_kwargs(model_cfg),
        reuse_backbone_classifier=bool(model_cfg.get("reuse_backbone_classifier", False)),
        conditioning=str(cdan_cfg.get("conditioning", "auto")),
        randomized_threshold=int(cdan_cfg.get("randomized_threshold", 4096)),
        random_dim=int(cdan_cfg.get("random_dim", 1024)),
        domain_hidden_dim=cdan_cfg.get("domain_hidden_dim"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        "Loaded CDAN checkpoint:",
        {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_metric": checkpoint.get("best_metric"),
            "fingerprint": checkpoint.get("model_state_fingerprint"),
        },
    )
    return model, checkpoint


def _domain_loss(
    domain_loss_fn: torch.nn.Module,
    domain_logits: torch.Tensor,
    domain_labels: torch.Tensor,
    entropy: torch.Tensor,
    source_batch_size: int,
    cdan_cfg: dict[str, Any],
) -> torch.Tensor:
    loss_each = domain_loss_fn(domain_logits, domain_labels)
    method = str(cdan_cfg.get("method", "cdan_e")).lower()
    if method in {"cdan_e", "cdan+e", "cdane"}:
        weights = 1.0 + torch.exp(-entropy.detach())
        weights = weights.view(-1, 1)
        if bool(cdan_cfg.get("normalize_entropy_weights", True)):
            weights = weights.clone()
            source_weights = weights[:source_batch_size]
            target_weights = weights[source_batch_size:]
            weights[:source_batch_size] = source_weights / source_weights.mean().clamp_min(1e-6)
            weights[source_batch_size:] = target_weights / target_weights.mean().clamp_min(1e-6)
        return (weights * loss_each).mean()
    if method == "cdan":
        return loss_each.mean()
    raise ValueError(f"Unsupported CDAN method: {method!r}")


def _entropy(probabilities: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities + eps)).sum(dim=1)


def _load_source_initialization(model: CDANModel, config: dict[str, Any], device: torch.device) -> None:
    checkpoint_value = config.get("cdan", {}).get("source_init_checkpoint")
    require_checkpoint = bool(config.get("cdan", {}).get("require_source_init_checkpoint", False))
    if checkpoint_value in (None, "", "null", "None"):
        if require_checkpoint:
            raise ValueError("CDAN source initialization checkpoint is required but not configured")
        return

    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(config.get("_base_dir", ".")) / checkpoint_path
    if not checkpoint_path.exists():
        if require_checkpoint:
            raise FileNotFoundError(f"Required CDAN source initialization checkpoint not found: {checkpoint_path}")
        print(f"Source initialization checkpoint not found, training CDAN from scratch: {checkpoint_path}")
        return

    checkpoint = _torch_load_checkpoint(checkpoint_path, device)
    expected_model = config.get("cdan", {}).get("source_init_expected_model")
    checkpoint_model = checkpoint.get("model_name") if isinstance(checkpoint, dict) else None
    if expected_model and checkpoint_model != expected_model:
        message = (
            f"CDAN source initialization checkpoint model mismatch: "
            f"expected {expected_model!r}, got {checkpoint_model!r} at {checkpoint_path}"
        )
        if require_checkpoint:
            raise ValueError(message)
        print(message)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    matched_keys = _count_matching_state_keys(model.feature_extractor, state_dict)
    if require_checkpoint and matched_keys == 0:
        raise ValueError(f"No matching feature extractor keys found in source checkpoint: {checkpoint_path}")
    missing, unexpected = model.feature_extractor.load_state_dict(state_dict, strict=False)
    copied_classifier = _copy_source_classifier(model)
    print(
        "Initialized CDAN feature extractor from source-only checkpoint:",
        {
            "path": str(checkpoint_path),
            "model_name": checkpoint_model,
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "best_epoch": checkpoint.get("best_epoch") if isinstance(checkpoint, dict) else None,
            "matched_keys": matched_keys,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "copied_classifier": copied_classifier,
            "reused_backbone_classifier": bool(getattr(model, "reuse_backbone_classifier", False)),
        },
    )


def _copy_source_classifier(model: CDANModel) -> bool:
    source_classifier = getattr(model.feature_extractor, "classifier", None)
    if source_classifier is None:
        return False
    if source_classifier is model.label_classifier:
        return True
    try:
        model.label_classifier.load_state_dict(source_classifier.state_dict())
    except RuntimeError:
        return False
    return True


def _cdan_param_groups(model: CDANModel, model_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    if str(model_cfg.get("backbone", "")).lower() != "clef_pretrained":
        return [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay, "name": "all"}]

    seen: set[int] = set()
    groups: list[dict[str, Any]] = []
    classifier_params = _unique_trainable_params(model.label_classifier.parameters(), seen)
    encoder = getattr(model.feature_extractor, "encoder", None)
    encoder_params = _unique_trainable_params(encoder.parameters(), seen) if encoder is not None else []
    domain_params = _unique_trainable_params(model.domain_classifier.parameters(), seen)

    if classifier_params:
        groups.append({"params": classifier_params, "lr": lr, "weight_decay": weight_decay, "name": "classifier"})
    if encoder_params:
        groups.append(
            {
                "params": encoder_params,
                "lr": float(model_cfg.get("encoder_lr", train_cfg.get("encoder_lr", lr))),
                "weight_decay": weight_decay,
                "name": "encoder",
            }
        )
    if domain_params:
        groups.append(
            {
                "params": domain_params,
                "lr": float(train_cfg.get("domain_lr", lr)),
                "weight_decay": weight_decay,
                "name": "domain",
            }
        )
    if not groups:
        raise ValueError("No trainable parameters found for CLEF-CDAN")
    print(
        "CDAN optimizer param groups:",
        [
            {
                "name": group["name"],
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "params": len(group["params"]),
            }
            for group in groups
        ],
    )
    return groups


def _unique_trainable_params(parameters, seen: set[int]) -> list[torch.nn.Parameter]:
    unique = []
    for param in parameters:
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        unique.append(param)
    return unique


def _payload(model, optimizer, scheduler, config, epoch, best_metric, best_epoch, stale_epochs, history):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_name": "cdan",
        "backbone": config["model"]["backbone"],
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "config": config,
        "class_names": config["data"]["class_names"],
        "model_state_fingerprint": _state_dict_fingerprint(model.state_dict()),
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


def _state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]
