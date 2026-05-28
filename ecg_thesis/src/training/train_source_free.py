from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.train import load_model_from_checkpoint
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb, should_log_artifacts


def train_source_free(
    target_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
    init_checkpoint: str | Path | None = None,
    unfreeze_top_ecgfm_layers: int = 0,
    model_kwargs_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapt_cfg = config["source_free"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    checkpoint_prefix = adapt_cfg.get("checkpoint_prefix", "sourcefree_ecgfm_leadbridge")
    wandb_run = init_wandb(
        config,
        job_type="train_source_free",
        default_name=checkpoint_prefix,
        extra_config={"output_dir": str(output_dir), "device": str(device)},
    )
    backup_dir = _checkpoint_backup_dir(config)
    if backup_dir is not None:
        ensure_dir(backup_dir)
        print(f"Checkpoint backup enabled: {backup_dir}")

    init_path = _resolve_checkpoint(init_checkpoint or adapt_cfg["init_checkpoint"], config)
    model, init_payload = load_model_from_checkpoint(init_path, device, model_kwargs_override=model_kwargs_override)
    trainable_summary = configure_source_free_trainable_layers(model, unfreeze_top_ecgfm_layers)
    print("Source-free trainable layers:", trainable_summary, flush=True)

    loader = DataLoader(
        target_dataset,
        batch_size=int(adapt_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = _build_optimizer(model, adapt_cfg, unfreeze_top_ecgfm_layers)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_path = ckpt_dir / f"{checkpoint_prefix}_best.pt"
    latest_path = ckpt_dir / f"{checkpoint_prefix}_latest.pt"
    total_epochs = int(adapt_cfg.get("epochs", 10))
    patience = int(adapt_cfg.get("early_stopping_patience", 3))
    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, total_epochs + 1):
        model.train()
        _set_source_free_train_mode(model, unfreeze_top_ecgfm_layers)
        rows = []
        progress = tqdm(loader, desc=f"source-free epoch {epoch}/{total_epochs}", leave=True, dynamic_ncols=True, mininterval=1.0)
        for batch in progress:
            inputs = _target_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss, parts = source_free_loss(logits, adapt_cfg)
            loss.backward()
            optimizer.step()
            row = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in parts.items()}
            row["loss"] = float(loss.detach().cpu())
            rows.append(row)
            progress.set_postfix(
                loss=f"{row['loss']:.4f}",
                ent=f"{row['entropy_loss']:.4f}",
                pseudo=f"{row['pseudo_loss']:.4f}",
                cov=f"{row['pseudo_coverage']:.2f}",
                refresh=False,
            )

        epoch_row = _mean_rows(rows)
        epoch_row.update(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "ecgfm_lr": float(optimizer.param_groups[-1]["lr"]) if len(optimizer.param_groups) > 1 else 0.0,
            }
        )
        history.append(epoch_row)
        wandb_run.log({f"train/{key}": value for key, value in epoch_row.items() if key != "epoch"}, step=epoch)
        scheduler.step(epoch_row["loss"])
        print(
            f"source-free epoch {epoch}/{total_epochs}: loss={epoch_row['loss']:.4f}, "
            f"entropy={epoch_row['entropy_loss']:.4f}, pseudo={epoch_row['pseudo_loss']:.4f}, "
            f"balance={epoch_row['balance_loss']:.4f}, coverage={epoch_row['pseudo_coverage']:.4f}",
            flush=True,
        )

        if epoch_row["loss"] < best_loss:
            best_loss = float(epoch_row["loss"])
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(
                _payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    init_payload=init_payload,
                    epoch=epoch,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    history=history,
                    trainable_summary=trainable_summary,
                ),
                best_path,
                backup_dir,
            )
        else:
            stale_epochs += 1

        _save_checkpoint(
            _payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                init_payload=init_payload,
                epoch=epoch,
                best_loss=best_loss,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
                trainable_summary=trainable_summary,
            ),
            latest_path,
            backup_dir,
        )
        if stale_epochs >= patience:
            break

    train_log_name = f"{checkpoint_prefix}_train_log.csv"
    _write_history_csv(history, log_dir / train_log_name)
    if backup_dir is not None:
        _copy_to_backup(log_dir / train_log_name, backup_dir)
    wandb_run.summary_update({"best_epoch": best_epoch, "best_adaptation_loss": best_loss})
    if should_log_artifacts(config):
        wandb_run.log_artifact(best_path, name=f"{checkpoint_prefix}_best", artifact_type="model")
    wandb_run.finish()
    return {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "checkpoint_backup_dir": str(backup_dir) if backup_dir is not None else None,
        "init_checkpoint": str(init_path),
        "best_epoch": best_epoch,
        "best_adaptation_loss": best_loss,
        "trainable_summary": trainable_summary,
        "history": history,
    }


def source_free_loss(logits: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    entropy_loss = -(probs * log_probs).sum(dim=1).mean()

    confidence, pseudo_labels = probs.max(dim=1)
    pseudo_mask = confidence >= float(config.get("pseudo_threshold", 0.90))
    if bool(pseudo_mask.any()):
        pseudo_loss = torch.nn.functional.cross_entropy(logits[pseudo_mask], pseudo_labels[pseudo_mask])
    else:
        pseudo_loss = logits.sum() * 0.0

    mean_probs = probs.mean(dim=0).clamp_min(1e-8)
    uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
    balance_loss = (mean_probs * (mean_probs.log() - uniform.log())).sum()

    loss = (
        float(config.get("pseudo_weight", 1.0)) * pseudo_loss
        + float(config.get("entropy_weight", 0.05)) * entropy_loss
        + float(config.get("balance_weight", 0.10)) * balance_loss
    )
    return loss, {
        "entropy_loss": entropy_loss,
        "pseudo_loss": pseudo_loss,
        "balance_loss": balance_loss,
        "pseudo_coverage": pseudo_mask.float().mean(),
        "mean_confidence": confidence.mean(),
        "class_prob_min": mean_probs.min(),
        "class_prob_max": mean_probs.max(),
    }


def configure_source_free_trainable_layers(model: torch.nn.Module, unfreeze_top_ecgfm_layers: int = 0) -> dict[str, Any]:
    for param in model.parameters():
        param.requires_grad = False
    _set_module_trainable(getattr(model, "lead_bridge"), True)
    _set_module_trainable(getattr(model, "classifier"), True)

    unfrozen_layer_names: list[str] = []
    if unfreeze_top_ecgfm_layers > 0:
        encoder_wrapper = getattr(model, "encoder", None)
        if encoder_wrapper is not None and hasattr(encoder_wrapper, "freeze"):
            encoder_wrapper.freeze = False
        layers, base_name = _find_ecgfm_encoder_layers(model)
        if len(layers) < unfreeze_top_ecgfm_layers:
            raise ValueError(
                f"Requested top {unfreeze_top_ecgfm_layers} ECG-FM layers, but only found {len(layers)} layers at {base_name}."
            )
        for idx, layer in list(enumerate(layers))[-int(unfreeze_top_ecgfm_layers):]:
            _set_module_trainable(layer, True)
            unfrozen_layer_names.append(f"{base_name}.{idx}")

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    return {
        "unfreeze_top_ecgfm_layers": int(unfreeze_top_ecgfm_layers),
        "unfrozen_ecgfm_layer_names": unfrozen_layer_names,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
    }


def _set_source_free_train_mode(model: torch.nn.Module, unfreeze_top_ecgfm_layers: int) -> None:
    if unfreeze_top_ecgfm_layers <= 0:
        return
    ecgfm = getattr(getattr(model, "encoder", None), "ecgfm", None)
    if ecgfm is not None:
        ecgfm.eval()
    layers, _base_name = _find_ecgfm_encoder_layers(model)
    for layer in list(layers)[-int(unfreeze_top_ecgfm_layers):]:
        layer.train()


def _find_ecgfm_encoder_layers(model: torch.nn.Module) -> tuple[torch.nn.ModuleList | list[torch.nn.Module], str]:
    ecgfm = getattr(getattr(model, "encoder", None), "ecgfm", None)
    direct_layers = getattr(getattr(ecgfm, "encoder", None), "layers", None)
    if _is_layer_sequence(direct_layers):
        return direct_layers, "encoder.ecgfm.encoder.layers"

    for name, module in model.named_modules():
        if name.endswith("encoder.layers") and _is_layer_sequence(module):
            return module, name
        layers = getattr(module, "layers", None)
        if name.endswith("encoder") and _is_layer_sequence(layers):
            return layers, f"{name}.layers"
    raise AttributeError("Could not find ECG-FM encoder layers. Expected model.encoder.ecgfm.encoder.layers.")


def _is_layer_sequence(value: Any) -> bool:
    return isinstance(value, (torch.nn.ModuleList, list, tuple)) and len(value) > 0


def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any], unfreeze_top_ecgfm_layers: int) -> torch.optim.Optimizer:
    base_params = []
    ecgfm_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if unfreeze_top_ecgfm_layers > 0 and ".ecgfm." in name:
            ecgfm_params.append(param)
        else:
            base_params.append(param)
    param_groups = [
        {
            "params": base_params,
            "lr": float(config.get("lr", 1e-4)),
            "weight_decay": float(config.get("weight_decay", 1e-4)),
        }
    ]
    if ecgfm_params:
        param_groups.append(
            {
                "params": ecgfm_params,
                "lr": float(config.get("ecgfm_lr", 1e-5)),
                "weight_decay": float(config.get("ecgfm_weight_decay", config.get("weight_decay", 1e-4))),
            }
        )
    return torch.optim.AdamW(param_groups)


def _target_batch_to_device(batch, device: torch.device) -> tuple[torch.Tensor, ...]:
    if len(batch) == 2:
        x, _y = batch
        return (x.to(device, non_blocking=True),)
    if len(batch) == 3:
        x, rr, _y = batch
        return (x.to(device, non_blocking=True), rr.to(device, non_blocking=True))
    if len(batch) == 4:
        x, rr, _y, _meta = batch
        return (x.to(device, non_blocking=True), rr.to(device, non_blocking=True))
    raise ValueError(f"Unsupported target batch length: {len(batch)}")


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _resolve_checkpoint(path: str | Path, config: dict[str, Any]) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_absolute():
        return checkpoint
    return (Path(config.get("_base_dir", ".")) / checkpoint).resolve()


def _payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    init_payload: dict[str, Any],
    epoch: int,
    best_loss: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    trainable_summary: dict[str, Any],
) -> dict[str, Any]:
    model_name = init_payload.get("model_name") or config.get("source_free", {}).get("model", "ecgfm_leadbridge")
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_name": model_name,
        "epoch": epoch,
        "best_adaptation_loss": best_loss,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "config": config,
        "class_names": config["data"]["class_names"],
        "source_free_init_checkpoint": config["source_free"]["init_checkpoint"],
        "trainable_summary": trainable_summary,
        "model_state_fingerprint": _state_dict_fingerprint(model.state_dict()),
    }


def _save_checkpoint(payload: dict[str, Any], path: str | Path, backup_dir: Path | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)
    print(
        f"Saved checkpoint {path} "
        f"(epoch={payload.get('epoch')}, best_epoch={payload.get('best_epoch')}, "
        f"best_adaptation_loss={payload.get('best_adaptation_loss'):.6f}, "
        f"fingerprint={payload.get('model_state_fingerprint')})"
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
    value = os.environ.get("ECG_PHASE4_CHECKPOINT_BACKUP_DIR") or config.get("paths", {}).get("checkpoint_backup_dir")
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
