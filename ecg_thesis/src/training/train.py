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

from src.models import build_model
from src.training.evaluate import predict_model
from src.training.metrics import classification_metrics
from src.utils.io import ensure_dir


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_source_only(
    train_dataset,
    val_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    train_cfg = config["training"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")

    model = build_model(train_cfg["model"], num_classes=3, **train_cfg.get("model_kwargs", {})).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    labels = _dataset_labels(train_dataset)
    if train_cfg.get("use_class_weights", True):
        class_weights = compute_class_weights(labels).to(device)
    else:
        class_weights = None
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_f1 = -1.0
    best_epoch = -1
    patience = int(train_cfg["early_stopping_patience"])
    stale_epochs = 0
    history = []
    checkpoint_prefix = train_cfg.get("checkpoint_prefix", "")
    if checkpoint_prefix:
        best_path = ckpt_dir / f"{checkpoint_prefix}_best.pt"
        latest_path = ckpt_dir / f"{checkpoint_prefix}_latest.pt"
    else:
        best_path = ckpt_dir / "best.pt"
        latest_path = ckpt_dir / "latest.pt"
    backup_dir = _checkpoint_backup_dir(config)
    if backup_dir is not None:
        ensure_dir(backup_dir)
        print(f"Checkpoint backup enabled: {backup_dir}")

    total_epochs = int(train_cfg["epochs"])
    for epoch in range(1, total_epochs + 1):
        should_stop = False
        model.train()
        losses = []
        train_true = []
        train_pred = []
        batch_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{total_epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for x, y in batch_bar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
            batch_bar.set_postfix(loss=f"{losses[-1]:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", refresh=False)

        train_metrics = classification_metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_result = predict_model(model, val_loader, device, desc=f"val epoch {epoch}")
        val_metrics = val_result["metrics"]
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{total_epochs}: "
            f"loss={row['loss']:.4f}, train_f1={row['train_macro_f1']:.4f}, "
            f"val_f1={row['val_macro_f1']:.4f}, lr={row['lr']:.2e}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            best_payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                model_name=train_cfg["model"],
                epoch=epoch,
                best_f1=best_f1,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
            )
            _save_checkpoint(best_payload, best_path, backup_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                should_stop = True

        latest_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            model_name=train_cfg["model"],
            epoch=epoch,
            best_f1=best_f1,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            history=history,
        )
        _save_checkpoint(latest_payload, latest_path, backup_dir)
        if should_stop:
            break

    _write_history_csv(history, log_dir / "train_log.csv")
    if backup_dir is not None:
        _copy_to_backup(log_dir / "train_log.csv", backup_dir)
    return {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "checkpoint_backup_dir": str(backup_dir) if backup_dir is not None else None,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
    }


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_name"], num_classes=3, **_checkpoint_model_kwargs(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    fingerprint = checkpoint.get("model_state_fingerprint") or _state_dict_fingerprint(checkpoint["model_state_dict"])
    print(
        "Loaded checkpoint:",
        {
            "path": str(checkpoint_path),
            "model_name": checkpoint.get("model_name"),
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_macro_f1": checkpoint.get("best_macro_f1"),
            "fingerprint": fingerprint,
        },
    )
    return model, checkpoint


def _checkpoint_model_kwargs(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    training_kwargs = config.get("training", {}).get("model_kwargs")
    if training_kwargs:
        return dict(training_kwargs)
    model_cfg = config.get("model", {})
    allowed = {
        "d_model",
        "num_heads",
        "dff",
        "num_transformer_layers",
        "attention_reduction",
        "dropout",
    }
    return {key: model_cfg[key] for key in allowed if key in model_cfg}


def _dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_labels = _dataset_labels(dataset.dataset)
        return parent_labels[np.asarray(dataset.indices)]
    return dataset.y


def _write_history_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    model_name: str,
    epoch: int,
    best_f1: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_name": model_name,
        "epoch": epoch,
        "best_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "config": config,
        "model_state_fingerprint": _state_dict_fingerprint(model.state_dict()),
    }


def _save_checkpoint(payload: dict[str, Any], path: str | Path, backup_dir: Path | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)
    print(
        f"Saved checkpoint {path} "
        f"(epoch={payload.get('epoch')}, best_epoch={payload.get('best_epoch')}, "
        f"best_macro_f1={payload.get('best_macro_f1'):.6f}, "
        f"fingerprint={payload.get('model_state_fingerprint')})"
    )
    if backup_dir is not None:
        _copy_to_backup(path, backup_dir)


def _copy_to_backup(path: str | Path, backup_dir: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    ensure_dir(backup_dir)
    shutil.copy2(path, backup_dir / path.name)


def _checkpoint_backup_dir(config: dict[str, Any]) -> Path | None:
    env_value = os.environ.get("ECG_PHASE2_CHECKPOINT_BACKUP_DIR") or os.environ.get("ECG_PHASE1_CHECKPOINT_BACKUP_DIR")
    config_value = config.get("paths", {}).get("checkpoint_backup_dir")
    value = env_value or config_value
    if value in (None, "", "null", "None"):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    base_dir = Path(config.get("_base_dir", "."))
    return (base_dir / path).resolve()


def _state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]
