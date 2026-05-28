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

from src.models.dann import DANNModel
from src.training.evaluate import predict_model
from src.training.metrics import classification_metrics
from src.training.train import compute_class_weights
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb, should_log_artifacts


def train_dann(
    source_train_dataset,
    source_val_dataset,
    target_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    model_cfg = config["model"]
    train_cfg = config["training"]
    dann_cfg = config["dann"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    checkpoint_prefix = train_cfg.get("checkpoint_prefix", "dann")
    wandb_run = init_wandb(
        config,
        job_type="train_dann",
        default_name=checkpoint_prefix,
        extra_config={"output_dir": str(output_dir), "device": str(device)},
    )
    backup_dir = _checkpoint_backup_dir(config)
    if backup_dir is not None:
        ensure_dir(backup_dir)
        print(f"Checkpoint backup enabled: {backup_dir}")

    model = DANNModel(
        backbone=model_cfg["backbone"],
        num_classes=int(model_cfg["num_classes"]),
        num_domains=int(model_cfg["num_domains"]),
        dropout=float(model_cfg["dropout"]),
        backbone_kwargs=_model_kwargs(model_cfg),
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
    if train_cfg.get("source_loss", "weighted_ce") == "focal":
        cls_loss_fn = FocalLoss(weight=class_weights, gamma=float(train_cfg.get("focal_gamma", 2.0)))
    else:
        cls_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    domain_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
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
        source_true = []
        source_pred = []
        domain_true = []
        domain_pred = []
        source_iter = cycle(source_loader) if len(source_loader) < steps_per_epoch else iter(source_loader)
        target_iter = cycle(target_loader) if len(target_loader) < steps_per_epoch else iter(target_loader)

        progress = tqdm(range(steps_per_epoch), desc=f"dann epoch {epoch}/{total_epochs}", leave=True, dynamic_ncols=True, mininterval=1.0)
        for _ in progress:
            global_step += 1
            lambd = dann_lambda(global_step, total_steps, dann_cfg)
            effective_alpha = float(dann_cfg["alpha"])
            if epoch <= int(dann_cfg.get("warmup_epochs", 0)):
                lambd = 0.0
                effective_alpha = 0.0
            source_inputs, y_s = _source_batch_to_device(next(source_iter), device)
            target_inputs = _target_batch_to_device(next(target_iter), device)
            domain_inputs = tuple(torch.cat([src, tgt], dim=0) for src, tgt in zip(source_inputs, target_inputs))
            y_domain = torch.cat([
                torch.zeros(source_inputs[0].shape[0], dtype=torch.long),
                torch.ones(target_inputs[0].shape[0], dtype=torch.long),
            ]).to(device)

            optimizer.zero_grad(set_to_none=True)
            class_logits = model(*source_inputs)
            domain_logits = model.forward_domain(*domain_inputs, lambd=lambd)
            loss_cls = cls_loss_fn(class_logits, y_s)
            loss_domain = domain_loss_fn(domain_logits, y_domain)
            loss = loss_cls + effective_alpha * loss_domain
            loss.backward()
            optimizer.step()

            losses_total.append(float(loss.detach().cpu()))
            losses_cls.append(float(loss_cls.detach().cpu()))
            losses_domain.append(float(loss_domain.detach().cpu()))
            source_true.append(y_s.detach().cpu().numpy())
            source_pred.append(class_logits.argmax(dim=1).detach().cpu().numpy())
            domain_true.append(y_domain.detach().cpu().numpy())
            domain_pred.append(domain_logits.argmax(dim=1).detach().cpu().numpy())
            progress.set_postfix(
                loss=f"{losses_total[-1]:.4f}",
                cls=f"{losses_cls[-1]:.4f}",
                dom=f"{losses_domain[-1]:.4f}",
                lam=f"{lambd:.3f}",
                alpha=f"{effective_alpha:.2f}",
                refresh=False,
            )

        train_metrics = classification_metrics(np.concatenate(source_true), np.concatenate(source_pred))
        domain_acc = float((np.concatenate(domain_true) == np.concatenate(domain_pred)).mean())
        val_result = predict_model(model, val_loader, device, desc=f"dann val epoch {epoch}")
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
            "lr": optimizer.param_groups[0]["lr"],
            "lambda": dann_lambda(global_step, total_steps, dann_cfg),
            "alpha": 0.0 if epoch <= int(dann_cfg.get("warmup_epochs", 0)) else float(dann_cfg["alpha"]),
        }
        history.append(row)
        wandb_run.log({f"train/{key}": value for key, value in row.items() if key != "epoch"}, step=epoch)
        print(
            f"dann epoch {epoch}/{total_epochs}: loss={row['loss']:.4f}, "
            f"cls={row['loss_cls']:.4f}, dom={row['loss_domain']:.4f}, "
            f"val_f1={row['source_val_macro_f1']:.4f}, domain_acc={row['domain_accuracy']:.4f}",
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


def dann_lambda(step: int, total_steps: int, config: dict[str, Any]) -> float:
    if config.get("lambda_schedule") == "fixed":
        return float(config.get("fixed_lambda", 1.0))
    p = min(max(step / max(total_steps, 1), 0.0), 1.0)
    gamma = float(config.get("gamma", 10.0))
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def _load_source_initialization(model: DANNModel, config: dict[str, Any], device: torch.device) -> None:
    checkpoint_value = config.get("dann", {}).get("source_init_checkpoint")
    if checkpoint_value in (None, "", "null", "None"):
        return

    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(config.get("_base_dir", ".")) / checkpoint_path
    if not checkpoint_path.exists():
        print(f"Source initialization checkpoint not found, training DANN from scratch: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.feature_extractor.load_state_dict(state_dict, strict=False)
    copied_classifier = _copy_source_classifier(model)
    print(
        "Initialized DANN feature extractor from source-only checkpoint:",
        {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "copied_classifier": copied_classifier,
        },
    )


def _copy_source_classifier(model: DANNModel) -> bool:
    source_classifier = getattr(model.feature_extractor, "classifier", None)
    if source_classifier is None:
        return False
    try:
        model.label_classifier.load_state_dict(source_classifier.state_dict())
    except RuntimeError:
        return False
    return True


class FocalLoss(torch.nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def load_dann_from_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["config"]["model"]
    model = DANNModel(
        backbone=model_cfg["backbone"],
        num_classes=int(model_cfg["num_classes"]),
        num_domains=int(model_cfg["num_domains"]),
        dropout=float(model_cfg["dropout"]),
        backbone_kwargs=_model_kwargs(model_cfg),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        "Loaded DANN checkpoint:",
        {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_metric": checkpoint.get("best_metric"),
            "fingerprint": checkpoint.get("model_state_fingerprint"),
        },
    )
    return model, checkpoint


def _dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_labels = _dataset_labels(dataset.dataset)
        return parent_labels[np.asarray(dataset.indices)]
    return dataset.y


def _model_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "d_model",
        "num_heads",
        "dff",
        "num_transformer_layers",
        "attention_reduction",
        "dropout",
        "rr_feature_dim",
        "rr_embedding_dim",
    }
    return {key: model_cfg[key] for key in allowed if key in model_cfg}


def _source_batch_to_device(batch, device: torch.device) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    if len(batch) == 2:
        x, y = batch
        return (x.to(device, non_blocking=True),), y.to(device, non_blocking=True)
    if len(batch) == 3:
        x, rr, y = batch
        return (x.to(device, non_blocking=True), rr.to(device, non_blocking=True)), y.to(device, non_blocking=True)
    if len(batch) == 4:
        x, rr, y, _meta = batch
        return (x.to(device, non_blocking=True), rr.to(device, non_blocking=True)), y.to(device, non_blocking=True)
    raise ValueError(f"Unsupported source batch length: {len(batch)}")


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


def _payload(model, optimizer, scheduler, config, epoch, best_metric, best_epoch, stale_epochs, history):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_name": "dann",
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
    value = os.environ.get("ECG_PHASE2_CHECKPOINT_BACKUP_DIR") or config.get("paths", {}).get("checkpoint_backup_dir")
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
