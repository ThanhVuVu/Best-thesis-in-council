from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

    model = build_model(train_cfg["model"], num_classes=3).to(device)
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
    best_path = ckpt_dir / "best.pt"

    epoch_bar = tqdm(range(1, int(train_cfg["epochs"]) + 1), desc="epochs")
    for epoch in epoch_bar:
        model.train()
        losses = []
        train_true = []
        train_pred = []
        batch_bar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
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
            batch_bar.set_postfix(loss=f"{losses[-1]:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

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
        print(row)
        epoch_bar.set_postfix(
            train_f1=f"{train_metrics['macro_f1']:.4f}",
            val_f1=f"{val_metrics['macro_f1']:.4f}",
            best_f1=f"{max(best_f1, val_metrics['macro_f1']):.4f}",
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": train_cfg["model"],
                "epoch": epoch,
                "best_macro_f1": best_f1,
                "config": config,
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    _write_history_csv(history, log_dir / "train_log.csv")
    return {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
    }


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_name"], num_classes=3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def _dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return dataset.dataset.y[np.asarray(dataset.indices)]
    return dataset.y


def _write_history_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
