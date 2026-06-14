from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset
from src.training.daeac_losses import weighted_cross_entropy_from_logits
from src.training.mcc_loss import minimum_class_confusion_loss
from src.training.train_daeac_paper import (
    _class_weights,
    build_daeac_model,
    evaluate_daeac_model,
    load_daeac_checkpoint,
    save_daeac_checkpoint,
)
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb


def train_daeac_mcc(
    source_dataset: DAEACDataset,
    val_dataset: DAEACDataset,
    target_dataset: DAEACTargetUnlabeledDataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = config["adaptation"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    prefix = str(cfg.get("checkpoint_prefix", "daeac_mcc"))
    model = build_daeac_model(config, device)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        load_daeac_checkpoint(init_checkpoint, config, device, model=model)

    source_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=True, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    class_weights = _class_weights(source_dataset, config, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    wandb_run = init_wandb(config, job_type="train_daeac_mcc", default_name=prefix)

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    best_macro_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        target_iter = _cycle(target_loader)
        epoch_rows: list[dict[str, float]] = []
        pred_counts = np.zeros(int(config["data"]["num_classes"]), dtype=np.int64)
        for x_s, y_s in source_loader:
            x_t = _batch_x(next(target_iter))
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            _, logits_s, _ = model(x_s, return_logits=True)
            _, logits_t, _ = model(x_t, return_logits=True)
            loss_cls = weighted_cross_entropy_from_logits(logits_s, y_s, class_weights)
            loss_mcc, diagnostics = minimum_class_confusion_loss(
                logits_t,
                temperature=float(cfg["mcc"]["temperature"]),
                return_diagnostics=True,
            )
            loss_total = loss_cls + float(cfg["mcc"]["mu"]) * loss_mcc

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            pred_counts += np.asarray(diagnostics["pred_counts"], dtype=np.int64)
            row = {
                "loss": float(loss_total.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_mcc": float(loss_mcc.detach().cpu()),
                "target_entropy": float(diagnostics["entropy_mean"]),
            }
            row.update(_soft_confusion_entries(diagnostics["soft_confusion"], config["data"]["class_names"]))
            epoch_rows.append(row)

        val_result = evaluate_daeac_model(model, val_loader, device, config["data"]["class_names"])
        row = _epoch_summary(epoch_rows)
        row.update(
            {
                "epoch": epoch,
                "val_accuracy": val_result["metrics"]["accuracy"],
                "val_macro_f1": val_result["metrics"]["macro_f1"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "target_pred_counts": pred_counts.astype(int).tolist(),
            }
        )
        history.append(row)
        log_row = {f"mcc/{k}": v for k, v in row.items() if k not in {"epoch", "target_pred_counts"}}
        for idx, count in enumerate(row["target_pred_counts"]):
            log_row[f"mcc/target_pred_count_{idx}"] = count
        wandb_run.log(log_row, step=epoch)
        if row["val_macro_f1"] >= best_macro_f1:
            best_macro_f1 = float(row["val_macro_f1"])
            best_epoch = epoch
            save_daeac_checkpoint(model, config, best_path, epoch, row)
        save_daeac_checkpoint(model, config, latest_path, epoch, row)
        print(
            f"[mcc epoch {epoch + 1}/{cfg['epochs']}] loss={row['loss']:.4f} "
            f"cls={row['loss_cls']:.4f} mcc={row['loss_mcc']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} target_pred={row['target_pred_counts']}"
        )

    summary = {
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "history": history,
    }
    wandb_run.summary_update(summary)
    wandb_run.finish()
    return summary


def _soft_confusion_entries(matrix: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    cpu = matrix.detach().cpu()
    for i, source_name in enumerate(class_names):
        for j, target_name in enumerate(class_names):
            if i == j:
                continue
            values[f"mcc_{source_name}_{target_name}"] = float(cpu[i, j])
    return values


def _batch_x(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _epoch_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
