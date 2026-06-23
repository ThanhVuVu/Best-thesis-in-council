from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset
from src.training.daeac_losses import weighted_cross_entropy_from_logits
from src.training.dev_validation import estimate_dev_risk
from src.training.mcc_loss import minimum_class_confusion_loss
from src.training.train_daeac_paper import (
    _class_weights,
    build_daeac_model,
    load_daeac_checkpoint,
    save_daeac_checkpoint,
)
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb


def train_daeac_mcc(
    source_dataset: DAEACDataset,
    source_val_dataset: DAEACDataset,
    target_dataset: DAEACTargetUnlabeledDataset,
    dev_target_dataset: DAEACTargetUnlabeledDataset,
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
    dev_batch_size = int(cfg.get("dev", {}).get("feature_batch_size", cfg["target_batch_size"]))
    dev_source_loader = DataLoader(source_dataset, batch_size=dev_batch_size, shuffle=False, num_workers=0)
    dev_val_loader = DataLoader(source_val_dataset, batch_size=dev_batch_size, shuffle=False, num_workers=0)
    dev_target_loader = DataLoader(dev_target_dataset, batch_size=dev_batch_size, shuffle=False, num_workers=0)
    class_weights = _class_weights(source_dataset, config, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    wandb_run = init_wandb(config, job_type="train_daeac_mcc", default_name=prefix)

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_dev_path = ckpt_dir / f"{prefix}_best_dev.pt"
    best_dev_risk = float("inf")
    best_dev_epoch = -1
    dev_cfg = dict(cfg.get("dev", {}))
    dev_interval = int(dev_cfg.get("interval_epochs", 1))
    history: list[dict[str, Any]] = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        source_iter = _cycle(source_loader)
        epoch_rows: list[dict[str, float]] = []
        pred_counts = np.zeros(int(config["data"]["num_classes"]), dtype=np.int64)
        source_samples_seen = 0
        target_samples_seen = 0
        # Target-driven epoch: every target sample is consumed exactly once.
        # Source batches are cycled only when the labeled source is smaller.
        for target_batch in target_loader:
            x_s, y_s = next(source_iter)
            x_t = _batch_x(target_batch)
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            # A single mixed-domain forward gives shared BatchNorm layers one
            # balanced view of source and target instead of updating their
            # running statistics sequentially with target always last.
            mixed_x = torch.cat((x_s, x_t), dim=0)
            _, mixed_logits, _ = model(mixed_x, return_logits=True)
            logits_s, logits_t = torch.split(mixed_logits, (x_s.size(0), x_t.size(0)), dim=0)
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

            source_samples_seen += int(x_s.size(0))
            target_samples_seen += int(x_t.size(0))
            pred_counts += np.asarray(diagnostics["pred_counts"], dtype=np.int64)
            row = {
                "loss": float(loss_total.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_mcc": float(loss_mcc.detach().cpu()),
                "target_entropy": float(diagnostics["entropy_mean"]),
            }
            row.update(_soft_confusion_entries(diagnostics["soft_confusion"], config["data"]["class_names"]))
            epoch_rows.append(row)

        row = _epoch_summary(epoch_rows)
        target_total = max(int(pred_counts.sum()), 1)
        row.update(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "target_pred_counts": pred_counts.astype(int).tolist(),
                "target_pred_ratios": (pred_counts / target_total).astype(float).tolist(),
                "source_samples_seen": source_samples_seen,
                "target_samples_seen": target_samples_seen,
                "steps": len(epoch_rows),
            }
        )
        if (epoch + 1) % dev_interval == 0 or epoch + 1 == int(cfg["epochs"]):
            dev_result = estimate_dev_risk(
                model,
                dev_source_loader,
                dev_val_loader,
                dev_target_loader,
                device,
                dev_cfg,
                seed=int(config.get("seed", 42)) + epoch,
            )
            row.update(dev_result)
            if row["dev_risk"] < best_dev_risk:
                best_dev_risk = float(row["dev_risk"])
                best_dev_epoch = epoch
                save_daeac_checkpoint(model, config, best_dev_path, epoch, row)
        history.append(row)
        log_row = {
            f"mcc/{k}": v
            for k, v in row.items()
            if k not in {"epoch", "target_pred_counts", "target_pred_ratios"}
        }
        for idx, count in enumerate(row["target_pred_counts"]):
            log_row[f"mcc/target_pred_count_{idx}"] = count
            log_row[f"mcc/target_pred_ratio_{idx}"] = row["target_pred_ratios"][idx]
        wandb_run.log(log_row, step=epoch)
        save_daeac_checkpoint(model, config, latest_path, epoch, row)
        print(
            f"[mcc epoch {epoch + 1}/{cfg['epochs']}] loss={row['loss']:.4f} "
            f"cls={row['loss_cls']:.4f} mcc={row['loss_mcc']:.4f} "
            f"entropy={row['target_entropy']:.4f} target_pred={row['target_pred_counts']} "
            f"target_ratio={[round(v, 4) for v in row['target_pred_ratios']]} "
            f"samples(src/tgt)={source_samples_seen}/{target_samples_seen} "
            f"dev_risk={row.get('dev_risk', float('nan')):.6f}"
        )

    summary = {
        "latest_checkpoint": str(latest_path),
        "best_dev_checkpoint": str(best_dev_path),
        "best_dev_epoch": best_dev_epoch,
        "best_dev_risk": best_dev_risk,
        "selection_policy": "minimum_deep_embedded_validation_risk",
        "epoch_driver": "target_once",
        "mixed_domain_batchnorm": True,
        "use_class_weights": bool(cfg.get("use_class_weights", True)),
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
