from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset
from src.training.daeac_losses import weighted_cross_entropy_from_logits
from src.training.dan_mkmmd import (
    beta_from_config,
    linear_mkmmd_loss,
    make_mkmmd_gammas,
    median_pairwise_squared_distance,
)
from src.training.train_daeac_paper import (
    _class_weights,
    build_daeac_model,
    evaluate_daeac_model,
    load_daeac_checkpoint,
    save_daeac_checkpoint,
)
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb


def train_daeac_dan_mkmmd(
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
    prefix = str(cfg.get("checkpoint_prefix", "daeac_dan_mkmmd"))
    model = build_daeac_model(config, device)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        load_daeac_checkpoint(init_checkpoint, config, device, model=model)

    source_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=True, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    class_weights = _class_weights(source_dataset, config, cfg, device)

    mkmmd_cfg = dict(cfg["mkmmd"])
    layer_weights = {str(k): float(v) for k, v in dict(mkmmd_cfg["layers"]).items() if float(v) != 0.0}
    gammas = estimate_mkmmd_gammas(model, source_loader, target_loader, layer_weights, mkmmd_cfg, device)
    beta = beta_from_config(mkmmd_cfg.get("beta", "uniform"), int(mkmmd_cfg["kernel_num"]), device, torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    wandb_run = init_wandb(config, job_type="train_daeac_dan_mkmmd", default_name=prefix)

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    best_macro_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    global_step = 0
    target_iter = _cycle(target_loader)
    for epoch in range(int(cfg["epochs"])):
        model.train()
        rows: list[dict[str, float]] = []
        for x_s, y_s in source_loader:
            x_t_batch = next(target_iter)
            x_t = _batch_x(x_t_batch)
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            source_layers = model.extract_feature_layers(x_s)
            target_layers = model.extract_feature_layers(x_t)
            logits_s, _ = model.classifier(source_layers["gap_embed"], return_logits=True)
            loss_cls = weighted_cross_entropy_from_logits(logits_s, y_s, class_weights)
            loss_mmd, layer_losses = _multi_layer_mkmmd_loss(source_layers, target_layers, layer_weights, gammas, beta)
            loss_total = loss_cls + float(cfg["lambda_mmd"]) * loss_mmd

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            row = {
                "loss": float(loss_total.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_mmd": float(loss_mmd.detach().cpu()),
            }
            row.update({f"mmd_{name}": float(value.detach().cpu()) for name, value in layer_losses.items()})
            rows.append(row)

        val_result = evaluate_daeac_model(model, val_loader, device, config["data"]["class_names"])
        row = _epoch_summary(rows)
        row.update(
            {
                "epoch": epoch,
                "val_accuracy": val_result["metrics"]["accuracy"],
                "val_macro_f1": val_result["metrics"]["macro_f1"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        history.append(row)
        wandb_run.log({f"dan_mkmmd/{k}": v for k, v in row.items() if k != "epoch"}, step=epoch)
        if row["val_macro_f1"] >= best_macro_f1:
            best_macro_f1 = float(row["val_macro_f1"])
            best_epoch = epoch
            save_daeac_checkpoint(model, config, best_path, epoch, row)
        save_daeac_checkpoint(model, config, latest_path, epoch, row)
        print(
            f"[dan-mkmmd epoch {epoch + 1}/{cfg['epochs']}] loss={row['loss']:.4f} "
            f"cls={row['loss_cls']:.4f} mmd={row['loss_mmd']:.4f} val_macro_f1={row['val_macro_f1']:.4f}"
        )

    summary = {
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "gammas": {name: values.detach().cpu().tolist() for name, values in gammas.items()},
        "history": history,
    }
    wandb_run.summary_update(summary)
    wandb_run.finish()
    return summary


def estimate_mkmmd_gammas(
    model,
    source_loader: DataLoader,
    target_loader: DataLoader,
    layer_weights: dict[str, float],
    mkmmd_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    mode = str(mkmmd_cfg.get("gamma_mode", "fixed_median_subset")).lower()
    if mode != "fixed_median_subset":
        raise ValueError(f"Unsupported DAN MK-MMD gamma_mode: {mode}")
    sample_size = int(mkmmd_cfg.get("gamma_sample_size", 4096))
    gamma_min = float(mkmmd_cfg.get("gamma_min", 1.0e-6))
    kernel_num = int(mkmmd_cfg["kernel_num"])
    kernel_mul = float(mkmmd_cfg["kernel_mul"])
    source_features = _collect_layer_features(model, source_loader, layer_weights.keys(), sample_size, device)
    target_features = _collect_layer_features(model, target_loader, layer_weights.keys(), sample_size, device)
    gammas: dict[str, torch.Tensor] = {}
    for layer_name in layer_weights:
        features = torch.cat([source_features[layer_name], target_features[layer_name]], dim=0)
        reference = median_pairwise_squared_distance(features, gamma_min=gamma_min)
        gammas[layer_name] = make_mkmmd_gammas(
            reference,
            kernel_num=kernel_num,
            kernel_mul=kernel_mul,
            gamma_min=gamma_min,
            device=device,
        )
        print(f"DAN MK-MMD {layer_name} median_gamma={reference:.6g}")
    return gammas


def _multi_layer_mkmmd_loss(
    source_layers: dict[str, torch.Tensor],
    target_layers: dict[str, torch.Tensor],
    layer_weights: dict[str, float],
    gammas: dict[str, torch.Tensor],
    beta: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for layer_name, weight in layer_weights.items():
        loss = linear_mkmmd_loss(source_layers[layer_name], target_layers[layer_name], gammas[layer_name], beta)
        losses[layer_name] = loss
        weighted = float(weight) * loss
        total = weighted if total is None else total + weighted
    if total is None:
        any_layer = next(iter(source_layers.values()))
        total = any_layer.sum() * 0.0
    return total, losses


def _collect_layer_features(
    model,
    loader: DataLoader,
    layer_names,
    max_samples: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    names = [str(name) for name in layer_names]
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = _batch_x(batch).to(device)
            layers = model.extract_feature_layers(x)
            take = min(int(x.shape[0]), max(int(max_samples) - count, 0))
            if take <= 0:
                break
            for name in names:
                rows[name].append(layers[name][:take].detach().cpu())
            count += take
            if count >= int(max_samples):
                break
    if count == 0:
        raise ValueError("Cannot estimate MK-MMD gammas from an empty dataset.")
    return {name: torch.cat(values, dim=0) for name, values in rows.items()}


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
