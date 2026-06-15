from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset
from src.models.daeac_paper import ClassifierH
from src.training.daeac_losses import (
    build_daeac_classification_loss,
    compacting_loss,
    distance_from_name,
    separating_loss,
)
from src.training.dan_mkmmd import beta_from_config, linear_mkmmd_loss
from src.training.mcc_loss import minimum_class_confusion_loss
from src.training.train_daeac_dan_mkmmd import estimate_mkmmd_gammas
from src.training.train_daeac_paper import (
    CenterMemory,
    _class_weights,
    _cluster_align_loss,
    _threshold_tensor,
    batch_centers,
    build_daeac_model,
    compute_global_source_centers,
    compute_global_target_centers,
    evaluate_daeac_model,
    load_daeac_checkpoint,
    save_daeac_checkpoint,
)
from src.training.train_daeac_mcc import _soft_confusion_entries
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb


def train_daeac_hybrid_mkmmd_mcc(
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
    prefix = str(cfg.get("checkpoint_prefix", "daeac_hybrid_mkmmd_mcc"))
    model = build_daeac_model(config, device)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        load_daeac_checkpoint(init_checkpoint, config, device, model=model)

    source_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=True, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    class_weights = _class_weights(source_dataset, config, cfg, device)
    cls_loss_fn = build_daeac_classification_loss(cfg, int(config["data"]["num_classes"]), class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    distance_fn = distance_from_name(str(cfg.get("distance", "l2")))
    thresholds = _threshold_tensor(config, cfg, device)

    center_memory = CenterMemory(int(config["data"]["num_classes"]), int(config["model"]["feature_dim"]), device)
    center_memory.source = compute_global_source_centers(model, source_loader, device, center_memory.num_classes)
    center_memory.target = compute_global_target_centers(model, target_loader, device, center_memory.num_classes, thresholds)
    center_memory.refresh_mixed()

    mkmmd_cfg = dict(cfg["mkmmd"])
    layer_weights = {str(k): float(v) for k, v in dict(mkmmd_cfg["layers"]).items() if float(v) != 0.0}
    gammas = estimate_mkmmd_gammas(model, source_loader, target_loader, layer_weights, mkmmd_cfg, device)
    beta = beta_from_config(mkmmd_cfg.get("beta", "uniform"), int(mkmmd_cfg["kernel_num"]), device, torch.float32)

    aux_classifier = ClassifierH(
        feature_dim=int(config["model"]["feature_dim"]),
        num_classes=int(config["data"]["num_classes"]),
        dropout=0.0,
    ).to(device)
    wandb_run = init_wandb(config, job_type="train_daeac_hybrid_mkmmd_mcc", default_name=prefix)

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    best_macro_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        aux_classifier.load_state_dict(copy.deepcopy(model.classifier.state_dict()))
        aux_classifier.eval()
        target_iter = _cycle(target_loader)
        epoch_rows: list[dict[str, float]] = []
        pseudo_counts = np.zeros(center_memory.num_classes, dtype=np.int64)
        target_pred_counts = np.zeros(center_memory.num_classes, dtype=np.int64)
        for x_s, y_s in source_loader:
            x_t = _batch_x(next(target_iter))
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            source_layers = model.extract_feature_layers(x_s)
            target_layers = model.extract_feature_layers(x_t)
            z_s = source_layers["gap_embed"]
            z_t_all = target_layers["gap_embed"]
            logits_s, _ = model.classifier(z_s, return_logits=True)
            logits_t, _ = model.classifier(z_t_all, return_logits=True)
            loss_cls = cls_loss_fn(logits_s, y_s)
            loss_mmd, layer_losses = _multi_layer_mkmmd_loss(source_layers, target_layers, layer_weights, gammas, beta)
            loss_mcc, mcc_diag = minimum_class_confusion_loss(
                logits_t,
                temperature=float(cfg["mcc"]["temperature"]),
                return_diagnostics=True,
            )

            with torch.no_grad():
                _, probs_t = aux_classifier(z_t_all.detach(), return_logits=True)
                conf_t, pseudo_t = probs_t.max(dim=1)
                top2 = probs_t.topk(k=min(2, probs_t.size(1)), dim=1).values
                margin_t = top2[:, 0] - top2[:, 1] if top2.size(1) > 1 else top2[:, 0]
                confident = conf_t >= thresholds[pseudo_t]
                confident = _apply_pseudo_filter(confident, pseudo_t, conf_t, margin_t, cfg, config["data"]["class_names"], epoch)

            if bool(confident.any()):
                selected_pseudo_t = pseudo_t[confident]
                z_t = z_t_all[confident]
                pseudo_counts += np.bincount(selected_pseudo_t.detach().cpu().numpy(), minlength=center_memory.num_classes)
            else:
                selected_pseudo_t = torch.empty(0, dtype=torch.long, device=device)
                z_t = torch.empty(0, center_memory.feature_dim, device=device)

            local_source = batch_centers(z_s, y_s, center_memory.num_classes)
            local_target = batch_centers(z_t, selected_pseudo_t, center_memory.num_classes)
            source_for_loss, target_for_loss, mixed_for_loss = center_memory.centers_for_loss(
                local_source,
                local_target,
                gamma=float(cfg["center_ema_gamma"]),
            )

            loss_align = _cluster_align_loss(source_for_loss, target_for_loss, cfg, distance_fn, device)
            if z_t.numel() > 0:
                z_mix = torch.cat([z_s, z_t], dim=0)
                y_mix = torch.cat([y_s, selected_pseudo_t], dim=0)
            else:
                z_mix = z_s
                y_mix = y_s
            loss_sep = separating_loss(mixed_for_loss, float(cfg["margin"]), distance_fn, device)
            loss_comp = compacting_loss(z_mix, y_mix, mixed_for_loss, distance_fn, device)
            loss_total = (
                loss_cls
                + float(cfg["beta1"]) * loss_align
                + float(cfg["beta2"]) * (loss_sep + loss_comp)
                + float(cfg["lambda_mmd"]) * loss_mmd
                + float(cfg["mcc"]["mu"]) * loss_mcc
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            center_memory.commit(source_for_loss, target_for_loss, mixed_for_loss)
            target_pred_counts += np.asarray(mcc_diag["pred_counts"], dtype=np.int64)
            row = {
                "loss": float(loss_total.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_align": float(loss_align.detach().cpu()),
                "loss_sep": float(loss_sep.detach().cpu()),
                "loss_comp": float(loss_comp.detach().cpu()),
                "loss_mmd": float(loss_mmd.detach().cpu()),
                "loss_mcc": float(loss_mcc.detach().cpu()),
                "target_entropy": float(mcc_diag["entropy_mean"]),
                "pseudo_selected": float(confident.sum().detach().cpu()),
            }
            row.update({f"mmd_{name}": float(value.detach().cpu()) for name, value in layer_losses.items()})
            row.update(_soft_confusion_entries(mcc_diag["soft_confusion"], config["data"]["class_names"]))
            epoch_rows.append(row)

        val_result = evaluate_daeac_model(model, val_loader, device, config["data"]["class_names"])
        row = _epoch_summary(epoch_rows)
        row.update(
            {
                "epoch": epoch,
                "val_accuracy": val_result["metrics"]["accuracy"],
                "val_macro_f1": val_result["metrics"]["macro_f1"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "pseudo_counts": pseudo_counts.astype(int).tolist(),
                "target_pred_counts": target_pred_counts.astype(int).tolist(),
            }
        )
        history.append(row)
        log_row = {
            f"hybrid_mkmmd_mcc/{k}": v
            for k, v in row.items()
            if k not in {"epoch", "pseudo_counts", "target_pred_counts"}
        }
        for idx, count in enumerate(row["pseudo_counts"]):
            log_row[f"hybrid_mkmmd_mcc/pseudo_count_{idx}"] = count
        for idx, count in enumerate(row["target_pred_counts"]):
            log_row[f"hybrid_mkmmd_mcc/target_pred_count_{idx}"] = count
        wandb_run.log(log_row, step=epoch)
        if row["val_macro_f1"] >= best_macro_f1:
            best_macro_f1 = float(row["val_macro_f1"])
            best_epoch = epoch
            save_daeac_checkpoint(model, config, best_path, epoch, row)
        save_daeac_checkpoint(model, config, latest_path, epoch, row)
        print(
            f"[hybrid-mkmmd-mcc epoch {epoch + 1}/{cfg['epochs']}] loss={row['loss']:.4f} "
            f"cls={row['loss_cls']:.4f} align={row['loss_align']:.4f} "
            f"sep={row['loss_sep']:.4f} comp={row['loss_comp']:.4f} "
            f"mmd={row['loss_mmd']:.4f} mcc={row['loss_mcc']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} pseudo={row['pseudo_counts']}"
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


def _apply_pseudo_filter(
    confident: torch.Tensor,
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
    margin: torch.Tensor,
    cfg: dict[str, Any],
    class_names: list[str],
    epoch: int,
) -> torch.Tensor:
    filter_cfg = dict(cfg.get("pseudo_filter", {}))
    if not bool(filter_cfg.get("enabled", False)):
        return confident
    keep = confident.clone()
    f_idx = class_names.index("F") if "F" in class_names else None
    if f_idx is not None and epoch < int(filter_cfg.get("f_warmup_epochs", 0)):
        keep &= pseudo != f_idx
    min_margin = {str(k): float(v) for k, v in dict(filter_cfg.get("min_margin", {})).items()}
    for class_name, threshold in min_margin.items():
        if class_name not in class_names:
            continue
        cls = class_names.index(class_name)
        keep &= ~((pseudo == cls) & (margin < threshold))
    max_batch_ratio = {str(k): float(v) for k, v in dict(filter_cfg.get("max_batch_ratio", {})).items()}
    for class_name, ratio in max_batch_ratio.items():
        if class_name not in class_names:
            continue
        cls = class_names.index(class_name)
        cls_idx = torch.nonzero(keep & (pseudo == cls), as_tuple=False).flatten()
        if cls_idx.numel() == 0:
            continue
        quota = int(math.floor(float(ratio) * float(pseudo.numel())))
        if ratio > 0 and quota < 1:
            quota = 1
        if cls_idx.numel() <= quota:
            continue
        _, order = torch.sort(confidence[cls_idx], descending=True)
        drop = cls_idx[order[quota:]]
        keep[drop] = False
    return keep


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
