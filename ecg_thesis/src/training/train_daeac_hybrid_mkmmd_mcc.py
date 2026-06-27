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
from src.training.dan_mkmmd import (
    beta_from_config,
    dan_qp_statistics,
    linear_mkmmd_loss,
    normalize_kernel_beta,
    solve_dan_kernel_qp,
)
from src.training.mcc_loss import minimum_class_confusion_loss
from src.training.train_daeac_dan_mkmmd import estimate_mkmmd_gammas
from src.training.train_daeac_paper import (
    CenterMemory,
    _class_weights,
    _cluster_align_loss,
    _threshold_tensor,
    _unpack_input_batch,
    _unpack_source_batch,
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
    dan_optimization_cfg = dict(cfg.get("dan_optimization", {}))
    frozen_module_names = configure_dan_finetuning(model, dan_optimization_cfg)
    optimizer = build_dan_sgd_optimizer(model, cfg, dan_optimization_cfg)
    total_steps = max(int(cfg["epochs"]) * len(source_loader), 1)
    scheduler = build_dan_annealing_scheduler(optimizer, dan_optimization_cfg, total_steps)
    distance_fn = distance_from_name(str(cfg.get("distance", "l2")))
    thresholds = _threshold_tensor(config, cfg, device)

    center_memory = CenterMemory(int(config["data"]["num_classes"]), int(config["model"]["feature_dim"]), device)
    center_memory.source = compute_global_source_centers(model, source_loader, device, center_memory.num_classes)
    center_memory.target = compute_global_target_centers(model, target_loader, device, center_memory.num_classes, thresholds)
    center_memory.refresh_mixed()

    mkmmd_cfg = dict(cfg["mkmmd"])
    layer_weights = {str(k): float(v) for k, v in dict(mkmmd_cfg["layers"]).items() if float(v) != 0.0}
    if set(layer_weights) != {"dan_fc"}:
        raise ValueError("DAN-faithful hybrid MK-MMD must be applied only to the final feature FC layer 'dan_fc'.")
    gammas = estimate_mkmmd_gammas(model, source_loader, target_loader, layer_weights, mkmmd_cfg, device)
    beta_mode = str(mkmmd_cfg.get("beta", "uniform")).lower()
    beta = beta_from_config("uniform" if beta_mode == "qp" else mkmmd_cfg.get("beta", "uniform"), int(mkmmd_cfg["kernel_num"]), device, torch.float32)
    qp_d_ema: torch.Tensor | None = None
    qp_q_ema: torch.Tensor | None = None
    global_step = 0

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
        set_frozen_dan_modules_eval(model, frozen_module_names)
        aux_classifier.load_state_dict(copy.deepcopy(model.classifier.state_dict()))
        aux_classifier.eval()
        target_iter = _cycle(target_loader)
        epoch_rows: list[dict[str, float]] = []
        pseudo_counts = np.zeros(center_memory.num_classes, dtype=np.int64)
        target_pred_counts = np.zeros(center_memory.num_classes, dtype=np.int64)
        for source_batch in source_loader:
            x_s, rr_s, y_s = _unpack_source_batch(source_batch, device)
            x_t, rr_t = _unpack_input_batch(next(target_iter), device)

            source_layers = model.extract_feature_layers(x_s, rr_features=rr_s)
            target_layers = model.extract_feature_layers(x_t, rr_features=rr_t)
            z_s = source_layers["gap_embed"]
            z_t_all = target_layers["gap_embed"]
            logits_s, _ = model.classifier(z_s, rr_s, return_logits=True)
            logits_t, _ = model.classifier(z_t_all, rr_t, return_logits=True)
            loss_cls = cls_loss_fn(logits_s, y_s)
            loss_mmd, layer_losses = _multi_layer_mkmmd_loss(source_layers, target_layers, layer_weights, gammas, beta)
            loss_mcc, mcc_diag = minimum_class_confusion_loss(
                logits_t,
                temperature=float(cfg["mcc"]["temperature"]),
                return_diagnostics=True,
            )

            with torch.no_grad():
                _, probs_t = aux_classifier(z_t_all.detach(), rr_t, return_logits=True)
                conf_t, pseudo_t = probs_t.max(dim=1)
                top2 = probs_t.topk(k=min(2, probs_t.size(1)), dim=1).values
                margin_t = top2[:, 0] - top2[:, 1] if top2.size(1) > 1 else top2[:, 0]
                confident = conf_t >= thresholds[pseudo_t]
                confident = _apply_pseudo_filter(confident, pseudo_t, conf_t, margin_t, cfg, config["data"]["class_names"], epoch)
                pseudo_guard = _pseudo_collapse_guard(confident, pseudo_t, cfg, config["data"]["class_names"])

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
            align_scale = float(pseudo_guard["align_scale"])
            comp_scale = float(pseudo_guard["comp_scale"])
            mcc_scale = float(pseudo_guard["mcc_scale"])
            loss_total = (
                loss_cls
                + float(cfg["beta1"]) * align_scale * loss_align
                + float(cfg["beta2"]) * (loss_sep + comp_scale * loss_comp)
                + float(cfg["lambda_mmd"]) * loss_mmd
                + float(cfg["mcc"]["mu"]) * mcc_scale * loss_mcc
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if beta_mode == "qp":
                d_batch, q_batch = dan_qp_statistics(
                    source_layers["dan_fc"].detach(),
                    target_layers["dan_fc"].detach(),
                    gammas["dan_fc"],
                )
                qp_momentum = float(mkmmd_cfg.get("qp_momentum", 0.9))
                qp_d_ema = d_batch if qp_d_ema is None else qp_momentum * qp_d_ema + (1.0 - qp_momentum) * d_batch
                qp_q_ema = q_batch if qp_q_ema is None else qp_momentum * qp_q_ema + (1.0 - qp_momentum) * q_batch
                if global_step % int(mkmmd_cfg.get("beta_update_interval", 1)) == 0:
                    raw_beta = solve_dan_kernel_qp(
                        qp_d_ema,
                        qp_q_ema,
                        epsilon=float(mkmmd_cfg.get("qp_epsilon", 1.0e-3)),
                        positivity_floor=float(mkmmd_cfg.get("qp_positivity_floor", 1.0e-8)),
                    )
                    beta = normalize_kernel_beta(raw_beta).to(device=device, dtype=torch.float32)
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
                "pseudo_guard_triggered": float(pseudo_guard["triggered"]),
                "pseudo_guard_max_ratio": float(pseudo_guard["max_ratio"]),
                "pseudo_guard_align_scale": align_scale,
                "pseudo_guard_comp_scale": comp_scale,
                "pseudo_guard_mcc_scale": mcc_scale,
                "beta_entropy": float((-(beta * beta.clamp_min(1.0e-12).log()).sum()).detach().cpu()),
            }
            row.update({f"kernel_beta_{idx}": float(value) for idx, value in enumerate(beta.detach().cpu().tolist())})
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
                "lr_classifier": float(next(group["lr"] for group in optimizer.param_groups if group.get("name") == "classifier")),
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
        "kernel_beta": beta.detach().cpu().tolist(),
        "mkmmd_layer": "dan_fc",
        "optimizer": "sgd_momentum_dan_annealing",
        "frozen_modules": frozen_module_names,
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
    max_per_class = {str(k): int(v) for k, v in dict(filter_cfg.get("max_per_class", {})).items()}
    for class_name, quota in max_per_class.items():
        if class_name not in class_names or int(quota) < 0:
            continue
        cls = class_names.index(class_name)
        cls_idx = torch.nonzero(keep & (pseudo == cls), as_tuple=False).flatten()
        if cls_idx.numel() <= int(quota):
            continue
        _, order = torch.sort(confidence[cls_idx], descending=True)
        drop = cls_idx[order[int(quota) :]]
        keep[drop] = False
    return keep


def _pseudo_collapse_guard(
    keep: torch.Tensor,
    pseudo: torch.Tensor,
    cfg: dict[str, Any],
    class_names: list[str],
) -> dict[str, float]:
    guard_cfg = dict(cfg.get("pseudo_collapse_guard", {}))
    if not bool(guard_cfg.get("enabled", False)):
        return {
            "triggered": 0.0,
            "max_ratio": 0.0,
            "align_scale": 1.0,
            "comp_scale": 1.0,
            "mcc_scale": 1.0,
        }
    selected = pseudo[keep]
    if selected.numel() == 0:
        return {
            "triggered": 1.0,
            "max_ratio": 1.0,
            "align_scale": float(guard_cfg.get("empty_align_scale", guard_cfg.get("align_scale", 0.0))),
            "comp_scale": float(guard_cfg.get("empty_comp_scale", guard_cfg.get("comp_scale", 0.0))),
            "mcc_scale": float(guard_cfg.get("empty_mcc_scale", guard_cfg.get("mcc_scale", 0.0))),
        }
    counts = torch.bincount(selected, minlength=len(class_names)).to(dtype=torch.float32)
    ratios = counts / counts.sum().clamp_min(1.0)
    max_ratio = float(ratios.max().detach().cpu())
    triggered = max_ratio > float(guard_cfg.get("max_class_ratio", 1.0))
    min_class_ratio = {str(k): float(v) for k, v in dict(guard_cfg.get("min_class_ratio", {})).items()}
    for class_name, threshold in min_class_ratio.items():
        if class_name not in class_names:
            continue
        if float(ratios[class_names.index(class_name)].detach().cpu()) < threshold:
            triggered = True
    if not triggered:
        return {
            "triggered": 0.0,
            "max_ratio": max_ratio,
            "align_scale": 1.0,
            "comp_scale": 1.0,
            "mcc_scale": 1.0,
        }
    scale = float(guard_cfg.get("scale", 0.0))
    return {
        "triggered": 1.0,
        "max_ratio": max_ratio,
        "align_scale": float(guard_cfg.get("align_scale", scale)),
        "comp_scale": float(guard_cfg.get("comp_scale", scale)),
        "mcc_scale": float(guard_cfg.get("mcc_scale", scale)),
    }


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


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _epoch_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def configure_dan_finetuning(model, cfg: dict[str, Any]) -> list[str]:
    """Map DAN's low-level freeze policy onto the DAEAC feature extractor."""
    names = [str(name) for name in cfg.get("freeze_modules", ["input_conv", "aspp_se_1", "residual_1"])]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for name in names:
        module = getattr(model.feature_extractor, name, None)
        if module is None:
            raise ValueError(f"Unknown DAEAC feature module in dan_optimization.freeze_modules: {name}")
        module.requires_grad_(False)
    return names


def set_frozen_dan_modules_eval(model, names: list[str]) -> None:
    for name in names:
        getattr(model.feature_extractor, name).eval()


def build_dan_sgd_optimizer(model, adaptation_cfg: dict[str, Any], dan_cfg: dict[str, Any]):
    base_lr = float(adaptation_cfg["lr"])
    classifier_multiplier = float(dan_cfg.get("classifier_lr_multiplier", 10.0))
    adaptation_fc_multiplier = float(dan_cfg.get("adaptation_fc_lr_multiplier", 1.0))
    classifier_ids = {id(parameter) for parameter in model.classifier.parameters() if parameter.requires_grad}
    adaptation_fc_ids = {id(parameter) for parameter in model.adaptation_fc.parameters() if parameter.requires_grad}
    backbone = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids and id(parameter) not in adaptation_fc_ids
    ]
    groups = [
        {"params": backbone, "lr": base_lr, "name": "backbone"},
        {"params": list(model.adaptation_fc.parameters()), "lr": base_lr * adaptation_fc_multiplier, "name": "dan_fc"},
        {"params": list(model.classifier.parameters()), "lr": base_lr * classifier_multiplier, "name": "classifier"},
    ]
    groups = [group for group in groups if group["params"]]
    return torch.optim.SGD(
        groups,
        lr=base_lr,
        momentum=float(dan_cfg.get("momentum", 0.9)),
        weight_decay=float(adaptation_cfg["weight_decay"]),
        nesterov=bool(dan_cfg.get("nesterov", False)),
    )


def build_dan_annealing_scheduler(optimizer, cfg: dict[str, Any], total_steps: int):
    alpha = float(cfg.get("annealing_alpha", 10.0))
    power = float(cfg.get("annealing_power", 0.75))

    def multiplier(step: int) -> float:
        progress = min(float(step) / max(int(total_steps), 1), 1.0)
        return (1.0 + alpha * progress) ** (-power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
