from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset
from src.models.daeac_paper import ClassifierH
from src.training.daeac_hybrid_ablation_losses import (
    class_balanced_conditional_mkmmd_loss,
    minority_class_weights,
    minority_weighted_mcc_loss,
    safe_topk_pseudolabel_mask,
    source_f_prototype_contrastive_loss,
    update_soft_prior_ema,
)
from src.training.daeac_losses import build_daeac_classification_loss, compacting_loss, distance_from_name, separating_loss
from src.training.dan_mkmmd import beta_from_config, linear_mkmmd_loss
from src.training.mcc_loss import minimum_class_confusion_loss
from src.training.train_daeac_dan_mkmmd import estimate_mkmmd_gammas
from src.training.train_daeac_mcc import _soft_confusion_entries
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
from src.utils.io import ensure_dir, write_json
from src.utils.wandb_logging import init_wandb


ABLATION_NAMES = {
    "class_balanced_mkmmd",
    "faware_pseudo_topk",
    "minority_weighted_mcc",
    "source_f_prototype",
}


def train_daeac_hybrid_ablation(
    source_dataset: DAEACDataset,
    val_dataset: DAEACDataset,
    target_dataset: DAEACTargetUnlabeledDataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = config["adaptation"]
    ablation_name = validate_ablation_config(config)
    class_names = list(config["data"]["class_names"])
    num_classes = int(config["data"]["num_classes"])
    output = Path(output_dir)
    metrics_dir = ensure_dir(output / "metrics")
    ckpt_dir = ensure_dir(output / "checkpoints")
    prefix = str(cfg["checkpoint_prefix"])
    write_json(config, output / "resolved_config.json")

    model = build_daeac_model(config, device)
    init_checkpoint = Path(str(cfg["init_checkpoint"]))
    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Split-correct focal-standard checkpoint not found: {init_checkpoint}")
    load_daeac_checkpoint(init_checkpoint, config, device, model=model)

    source_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=True, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    class_weights = _class_weights(source_dataset, config, cfg, device)
    cls_loss_fn = build_daeac_classification_loss(cfg, num_classes, class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    distance_fn = distance_from_name(str(cfg.get("distance", "l2")))
    thresholds = _threshold_tensor(config, cfg, device)

    center_memory = CenterMemory(num_classes, int(config["model"]["feature_dim"]), device)
    center_memory.source = compute_global_source_centers(model, source_loader, device, num_classes)
    center_memory.target = compute_global_target_centers(model, target_loader, device, num_classes, thresholds)
    center_memory.refresh_mixed()
    source_proto_memory = [center.detach().clone() if center is not None else None for center in center_memory.source]

    mkmmd_cfg = dict(cfg["mkmmd"])
    layer_weights = {str(k): float(v) for k, v in dict(mkmmd_cfg["layers"]).items() if float(v) != 0.0}
    gammas = estimate_mkmmd_gammas(model, source_loader, target_loader, layer_weights, mkmmd_cfg, device)
    beta = beta_from_config(mkmmd_cfg.get("beta", "uniform"), int(mkmmd_cfg["kernel_num"]), device, torch.float32)
    conditional_weights = _class_tensor(
        class_names,
        dict(cfg.get("class_balanced_mkmmd", {}).get("class_weights", {})),
        device,
        default=1.0,
    )

    aux_classifier = ClassifierH(
        feature_dim=int(config["model"]["feature_dim"]),
        num_classes=num_classes,
        dropout=0.0,
    ).to(device)
    target_prior = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32, device=device)
    wandb_run = init_wandb(config, job_type="train_daeac_hybrid_ablation", default_name=prefix)
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
        for x_s, y_s in source_loader:
            x_t = _batch_x(next(target_iter))
            x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)
            source_layers = model.extract_feature_layers(x_s)
            target_layers = model.extract_feature_layers(x_t)
            z_s, z_t_all = source_layers["gap_embed"], target_layers["gap_embed"]
            logits_s, _ = model.classifier(z_s, return_logits=True)
            logits_t, _ = model.classifier(z_t_all, return_logits=True)
            loss_cls = cls_loss_fn(logits_s, y_s)

            with torch.no_grad():
                _, probs_t = aux_classifier(z_t_all.detach(), return_logits=True)
                conf_t, pseudo_t = probs_t.max(dim=1)
                top2 = probs_t.topk(k=min(2, probs_t.size(1)), dim=1).values
                margin_t = top2[:, 0] - top2[:, 1] if top2.size(1) > 1 else top2[:, 0]

            if ablation_name == "class_balanced_mkmmd":
                loss_mmd, mmd_diag = _conditional_multilayer_mkmmd(
                    source_layers,
                    target_layers,
                    y_s,
                    probs_t,
                    layer_weights,
                    gammas,
                    beta,
                    conditional_weights,
                    class_names,
                    float(cfg["class_balanced_mkmmd"].get("min_target_mass", 1.0)),
                )
            else:
                loss_mmd, layer_losses = _global_multilayer_mkmmd(source_layers, target_layers, layer_weights, gammas, beta)
                mmd_diag = {f"mmd_{name}": float(value.detach().cpu()) for name, value in layer_losses.items()}

            if ablation_name == "faware_pseudo_topk":
                topk_cfg = dict(cfg["faware_pseudo_topk"])
                confident, pseudo_diag = safe_topk_pseudolabel_mask(
                    pseudo_t,
                    conf_t,
                    margin_t,
                    class_names,
                    dict(topk_cfg["min_confidence"]),
                    dict(topk_cfg["min_margin"]),
                    dict(topk_cfg["max_per_batch"]),
                )
            else:
                confident = conf_t >= thresholds[pseudo_t]
                pseudo_diag = _pseudo_diagnostics(confident, pseudo_t, conf_t, margin_t, class_names)

            selected_pseudo = pseudo_t[confident]
            z_t = z_t_all[confident]
            local_source = batch_centers(z_s, y_s, num_classes)
            local_target = batch_centers(z_t, selected_pseudo, num_classes)
            source_for_loss, target_for_loss, mixed_for_loss = center_memory.centers_for_loss(
                local_source,
                local_target,
                gamma=float(cfg["center_ema_gamma"]),
            )
            loss_align = _cluster_align_loss(source_for_loss, target_for_loss, cfg, distance_fn, device)
            z_mix = torch.cat([z_s, z_t], dim=0) if z_t.numel() else z_s
            y_mix = torch.cat([y_s, selected_pseudo], dim=0) if z_t.numel() else y_s
            loss_sep = separating_loss(mixed_for_loss, float(cfg["margin"]), distance_fn, device)
            loss_comp = compacting_loss(z_mix, y_mix, mixed_for_loss, distance_fn, device)

            if ablation_name == "minority_weighted_mcc":
                wmcc_cfg = dict(cfg["minority_weighted_mcc"])
                mcc_probs = torch.softmax(logits_t.detach() / float(cfg["mcc"]["temperature"]), dim=1)
                target_prior = update_soft_prior_ema(target_prior, mcc_probs, float(wmcc_cfg["prior_ema_decay"]))
                multipliers = _class_tensor(class_names, dict(wmcc_cfg["multipliers"]), device, default=1.0)
                mcc_weights = minority_class_weights(
                    target_prior,
                    exponent=float(wmcc_cfg["inverse_prior_exponent"]),
                    min_weight=float(wmcc_cfg["min_weight"]),
                    max_weight=float(wmcc_cfg["max_weight"]),
                    multipliers=multipliers,
                )
                loss_mcc, mcc_diag = minority_weighted_mcc_loss(
                    logits_t,
                    mcc_weights,
                    temperature=float(cfg["mcc"]["temperature"]),
                    return_diagnostics=True,
                )
            else:
                loss_mcc, mcc_diag = minimum_class_confusion_loss(
                    logits_t,
                    temperature=float(cfg["mcc"]["temperature"]),
                    return_diagnostics=True,
                )
                mcc_weights = torch.ones(num_classes, device=device)

            if ablation_name == "source_f_prototype":
                proto_cfg = dict(cfg["source_f_prototype"])
                loss_proto, proto_diag = source_f_prototype_contrastive_loss(
                    z_s,
                    y_s,
                    source_proto_memory,
                    class_names,
                    temperature=float(proto_cfg["temperature"]),
                    sample_weights=dict(proto_cfg["sample_weights"]),
                )
            else:
                loss_proto = z_s.sum() * 0.0
                proto_diag = {"cosine_F_N": 0.0, "cosine_F_V": 0.0, "active_samples": 0.0}

            loss_total = (
                loss_cls
                + float(cfg["beta1"]) * loss_align
                + float(cfg["beta2"]) * (loss_sep + loss_comp)
                + float(cfg["lambda_mmd"]) * loss_mmd
                + float(cfg["mcc"]["mu"]) * loss_mcc
                + float(cfg.get("source_f_prototype", {}).get("lambda", 0.0)) * loss_proto
            )
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            center_memory.commit(source_for_loss, target_for_loss, mixed_for_loss)
            if ablation_name == "source_f_prototype":
                source_proto_memory = _update_prototype_memory(
                    source_proto_memory,
                    local_source,
                    float(cfg["source_f_prototype"].get("prototype_ema_gamma", 0.1)),
                )

            row = {
                "loss": float(loss_total.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_align": float(loss_align.detach().cpu()),
                "loss_sep": float(loss_sep.detach().cpu()),
                "loss_comp": float(loss_comp.detach().cpu()),
                "loss_mmd": float(loss_mmd.detach().cpu()),
                "loss_mcc": float(loss_mcc.detach().cpu()),
                "loss_source_proto": float(loss_proto.detach().cpu()),
                "target_entropy": float(mcc_diag["entropy_mean"]),
                "pseudo_selected": float(confident.sum().detach().cpu()),
                **mmd_diag,
                **pseudo_diag,
                **{f"proto_{key}": float(value) for key, value in proto_diag.items()},
            }
            for idx, name in enumerate(class_names):
                row[f"target_prior_{name}"] = float(target_prior[idx].detach().cpu())
                row[f"mcc_weight_{name}"] = float(mcc_weights[idx].detach().cpu())
                row[f"target_pred_count_{name}"] = float(mcc_diag["pred_counts"][idx])
                if "offdiag_by_class" in mcc_diag:
                    row[f"mcc_offdiag_{name}"] = float(mcc_diag["offdiag_by_class"][idx].detach().cpu())
            row.update(_soft_confusion_entries(mcc_diag["soft_confusion"], class_names))
            epoch_rows.append(row)

        val_result = evaluate_daeac_model(model, val_loader, device, class_names)
        epoch_row = _epoch_summary(epoch_rows)
        epoch_row.update(
            {
                "epoch": epoch,
                "val_accuracy": float(val_result["metrics"]["accuracy"]),
                "val_macro_f1": float(val_result["metrics"]["macro_f1"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        history.append(epoch_row)
        _write_history_csv(history, metrics_dir / f"{prefix}_train_log.csv")
        wandb_run.log({f"{prefix}/{key}": value for key, value in epoch_row.items() if key != "epoch"}, step=epoch)
        if epoch_row["val_macro_f1"] >= best_macro_f1:
            best_macro_f1 = epoch_row["val_macro_f1"]
            best_epoch = epoch
            save_daeac_checkpoint(model, config, best_path, epoch, epoch_row)
        save_daeac_checkpoint(model, config, latest_path, epoch, epoch_row)
        print(
            f"[{prefix} {epoch + 1}/{cfg['epochs']}] loss={epoch_row['loss']:.4f} "
            f"mmd={epoch_row['loss_mmd']:.4f} mcc={epoch_row['loss_mcc']:.4f} "
            f"proto={epoch_row['loss_source_proto']:.4f} val_macro_f1={epoch_row['val_macro_f1']:.4f}"
        )

    summary = {
        "ablation": ablation_name,
        "init_checkpoint": str(init_checkpoint),
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "gammas": {name: values.detach().cpu().tolist() for name, values in gammas.items()},
        "history": history,
    }
    write_json(summary, metrics_dir / f"{prefix}_train_summary.json")
    wandb_run.summary_update({key: value for key, value in summary.items() if key != "history"})
    wandb_run.finish()
    return summary


def validate_ablation_config(config: dict[str, Any]) -> str:
    name = str(config.get("ablation", {}).get("name", ""))
    if name not in ABLATION_NAMES:
        raise ValueError(f"ablation.name must be exactly one of {sorted(ABLATION_NAMES)}, got {name!r}.")
    block_by_name = {
        "class_balanced_mkmmd": "class_balanced_mkmmd",
        "faware_pseudo_topk": "faware_pseudo_topk",
        "minority_weighted_mcc": "minority_weighted_mcc",
        "source_f_prototype": "source_f_prototype",
    }
    required = block_by_name[name]
    present = [block for block in block_by_name.values() if block in config["adaptation"]]
    if present != [required]:
        raise ValueError(f"Ablation {name!r} must define only adaptation.{required}; found {present}.")
    if required not in config["adaptation"]:
        raise ValueError(f"Ablation {name!r} requires adaptation.{required} config.")
    return name


def _conditional_multilayer_mkmmd(
    source_layers,
    target_layers,
    source_labels,
    target_probs,
    layer_weights,
    gammas,
    beta,
    class_weights,
    class_names,
    min_target_mass,
):
    total = None
    diagnostics: dict[str, float] = {}
    for layer_name, layer_weight in layer_weights.items():
        loss, class_losses, diag = class_balanced_conditional_mkmmd_loss(
            source_layers[layer_name],
            target_layers[layer_name],
            source_labels,
            target_probs,
            gammas[layer_name],
            beta,
            class_weights,
            min_target_mass=min_target_mass,
        )
        total = float(layer_weight) * loss if total is None else total + float(layer_weight) * loss
        diagnostics[f"mmd_{layer_name}"] = float(loss.detach().cpu())
        for idx, name in enumerate(class_names):
            diagnostics[f"mmd_{layer_name}_{name}"] = float(class_losses.get(str(idx), loss * 0.0).detach().cpu())
            diagnostics[f"mmd_active_{layer_name}_{name}"] = float(diag[f"active_{idx}"].detach().cpu())
            diagnostics[f"mmd_target_mass_{layer_name}_{name}"] = float(diag[f"target_mass_{idx}"].detach().cpu())
    if total is None:
        total = next(iter(source_layers.values())).sum() * 0.0
    return total, diagnostics


def _global_multilayer_mkmmd(source_layers, target_layers, layer_weights, gammas, beta):
    losses = {}
    total = None
    for layer_name, weight in layer_weights.items():
        loss = linear_mkmmd_loss(source_layers[layer_name], target_layers[layer_name], gammas[layer_name], beta)
        losses[layer_name] = loss
        total = float(weight) * loss if total is None else total + float(weight) * loss
    if total is None:
        total = next(iter(source_layers.values())).sum() * 0.0
    return total, losses


def _pseudo_diagnostics(mask, pseudo, confidence, margin, class_names):
    values = {}
    for idx, name in enumerate(class_names):
        selected = mask & (pseudo == idx)
        values[f"candidate_{name}"] = float((pseudo == idx).sum().detach().cpu())
        values[f"selected_{name}"] = float(selected.sum().detach().cpu())
        values[f"selected_confidence_{name}"] = float(confidence[selected].mean().detach().cpu()) if bool(selected.any()) else 0.0
        values[f"selected_margin_{name}"] = float(margin[selected].mean().detach().cpu()) if bool(selected.any()) else 0.0
    return values


def _class_tensor(class_names, values, device, default):
    return torch.as_tensor([float(values.get(name, default)) for name in class_names], dtype=torch.float32, device=device)


def _update_prototype_memory(memory, local, gamma):
    updated = []
    for old, current in zip(memory, local):
        if old is None and current is None:
            updated.append(None)
        elif old is None:
            updated.append(current.detach())
        elif current is None:
            updated.append(old.detach())
        else:
            updated.append(((1.0 - float(gamma)) * old.detach() + float(gamma) * current.detach()).detach())
    return updated


def _batch_x(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _epoch_summary(rows):
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {}
    for key in keys:
        values = [row.get(key, 0.0) for row in rows]
        is_count = (
            key == "pseudo_selected"
            or key.startswith("candidate_")
            or key.startswith("target_pred_count_")
            or (key.startswith("selected_") and not key.startswith(("selected_confidence_", "selected_margin_")))
        )
        summary[key] = float(np.sum(values) if is_count else np.mean(values))
    return summary


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted(set().union(*(row.keys() for row in history)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
