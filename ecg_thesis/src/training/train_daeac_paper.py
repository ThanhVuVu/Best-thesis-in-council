from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.daeac_dataset import DAEACDataset, DAEACPseudoLabeledDataset, DAEACTargetUnlabeledDataset
from src.models.daeac_paper import ClassifierH, DAEACNetwork, DualClassifierH
from src.training.daeac_losses import (
    build_daeac_classification_loss,
    cluster_aligning_loss,
    compacting_loss,
    distance_from_name,
    separating_loss,
    weighted_cross_entropy_from_logits,
)
from src.training.mk_mmd import center_cluster_mk_mmd_loss, center_pair_reference_distance
from src.training.metrics import classification_metrics
from src.training.v_measure_validation import aggregate_v_measure, ericsson_v_measure, save_v_measure_assignments
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb


def build_daeac_model(config: dict[str, Any], device: torch.device) -> DAEACNetwork:
    model_cfg = config["model"]
    dual_head_cfg = dict(config.get("rtd_daeac", {}).get("dual_head", {}))
    return DAEACNetwork(
        num_classes=int(model_cfg["num_classes"]),
        input_channels=int(model_cfg.get("input_channels", 1)),
        initial_channels=int(model_cfg.get("initial_channels", 4)),
        feature_dim=int(model_cfg.get("feature_dim", 256)),
        dilations=tuple(int(v) for v in model_cfg.get("dilations", [1, 6, 12, 18])),
        se_reduction=int(model_cfg.get("se_reduction", 16)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        adaptation_fc=bool(dict(model_cfg.get("adaptation_fc", {})).get("enabled", False)),
        dual_head=bool(dual_head_cfg.get("enabled", False)),
    ).to(device)


def train_daeac_base(
    train_dataset: DAEACDataset,
    val_dataset: DAEACDataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = config["training"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    prefix = str(cfg.get("checkpoint_prefix", "daeac_base"))
    model = build_daeac_model(config, device)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    class_weights = _class_weights(train_dataset, config, cfg, device)
    cls_loss_fn = build_daeac_classification_loss(cfg, int(config["data"]["num_classes"]), class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    wandb_run = init_wandb(config, job_type="train_daeac_base", default_name=prefix)

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    best_macro_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(int(cfg["epochs"])):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x, return_dict=True)
            loss = _source_classification_loss(output, y, class_weights, cls_loss_fn, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))

        val_result = evaluate_daeac_model(model, val_loader, device, config["data"]["class_names"])
        row = {
            "epoch": epoch,
            "loss_cls": float(np.mean(losses)) if losses else 0.0,
            "val_accuracy": val_result["metrics"]["accuracy"],
            "val_macro_f1": val_result["metrics"]["macro_f1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        wandb_run.log({f"train/{k}": v for k, v in row.items() if k != "epoch"}, step=epoch)
        if row["val_macro_f1"] >= best_macro_f1:
            best_macro_f1 = float(row["val_macro_f1"])
            best_epoch = epoch
            save_daeac_checkpoint(model, config, best_path, epoch, row)
        save_daeac_checkpoint(model, config, latest_path, epoch, row)
        print(f"[base epoch {epoch + 1}/{cfg['epochs']}] loss={row['loss_cls']:.4f} val_macro_f1={row['val_macro_f1']:.4f}")

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


def adapt_daeac(
    source_dataset: DAEACDataset,
    target_dataset: DAEACTargetUnlabeledDataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
    source_val_dataset: DAEACDataset | None = None,
    target_val_dataset: DAEACTargetUnlabeledDataset | None = None,
) -> dict[str, Any]:
    if source_val_dataset is None or target_val_dataset is None:
        raise ValueError("Ericsson V-Measure adaptation requires disjoint source_val_dataset and target_val_dataset")
    cfg = config["adaptation"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    prefix = str(cfg.get("checkpoint_prefix", "daeac_uda"))
    model = build_daeac_model(config, device)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        load_daeac_checkpoint(init_checkpoint, config, device, model=model)

    source_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=True, num_workers=0)
    target_inference_loader = DataLoader(target_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=False, num_workers=0)
    source_val_loader = DataLoader(source_val_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    target_val_loader = DataLoader(target_val_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=False, num_workers=0)
    class_weights = _class_weights(source_dataset, config, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    distance_fn = distance_from_name(str(cfg.get("distance", "l2")))
    thresholds = _threshold_tensor(config, cfg, device)
    aux_classifier = copy.deepcopy(model.classifier).to(device).eval()
    pseudo_dataset = build_pseudo_labeled_target_dataset(
        model, aux_classifier, target_dataset, target_inference_loader, thresholds, device
    )
    center_memory = CenterMemory(int(config["data"]["num_classes"]), int(config["model"]["feature_dim"]), device)
    center_memory.source = compute_global_source_centers(model, source_loader, device, center_memory.num_classes)
    center_memory.target = compute_global_pseudo_target_centers(
        model,
        DataLoader(pseudo_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=False, num_workers=0),
        device,
        center_memory.num_classes,
    )
    center_memory.refresh_mixed()
    _prepare_center_mkmmd_config(cfg, center_memory)
    wandb_run = init_wandb(config, job_type="adapt_daeac", default_name=prefix)

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    log_dir = ensure_dir(output_dir / "logs")
    history_csv = log_dir / f"{prefix}_adapt_history.csv"
    history: list[dict[str, Any]] = []
    global_step = 0
    first_stable_epoch: int | None = None
    pseudo_snapshot_origin = "initial_before_epoch_1"
    class_names = list(config["data"]["class_names"])
    best_v_measure = -1.0
    best_epoch = -1
    stale_epochs = 0
    print(
        f"[uda setup] protocol=unlabeled_target_loss_monitoring source_samples={len(source_dataset)} "
        f"target_total={len(target_dataset)} epochs={cfg['epochs']} source_batch={cfg['source_batch_size']} "
        f"target_batch={cfg['target_batch_size']} beta1={cfg['beta1']} beta2={cfg['beta2']} "
        f"distance={cfg.get('distance', 'l2')} reduction={cfg.get('cluster_loss_reduction', 'sum')}"
    )
    print(
        "[uda setup] pseudo_thresholds="
        + ", ".join(f"{name}>{float(cfg['pseudo_thresholds'][name]):.4f}" for name in class_names)
        + " checkpoint_policy=maximum_ericsson_v_measure"
    )
    for epoch in range(int(cfg["epochs"])):
        epoch_started = time.perf_counter()
        model.train()
        target_loader = DataLoader(
            pseudo_dataset,
            batch_size=int(cfg["target_batch_size"]),
            shuffle=True,
            num_workers=0,
        )
        source_iter = _cycle(source_loader)
        epoch_rows: list[dict[str, float]] = []
        pseudo_counts = np.bincount(pseudo_dataset.labels.numpy(), minlength=center_memory.num_classes)
        for x_t, pseudo_t, _, _ in target_loader:
            x_s, y_s = next(source_iter)
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)
            selected_pseudo_t = pseudo_t.to(device)

            source_output = model(x_s, return_dict=True)
            z_s = source_output["features"]
            loss_cls = _source_classification_loss(source_output, y_s, class_weights, None, config)
            z_t = model.extract_features(x_t)

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
            reduction = str(cfg.get("cluster_loss_reduction", "sum"))
            loss_sep = separating_loss(mixed_for_loss, float(cfg["margin"]), distance_fn, device, reduction=reduction)
            loss_comp = compacting_loss(z_mix, y_mix, mixed_for_loss, distance_fn, device, reduction=reduction)
            loss_total = loss_cls + float(cfg["beta1"]) * loss_align + float(cfg["beta2"]) * (loss_sep + loss_comp)

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            center_memory.commit(source_for_loss, target_for_loss, mixed_for_loss)
            global_step += 1
            epoch_rows.append(
                {
                    "loss": float(loss_total.detach().cpu()),
                    "loss_cls": float(loss_cls.detach().cpu()),
                    "loss_align": float(loss_align.detach().cpu()),
                    "loss_sep": float(loss_sep.detach().cpu()),
                    "loss_comp": float(loss_comp.detach().cpu()),
                    "pseudo_selected": float(len(selected_pseudo_t)),
                    "source_batch_size": float(len(y_s)),
                    "target_batch_size": float(len(selected_pseudo_t)),
                }
            )

        row = _detailed_epoch_loss_summary(epoch_rows)
        loss_main = float(row["loss_sep"] + row["loss_comp"])
        pseudo_diag = _pseudo_snapshot_diagnostics(pseudo_dataset, len(target_dataset), class_names, cfg)
        center_diag = _center_diagnostics(center_memory, class_names, distance_fn)
        row.update(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "iterations": len(epoch_rows),
                "source_samples_seen": int(sum(item["source_batch_size"] for item in epoch_rows)),
                "target_pseudo_samples_seen": int(sum(item["target_batch_size"] for item in epoch_rows)),
                "loss_main": loss_main,
                "weighted_align": float(cfg["beta1"]) * float(row["loss_align"]),
                "weighted_sep": float(cfg["beta2"]) * float(row["loss_sep"]),
                "weighted_comp": float(cfg["beta2"]) * float(row["loss_comp"]),
                "weighted_main": float(cfg["beta2"]) * loss_main,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "pseudo_counts": pseudo_counts.astype(int).tolist(),
                "pseudo_snapshot_origin": pseudo_snapshot_origin,
                **pseudo_diag,
                **center_diag,
            }
        )
        stability = _stability_diagnostics(history, row, cfg)
        row.update(stability)
        source_result = evaluate_daeac_model(model, source_val_loader, device, class_names)
        target_logits = _daeac_target_logits(model, target_val_loader, device)
        v_result = ericsson_v_measure(
            source_result["logits"], source_result["y_true"], target_logits,
            num_classes=int(config["data"]["num_classes"]), random_state=int(config.get("seed", 42)),
        )
        row.update(aggregate_v_measure(v_result))
        save_v_measure_assignments(ckpt_dir / f"{prefix}_latest_v_measure_assignments.npz", v_result)
        min_delta = float(config.get("validation", {}).get("min_delta", 1e-4))
        if bool(row["valid"]) and row["v_measure"] > best_v_measure + min_delta:
            best_v_measure = float(row["v_measure"])
            best_epoch = epoch + 1
            stale_epochs = 0
            save_daeac_checkpoint(model, config, best_path, epoch + 1, row)
            save_v_measure_assignments(ckpt_dir / f"{prefix}_best_v_measure_assignments.npz", v_result)
        else:
            stale_epochs += 1
        if bool(row["losses_stable"]) and first_stable_epoch is None:
            first_stable_epoch = epoch + 1
        row["first_stable_epoch"] = first_stable_epoch
        row["epochs_since_first_stable"] = 0 if first_stable_epoch is None else epoch + 1 - first_stable_epoch
        row["epoch_seconds"] = float(time.perf_counter() - epoch_started)
        history.append(row)
        _write_history_csv(history, history_csv)
        log_row = {
            f"adapt/{k}": v
            for k, v in row.items()
            if k not in {"epoch", "pseudo_counts"} and isinstance(v, (int, float, bool)) and v is not None
        }
        for idx, count in enumerate(row["pseudo_counts"]):
            log_row[f"adapt/pseudo_count_{idx}"] = count
        wandb_run.log(log_row, step=epoch)
        save_daeac_checkpoint(model, config, latest_path, epoch + 1, row)
        print(
            f"[uda epoch {epoch + 1}/{cfg['epochs']}] steps={row['iterations']} global_step={global_step} "
            f"lr={row['lr']:.8g} seconds={row['epoch_seconds']:.2f}\n"
            f"  losses(mean): total={row['loss']:.6f} cls={row['loss_cls']:.6f} "
            f"align={row['loss_align']:.6f} sep={row['loss_sep']:.6f} "
            f"comp={row['loss_comp']:.6f} main={row['loss_main']:.6f}\n"
            f"  weighted: beta1*align={row['weighted_align']:.6f} "
            f"beta2*sep={row['weighted_sep']:.6f} beta2*comp={row['weighted_comp']:.6f} "
            f"beta2*main={row['weighted_main']:.6f}\n"
            f"  pseudo: snapshot={row['pseudo_snapshot_origin']} selected={row['pseudo_total']}/{row['target_total']} "
            f"coverage={row['pseudo_coverage']:.6f} active_classes={row['pseudo_active_classes']} "
            f"mean_conf={row['pseudo_mean_confidence']:.6f} mean_entropy={row['pseudo_mean_normalized_entropy']:.6f}\n"
            f"  pseudo_by_class: "
            + " ".join(
                f"{name}={row[f'pseudo_count_{name}']}({row[f'pseudo_rate_{name}']:.4f})"
                for name in class_names
            )
            + "\n  center_align_by_class: "
            + " ".join(
                f"{name}={row[f'center_align_{name}']:.6f}" if row[f'center_align_{name}'] is not None else f"{name}=NA"
                for name in class_names
            )
            + f"\n  validation: v_measure={row['v_measure']:.6f} valid={row['valid']}"
        )
        # Epoch boundary: synchronize h <- H, then freeze a complete target
        # pseudo-label snapshot only when another epoch will consume it.
        if epoch + 1 < int(cfg["epochs"]):
            aux_classifier = copy.deepcopy(model.classifier).to(device).eval()
            try:
                pseudo_dataset = build_pseudo_labeled_target_dataset(
                    model, aux_classifier, target_dataset, target_inference_loader, thresholds, device
                )
                pseudo_snapshot_origin = f"refreshed_after_epoch_{epoch + 1}"
            except RuntimeError as exc:
                if "No target samples passed" not in str(exc):
                    raise
                print(
                    f"[uda epoch {epoch + 1}] WARNING: refreshed pseudo-label set is empty; "
                    "retaining the previous epoch's valid snapshot."
                )
                pseudo_snapshot_origin = f"retained_after_empty_refresh_epoch_{epoch + 1}"
        if epoch + 1 >= int(config.get("validation", {}).get("min_epochs", 20)) and stale_epochs >= int(config.get("validation", {}).get("patience", 10)):
            break

    summary = {
        "latest_checkpoint": str(latest_path),
        "official_checkpoint": str(best_path),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_v_measure": best_v_measure,
        "checkpoint_policy": "maximum_ericsson_v_measure_source_val_plus_target_val_logits",
        "adaptation_monitoring": "losses_pseudo_labels_and_ericsson_v_measure",
        "history_csv": str(history_csv),
        "first_stable_epoch": first_stable_epoch,
        "history": history,
    }
    wandb_run.summary_update(summary)
    wandb_run.finish()
    return summary


@torch.no_grad()
def _daeac_target_logits(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        _, logits, _ = model(x.to(device), return_logits=True)
        values.append(logits.detach().cpu().numpy())
    return np.concatenate(values)


def evaluate_daeac_model(
    model: DAEACNetwork,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    features_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[:2]
            x = x.to(device)
            features, logits, probs = model(x, return_logits=True)
            probs_cpu = probs.detach().cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(probs_cpu.argmax(axis=1))
            probs_all.append(probs_cpu)
            logits_all.append(logits.detach().cpu().numpy())
            features_all.append(features.detach().cpu().numpy())
    true = np.concatenate(y_true) if y_true else np.zeros(0, dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=np.int64)
    probs = np.concatenate(probs_all) if probs_all else np.zeros((0, len(class_names)), dtype=np.float32)
    logits = np.concatenate(logits_all) if logits_all else np.zeros((0, len(class_names)), dtype=np.float32)
    features = np.concatenate(features_all) if features_all else np.zeros((0, 256), dtype=np.float32)
    return {
        "y_true": true,
        "y_pred": pred,
        "probabilities": probs,
        "logits": logits,
        "features": features,
        "metrics": daeac_metrics(true, pred, class_names),
    }


def daeac_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_pred, class_names)
    per_class = {}
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        se = tp / max(tp + fn, 1)
        pp = tp / max(tp + fp, 1)
        f1 = 2 * se * pp / max(se + pp, 1e-12)
        per_class[name] = {"Se": float(se), "Pp": float(pp), "F1": float(f1), "support": int(cm[idx, :].sum())}
    metrics["paper_metrics"] = {"accuracy": metrics["accuracy"], "per_class": per_class}
    return metrics


def save_daeac_checkpoint(model: DAEACNetwork, config: dict[str, Any], path: str | Path, epoch: int, metrics: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": int(epoch),
            "metrics": metrics,
        },
        path,
    )


def load_daeac_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
    model: DAEACNetwork | None = None,
) -> DAEACNetwork:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model or build_daeac_model(config, device)
    state_dict = _prepare_checkpoint_state_dict_for_model(checkpoint["model_state_dict"], model)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"adaptation_fc.weight", "adaptation_fc.bias"} if model.adaptation_fc_enabled else set()
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model state mismatch: "
            f"missing={sorted(unexpected_missing)}, unexpected={sorted(incompatible.unexpected_keys)}"
        )
    model.to(device)
    return model


def _source_classification_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    cls_loss_fn: nn.Module | None,
    config: dict[str, Any],
) -> torch.Tensor:
    dual_cfg = dict(config.get("rtd_daeac", {}).get("dual_head", {}))
    if bool(dual_cfg.get("enabled", False)) and "logits_1" in output and "logits_2" in output:
        loss_1 = weighted_cross_entropy_from_logits(output["logits_1"], labels, class_weights)
        loss_2 = weighted_cross_entropy_from_logits(output["logits_2"], labels, class_weights)
        loss = 0.5 * (loss_1 + loss_2)
        consistency_weight = float(dual_cfg.get("consistency_weight", 0.0))
        if consistency_weight > 0.0:
            probs_1 = torch.softmax(output["logits_1"], dim=1)
            probs_2 = torch.softmax(output["logits_2"], dim=1)
            loss = loss + consistency_weight * F.mse_loss(probs_1, probs_2)
        return loss
    if cls_loss_fn is not None:
        return cls_loss_fn(output["logits"], labels)
    return weighted_cross_entropy_from_logits(output["logits"], labels, class_weights)


def _prepare_checkpoint_state_dict_for_model(state_dict: dict[str, torch.Tensor], model: DAEACNetwork) -> dict[str, torch.Tensor]:
    prepared = dict(state_dict)
    if isinstance(model.classifier, DualClassifierH):
        copies = {
            "classifier.fc2.weight": "classifier.fc.weight",
            "classifier.fc2.bias": "classifier.fc.bias",
        }
        for missing_key, source_key in copies.items():
            if missing_key not in prepared and source_key in prepared:
                prepared[missing_key] = prepared[source_key].clone()
    return prepared


class CenterMemory:
    def __init__(self, num_classes: int, feature_dim: int, device: torch.device):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.device = device
        self.source: list[torch.Tensor | None] = [None for _ in range(self.num_classes)]
        self.target: list[torch.Tensor | None] = [None for _ in range(self.num_classes)]
        self.mixed: list[torch.Tensor | None] = [None for _ in range(self.num_classes)]

    def refresh_mixed(self) -> None:
        self.mixed = [
            (cs + ct) / 2 if cs is not None and ct is not None else None
            for cs, ct in zip(self.source, self.target)
        ]

    def centers_for_loss(
        self,
        local_source: list[torch.Tensor | None],
        local_target: list[torch.Tensor | None],
        gamma: float,
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], list[torch.Tensor | None]]:
        source: list[torch.Tensor | None] = []
        target: list[torch.Tensor | None] = []
        mixed: list[torch.Tensor | None] = []
        for cls in range(self.num_classes):
            cs = _ema_center(self.source[cls], local_source[cls], gamma)
            ct = _ema_center(self.target[cls], local_target[cls], gamma)
            source.append(cs)
            target.append(ct)
            mixed.append((cs + ct) / 2 if cs is not None and ct is not None else None)
        return source, target, mixed

    def commit(
        self,
        source: list[torch.Tensor | None],
        target: list[torch.Tensor | None],
        mixed: list[torch.Tensor | None],
    ) -> None:
        self.source = [center.detach() if center is not None else None for center in source]
        self.target = [center.detach() if center is not None else None for center in target]
        self.mixed = [center.detach() if center is not None else None for center in mixed]


def compute_global_source_centers(
    model: DAEACNetwork,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> list[torch.Tensor | None]:
    sums = [torch.zeros(model.feature_dim, device=device) for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            features = model.extract_features(x.to(device))
            y = y.to(device)
            for cls in range(num_classes):
                mask = y == cls
                if bool(mask.any()):
                    sums[cls] += features[mask].sum(dim=0)
                    counts[cls] += int(mask.sum().item())
    return [sums[cls] / counts[cls] if counts[cls] > 0 else None for cls in range(num_classes)]


def compute_global_target_centers(
    model: DAEACNetwork,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    thresholds: torch.Tensor,
) -> list[torch.Tensor | None]:
    sums = [torch.zeros(model.feature_dim, device=device) for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            features, _, probs = model(x.to(device), return_logits=True)
            conf, pseudo = probs.max(dim=1)
            confident = conf > thresholds[pseudo]
            for cls in range(num_classes):
                mask = (pseudo == cls) & confident
                if bool(mask.any()):
                    sums[cls] += features[mask].sum(dim=0)
                    counts[cls] += int(mask.sum().item())
    return [sums[cls] / counts[cls] if counts[cls] > 0 else None for cls in range(num_classes)]


def compute_global_pseudo_target_centers(
    model: DAEACNetwork,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> list[torch.Tensor | None]:
    sums = [torch.zeros(model.feature_dim, device=device) for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for x, pseudo, *_ in loader:
            features = model.extract_features(x.to(device))
            pseudo = pseudo.to(device)
            for cls in range(num_classes):
                mask = pseudo == cls
                if bool(mask.any()):
                    sums[cls] += features[mask].sum(dim=0)
                    counts[cls] += int(mask.sum().item())
    return [sums[cls] / counts[cls] if counts[cls] > 0 else None for cls in range(num_classes)]


def build_pseudo_labeled_target_dataset(
    model: DAEACNetwork,
    aux_classifier: ClassifierH,
    target_dataset: Dataset,
    inference_loader: DataLoader,
    thresholds: torch.Tensor,
    device: torch.device,
) -> DAEACPseudoLabeledDataset:
    """Infer all target samples once and return an immutable confident subset."""
    positions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    offset = 0
    model.eval()
    aux_classifier.eval()
    with torch.no_grad():
        for batch in inference_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            features = model.extract_features(x)
            _, probabilities = aux_classifier(features, return_logits=True)
            confidence, pseudo = probabilities.max(dim=1)
            accepted = confidence > thresholds[pseudo]
            entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=1)
            entropy = entropy / np.log(probabilities.shape[1])
            local_positions = torch.arange(offset, offset + len(x), device=device)
            positions.append(local_positions[accepted].cpu())
            labels.append(pseudo[accepted].cpu())
            confidences.append(confidence[accepted].cpu())
            entropies.append(entropy[accepted].cpu())
            offset += len(x)
    accepted_count = sum(len(value) for value in labels)
    if accepted_count == 0:
        raise RuntimeError("No target samples passed the strict class-specific pseudo-label thresholds.")
    return DAEACPseudoLabeledDataset(
        target_dataset,
        torch.cat(positions),
        torch.cat(labels),
        torch.cat(confidences),
        torch.cat(entropies),
    )


def batch_centers(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> list[torch.Tensor | None]:
    centers: list[torch.Tensor | None] = []
    for cls in range(num_classes):
        mask = labels == cls
        centers.append(features[mask].mean(dim=0) if bool(mask.any()) else None)
    return centers


def _ema_center(old: torch.Tensor | None, local: torch.Tensor | None, gamma: float) -> torch.Tensor | None:
    if old is None and local is None:
        return None
    if old is None:
        return local
    if local is None:
        return old.detach()
    return (1.0 - float(gamma)) * old.detach() + float(gamma) * local


def _class_weights(dataset: DAEACDataset | Subset, config: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    if not bool(cfg.get("use_class_weights", True)):
        return None
    labels = _dataset_labels(dataset)
    if labels is None:
        return None
    num_classes = int(config["data"]["num_classes"])
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _dataset_labels(dataset: DAEACDataset | Subset) -> np.ndarray | None:
    if isinstance(dataset, Subset):
        parent = dataset.dataset
        if isinstance(parent, DAEACDataset) and parent.y is not None:
            return parent.y[np.asarray(dataset.indices, dtype=np.int64)]
        return None
    if isinstance(dataset, DAEACDataset):
        return dataset.y
    return None


def _threshold_tensor(config: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    class_names = list(config["data"]["class_names"])
    values = [float(cfg["pseudo_thresholds"][name]) for name in class_names]
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _cluster_align_loss(
    source_centers: list[torch.Tensor | None],
    target_centers: list[torch.Tensor | None],
    cfg: dict[str, Any],
    distance_fn,
    device: torch.device,
) -> torch.Tensor:
    align_loss = str(cfg.get("align_loss", "l2")).lower()
    if align_loss in {"l2", "distance"}:
        return cluster_aligning_loss(
            source_centers,
            target_centers,
            distance_fn,
            device,
            reduction=str(cfg.get("cluster_loss_reduction", "sum")),
        )
    if align_loss == "mkmmd_center":
        return center_cluster_mk_mmd_loss(source_centers, target_centers, dict(cfg.get("mkmmd", {})), device)
    raise ValueError(f"Unknown DAEAC align_loss: {align_loss}")


def _prepare_center_mkmmd_config(cfg: dict[str, Any], center_memory: CenterMemory) -> None:
    if str(cfg.get("align_loss", "l2")).lower() != "mkmmd_center":
        return
    mkmmd_cfg = cfg.setdefault("mkmmd", {})
    mode = str(mkmmd_cfg.get("gamma_mode", "adaptive")).lower()
    if mode != "fixed_from_initial_centers":
        return
    gamma_min = float(mkmmd_cfg.get("gamma_min", 1.0e-6))
    reference = center_pair_reference_distance(center_memory.source, center_memory.target, gamma_min=gamma_min)
    fixed_gamma = float(reference.detach().cpu()) if reference is not None else gamma_min
    mkmmd_cfg["fixed_gamma"] = fixed_gamma
    print(f"Center MK-MMD fixed_gamma={fixed_gamma:.6g}")


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _epoch_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"loss": 0.0, "loss_cls": 0.0, "loss_align": 0.0, "loss_sep": 0.0, "loss_comp": 0.0, "pseudo_selected": 0.0}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _detailed_epoch_loss_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    loss_keys = ("loss", "loss_cls", "loss_align", "loss_sep", "loss_comp")
    if not rows:
        result = {key: 0.0 for key in loss_keys}
        for key in loss_keys:
            result.update({f"{key}_std": 0.0, f"{key}_min": 0.0, f"{key}_max": 0.0})
        return result
    result: dict[str, float] = {}
    for key in loss_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = float(values.mean())
        result[f"{key}_std"] = float(values.std())
        result[f"{key}_min"] = float(values.min())
        result[f"{key}_max"] = float(values.max())
    result["pseudo_selected_per_batch"] = float(np.mean([row["pseudo_selected"] for row in rows]))
    return result


def _pseudo_snapshot_diagnostics(
    pseudo_dataset: DAEACPseudoLabeledDataset,
    target_total: int,
    class_names: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    labels = pseudo_dataset.labels.numpy()
    confidences = pseudo_dataset.confidence.numpy()
    entropies = pseudo_dataset.normalized_entropy.numpy()
    counts = np.bincount(labels, minlength=len(class_names)).astype(np.int64)
    selected = int(len(labels))
    result: dict[str, Any] = {
        "target_total": int(target_total),
        "pseudo_total": selected,
        "pseudo_coverage": float(selected / max(int(target_total), 1)),
        "pseudo_active_classes": int(np.count_nonzero(counts)),
        "pseudo_mean_confidence": float(confidences.mean()) if selected else 0.0,
        "pseudo_min_confidence": float(confidences.min()) if selected else 0.0,
        "pseudo_max_confidence": float(confidences.max()) if selected else 0.0,
        "pseudo_mean_normalized_entropy": float(entropies.mean()) if selected else 0.0,
        "pseudo_min_normalized_entropy": float(entropies.min()) if selected else 0.0,
        "pseudo_max_normalized_entropy": float(entropies.max()) if selected else 0.0,
    }
    for idx, name in enumerate(class_names):
        mask = labels == idx
        class_conf = confidences[mask]
        class_entropy = entropies[mask]
        result[f"pseudo_threshold_{name}"] = float(cfg["pseudo_thresholds"][name])
        result[f"pseudo_count_{name}"] = int(counts[idx])
        result[f"pseudo_rate_{name}"] = float(counts[idx] / max(selected, 1))
        result[f"pseudo_coverage_{name}"] = float(counts[idx] / max(int(target_total), 1))
        result[f"pseudo_mean_confidence_{name}"] = float(class_conf.mean()) if len(class_conf) else 0.0
        result[f"pseudo_mean_entropy_{name}"] = float(class_entropy.mean()) if len(class_entropy) else 0.0
    return result


def _center_diagnostics(center_memory: CenterMemory, class_names: list[str], distance_fn) -> dict[str, Any]:
    result: dict[str, Any] = {}
    valid_pairs = 0
    for idx, name in enumerate(class_names):
        cs = center_memory.source[idx]
        ct = center_memory.target[idx]
        cm = center_memory.mixed[idx]
        result[f"center_source_present_{name}"] = cs is not None
        result[f"center_target_present_{name}"] = ct is not None
        result[f"center_source_norm_{name}"] = float(torch.linalg.vector_norm(cs).cpu()) if cs is not None else None
        result[f"center_target_norm_{name}"] = float(torch.linalg.vector_norm(ct).cpu()) if ct is not None else None
        result[f"center_mixed_norm_{name}"] = float(torch.linalg.vector_norm(cm).cpu()) if cm is not None else None
        if cs is not None and ct is not None:
            result[f"center_align_{name}"] = float(distance_fn(cs, ct).detach().cpu())
            valid_pairs += 1
        else:
            result[f"center_align_{name}"] = None
    result["center_valid_class_pairs"] = valid_pairs
    return result


def _stability_diagnostics(history: list[dict[str, Any]], row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    monitor_cfg = dict(cfg.get("monitoring", {}))
    window = int(monitor_cfg.get("stability_window", 10))
    min_epoch = int(monitor_cfg.get("stability_min_epoch", 20))
    tolerance = float(monitor_cfg.get("stability_cv_tolerance", 0.05))
    rows = [*history, row]
    ready = len(rows) >= window and int(row["epoch"]) >= min_epoch
    if not ready:
        return {
            "stability_window": window,
            "stability_window_ready": False,
            "stability_cv_tolerance": tolerance,
            "loss_align_cv": -1.0,
            "loss_main_cv": -1.0,
            "losses_stable": False,
        }
    recent = rows[-window:]

    def cv(key: str) -> float:
        values = np.asarray([float(item[key]) for item in recent], dtype=np.float64)
        return float(values.std() / max(abs(values.mean()), 1.0e-12))

    align_cv = cv("loss_align")
    main_cv = cv("loss_main")
    return {
        "stability_window": window,
        "stability_window_ready": True,
        "stability_cv_tolerance": tolerance,
        "loss_align_cv": align_cv,
        "loss_main_cv": main_cv,
        "losses_stable": bool(align_cv <= tolerance and main_cv <= tolerance),
    }


def _write_history_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
