from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset
from src.models.daeac_paper import ClassifierH, DAEACNetwork
from src.training.daeac_losses import (
    cluster_aligning_loss,
    compacting_loss,
    distance_from_name,
    separating_loss,
    weighted_cross_entropy_from_logits,
)
from src.training.mk_mmd import center_cluster_mk_mmd_loss, center_pair_reference_distance
from src.training.metrics import classification_metrics
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb


def build_daeac_model(config: dict[str, Any], device: torch.device) -> DAEACNetwork:
    model_cfg = config["model"]
    return DAEACNetwork(
        num_classes=int(model_cfg["num_classes"]),
        input_channels=int(model_cfg.get("input_channels", 1)),
        initial_channels=int(model_cfg.get("initial_channels", 4)),
        feature_dim=int(model_cfg.get("feature_dim", 256)),
        dilations=tuple(int(v) for v in model_cfg.get("dilations", [1, 6, 12, 18])),
        se_reduction=int(model_cfg.get("se_reduction", 16)),
        dropout=float(model_cfg.get("dropout", 0.0)),
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
            _, logits, _ = model(x, return_logits=True)
            loss = weighted_cross_entropy_from_logits(logits, y, class_weights)
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
    val_dataset: DAEACDataset,
    target_dataset: DAEACTargetUnlabeledDataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = config["adaptation"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    prefix = str(cfg.get("checkpoint_prefix", "daeac_uda"))
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
    distance_fn = distance_from_name(str(cfg.get("distance", "l2")))
    thresholds = _threshold_tensor(config, cfg, device)
    center_memory = CenterMemory(int(config["data"]["num_classes"]), int(config["model"]["feature_dim"]), device)
    center_memory.source = compute_global_source_centers(model, source_loader, device, center_memory.num_classes)
    center_memory.target = compute_global_target_centers(model, target_loader, device, center_memory.num_classes, thresholds)
    center_memory.refresh_mixed()
    _prepare_center_mkmmd_config(cfg, center_memory)
    aux_classifier = ClassifierH(
        feature_dim=int(config["model"]["feature_dim"]),
        num_classes=int(config["data"]["num_classes"]),
        dropout=0.0,
    ).to(device)
    wandb_run = init_wandb(config, job_type="adapt_daeac", default_name=prefix)

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    best_macro_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(int(cfg["epochs"])):
        model.train()
        aux_classifier.load_state_dict(copy.deepcopy(model.classifier.state_dict()))
        aux_classifier.eval()
        target_iter = _cycle(target_loader)
        epoch_rows: list[dict[str, float]] = []
        pseudo_counts = np.zeros(center_memory.num_classes, dtype=np.int64)
        for x_s, y_s in source_loader:
            x_t_batch = next(target_iter)
            x_t = x_t_batch[0] if isinstance(x_t_batch, (tuple, list)) else x_t_batch
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            z_s, logits_s, _ = model(x_s, return_logits=True)
            loss_cls = weighted_cross_entropy_from_logits(logits_s, y_s, class_weights)

            with torch.no_grad():
                z_t_all = model.extract_features(x_t)
                _, probs_t = aux_classifier(z_t_all, return_logits=True)
                conf_t, pseudo_t = probs_t.max(dim=1)
                confident = conf_t >= thresholds[pseudo_t]

            if bool(confident.any()):
                selected_x_t = x_t[confident]
                selected_pseudo_t = pseudo_t[confident]
                z_t = model.extract_features(selected_x_t)
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
                    "pseudo_selected": float(confident.sum().detach().cpu()),
                }
            )

        val_result = evaluate_daeac_model(model, val_loader, device, config["data"]["class_names"])
        row = _epoch_summary(epoch_rows)
        row.update(
            {
                "epoch": epoch,
                "val_accuracy": val_result["metrics"]["accuracy"],
                "val_macro_f1": val_result["metrics"]["macro_f1"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "pseudo_counts": pseudo_counts.astype(int).tolist(),
            }
        )
        history.append(row)
        log_row = {f"adapt/{k}": v for k, v in row.items() if k not in {"epoch", "pseudo_counts"}}
        for idx, count in enumerate(row["pseudo_counts"]):
            log_row[f"adapt/pseudo_count_{idx}"] = count
        wandb_run.log(log_row, step=epoch)
        if row["val_macro_f1"] >= best_macro_f1:
            best_macro_f1 = float(row["val_macro_f1"])
            best_epoch = epoch
            save_daeac_checkpoint(model, config, best_path, epoch, row)
        save_daeac_checkpoint(model, config, latest_path, epoch, row)
        print(
            f"[uda epoch {epoch + 1}/{cfg['epochs']}] loss={row['loss']:.4f} "
            f"align={row['loss_align']:.4f} sep={row['loss_sep']:.4f} comp={row['loss_comp']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} pseudo={row['pseudo_counts']}"
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
    features_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[:2]
            x = x.to(device)
            features, _, probs = model(x, return_logits=True)
            probs_cpu = probs.detach().cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(probs_cpu.argmax(axis=1))
            probs_all.append(probs_cpu)
            features_all.append(features.detach().cpu().numpy())
    true = np.concatenate(y_true) if y_true else np.zeros(0, dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=np.int64)
    probs = np.concatenate(probs_all) if probs_all else np.zeros((0, len(class_names)), dtype=np.float32)
    features = np.concatenate(features_all) if features_all else np.zeros((0, 256), dtype=np.float32)
    return {
        "y_true": true,
        "y_pred": pred,
        "probabilities": probs,
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


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
            confident = conf >= thresholds[pseudo]
            for cls in range(num_classes):
                mask = (pseudo == cls) & confident
                if bool(mask.any()):
                    sums[cls] += features[mask].sum(dim=0)
                    counts[cls] += int(mask.sum().item())
    return [sums[cls] / counts[cls] if counts[cls] > 0 else None for cls in range(num_classes)]


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
        return cluster_aligning_loss(source_centers, target_centers, distance_fn, device)
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
