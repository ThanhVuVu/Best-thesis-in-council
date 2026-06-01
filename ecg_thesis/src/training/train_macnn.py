from __future__ import annotations

import copy
import csv
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import build_model
from src.training.metrics import classification_metrics
from src.training.train import compute_class_weights
from src.utils.io import ensure_dir


def macnn_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[1]
    return out


def macnn_features_logits(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(x)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    if hasattr(model, "forward_features") and hasattr(model, "classifier"):
        features = model.forward_features(x)
        return features, model.classifier(features)
    raise TypeError("Model does not expose MACNN features/logits")


def train_macnn_source_only(train_dataset, val_dataset, config: dict[str, Any], output_dir: str | Path, device: torch.device) -> dict[str, Any]:
    cfg = config["source_only"]
    model_cfg = config["model"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    prefix = cfg.get("checkpoint_prefix", "macnn_se_source_only")
    model = build_model(cfg["model"], num_classes=int(model_cfg["num_classes"]), **_model_kwargs(model_cfg)).to(device)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    labels = _dataset_labels(train_dataset)
    weights = compute_class_weights(labels, num_classes=int(model_cfg["num_classes"])).to(device) if cfg.get("use_class_weights", True) else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)

    best_f1 = -1.0
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses, y_true, y_pred = [], [], []
        for x, y in tqdm(train_loader, desc=f"{prefix} epoch {epoch}", dynamic_ncols=True):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = macnn_logits(model, x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(logits.argmax(1).detach().cpu().numpy())
        train_metrics = classification_metrics(np.concatenate(y_true), np.concatenate(y_pred), config["data"]["class_names"])
        val_result = evaluate_macnn_model(model, val_loader, device, config["data"]["class_names"], desc=f"{prefix} val")
        val_metrics = val_result["metrics"]
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
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale = 0
            _save_checkpoint(model, optimizer, scheduler, config, cfg["model"], epoch, best_f1, best_epoch, history, best_path)
        else:
            stale += 1
        _save_checkpoint(model, optimizer, scheduler, config, cfg["model"], epoch, best_f1, best_epoch, history, latest_path)
        print(f"{prefix} epoch {epoch}: val_macro_f1={val_metrics['macro_f1']:.4f}, best={best_f1:.4f}")
        if stale >= int(cfg["early_stopping_patience"]):
            break
    _write_history_csv(history, log_dir / f"{prefix}_train_log.csv")
    return {"best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path), "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "history": history}


def train_macnn_daeac(source_dataset, source_val_dataset, target_dataset, config: dict[str, Any], output_dir: str | Path, device: torch.device) -> dict[str, Any]:
    cfg = config["daeac"]
    model_cfg = config["model"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    prefix = cfg.get("checkpoint_prefix", "macnn_se_daeac")
    model = build_model("macnn_se", num_classes=int(model_cfg["num_classes"]), **_model_kwargs(model_cfg)).to(device)
    _load_macnn_init(model, cfg.get("init_checkpoint"), config, device)
    aux = copy.deepcopy(model).to(device)
    for param in aux.parameters():
        param.requires_grad = False

    source_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    target_loader = DataLoader(target_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(source_val_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    labels = _dataset_labels(source_dataset)
    weights = compute_class_weights(labels, num_classes=int(model_cfg["num_classes"])).to(device)
    cls_loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    feature_params = [p for name, p in model.named_parameters() if not name.startswith("classifier.")]
    classifier_params = list(model.classifier.parameters())
    optimizer_f = torch.optim.Adam(feature_params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    optimizer_h = torch.optim.Adam(classifier_params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=int(cfg["lr_decay_every_steps"]), gamma=float(cfg["lr_decay_gamma"]))
    scheduler_h = torch.optim.lr_scheduler.StepLR(optimizer_h, step_size=int(cfg["lr_decay_every_steps"]), gamma=float(cfg["lr_decay_gamma"]))
    prototypes = PrototypeBank(int(model_cfg["num_classes"]), int(model_cfg["embedding_dim"]), float(cfg["ema_momentum"]), device)
    thresholds = _threshold_tensor(cfg["pseudo_thresholds"], config["data"]["class_names"], device)

    best_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    steps = max(len(source_loader), len(target_loader))
    for epoch in range(1, int(cfg["epochs"]) + 1):
        aux.load_state_dict(model.state_dict())
        aux.eval()
        model.train()
        source_iter = cycle(source_loader) if len(source_loader) < steps else iter(source_loader)
        target_iter = cycle(target_loader) if len(target_loader) < steps else iter(target_loader)
        rows = []
        for _ in tqdm(range(steps), desc=f"{prefix} epoch {epoch}", dynamic_ncols=True):
            x_s, y_s = next(source_iter)
            x_t, _ = next(target_iter)
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)
            with torch.no_grad():
                _, aux_logits_t = macnn_features_logits(aux, x_t)
                probs_t = torch.softmax(aux_logits_t, dim=1)
                conf_t, pseudo_t = probs_t.max(dim=1)
                confident = conf_t >= thresholds[pseudo_t]

            optimizer_f.zero_grad(set_to_none=True)
            optimizer_h.zero_grad(set_to_none=True)
            f_s, logits_s = macnn_features_logits(model, x_s)
            f_t, _ = macnn_features_logits(model, x_t)
            loss_cls = cls_loss_fn(logits_s, y_s)
            prototypes.update(f_s.detach(), y_s, f_t.detach(), pseudo_t, confident)
            centers = _batch_centers(f_s, y_s, f_t, pseudo_t, confident, int(model_cfg["num_classes"]))
            loss_align = _align_loss(centers) if cfg.get("use_align", True) else f_s.sum() * 0.0
            loss_sep = _separation_loss(centers, float(cfg["margin"])) if cfg.get("use_separation", True) else f_s.sum() * 0.0
            loss_comp = _compactness_loss(f_s, y_s, f_t, pseudo_t, confident, centers) if cfg.get("use_compact", True) else f_s.sum() * 0.0
            loss_proto = float(cfg["beta1"]) * loss_align + float(cfg["beta2"]) * (loss_sep + loss_comp)
            loss_total = loss_cls + loss_proto
            loss_total.backward()
            optimizer_f.step()
            optimizer_h.step()
            scheduler_f.step()
            scheduler_h.step()
            rows.append({
                "loss": float(loss_total.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_align": float(loss_align.detach().cpu()),
                "loss_sep": float(loss_sep.detach().cpu()),
                "loss_comp": float(loss_comp.detach().cpu()),
                "confident_target": int(confident.sum().detach().cpu()),
            })
        val_result = evaluate_macnn_model(model, val_loader, device, config["data"]["class_names"], desc=f"{prefix} val")
        val_metrics = val_result["metrics"]
        row = {
            "epoch": epoch,
            "source_val_macro_f1": val_metrics["macro_f1"],
            "source_val_accuracy": val_metrics["accuracy"],
            "lr": optimizer_f.param_groups[0]["lr"],
        }
        for key in ("loss", "loss_cls", "loss_align", "loss_sep", "loss_comp", "confident_target"):
            row[key] = float(np.mean([r[key] for r in rows]))
        history.append(row)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            _save_checkpoint(model, optimizer_f, scheduler_f, config, "macnn_se", epoch, best_f1, best_epoch, history, best_path)
        _save_checkpoint(model, optimizer_f, scheduler_f, config, "macnn_se", epoch, best_f1, best_epoch, history, latest_path)
        print(f"{prefix} epoch {epoch}: val_macro_f1={val_metrics['macro_f1']:.4f}, confident_target={row['confident_target']:.1f}")
    _write_history_csv(history, log_dir / f"{prefix}_train_log.csv")
    return {"best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path), "best_epoch": best_epoch, "best_source_val_macro_f1": best_f1, "history": history}


@torch.no_grad()
def evaluate_macnn_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, class_names: list[str], desc: str = "evaluate", collect_embeddings: bool = False) -> dict[str, Any]:
    model.to(device)
    model.eval()
    y_true, y_pred, probs, embeddings = [], [], [], []
    for x, y in tqdm(loader, desc=desc, dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        features, logits = macnn_features_logits(model, x)
        prob = torch.softmax(logits, dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(prob.argmax(1).detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
        if collect_embeddings:
            embeddings.append(features.detach().cpu().numpy())
    result = {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "probabilities": np.concatenate(probs),
    }
    if collect_embeddings:
        result["embeddings"] = np.concatenate(embeddings)
    result["metrics"] = classification_metrics(result["y_true"], result["y_pred"], class_names)
    return result


def load_macnn_checkpoint(checkpoint_path: str | Path, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint.get("config", config).get("model", config["model"])
    model = build_model("macnn_se", num_classes=int(model_cfg["num_classes"]), **_model_kwargs(model_cfg)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class PrototypeBank:
    def __init__(self, num_classes: int, dim: int, momentum: float, device: torch.device):
        self.num_classes = num_classes
        self.dim = dim
        self.momentum = momentum
        self.source = torch.zeros(num_classes, dim, device=device)
        self.target = torch.zeros(num_classes, dim, device=device)
        self.source_seen = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.target_seen = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, f_s: torch.Tensor, y_s: torch.Tensor, f_t: torch.Tensor, pseudo_t: torch.Tensor, confident: torch.Tensor) -> dict[str, torch.Tensor]:
        for cls in range(self.num_classes):
            mask_s = y_s == cls
            if mask_s.any():
                center = f_s[mask_s].mean(dim=0)
                self.source[cls] = self._ema(self.source[cls], center, bool(self.source_seen[cls]))
                self.source_seen[cls] = True
            mask_t = (pseudo_t == cls) & confident
            if mask_t.any():
                center = f_t[mask_t].mean(dim=0)
                self.target[cls] = self._ema(self.target[cls], center, bool(self.target_seen[cls]))
                self.target_seen[cls] = True
        both = self.source_seen & self.target_seen
        combined = (self.source + self.target) / 2.0
        return {"source": self.source.clone(), "target": self.target.clone(), "combined": combined.clone(), "both": both.clone(), "source_seen": self.source_seen.clone(), "target_seen": self.target_seen.clone()}

    def _ema(self, old: torch.Tensor, new: torch.Tensor, seen: bool) -> torch.Tensor:
        if not seen:
            return new
        return self.momentum * old + (1.0 - self.momentum) * new


def _align_loss(centers: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = centers["both"]
    if not mask.any():
        return centers["source"].sum() * 0.0
    return torch.norm(centers["source"][mask] - centers["target"][mask], p=2, dim=1).mean()


def _batch_centers(
    f_s: torch.Tensor,
    y_s: torch.Tensor,
    f_t: torch.Tensor,
    pseudo_t: torch.Tensor,
    confident: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    source_centers = []
    target_centers = []
    source_seen = []
    target_seen = []
    for cls in range(num_classes):
        mask_s = y_s == cls
        mask_t = (pseudo_t == cls) & confident
        source_seen.append(mask_s.any())
        target_seen.append(mask_t.any())
        source_centers.append(f_s[mask_s].mean(dim=0) if mask_s.any() else torch.zeros_like(f_s[0]))
        target_centers.append(f_t[mask_t].mean(dim=0) if mask_t.any() else torch.zeros_like(f_t[0]))
    source = torch.stack(source_centers)
    target = torch.stack(target_centers)
    source_seen_t = torch.stack(source_seen)
    target_seen_t = torch.stack(target_seen)
    both = source_seen_t & target_seen_t
    combined = torch.where(
        both[:, None],
        (source + target) / 2.0,
        torch.where(source_seen_t[:, None], source, target),
    )
    return {
        "source": source,
        "target": target,
        "combined": combined,
        "both": both,
        "source_seen": source_seen_t,
        "target_seen": target_seen_t,
    }


def _separation_loss(centers: dict[str, torch.Tensor], margin: float) -> torch.Tensor:
    valid = centers["source_seen"] | centers["target_seen"]
    idx = torch.where(valid)[0]
    if len(idx) < 2:
        return centers["combined"].sum() * 0.0
    losses = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dist = torch.norm(centers["combined"][idx[i]] - centers["combined"][idx[j]], p=2)
            losses.append(torch.relu(torch.tensor(margin, device=dist.device) - dist))
    return torch.stack(losses).mean() if losses else centers["combined"].sum() * 0.0


def _compactness_loss(f_s: torch.Tensor, y_s: torch.Tensor, f_t: torch.Tensor, pseudo_t: torch.Tensor, confident: torch.Tensor, centers: dict[str, torch.Tensor]) -> torch.Tensor:
    combined = centers["combined"]
    losses = [torch.norm(f_s - combined[y_s], p=2, dim=1).mean()]
    if confident.any():
        losses.append(torch.norm(f_t[confident] - combined[pseudo_t[confident]], p=2, dim=1).mean())
    return torch.stack(losses).mean()


def _model_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    allowed = {"input_channels", "channels", "embedding_dim", "se_reduction", "dropout"}
    return {key: model_cfg[key] for key in allowed if key in model_cfg}


def _dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_labels = _dataset_labels(dataset.dataset)
        return parent_labels[np.asarray(dataset.indices)]
    return dataset.y


def _threshold_tensor(thresholds: dict[str, float], class_names: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor([float(thresholds[name]) for name in class_names], dtype=torch.float32, device=device)


def _load_macnn_init(model: torch.nn.Module, checkpoint_value: str | None, config: dict[str, Any], device: torch.device) -> None:
    if checkpoint_value in (None, "", "null", "None"):
        return
    path = Path(str(checkpoint_value))
    if not path.is_absolute():
        path = Path(config.get("_base_dir", ".")) / path
    if not path.exists():
        print(f"MACNN init checkpoint not found, training from scratch: {path}")
        return
    checkpoint = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Initialized MACNN from {path} (missing={len(missing)}, unexpected={len(unexpected)})")


def _save_checkpoint(model, optimizer, scheduler, config, model_name: str, epoch: int, best_f1: float, best_epoch: int, history: list[dict[str, Any]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_name": model_name,
            "epoch": epoch,
            "best_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "history": history,
            "config": config,
        },
        path,
    )


def _write_history_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
