from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.training.train_macnn import load_macnn_checkpoint, macnn_features_logits
from src.utils.io import ensure_dir

EPS = 1e-8


class IndexedTargetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        x = item[0]
        return x, torch.tensor(idx, dtype=torch.long)


def train_macnn_shot(
    target_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = config["shot"]
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    prefix = str(cfg.get("checkpoint_prefix", "macnn_se_shot"))

    model = load_macnn_checkpoint(_resolve_path(cfg["init_checkpoint"], config), config, device)
    for param in model.classifier.parameters():
        param.requires_grad = False
    model.classifier.eval()

    indexed_dataset = IndexedTargetDataset(target_dataset)
    loader = DataLoader(
        indexed_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    pseudo_loader = DataLoader(
        indexed_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    feature_params = [p for name, p in model.named_parameters() if not name.startswith("classifier.") and p.requires_grad]
    optimizer = torch.optim.Adam(feature_params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.get("lr_decay_every_steps", 200)),
        gamma=float(cfg.get("lr_decay_gamma", 0.99)),
    )

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_path = ckpt_dir / f"{prefix}_best.pt"
    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    total_epochs = int(cfg["epochs"])

    for epoch in range(1, total_epochs + 1):
        if bool(cfg.get("pseudo_labeling", True)):
            pseudo_labels, pseudo_summary = compute_shot_pseudo_labels(model, pseudo_loader, device, int(config["data"]["num_classes"]))
        else:
            pseudo_labels = None
            pseudo_summary = {"pseudo_counts": [0] * int(config["data"]["num_classes"])}

        model.train()
        model.classifier.eval()
        rows = []
        for x, idx in tqdm(loader, desc=f"{prefix} epoch {epoch}/{total_epochs}", dynamic_ncols=True):
            x = x.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _features, logits = macnn_features_logits(model, x)
            probs = torch.softmax(logits, dim=1)
            loss_ent = entropy_loss(probs)
            loss_div = diversity_loss(
                probs,
                mode=str(cfg.get("diversity_target", "uniform")),
                source_prior=cfg.get("source_prior"),
                device=device,
            )
            if pseudo_labels is not None and float(cfg.get("beta", 0.0)) > 0:
                pseudo_batch = torch.as_tensor(pseudo_labels[idx.detach().cpu().numpy()], dtype=torch.long, device=device)
                loss_pl = torch.nn.functional.cross_entropy(logits, pseudo_batch)
            else:
                loss_pl = logits.sum() * 0.0
            loss = (
                float(cfg.get("entropy_weight", 1.0)) * loss_ent
                + float(cfg.get("diversity_weight", 1.0)) * loss_div
                + float(cfg.get("beta", 0.3)) * loss_pl
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "loss_entropy": float(loss_ent.detach().cpu()),
                    "loss_diversity": float(loss_div.detach().cpu()),
                    "loss_pseudo": float(loss_pl.detach().cpu()),
                }
            )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "pseudo_counts": pseudo_summary["pseudo_counts"],
        }
        for key in ("loss", "loss_entropy", "loss_diversity", "loss_pseudo"):
            row[key] = float(np.mean([r[key] for r in rows]))
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = epoch
            _save_shot_checkpoint(model, optimizer, scheduler, config, epoch, best_loss, best_epoch, history, best_path)
        _save_shot_checkpoint(model, optimizer, scheduler, config, epoch, best_loss, best_epoch, history, latest_path)
        print(
            f"{prefix} epoch {epoch}: loss={row['loss']:.4f}, "
            f"ent={row['loss_entropy']:.4f}, div={row['loss_diversity']:.4f}, "
            f"pl={row['loss_pseudo']:.4f}, pseudo_counts={row['pseudo_counts']}"
        )

    _write_history_csv(history, log_dir / f"{prefix}_train_log.csv")
    return {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "best_epoch": best_epoch,
        "best_adaptation_loss": best_loss,
        "history": history,
        "selection_note": "best uses unlabeled adaptation loss only; latest is the paper-style fixed-epoch checkpoint",
    }


@torch.no_grad()
def compute_shot_pseudo_labels(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    features_all = []
    probs_all = []
    indices_all = []
    for x, idx in tqdm(loader, desc="SHOT pseudo labels", dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        features, logits = macnn_features_logits(model, x)
        features = torch.nn.functional.normalize(features, dim=1)
        probs = torch.softmax(logits, dim=1)
        features_all.append(features.detach().cpu())
        probs_all.append(probs.detach().cpu())
        indices_all.append(idx.detach().cpu())

    features = torch.cat(features_all, dim=0)
    probs = torch.cat(probs_all, dim=0)
    indices = torch.cat(indices_all, dim=0).numpy()
    centroids = _weighted_centroids(features, probs, num_classes)
    pseudo = _nearest_centroid(features, centroids)
    centroids = _hard_centroids(features, pseudo, centroids, num_classes)
    pseudo = _nearest_centroid(features, centroids).numpy().astype(np.int64)
    ordered = np.zeros(len(indices), dtype=np.int64)
    ordered[indices] = pseudo
    return ordered, {"pseudo_counts": np.bincount(ordered, minlength=num_classes).astype(int).tolist()}


def entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=1).mean()


def diversity_loss(
    probs: torch.Tensor,
    mode: str,
    source_prior: list[float] | None,
    device: torch.device,
) -> torch.Tensor:
    mode = mode.lower()
    if mode == "none":
        return probs.sum() * 0.0
    p_bar = probs.mean(dim=0).clamp_min(EPS)
    if mode == "source_prior":
        if source_prior is None:
            raise ValueError("SHOT diversity_target=source_prior requires shot.source_prior")
        prior = torch.as_tensor(source_prior, dtype=torch.float32, device=device)
        prior = prior / prior.sum().clamp_min(EPS)
        return (p_bar * (torch.log(p_bar) - torch.log(prior.clamp_min(EPS)))).sum()
    if mode == "uniform":
        return (p_bar * torch.log(p_bar)).sum()
    raise ValueError(f"Unsupported SHOT diversity_target: {mode}")


def _weighted_centroids(features: torch.Tensor, probs: torch.Tensor, num_classes: int) -> torch.Tensor:
    centroids = []
    for cls in range(num_classes):
        weight = probs[:, cls : cls + 1]
        denom = weight.sum().clamp_min(EPS)
        centroids.append((weight * features).sum(dim=0) / denom)
    return torch.nn.functional.normalize(torch.stack(centroids, dim=0), dim=1)


def _hard_centroids(features: torch.Tensor, pseudo: torch.Tensor, fallback: torch.Tensor, num_classes: int) -> torch.Tensor:
    centroids = []
    for cls in range(num_classes):
        mask = pseudo == cls
        if mask.any():
            centroids.append(features[mask].mean(dim=0))
        else:
            centroids.append(fallback[cls])
    return torch.nn.functional.normalize(torch.stack(centroids, dim=0), dim=1)


def _nearest_centroid(features: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    sims = features @ centroids.T
    return sims.argmax(dim=1)


def _resolve_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config.get("_base_dir", ".")) / path


def _save_shot_checkpoint(model, optimizer, scheduler, config, epoch: int, best_loss: float, best_epoch: int, history: list[dict[str, Any]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_name": "macnn_se",
            "epoch": epoch,
            "best_adaptation_loss": best_loss,
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
    flat_rows = []
    for row in rows:
        flat = dict(row)
        if isinstance(flat.get("pseudo_counts"), list):
            for i, value in enumerate(flat.pop("pseudo_counts")):
                flat[f"pseudo_count_{i}"] = value
        flat_rows.append(flat)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
