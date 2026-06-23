from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class CustomFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | list[float] | tuple[float, ...] | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported focal loss reduction: {reduction}")
        self.gamma = float(gamma)
        self.reduction = str(reduction)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Focal loss expects logits shaped [B, C], got {tuple(logits.shape)}.")
        if labels.ndim != 1:
            raise ValueError(f"Focal loss expects labels shaped [B], got {tuple(labels.shape)}.")
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            if self.alpha.numel() != logits.size(1):
                raise ValueError(
                    f"Focal alpha length must match num_classes={logits.size(1)}, got {self.alpha.numel()}."
                )
            focal_loss = self.alpha.to(device=logits.device, dtype=logits.dtype)[labels] * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WeightedCrossEntropyByBatchSize(nn.Module):
    """Weighted CE matching Algorithm 1: -(1/N) sum_i w[y_i] log p_i[y_i]."""

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.numel() == 0:
            return logits.sum() * 0.0
        weighted_sum = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="sum",
        )
        return weighted_sum / labels.numel()


def weighted_cross_entropy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, labels, weight=class_weights, reduction="sum") / labels.numel()


def build_daeac_classification_loss(
    cfg: dict[str, Any],
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    source_loss = str(cfg.get("source_loss", "weighted_ce")).lower()
    if source_loss == "weighted_ce":
        return WeightedCrossEntropyByBatchSize(class_weights)
    if source_loss == "focal":
        alpha_cfg = cfg.get("focal_alpha")
        alpha = class_weights if alpha_cfg is None else torch.as_tensor(alpha_cfg, dtype=torch.float32)
        if alpha is not None and int(alpha.numel()) != int(num_classes):
            raise ValueError(f"focal_alpha length must match num_classes={num_classes}, got {int(alpha.numel())}.")
        return CustomFocalLoss(alpha=alpha, gamma=float(cfg.get("focal_gamma", 2.0)))
    raise ValueError(f"Unsupported DAEAC source_loss: {source_loss}")


def l2_distance(x: torch.Tensor, y: torch.Tensor, dim: int = -1, mean: bool = True) -> torch.Tensor:
    dist = torch.linalg.vector_norm(x - y, ord=2, dim=dim)
    return dist.mean() if mean else dist


def cosine_distance(x: torch.Tensor, y: torch.Tensor, dim: int = -1, mean: bool = True) -> torch.Tensor:
    dist = 1.0 - F.cosine_similarity(x, y, dim=dim)
    return dist.mean() if mean else dist


def cluster_aligning_loss(
    source_centers: list[torch.Tensor | None],
    target_centers: list[torch.Tensor | None],
    distance_fn,
    device: torch.device,
    reduction: str = "sum",
) -> torch.Tensor:
    losses = [
        distance_fn(cs, ct)
        for cs, ct in zip(source_centers, target_centers)
        if cs is not None and ct is not None
    ]
    return _reduce_losses(losses, reduction, device)


def separating_loss(
    mixed_centers: list[torch.Tensor | None],
    margin: float,
    distance_fn,
    device: torch.device,
    reduction: str = "sum",
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for i, ci in enumerate(mixed_centers):
        if ci is None:
            continue
        for j, cj in enumerate(mixed_centers):
            if i == j or cj is None:
                continue
            losses.append(torch.relu(torch.as_tensor(float(margin), device=device) - distance_fn(ci, cj)))
    return _reduce_losses(losses, reduction, device)


def compacting_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    mixed_centers: list[torch.Tensor | None],
    distance_fn,
    device: torch.device,
    reduction: str = "sum",
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for cls, center in enumerate(mixed_centers):
        if center is None:
            continue
        mask = labels == cls
        if bool(mask.any()):
            distances = distance_fn(features[mask], center.expand_as(features[mask]), mean=False)
            losses.extend(distances.unbind())
    return _reduce_losses(losses, reduction, device)


def _reduce_losses(losses: list[torch.Tensor], reduction: str, device: torch.device) -> torch.Tensor:
    if not losses:
        return torch.zeros((), device=device)
    values = torch.stack(losses)
    reduction = str(reduction).lower()
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"Unknown DAEAC loss reduction: {reduction}")


def distance_from_name(name: str):
    name = str(name).lower()
    if name == "l2":
        return l2_distance
    if name == "cosine":
        return cosine_distance
    raise ValueError(f"Unknown DAEAC distance: {name}")
