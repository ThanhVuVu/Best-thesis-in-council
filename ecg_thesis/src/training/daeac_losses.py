from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_cross_entropy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=class_weights, reduction="mean")


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
) -> torch.Tensor:
    losses = [
        distance_fn(cs, ct)
        for cs, ct in zip(source_centers, target_centers)
        if cs is not None and ct is not None
    ]
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def separating_loss(
    mixed_centers: list[torch.Tensor | None],
    margin: float,
    distance_fn,
    device: torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for i, ci in enumerate(mixed_centers):
        if ci is None:
            continue
        for j, cj in enumerate(mixed_centers):
            if i == j or cj is None:
                continue
            losses.append(torch.relu(torch.as_tensor(float(margin), device=device) - distance_fn(ci, cj)))
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def compacting_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    mixed_centers: list[torch.Tensor | None],
    distance_fn,
    device: torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for cls, center in enumerate(mixed_centers):
        if center is None:
            continue
        mask = labels == cls
        if bool(mask.any()):
            losses.append(distance_fn(features[mask], center.expand_as(features[mask])))
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def distance_from_name(name: str):
    name = str(name).lower()
    if name == "l2":
        return l2_distance
    if name == "cosine":
        return cosine_distance
    raise ValueError(f"Unknown DAEAC distance: {name}")
