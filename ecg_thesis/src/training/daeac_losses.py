from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DynamicWeightState:
    mmd: float
    lda: float
    tau: float
    lambda_align: float
    lambda_sep: float
    lambda_comp: float

    def as_dict(self) -> dict[str, float]:
        return {
            "dynamic_mmd": self.mmd,
            "dynamic_lda": self.lda,
            "dynamic_tau": self.tau,
            "dynamic_lambda_align": self.lambda_align,
            "dynamic_lambda_sep": self.lambda_sep,
            "dynamic_lambda_comp": self.lambda_comp,
        }


class DynamicWeightController:
    """EMA-smoothed scalar controller for DAEAC alignment/separation weights."""

    def __init__(
        self,
        beta1: float,
        beta2: float,
        ema_momentum: float = 0.9,
        rampup_epochs: int = 10,
        clip_min: float = 0.0,
        clip_max: float = 0.1,
        eps: float = 1.0e-8,
    ) -> None:
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.ema_momentum = float(ema_momentum)
        self.rampup_epochs = max(int(rampup_epochs), 1)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.eps = float(eps)
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("DynamicWeightController ema_momentum must satisfy 0 <= momentum < 1.")
        if self.clip_min > self.clip_max:
            raise ValueError("DynamicWeightController requires clip_min <= clip_max.")
        self._mmd_ema: float | None = None
        self._lda_ema: float | None = None
        self._mmd_min: float | None = None
        self._mmd_max: float | None = None
        self._lda_min: float | None = None
        self._lda_max: float | None = None

    def update(self, z_s: torch.Tensor, z_t: torch.Tensor, y_s: torch.Tensor, epoch: int) -> DynamicWeightState:
        if z_s.ndim != 2:
            raise ValueError(f"Dynamic weighting expects source features shaped [B, D], got {tuple(z_s.shape)}.")
        if z_t.ndim != 2:
            raise ValueError(f"Dynamic weighting expects target features shaped [B, D], got {tuple(z_t.shape)}.")
        if y_s.ndim != 1:
            raise ValueError(f"Dynamic weighting expects source labels shaped [B], got {tuple(y_s.shape)}.")
        if z_s.shape[0] != y_s.shape[0]:
            raise ValueError(f"Dynamic weighting batch mismatch: z_s={tuple(z_s.shape)}, y_s={tuple(y_s.shape)}.")
        if z_s.shape[1] != z_t.shape[1]:
            raise ValueError(f"Dynamic weighting feature mismatch: z_s={tuple(z_s.shape)}, z_t={tuple(z_t.shape)}.")

        with torch.no_grad():
            source = z_s.detach().float()
            target = z_t.detach().float()
            labels = y_s.detach().long()
            mmd_raw = self._mean_feature_distance(source, target)
            lda_raw = self._source_lda(source, labels)

        mmd = self._update_ema("mmd", mmd_raw)
        lda = self._update_ema("lda", lda_raw)
        mmd_norm = self._normalize("mmd", mmd)
        lda_norm = self._normalize("lda", lda)
        tau = mmd_norm / (mmd_norm + (1.0 - lda_norm) + self.eps)
        tau = min(max(float(tau), 0.0), 1.0)
        ramp = min(1.0, max(float(epoch), 0.0) / float(self.rampup_epochs))
        lambda_align = self._clip(self.beta1 * ramp * tau)
        lambda_sep = self._clip(self.beta2 * ramp * (1.0 - tau))
        lambda_comp = self._clip(self.beta2 * ramp * (1.0 - tau))
        return DynamicWeightState(
            mmd=float(mmd),
            lda=float(lda),
            tau=float(tau),
            lambda_align=float(lambda_align),
            lambda_sep=float(lambda_sep),
            lambda_comp=float(lambda_comp),
        )

    def fixed_state(self) -> DynamicWeightState:
        return DynamicWeightState(
            mmd=0.0,
            lda=0.0,
            tau=0.0,
            lambda_align=self.beta1,
            lambda_sep=self.beta2,
            lambda_comp=self.beta2,
        )

    def _mean_feature_distance(self, source: torch.Tensor, target: torch.Tensor) -> float:
        if source.numel() == 0 or target.numel() == 0:
            return 0.0
        value = torch.linalg.vector_norm(source.mean(dim=0) - target.mean(dim=0), ord=2)
        return self._finite_float(value)

    def _source_lda(self, source: torch.Tensor, labels: torch.Tensor) -> float:
        if source.numel() == 0 or labels.numel() == 0:
            return 0.0
        global_mean = source.mean(dim=0)
        s_between = source.new_zeros(())
        s_within = source.new_zeros(())
        for cls in torch.unique(labels):
            mask = labels == cls
            if not bool(mask.any()):
                continue
            class_features = source[mask]
            class_mean = class_features.mean(dim=0)
            s_between = s_between + float(class_features.shape[0]) * torch.sum((class_mean - global_mean) ** 2)
            s_within = s_within + torch.sum((class_features - class_mean) ** 2)
        value = s_between / (s_within + self.eps)
        return self._finite_float(value)

    def _update_ema(self, name: str, raw_value: float) -> float:
        value = float(raw_value)
        ema_attr = f"_{name}_ema"
        previous = getattr(self, ema_attr)
        ema = value if previous is None else self.ema_momentum * float(previous) + (1.0 - self.ema_momentum) * value
        setattr(self, ema_attr, ema)
        min_attr = f"_{name}_min"
        max_attr = f"_{name}_max"
        current_min = getattr(self, min_attr)
        current_max = getattr(self, max_attr)
        setattr(self, min_attr, ema if current_min is None else min(float(current_min), ema))
        setattr(self, max_attr, ema if current_max is None else max(float(current_max), ema))
        return float(ema)

    def _normalize(self, name: str, value: float) -> float:
        min_value = getattr(self, f"_{name}_min")
        max_value = getattr(self, f"_{name}_max")
        if min_value is None or max_value is None:
            return 0.0
        normalized = (float(value) - float(min_value)) / (float(max_value) - float(min_value) + self.eps)
        return min(max(normalized, 0.0), 1.0)

    def _clip(self, value: float) -> float:
        return min(max(float(value), self.clip_min), self.clip_max)

    @staticmethod
    def _finite_float(value: torch.Tensor | float) -> float:
        scalar = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        if scalar != scalar or scalar in {float("inf"), float("-inf")}:
            return 0.0
        return scalar


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


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: torch.Tensor | list[float] | tuple[float, ...],
        beta: float = 0.9999,
        gamma: float = 2.35,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported class-balanced focal loss reduction: {reduction}")
        beta = float(beta)
        if beta < 0.0 or beta >= 1.0:
            raise ValueError("class_balanced_focal beta must satisfy 0 <= beta < 1.")
        counts = torch.as_tensor(class_counts, dtype=torch.float32).flatten().clamp_min(1.0)
        if counts.numel() == 0:
            raise ValueError("class_balanced_focal requires non-empty class_counts.")
        effective_weights = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
        alpha = effective_weights * (float(counts.numel()) / effective_weights.sum().clamp_min(torch.finfo(counts.dtype).eps))
        self.gamma = float(gamma)
        self.reduction = str(reduction)
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Class-balanced focal loss expects logits shaped [B, C], got {tuple(logits.shape)}.")
        if labels.ndim != 1:
            raise ValueError(f"Class-balanced focal loss expects labels shaped [B], got {tuple(labels.shape)}.")
        if self.alpha.numel() != logits.size(1):
            raise ValueError(
                f"class_counts length must match num_classes={logits.size(1)}, got {self.alpha.numel()}."
            )
        if labels.numel() == 0:
            return logits.sum() * 0.0
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha.to(device=logits.device, dtype=logits.dtype)[labels]
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def weighted_cross_entropy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, labels, weight=class_weights, reduction="sum") / labels.numel()


def task_positive_features_from_logits(
    logits: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1.0e-8,
    detach_task_mask: bool = True,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Task-positive alignment expects logits shaped [B, C], got {tuple(logits.shape)}.")
    if features.ndim != 2:
        raise ValueError(f"Task-positive alignment expects features shaped [B, D], got {tuple(features.shape)}.")
    if labels.ndim != 1:
        raise ValueError(f"Task-positive alignment expects labels shaped [B], got {tuple(labels.shape)}.")
    if logits.shape[0] != features.shape[0] or labels.shape[0] != features.shape[0]:
        raise ValueError(
            "Task-positive alignment batch mismatch: "
            f"logits={tuple(logits.shape)}, features={tuple(features.shape)}, labels={tuple(labels.shape)}."
        )
    if labels.numel() == 0:
        return features
    if not features.requires_grad:
        raise ValueError("Task-positive alignment requires features to require gradients.")

    gt_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    task_mask = torch.autograd.grad(
        outputs=gt_logits.sum(),
        inputs=features,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    if detach_task_mask:
        task_mask = task_mask.detach()
    weighted = task_mask * features
    numerator = torch.linalg.vector_norm(features.detach(), ord=2, dim=1, keepdim=True)
    denominator = torch.linalg.vector_norm(weighted, ord=2, dim=1, keepdim=True).clamp_min(float(eps))
    positive = (numerator / denominator) * weighted
    if not torch.isfinite(positive).all():
        raise ValueError("Task-positive alignment produced NaN or Inf features.")
    return positive


def build_daeac_classification_loss(
    cfg: dict[str, Any],
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    class_counts: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> nn.Module:
    losses_cfg = dict(cfg.get("losses", {}))
    source_loss = str(losses_cfg.get("source_cls_loss", cfg.get("source_loss", "weighted_ce"))).lower()
    if source_loss == "weighted_ce":
        return WeightedCrossEntropyByBatchSize(class_weights)
    if source_loss == "focal":
        alpha_cfg = cfg.get("focal_alpha")
        alpha = class_weights if alpha_cfg is None else torch.as_tensor(alpha_cfg, dtype=torch.float32)
        if alpha is not None and int(alpha.numel()) != int(num_classes):
            raise ValueError(f"focal_alpha length must match num_classes={num_classes}, got {int(alpha.numel())}.")
        return CustomFocalLoss(alpha=alpha, gamma=float(cfg.get("focal_gamma", 2.0)))
    if source_loss == "class_balanced_focal":
        if class_counts is None:
            raise ValueError("class_balanced_focal requires source class_counts.")
        cb_cfg = dict(losses_cfg.get("class_balanced_focal", cfg.get("class_balanced_focal", {})))
        return ClassBalancedFocalLoss(
            class_counts,
            beta=float(cb_cfg.get("beta", 0.9999)),
            gamma=float(cb_cfg.get("gamma", 2.35)),
        )
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
