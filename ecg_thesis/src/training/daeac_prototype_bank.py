from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PrototypeCandidates:
    source: torch.Tensor
    target: torch.Tensor
    global_: torch.Tensor
    source_valid: torch.Tensor
    target_valid: torch.Tensor
    global_valid: torch.Tensor
    source_update_mask: torch.Tensor
    target_update_mask: torch.Tensor
    source_batch_counts: torch.Tensor
    target_batch_counts: torch.Tensor
    beta: torch.Tensor


class ReliabilityWeightedPrototypeBank(nn.Module):
    """Non-trainable source/target/global prototype statistics."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        source_momentum: float = 0.90,
        target_momentum: float = 0.99,
        reliability_momentum: float = 0.90,
        min_target_count: int = 4,
        beta_max: float = 0.30,
        rampup_epochs: int = 10,
    ) -> None:
        super().__init__()
        _validate_unit_interval("source_momentum", source_momentum)
        _validate_unit_interval("target_momentum", target_momentum)
        _validate_unit_interval("reliability_momentum", reliability_momentum)
        if int(min_target_count) < 1:
            raise ValueError("min_target_count must be at least 1.")
        if not 0.0 <= float(beta_max) <= 1.0:
            raise ValueError("beta_max must be in [0, 1].")
        if int(rampup_epochs) < 1:
            raise ValueError("rampup_epochs must be at least 1.")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.source_momentum = float(source_momentum)
        self.target_momentum = float(target_momentum)
        self.reliability_momentum = float(reliability_momentum)
        self.min_target_count = int(min_target_count)
        self.beta_max = float(beta_max)
        self.rampup_epochs = int(rampup_epochs)

        shape = (self.num_classes, self.feature_dim)
        self.register_buffer("source_prototypes", torch.zeros(shape))
        self.register_buffer("target_prototypes", torch.zeros(shape))
        self.register_buffer("global_prototypes", torch.zeros(shape))
        self.register_buffer("target_reliability", torch.zeros(self.num_classes))
        self.register_buffer("beta", torch.zeros(self.num_classes))
        self.register_buffer("source_valid", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("target_valid", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("source_counts", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("target_counts", torch.zeros(self.num_classes, dtype=torch.long))

    @torch.no_grad()
    def initialize_source(self, prototypes: torch.Tensor, counts: torch.Tensor) -> None:
        self._validate_prototypes(prototypes, "prototypes")
        self._validate_counts(counts, "counts")
        valid = counts.to(self.device) > 0
        self.source_prototypes.zero_()
        self.source_prototypes[valid] = prototypes.to(self.device, self.dtype)[valid]
        self.source_counts.copy_(counts.to(self.device, torch.long))
        self.source_valid.copy_(valid)
        self.target_prototypes.zero_()
        self.target_counts.zero_()
        self.target_valid.zero_()
        self.target_reliability.zero_()
        self.beta.zero_()
        self.refresh_global()

    def candidates(
        self,
        source_batch_prototypes: torch.Tensor,
        source_batch_counts: torch.Tensor,
        target_batch_prototypes: torch.Tensor,
        target_batch_counts: torch.Tensor,
    ) -> PrototypeCandidates:
        self._validate_prototypes(source_batch_prototypes, "source_batch_prototypes")
        self._validate_prototypes(target_batch_prototypes, "target_batch_prototypes")
        self._validate_counts(source_batch_counts, "source_batch_counts")
        self._validate_counts(target_batch_counts, "target_batch_counts")
        device, dtype = self.device, self.dtype
        source_local = source_batch_prototypes.to(device=device, dtype=dtype)
        target_local = target_batch_prototypes.to(device=device, dtype=dtype)
        source_counts = source_batch_counts.to(device=device, dtype=torch.long)
        target_counts = target_batch_counts.to(device=device, dtype=torch.long)
        source_update = source_counts > 0
        target_update = target_counts >= self.min_target_count

        source_ema = self.source_momentum * self.source_prototypes.detach() + (1.0 - self.source_momentum) * source_local
        target_ema = self.target_momentum * self.target_prototypes.detach() + (1.0 - self.target_momentum) * target_local
        source_first = source_local
        target_first = target_local
        source_updated = torch.where(self.source_valid[:, None], source_ema, source_first)
        target_updated = torch.where(self.target_valid[:, None], target_ema, target_first)
        source_candidate = torch.where(source_update[:, None], source_updated, self.source_prototypes.detach())
        target_candidate = torch.where(target_update[:, None], target_updated, self.target_prototypes.detach())
        source_valid = self.source_valid | source_update
        target_valid = self.target_valid | target_update
        both_valid = source_valid & target_valid
        effective_beta = self.beta.detach() * both_valid.to(dtype)
        global_candidate = (
            (1.0 - effective_beta[:, None]) * source_candidate
            + effective_beta[:, None] * target_candidate
        )
        global_candidate = torch.where(source_valid[:, None], global_candidate, torch.zeros_like(global_candidate))
        return PrototypeCandidates(
            source=source_candidate,
            target=target_candidate,
            global_=global_candidate,
            source_valid=source_valid,
            target_valid=target_valid,
            global_valid=source_valid,
            source_update_mask=source_update,
            target_update_mask=target_update,
            source_batch_counts=source_counts,
            target_batch_counts=target_counts,
            beta=effective_beta,
        )

    @torch.no_grad()
    def commit(self, candidates: PrototypeCandidates) -> None:
        self.source_prototypes.copy_(candidates.source.detach())
        self.target_prototypes.copy_(candidates.target.detach())
        self.global_prototypes.copy_(candidates.global_.detach())
        self.source_valid.copy_(candidates.source_valid.detach())
        self.target_valid.copy_(candidates.target_valid.detach())
        self.source_counts.add_(candidates.source_batch_counts * candidates.source_update_mask.to(torch.long))
        self.target_counts.add_(candidates.target_batch_counts * candidates.target_update_mask.to(torch.long))

    @torch.no_grad()
    def update_reliability(
        self,
        predicted_counts: torch.Tensor,
        accepted_counts: torch.Tensor,
        accepted_confidence_sums: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor | float]:
        self._validate_counts(predicted_counts, "predicted_counts")
        self._validate_counts(accepted_counts, "accepted_counts")
        if tuple(accepted_confidence_sums.shape) != (self.num_classes,):
            raise ValueError(
                f"accepted_confidence_sums must have shape [{self.num_classes}], "
                f"got {tuple(accepted_confidence_sums.shape)}."
            )
        predicted = predicted_counts.to(self.device, self.dtype)
        accepted = accepted_counts.to(self.device, self.dtype)
        confidence_sums = accepted_confidence_sums.to(self.device, self.dtype)
        coverage = torch.where(predicted > 0, accepted / predicted.clamp_min(1.0), torch.zeros_like(predicted))
        mean_confidence = torch.where(
            accepted > 0,
            confidence_sums / accepted.clamp_min(1.0),
            torch.zeros_like(accepted),
        )
        observed = (coverage * mean_confidence).clamp(0.0, 1.0)
        self.target_reliability.mul_(self.reliability_momentum).add_(
            observed,
            alpha=1.0 - self.reliability_momentum,
        )
        ramp = min(max(float(epoch) / float(self.rampup_epochs), 0.0), 1.0)
        valid = self.source_valid & self.target_valid
        new_beta = ramp * self.beta_max * self.target_reliability
        self.beta.copy_(torch.where(valid, new_beta, torch.zeros_like(new_beta)))
        self.refresh_global()
        return {
            "coverage": coverage.detach().clone(),
            "mean_confidence": mean_confidence.detach().clone(),
            "observed_reliability": observed.detach().clone(),
            "ramp": ramp,
        }

    @torch.no_grad()
    def refresh_global(self) -> None:
        valid = self.source_valid & self.target_valid
        effective_beta = torch.where(valid, self.beta, torch.zeros_like(self.beta))
        global_values = (
            (1.0 - effective_beta[:, None]) * self.source_prototypes
            + effective_beta[:, None] * self.target_prototypes
        )
        self.global_prototypes.copy_(
            torch.where(self.source_valid[:, None], global_values, torch.zeros_like(global_values))
        )

    @torch.no_grad()
    def diagnostics(self) -> dict[str, torch.Tensor]:
        ps_pt = torch.linalg.vector_norm(self.source_prototypes - self.target_prototypes, dim=1)
        pg_ps = torch.linalg.vector_norm(self.global_prototypes - self.source_prototypes, dim=1)
        return {
            "source_norm": torch.linalg.vector_norm(self.source_prototypes, dim=1),
            "target_norm": torch.linalg.vector_norm(self.target_prototypes, dim=1),
            "global_norm": torch.linalg.vector_norm(self.global_prototypes, dim=1),
            "source_target_l2": torch.where(self.source_valid & self.target_valid, ps_pt, torch.zeros_like(ps_pt)),
            "global_source_l2": torch.where(self.source_valid, pg_ps, torch.zeros_like(pg_ps)),
            "source_valid": self.source_valid.clone(),
            "target_valid": self.target_valid.clone(),
        }

    @property
    def device(self) -> torch.device:
        return self.source_prototypes.device

    @property
    def dtype(self) -> torch.dtype:
        return self.source_prototypes.dtype

    def _validate_prototypes(self, values: torch.Tensor, name: str) -> None:
        expected = (self.num_classes, self.feature_dim)
        if tuple(values.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(values.shape)}.")

    def _validate_counts(self, values: torch.Tensor, name: str) -> None:
        if tuple(values.shape) != (self.num_classes,):
            raise ValueError(f"{name} must have shape [{self.num_classes}], got {tuple(values.shape)}.")


def dense_batch_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"features must be [B,D], got {tuple(features.shape)}.")
    prototypes = []
    counts = []
    for class_id in range(int(num_classes)):
        mask = labels == class_id
        count = mask.sum()
        counts.append(count)
        if bool(mask.any()):
            prototypes.append(features[mask].mean(dim=0))
        else:
            prototypes.append(features.sum(dim=0) * 0.0)
    return torch.stack(prototypes), torch.stack(counts).to(torch.long)


def candidate_lists(values: torch.Tensor, valid: torch.Tensor) -> list[torch.Tensor | None]:
    return [values[index] if bool(valid[index]) else None for index in range(len(valid))]


def pseudo_distribution_flags(
    accepted_counts: torch.Tensor,
    near_all_n_ratio: float = 0.95,
    normal_class_index: int = 0,
) -> dict[str, float | bool | int]:
    total = int(accepted_counts.sum().item())
    normal = int(accepted_counts[int(normal_class_index)].item()) if total else 0
    ratio = float(normal / total) if total else 0.0
    return {
        "accepted_total": total,
        "normal_accepted": normal,
        "normal_ratio": ratio,
        "all_n": bool(total > 0 and normal == total),
        "near_all_n": bool(total > 0 and ratio >= float(near_all_n_ratio)),
    }


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= float(value) < 1.0:
        raise ValueError(f"{name} must be in [0, 1).")
