from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from src.training.daeac_prototype_bank import pseudo_distribution_flags


VALID_FILTER_MODES = {"none", "confidence_global", "confidence_entropy", "class_specific"}


@dataclass(frozen=True)
class PseudoFilterResult:
    pseudo_labels: torch.Tensor
    confidence: torch.Tensor
    normalized_entropy: torch.Tensor
    accepted_mask: torch.Tensor
    rejected_confidence_mask: torch.Tensor
    rejected_entropy_mask: torch.Tensor
    rejected_both_mask: torch.Tensor


def update_pseudo_safety_state(
    accepted_counts: torch.Tensor,
    *,
    previous_empty_streak: int,
    previous_all_n_streak: int,
    near_all_n_ratio: float,
) -> dict[str, float | bool | int]:
    state = pseudo_distribution_flags(accepted_counts, near_all_n_ratio=near_all_n_ratio)
    empty = int(state["accepted_total"]) == 0
    state["empty_acceptance"] = empty
    state["empty_acceptance_streak"] = int(previous_empty_streak) + 1 if empty else 0
    state["all_n_streak"] = int(previous_all_n_streak) + 1 if bool(state["all_n"]) else 0
    return state


def pseudo_safety_reason(
    state: dict[str, float | bool | int],
    *,
    fail_on_empty: bool,
    fail_on_all_n: bool,
    patience: int,
) -> str | None:
    if int(patience) < 1:
        raise ValueError("patience must be at least 1.")
    if fail_on_empty and int(state["empty_acceptance_streak"]) >= int(patience):
        return "no_target_pseudo_labels_accepted"
    if fail_on_all_n and int(state["all_n_streak"]) >= int(patience):
        return "all_accepted_target_pseudo_labels_are_N"
    return None


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Return entropy in [0, 1] for a [B,K] probability tensor."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"probabilities must be [B,K] with K >= 2, got {tuple(probabilities.shape)}.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values.")
    if bool((probabilities < 0).any()):
        raise ValueError("probabilities must be non-negative.")
    row_sums = probabilities.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4):
        raise ValueError("probability rows must sum to 1.")
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    entropy = -(probabilities * safe.log()).sum(dim=1) / math.log(probabilities.shape[1])
    return entropy.clamp(0.0, 1.0)


def filter_target_pseudolabels(
    probabilities: torch.Tensor,
    *,
    mode: str,
    global_confidence_threshold: float,
    class_confidence_thresholds: torch.Tensor,
    max_normalized_entropy: float,
) -> PseudoFilterResult:
    mode = str(mode).lower()
    if mode not in VALID_FILTER_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_FILTER_MODES)}, got '{mode}'.")
    num_classes = probabilities.shape[1] if probabilities.ndim == 2 else 0
    if tuple(class_confidence_thresholds.shape) != (num_classes,):
        raise ValueError(
            f"class_confidence_thresholds must have shape [{num_classes}], "
            f"got {tuple(class_confidence_thresholds.shape)}."
        )
    _validate_unit_interval("global_confidence_threshold", global_confidence_threshold)
    _validate_unit_interval("max_normalized_entropy", max_normalized_entropy)
    thresholds = class_confidence_thresholds.to(device=probabilities.device, dtype=probabilities.dtype)
    if bool(((thresholds < 0) | (thresholds > 1)).any()):
        raise ValueError("class confidence thresholds must be in [0, 1].")

    entropy = normalized_entropy(probabilities)
    confidence, pseudo_labels = probabilities.max(dim=1)
    confidence_ok = torch.ones_like(confidence, dtype=torch.bool)
    entropy_ok = torch.ones_like(confidence, dtype=torch.bool)
    if mode in {"confidence_global", "confidence_entropy"}:
        confidence_ok = confidence > float(global_confidence_threshold)
    elif mode == "class_specific":
        confidence_ok = confidence > thresholds[pseudo_labels]
    if mode in {"confidence_entropy", "class_specific"}:
        entropy_ok = entropy <= float(max_normalized_entropy)

    accepted = confidence_ok & entropy_ok
    return PseudoFilterResult(
        pseudo_labels=pseudo_labels,
        confidence=confidence,
        normalized_entropy=entropy,
        accepted_mask=accepted,
        rejected_confidence_mask=(~confidence_ok) & entropy_ok,
        rejected_entropy_mask=confidence_ok & (~entropy_ok),
        rejected_both_mask=(~confidence_ok) & (~entropy_ok),
    )


def validate_pseudo_filter_config(config: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    cfg = dict(config.get("pseudo_filter", {}))
    if not cfg:
        return {"enabled": False, "mode": "legacy_class_specific"}
    if not bool(cfg.get("enabled", False)):
        raise ValueError("pseudo_filter.enabled must be true when the pseudo_filter section is present.")
    mode = str(cfg.get("mode", "")).lower()
    if mode not in VALID_FILTER_MODES:
        raise ValueError(f"pseudo_filter.mode must be one of {sorted(VALID_FILTER_MODES)}, got '{mode}'.")
    _validate_unit_interval("pseudo_filter.global_confidence_threshold", cfg.get("global_confidence_threshold"))
    _validate_unit_interval("pseudo_filter.max_normalized_entropy", cfg.get("max_normalized_entropy"))
    thresholds = dict(cfg.get("class_confidence_thresholds", {}))
    missing = [name for name in class_names if name not in thresholds]
    extra = sorted(set(thresholds) - set(class_names))
    if missing or extra:
        raise ValueError(f"class confidence thresholds mismatch: missing={missing}, extra={extra}.")
    for name in class_names:
        _validate_unit_interval(f"class_confidence_thresholds.{name}", thresholds[name])
    _validate_unit_interval("pseudo_filter.near_all_n_ratio", cfg.get("near_all_n_ratio"), lower_open=True)
    if int(cfg.get("safety_patience_epochs", 0)) < 1:
        raise ValueError("pseudo_filter.safety_patience_epochs must be at least 1.")
    return cfg


def class_threshold_tensor(cfg: dict[str, Any], class_names: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(cfg["class_confidence_thresholds"][name]) for name in class_names],
        dtype=torch.float32,
        device=device,
    )


def _validate_unit_interval(name: str, value: Any, lower_open: bool = False) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in [0, 1].") from exc
    lower_valid = number > 0.0 if lower_open else number >= 0.0
    if not lower_valid or number > 1.0:
        interval = "(0, 1]" if lower_open else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}.")
