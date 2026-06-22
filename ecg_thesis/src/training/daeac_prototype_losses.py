from __future__ import annotations

from typing import Any

import torch


VALID_PROTOTYPE_LOSS_MODES = {"legacy", "replacement"}


def validate_prototype_loss_config(config: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    raw = config.get("prototype_losses")
    if raw is None:
        return {"enabled": False, "mode": "legacy"}
    cfg = dict(raw)
    mode = str(cfg.get("mode", "replacement")).lower()
    if mode not in VALID_PROTOTYPE_LOSS_MODES:
        raise ValueError(f"prototype_losses.mode must be one of {sorted(VALID_PROTOTYPE_LOSS_MODES)}.")
    enabled = bool(cfg.get("enabled", False))
    if mode == "replacement" and not enabled:
        raise ValueError("prototype_losses.enabled must be true for replacement mode.")
    if mode == "legacy":
        return {**cfg, "enabled": enabled, "mode": mode}
    if str(config.get("adaptation", {}).get("distance", "l2")).lower() != "l2":
        raise ValueError("PLAN 3 replacement prototype losses support only adaptation.distance=l2.")
    if str(cfg.get("target_weight_mode", "")) != "confidence_x_inverse_entropy":
        raise ValueError("prototype_losses.target_weight_mode must be confidence_x_inverse_entropy.")
    for name in ("lambda_proto_align", "lambda_comp_source", "lambda_comp_target", "lambda_sep_margin"):
        if float(cfg.get(name, -1.0)) < 0.0:
            raise ValueError(f"prototype_losses.{name} must be non-negative.")
    ramps = dict(cfg.get("rampup_epochs", {}))
    for name in ("proto_align", "comp_source", "comp_target", "sep_margin"):
        if int(ramps.get(name, 0)) < 1:
            raise ValueError(f"prototype_losses.rampup_epochs.{name} must be at least 1.")
    if bool(cfg.get("use_pair_margin", False)) and not bool(cfg.get("use_sep_margin", False)):
        raise ValueError("use_pair_margin requires use_sep_margin=true.")
    build_margin_matrix(cfg, class_names, device=torch.device("cpu"), dtype=torch.float32)
    return {**cfg, "enabled": enabled, "mode": mode, "rampup_epochs": ramps}


def linear_ramp(epoch: int, rampup_epochs: int) -> float:
    if int(rampup_epochs) < 1:
        raise ValueError("rampup_epochs must be at least 1.")
    return min(max(float(epoch) / float(rampup_epochs), 0.0), 1.0)


def build_margin_matrix(
    cfg: dict[str, Any],
    class_names: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    k = len(class_names)
    if bool(cfg.get("use_pair_margin", False)):
        raw = dict(cfg.get("pair_margin", {}))
        missing = [name for name in class_names if name not in raw]
        extra = sorted(set(raw) - set(class_names))
        if missing or extra:
            raise ValueError(f"prototype pair-margin rows mismatch: missing={missing}, extra={extra}.")
        rows = []
        for row_name in class_names:
            row = dict(raw[row_name])
            row_missing = [name for name in class_names if name not in row]
            row_extra = sorted(set(row) - set(class_names))
            if row_missing or row_extra:
                raise ValueError(
                    f"prototype pair-margin row {row_name} mismatch: missing={row_missing}, extra={row_extra}."
                )
            rows.append([float(row[name]) for name in class_names])
        matrix = torch.tensor(rows, device=device, dtype=dtype)
        if bool((matrix < 0).any()):
            raise ValueError("prototype pair margins must be non-negative.")
        if not torch.allclose(matrix, matrix.transpose(0, 1)):
            raise ValueError("prototype pair-margin matrix must be symmetric.")
        if not torch.allclose(torch.diagonal(matrix), torch.zeros(k, device=device, dtype=dtype)):
            raise ValueError("prototype pair-margin diagonal must be zero.")
        return matrix
    uniform = float(cfg.get("uniform_margin", -1.0))
    if uniform < 0.0:
        raise ValueError("prototype_losses.uniform_margin must be non-negative.")
    matrix = torch.full((k, k), uniform, device=device, dtype=dtype)
    matrix.fill_diagonal_(0.0)
    return matrix


def target_reliability_weights(confidence: torch.Tensor, normalized_entropy: torch.Tensor) -> torch.Tensor:
    if confidence.ndim != 1 or normalized_entropy.shape != confidence.shape:
        raise ValueError("confidence and normalized_entropy must have matching shape [B].")
    return (confidence * (1.0 - normalized_entropy)).clamp_min(0.0).detach()


def source_compactness_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    source_prototypes: torch.Tensor,
    source_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_shapes(features, labels, source_prototypes, source_valid)
    mask = _valid_sample_mask(labels, source_valid)
    if not bool(mask.any()):
        return _zero(features), {"active_samples": _scalar(features, 0.0)}
    anchors = source_prototypes.detach()[labels[mask]]
    distances = torch.linalg.vector_norm(features[mask] - anchors, dim=1)
    return distances.mean(), {"active_samples": _scalar(features, float(mask.sum()))}


def target_compactness_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    global_prototypes: torch.Tensor,
    global_valid: torch.Tensor,
    sample_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_shapes(features, labels, global_prototypes, global_valid)
    _validate_weights(features, sample_weights)
    mask = _valid_sample_mask(labels, global_valid) & (sample_weights > 0)
    if not bool(mask.any()):
        return _zero(features), {"active_samples": _scalar(features, 0.0), "weight_sum": _scalar(features, 0.0)}
    anchors = global_prototypes.detach()[labels[mask]]
    distances = torch.linalg.vector_norm(features[mask] - anchors, dim=1)
    weights = sample_weights[mask].detach()
    return _weighted_mean(distances, weights), {
        "active_samples": _scalar(features, float(mask.sum())),
        "weight_sum": weights.sum().detach(),
    }


def directed_target_alignment_loss(
    target_features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    sample_weights: torch.Tensor,
    source_prototypes: torch.Tensor,
    source_valid: torch.Tensor,
    min_target_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_shapes(target_features, pseudo_labels, source_prototypes, source_valid)
    _validate_weights(target_features, sample_weights)
    class_losses = []
    class_weights = []
    active_mask = torch.zeros_like(source_valid, dtype=torch.bool)
    for class_id in range(int(source_prototypes.shape[0])):
        mask = pseudo_labels == class_id
        if not bool(source_valid[class_id]) or int(mask.sum()) < int(min_target_count):
            continue
        weights = sample_weights[mask].detach()
        if float(weights.sum()) <= 0.0:
            continue
        target_batch_prototype = target_features[mask].mean(dim=0)
        class_losses.append(torch.linalg.vector_norm(target_batch_prototype - source_prototypes[class_id].detach()))
        class_weights.append(weights.mean())
        active_mask[class_id] = True
    if not class_losses:
        return _zero(target_features), {
            "active_classes": _scalar(target_features, 0.0),
            "active_class_mask": active_mask,
        }
    losses = torch.stack(class_losses)
    weights = torch.stack(class_weights).detach()
    return _weighted_mean(losses, weights), {
        "active_classes": _scalar(target_features, float(active_mask.sum())),
        "active_class_mask": active_mask,
    }


def sample_prototype_margin_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_valid: torch.Tensor,
    margin_matrix: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_shapes(features, labels, prototypes, prototype_valid)
    k = int(prototypes.shape[0])
    if tuple(margin_matrix.shape) != (k, k):
        raise ValueError(f"margin_matrix must have shape [{k},{k}].")
    if sample_weights is None:
        sample_weights = torch.ones(features.shape[0], device=features.device, dtype=features.dtype)
    _validate_weights(features, sample_weights)
    pair_counts = torch.zeros((k, k), device=features.device, dtype=features.dtype)
    pair_violations = torch.zeros_like(pair_counts)
    weighted_losses = []
    weights = []
    active_samples = torch.zeros(features.shape[0], device=features.device, dtype=torch.bool)
    anchors = prototypes.detach()
    for positive in range(k):
        sample_mask = (labels == positive) & (sample_weights > 0)
        if not bool(prototype_valid[positive]) or not bool(sample_mask.any()):
            continue
        positive_distance = torch.linalg.vector_norm(features[sample_mask] - anchors[positive], dim=1)
        current_weights = sample_weights[sample_mask].detach()
        for negative in range(k):
            if negative == positive or not bool(prototype_valid[negative]):
                continue
            negative_distance = torch.linalg.vector_norm(features[sample_mask] - anchors[negative], dim=1)
            hinge = torch.relu(positive_distance - negative_distance + margin_matrix[positive, negative])
            weighted_losses.append(hinge * current_weights)
            weights.append(current_weights)
            pair_counts[positive, negative] += current_weights.sum()
            pair_violations[positive, negative] += (hinge.detach() > 0).to(features.dtype).mul(current_weights).sum()
            active_samples[sample_mask] = True
    if not weighted_losses:
        loss = _zero(features)
    else:
        loss = torch.cat(weighted_losses).sum() / torch.cat(weights).sum().clamp_min(torch.finfo(features.dtype).eps)
    total_pairs = pair_counts.sum()
    return loss, {
        "active_samples": active_samples.sum().to(features.dtype),
        "pair_counts": pair_counts.detach(),
        "pair_violations": pair_violations.detach(),
        "violation_ratio": torch.where(
            total_pairs > 0,
            pair_violations.sum() / total_pairs.clamp_min(torch.finfo(features.dtype).eps),
            _scalar(features, 0.0),
        ).detach(),
    }


def _valid_sample_mask(labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError("labels must be shaped [B].")
    if labels.numel() == 0:
        return torch.zeros_like(labels, dtype=torch.bool)
    if bool(((labels < 0) | (labels >= len(valid))).any()):
        raise ValueError("labels contain an out-of-range class id.")
    return valid[labels]


def _validate_shapes(features, labels, prototypes, valid) -> None:
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features/labels must be [B,D] and [B].")
    if prototypes.ndim != 2 or prototypes.shape[1] != features.shape[1]:
        raise ValueError("prototypes must be [K,D] with the same D as features.")
    if tuple(valid.shape) != (prototypes.shape[0],):
        raise ValueError("prototype_valid must be shaped [K].")
    _valid_sample_mask(labels, valid)


def _validate_weights(features: torch.Tensor, weights: torch.Tensor) -> None:
    if tuple(weights.shape) != (features.shape[0],):
        raise ValueError("sample_weights must be shaped [B].")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("sample_weights must be finite and non-negative.")


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).eps)


def _zero(features: torch.Tensor) -> torch.Tensor:
    return features.sum() * 0.0


def _scalar(features: torch.Tensor, value: float) -> torch.Tensor:
    return torch.as_tensor(value, device=features.device, dtype=features.dtype)
