from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def class_balanced_conditional_mkmmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_probabilities: torch.Tensor,
    gammas: torch.Tensor,
    beta: torch.Tensor,
    class_weights: torch.Tensor,
    min_target_mass: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    _validate_features(source_features, target_features)
    num_classes = int(target_probabilities.shape[1])
    if int(class_weights.numel()) != num_classes:
        raise ValueError("class_weights length must equal the target probability class dimension.")

    kernels_ss = _multi_kernel_matrix(source_features, source_features, gammas, beta)
    kernels_tt = _multi_kernel_matrix(target_features, target_features, gammas, beta)
    kernels_st = _multi_kernel_matrix(source_features, target_features, gammas, beta)
    probabilities = target_probabilities.detach()
    losses: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, torch.Tensor] = {}
    active_losses: list[torch.Tensor] = []
    active_weights: list[torch.Tensor] = []
    for cls in range(num_classes):
        source_membership = (source_labels == cls).to(source_features.dtype)
        target_membership = probabilities[:, cls].to(target_features.dtype)
        source_count = source_membership.sum()
        target_mass = target_membership.sum()
        diagnostics[f"source_count_{cls}"] = source_count.detach()
        diagnostics[f"target_mass_{cls}"] = target_mass.detach()
        active = bool(source_count.item() >= 1.0 and target_mass.item() >= float(min_target_mass))
        diagnostics[f"active_{cls}"] = torch.as_tensor(float(active), device=source_features.device)
        if not active:
            continue
        ws = source_membership / source_count.clamp_min(1.0)
        wt = target_membership / target_mass.clamp_min(torch.finfo(target_features.dtype).eps)
        loss = ws @ kernels_ss @ ws + wt @ kernels_tt @ wt - 2.0 * (ws @ kernels_st @ wt)
        loss = loss.clamp_min(0.0)
        losses[str(cls)] = loss
        active_losses.append(loss)
        active_weights.append(class_weights[cls].to(loss.device, loss.dtype))

    if not active_losses:
        zero = (source_features.sum() + target_features.sum()) * 0.0
        return zero, losses, diagnostics
    stacked_losses = torch.stack(active_losses)
    stacked_weights = torch.stack(active_weights)
    total = torch.sum(stacked_losses * stacked_weights) / stacked_weights.sum().clamp_min(torch.finfo(stacked_losses.dtype).eps)
    return total, losses, diagnostics


def safe_topk_pseudolabel_mask(
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
    margin: torch.Tensor,
    class_names: list[str],
    min_confidence: dict[str, float],
    min_margin: dict[str, float],
    max_per_batch: dict[str, int],
) -> tuple[torch.Tensor, dict[str, float]]:
    keep = torch.zeros_like(pseudo, dtype=torch.bool)
    diagnostics: dict[str, float] = {}
    for cls, name in enumerate(class_names):
        candidate = (pseudo == cls) & (confidence >= float(min_confidence[name])) & (margin >= float(min_margin[name]))
        indices = torch.nonzero(candidate, as_tuple=False).flatten()
        quota = max(0, int(max_per_batch[name]))
        if quota and indices.numel() > quota:
            order = torch.argsort(confidence[indices], descending=True)
            indices = indices[order[:quota]]
        elif quota == 0:
            indices = indices[:0]
        keep[indices] = True
        diagnostics[f"candidate_{name}"] = float(candidate.sum().detach().cpu())
        diagnostics[f"selected_{name}"] = float(indices.numel())
        diagnostics[f"selected_confidence_{name}"] = float(confidence[indices].mean().detach().cpu()) if indices.numel() else 0.0
        diagnostics[f"selected_margin_{name}"] = float(margin[indices].mean().detach().cpu()) if indices.numel() else 0.0
    return keep, diagnostics


def minority_class_weights(
    soft_prior: torch.Tensor,
    exponent: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
    multipliers: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = soft_prior.detach().clamp_min(1.0e-6).pow(-float(exponent))
    weights = weights.clamp(min=float(min_weight), max=float(max_weight))
    if multipliers is not None:
        weights = weights * multipliers.to(weights.device, weights.dtype)
    return weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)


def update_soft_prior_ema(previous: torch.Tensor, probabilities: torch.Tensor, decay: float) -> torch.Tensor:
    batch_prior = probabilities.detach().mean(dim=0)
    return float(decay) * previous.detach() + (1.0 - float(decay)) * batch_prior


def minority_weighted_mcc_loss(
    logits: torch.Tensor,
    class_weights: torch.Tensor,
    temperature: float = 2.5,
    eps: float = 1.0e-5,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    if logits.ndim != 2:
        raise ValueError(f"MCC expects logits shaped [B, C], got {tuple(logits.shape)}.")
    if logits.size(0) == 0:
        loss = logits.sum() * 0.0
        return (loss, {}) if return_diagnostics else loss
    if class_weights.numel() != logits.size(1):
        raise ValueError("MCC class_weights length must equal logits class dimension.")

    probs = F.softmax(logits / float(temperature), dim=1)
    entropy = -(probs * torch.log(probs + float(eps))).sum(dim=1)
    sample_weights = (1.0 + torch.exp(-entropy)).detach()
    sample_weights = logits.size(0) * sample_weights / sample_weights.sum().clamp_min(float(eps))
    correlation = probs.mul(sample_weights.view(-1, 1)).transpose(1, 0).mm(probs)
    soft_confusion = correlation / correlation.sum(dim=1, keepdim=True).clamp_min(float(eps))
    offdiag = soft_confusion.sum(dim=1) - torch.diagonal(soft_confusion)
    row_weights = class_weights.to(logits.device, logits.dtype)
    loss = torch.sum(row_weights * offdiag) / row_weights.sum().clamp_min(float(eps))
    if not return_diagnostics:
        return loss
    return loss, {
        "entropy_mean": float(entropy.detach().mean().cpu()),
        "pred_counts": torch.bincount(probs.argmax(dim=1), minlength=logits.size(1)).detach().cpu().tolist(),
        "soft_confusion": soft_confusion.detach(),
        "offdiag_by_class": offdiag.detach(),
        "class_weights": row_weights.detach(),
        "probabilities": probs.detach(),
    }


def source_f_prototype_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: list[torch.Tensor | None],
    class_names: list[str],
    temperature: float = 0.1,
    sample_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    for required in ("N", "V", "F"):
        if required not in class_names:
            raise ValueError(f"Prototype loss requires class {required!r}.")
    n_idx, v_idx, f_idx = (class_names.index(name) for name in ("N", "V", "F"))
    required_indices = (n_idx, v_idx, f_idx)
    if any(prototypes[idx] is None for idx in required_indices):
        zero = features.sum() * 0.0
        return zero, {"cosine_F_N": 0.0, "cosine_F_V": 0.0, "active_samples": 0.0}

    proto = torch.stack([prototypes[idx].detach() for idx in range(len(prototypes)) if prototypes[idx] is not None])
    if len(proto) != len(prototypes):
        zero = features.sum() * 0.0
        return zero, {"cosine_F_N": 0.0, "cosine_F_V": 0.0, "active_samples": 0.0}
    normalized_features = F.normalize(features, dim=1)
    normalized_proto = F.normalize(proto.to(features.device, features.dtype), dim=1)
    weights_cfg = sample_weights or {"N": 1.0, "V": 1.0, "F": 2.0}
    weighted_losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    groups = {
        n_idx: [n_idx, f_idx],
        v_idx: [v_idx, f_idx],
        f_idx: [f_idx, n_idx, v_idx],
    }
    for true_class, candidates in groups.items():
        mask = labels == true_class
        if not bool(mask.any()):
            continue
        logits = normalized_features[mask] @ normalized_proto[candidates].transpose(0, 1) / float(temperature)
        target = torch.zeros(int(mask.sum()), dtype=torch.long, device=features.device)
        losses = F.cross_entropy(logits, target, reduction="none")
        class_weight = torch.as_tensor(float(weights_cfg[class_names[true_class]]), device=features.device, dtype=features.dtype)
        weighted_losses.append(losses * class_weight)
        weights.append(torch.ones_like(losses) * class_weight)
    if not weighted_losses:
        total = features.sum() * 0.0
        active_samples = 0.0
    else:
        total = torch.cat(weighted_losses).sum() / torch.cat(weights).sum().clamp_min(torch.finfo(features.dtype).eps)
        active_samples = float(sum(int(value.numel()) for value in weighted_losses))
    diagnostics = {
        "cosine_F_N": float(torch.dot(normalized_proto[f_idx], normalized_proto[n_idx]).detach().cpu()),
        "cosine_F_V": float(torch.dot(normalized_proto[f_idx], normalized_proto[v_idx]).detach().cpu()),
        "active_samples": active_samples,
    }
    return total, diagnostics


def _multi_kernel_matrix(x: torch.Tensor, y: torch.Tensor, gammas: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(x, y, p=2).pow(2)
    gamma_values = gammas.to(x.device, x.dtype)
    beta_values = beta.to(x.device, x.dtype)
    beta_values = beta_values / beta_values.sum().clamp_min(torch.finfo(x.dtype).eps)
    kernel = torch.zeros_like(distances)
    for gamma, weight in zip(gamma_values, beta_values):
        kernel = kernel + weight * torch.exp(-distances / gamma.clamp_min(torch.finfo(x.dtype).eps))
    return kernel


def _validate_features(source: torch.Tensor, target: torch.Tensor) -> None:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("Conditional MKMMD expects [B, D] source and target features.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target feature dimensions must match.")
