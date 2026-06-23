from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def minimum_class_confusion_loss(
    logits: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1.0e-5,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Minimum Class Confusion loss from target logits.

    This follows the official PyTorch reference pattern: soften target logits
    with temperature, compute entropy weights with detached entropy, build a
    class-correlation matrix, row-normalize it, then penalize off-diagonal mass.
    """

    if logits.ndim != 2:
        raise ValueError(f"MCC expects logits shaped [B, C], got {tuple(logits.shape)}.")
    if logits.size(0) == 0:
        loss = logits.sum() * 0.0
        if return_diagnostics:
            return loss, {
                "entropy_mean": 0.0,
                "pred_counts": [0 for _ in range(int(logits.size(1)))],
                "soft_confusion": torch.zeros(logits.size(1), logits.size(1), device=logits.device),
            }
        return loss
    if temperature <= 0:
        raise ValueError(f"MCC temperature must be positive, got {temperature}.")

    probs = F.softmax(logits / float(temperature), dim=1)
    entropy = -(probs * torch.log(probs + float(eps))).sum(dim=1)
    weights = (1.0 + torch.exp(-entropy)).detach()
    weights = logits.size(0) * weights / weights.sum().clamp_min(float(eps))

    correlation = probs.mul(weights.view(-1, 1)).transpose(1, 0).mm(probs)
    row_sums = correlation.sum(dim=1, keepdim=True).clamp_min(float(eps))
    soft_confusion = correlation / row_sums
    loss = (soft_confusion.sum() - torch.trace(soft_confusion)) / logits.size(1)

    if not return_diagnostics:
        return loss

    with torch.no_grad():
        pred_counts = torch.bincount(probs.argmax(dim=1), minlength=logits.size(1)).detach().cpu().tolist()
    return loss, {
        "entropy_mean": float(entropy.detach().mean().cpu()),
        "pred_counts": [int(v) for v in pred_counts],
        "soft_confusion": soft_confusion.detach(),
    }
