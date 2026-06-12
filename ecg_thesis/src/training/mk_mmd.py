from __future__ import annotations

from typing import Any

import torch


def gaussian_kernel_distance2(x: torch.Tensor, y: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    """Gaussian kernels for a pair of vectors using exp(-||x-y||^2 / gamma)."""
    dist2 = torch.sum((x - y) ** 2)
    return torch.exp(-dist2 / gammas.clamp_min(torch.finfo(x.dtype).eps))


def make_dan_gammas(
    reference_distance: torch.Tensor | float,
    kernel_num: int = 5,
    kernel_mul: float = 2.0,
    gamma_min: float = 1.0e-6,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build the DAN-style bandwidth grid centered on a reference distance."""
    if isinstance(reference_distance, torch.Tensor):
        ref = reference_distance.detach()
        device = device or ref.device
        dtype = dtype or ref.dtype
    else:
        ref = torch.as_tensor(float(reference_distance), device=device, dtype=dtype or torch.float32)
    ref = ref.clamp_min(float(gamma_min))
    offsets = torch.arange(int(kernel_num), device=device or ref.device, dtype=dtype or ref.dtype)
    offsets = offsets - int(kernel_num) // 2
    return ref * (float(kernel_mul) ** offsets)


def center_pair_mk_mmd(
    source_center: torch.Tensor,
    target_center: torch.Tensor,
    gammas: torch.Tensor,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact MK-MMD between two single-point center distributions."""
    kernels = gaussian_kernel_distance2(source_center, target_center, gammas.to(source_center.device, source_center.dtype))
    if beta is None:
        beta = torch.full_like(kernels, 1.0 / max(int(kernels.numel()), 1))
    else:
        beta = beta.to(source_center.device, source_center.dtype)
        beta = beta / beta.sum().clamp_min(torch.finfo(beta.dtype).eps)
    return torch.sum(beta * (2.0 - 2.0 * kernels))


def center_cluster_mk_mmd_loss(
    source_centers: list[torch.Tensor | None],
    target_centers: list[torch.Tensor | None],
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    valid_pairs = [(cs, ct) for cs, ct in zip(source_centers, target_centers) if cs is not None and ct is not None]
    if not valid_pairs:
        return torch.zeros((), device=device)

    first = valid_pairs[0][0]
    cfg = dict(config or {})
    kernel_num = int(cfg.get("kernel_num", 5))
    kernel_mul = float(cfg.get("kernel_mul", 2.0))
    gamma_min = float(cfg.get("gamma_min", 1.0e-6))
    gammas = _gammas_for_center_pairs(valid_pairs, cfg, kernel_num, kernel_mul, gamma_min, first.device, first.dtype)
    beta = _beta_from_config(cfg, kernel_num, first.device, first.dtype)
    losses = [center_pair_mk_mmd(cs, ct, gammas, beta) for cs, ct in valid_pairs]
    return torch.stack(losses).mean()


def center_pair_reference_distance(
    source_centers: list[torch.Tensor | None],
    target_centers: list[torch.Tensor | None],
    gamma_min: float = 1.0e-6,
) -> torch.Tensor | None:
    valid = [
        torch.sum((cs.detach() - ct.detach()) ** 2)
        for cs, ct in zip(source_centers, target_centers)
        if cs is not None and ct is not None
    ]
    if not valid:
        return None
    return torch.stack(valid).mean().clamp_min(float(gamma_min))


def _gammas_for_center_pairs(
    valid_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    kernel_num: int,
    kernel_mul: float,
    gamma_min: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mode = str(cfg.get("gamma_mode", "adaptive")).lower()
    if mode in {"fixed", "fixed_from_initial_centers"}:
        value = cfg.get("fixed_gamma", cfg.get("gamma", None))
        if value is None:
            raise ValueError(f"MK-MMD gamma_mode={mode!r} requires 'fixed_gamma'.")
        reference = torch.as_tensor(float(value), device=device, dtype=dtype)
    elif mode in {"adaptive", "adaptive_from_valid_center_pairs"}:
        distances = [torch.sum((cs.detach() - ct.detach()) ** 2) for cs, ct in valid_pairs]
        reference = torch.stack(distances).mean()
    else:
        raise ValueError(f"Unknown MK-MMD gamma_mode: {mode}")
    return make_dan_gammas(reference, kernel_num=kernel_num, kernel_mul=kernel_mul, gamma_min=gamma_min, device=device, dtype=dtype)


def _beta_from_config(cfg: dict[str, Any], kernel_num: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    value = cfg.get("beta", "uniform")
    if value in (None, "uniform"):
        return None
    beta = torch.as_tensor(value, device=device, dtype=dtype)
    if beta.numel() != int(kernel_num):
        raise ValueError(f"MK-MMD beta length must equal kernel_num={kernel_num}, got {beta.numel()}.")
    return beta
