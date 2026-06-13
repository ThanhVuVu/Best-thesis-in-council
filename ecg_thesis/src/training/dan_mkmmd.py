from __future__ import annotations

import torch


def make_mkmmd_gammas(
    reference_distance: float | torch.Tensor,
    kernel_num: int,
    kernel_mul: float,
    gamma_min: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(reference_distance, torch.Tensor):
        ref = reference_distance.detach().to(device=device or reference_distance.device, dtype=dtype)
    else:
        ref = torch.as_tensor(float(reference_distance), device=device, dtype=dtype)
    ref = ref.clamp_min(float(gamma_min))
    offsets = torch.arange(int(kernel_num), device=ref.device, dtype=ref.dtype)
    offsets = offsets - int(kernel_num) // 2
    return ref * (float(kernel_mul) ** offsets)


def linear_mkmmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    gammas: torch.Tensor,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear-time unbiased MK-MMD estimate using DAN-style quad-tuples.

    The estimator may be negative on a mini-batch, which is expected for the
    unbiased form used by DAN.
    """
    _validate_feature_matrix(source_features, "source_features")
    _validate_feature_matrix(target_features, "target_features")
    n = min(int(source_features.shape[0]), int(target_features.shape[0]))
    n = n - (n % 2)
    if n < 2:
        return (source_features.sum() + target_features.sum()) * 0.0

    source = source_features[:n]
    target = target_features[:n]
    s1, s2 = source[0::2], source[1::2]
    t1, t2 = target[0::2], target[1::2]
    weights = beta_from_config(beta, int(gammas.numel()), source.device, source.dtype)
    gammas = gammas.to(source.device, source.dtype)

    g = (
        _multi_kernel_pairs(s1, s2, gammas, weights)
        + _multi_kernel_pairs(t1, t2, gammas, weights)
        - _multi_kernel_pairs(s1, t2, gammas, weights)
        - _multi_kernel_pairs(s2, t1, gammas, weights)
    )
    return g.mean()


def median_pairwise_squared_distance(features: torch.Tensor, gamma_min: float = 1.0e-6) -> float:
    _validate_feature_matrix(features, "features")
    if int(features.shape[0]) < 2:
        return float(gamma_min)
    with torch.no_grad():
        values = torch.pdist(features.detach().cpu().float(), p=2).pow(2)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return float(gamma_min)
        return float(values.median().clamp_min(float(gamma_min)).item())


def beta_from_config(
    beta: str | list[float] | torch.Tensor | None,
    kernel_num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if beta is None or (isinstance(beta, str) and beta == "uniform"):
        return torch.full((int(kernel_num),), 1.0 / max(int(kernel_num), 1), device=device, dtype=dtype)
    weights = torch.as_tensor(beta, device=device, dtype=dtype)
    if weights.numel() != int(kernel_num):
        raise ValueError(f"MK-MMD beta length must equal kernel_num={kernel_num}, got {weights.numel()}.")
    return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)


def _multi_kernel_pairs(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    dist2 = torch.sum((x - y) ** 2, dim=1)
    kernels = torch.exp(-dist2.unsqueeze(0) / gammas.clamp_min(torch.finfo(x.dtype).eps).unsqueeze(1))
    return torch.sum(beta.unsqueeze(1) * kernels, dim=0)


def _validate_feature_matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D], got {tuple(value.shape)}.")
