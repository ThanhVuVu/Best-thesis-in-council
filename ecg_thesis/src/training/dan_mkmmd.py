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

    weights = beta_from_config(beta, int(gammas.numel()), source_features.device, source_features.dtype)
    per_kernel = linear_mkmmd_quadruple_values(source_features[:n], target_features[:n], gammas)
    return torch.mean(per_kernel @ weights)


def linear_mkmmd_quadruple_values(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    """Return g_k(z_i) for every DAN quad-tuple and every Gaussian kernel.

    Shape is ``[N/2, M]`` where ``M`` is the number of kernels. Keeping the
    per-kernel values is required by DAN's alternating QP update for beta.
    """
    _validate_feature_matrix(source_features, "source_features")
    _validate_feature_matrix(target_features, "target_features")
    n = min(int(source_features.shape[0]), int(target_features.shape[0]))
    n -= n % 2
    if n < 2:
        return source_features.new_zeros((0, int(gammas.numel())))
    source = source_features[:n]
    target = target_features[:n]
    s1, s2 = source[0::2], source[1::2]
    t1, t2 = target[0::2], target[1::2]
    gammas = gammas.to(source.device, source.dtype)
    return (
        _kernel_pairs_per_kernel(s1, s2, gammas)
        + _kernel_pairs_per_kernel(t1, t2, gammas)
        - _kernel_pairs_per_kernel(s1, t2, gammas)
        - _kernel_pairs_per_kernel(s2, t1, gammas)
    )


def dan_qp_statistics(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    gammas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DAN's d vector and covariance Q from linear-time statistics."""
    g = linear_mkmmd_quadruple_values(source_features, target_features, gammas)
    if int(g.shape[0]) == 0:
        m = int(gammas.numel())
        return gammas.new_zeros(m), gammas.new_zeros((m, m))
    d = g.mean(dim=0)
    pair_count = int(g.shape[0]) - (int(g.shape[0]) % 2)
    if pair_count < 2:
        return d, torch.eye(int(gammas.numel()), device=g.device, dtype=g.dtype)
    delta = g[:pair_count:2] - g[1:pair_count:2]
    q = delta.transpose(0, 1) @ delta / max(int(delta.shape[0]), 1)
    return d, q


def solve_dan_kernel_qp(
    d: torch.Tensor,
    q: torch.Tensor,
    epsilon: float = 1.0e-3,
    positivity_floor: float = 1.0e-8,
) -> torch.Tensor:
    """Solve min beta' (Q+eps I) beta, s.t. d' beta=1, beta>=0.

    The active-set solution is dependency-free and deterministic. The returned
    vector satisfies the QP scale constraint; call ``normalize_kernel_beta``
    before using it as convex mixture weights.
    """
    if d.ndim != 1 or q.shape != (d.numel(), d.numel()):
        raise ValueError("DAN kernel QP expects d=[M] and Q=[M,M].")
    dtype, device = q.dtype, q.device
    d_safe = d.detach().to(device=device, dtype=dtype).clamp_min(float(positivity_floor))
    a = q.detach().to(dtype=dtype)
    a = 0.5 * (a + a.transpose(0, 1))
    a = a + float(epsilon) * torch.eye(d.numel(), device=device, dtype=dtype)
    active = torch.ones(d.numel(), dtype=torch.bool, device=device)
    beta = torch.zeros_like(d_safe)
    while bool(active.any()):
        idx = torch.nonzero(active, as_tuple=False).flatten()
        a_free = a.index_select(0, idx).index_select(1, idx)
        d_free = d_safe.index_select(0, idx)
        solution = torch.linalg.solve(a_free, d_free)
        denominator = torch.dot(d_free, solution).clamp_min(torch.finfo(dtype).eps)
        candidate = solution / denominator
        if bool((candidate > 0).all()):
            beta[idx] = candidate
            break
        most_negative = idx[torch.argmin(candidate)]
        active[most_negative] = False
    if not bool(beta.gt(0).any()):
        beta = torch.ones_like(d_safe) / d_safe.sum().clamp_min(torch.finfo(dtype).eps)
    beta = beta.clamp_min(float(positivity_floor))
    return beta / torch.dot(d_safe, beta).clamp_min(torch.finfo(dtype).eps)


def normalize_kernel_beta(beta: torch.Tensor) -> torch.Tensor:
    beta = beta.clamp_min(0.0)
    return beta / beta.sum().clamp_min(torch.finfo(beta.dtype).eps)


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


def _kernel_pairs_per_kernel(x: torch.Tensor, y: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    dist2 = torch.sum((x - y) ** 2, dim=1)
    kernels = torch.exp(-dist2.unsqueeze(1) / gammas.clamp_min(torch.finfo(x.dtype).eps).unsqueeze(0))
    return kernels


def _validate_feature_matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D], got {tuple(value.shape)}.")
