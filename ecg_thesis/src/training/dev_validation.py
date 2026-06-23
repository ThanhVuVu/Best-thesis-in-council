from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class DEVDiscriminator(nn.Module):
    """Two-layer domain logistic regressor used by Deep Embedded Validation."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(1)


@dataclass(frozen=True)
class DEVArrays:
    source_train_features: torch.Tensor
    target_features: torch.Tensor
    source_val_features: torch.Tensor
    source_val_losses: torch.Tensor


def estimate_dev_risk(
    model: nn.Module,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """Estimate target risk without exposing any target label.

    The implementation follows DEV directly: train M on embedded source-train
    versus unlabeled target-test features, estimate density ratios on held-out
    labeled source validation features, then apply the control variate.
    """

    arrays = extract_dev_arrays(model, source_train_loader, source_val_loader, target_loader, device)
    dev_cfg = dict(config)
    discriminator, domain_metrics = _fit_domain_discriminator(
        arrays.source_train_features,
        arrays.target_features,
        device=device,
        hidden_dim=int(dev_cfg.get("hidden_dim", arrays.source_train_features.size(1))),
        epochs=int(dev_cfg.get("discriminator_epochs", 10)),
        batch_size=int(dev_cfg.get("batch_size", 2048)),
        lr=float(dev_cfg.get("lr", 1.0e-3)),
        weight_decay=float(dev_cfg.get("weight_decay", 0.0)),
        seed=int(seed),
    )

    eps = float(dev_cfg.get("probability_eps", 1.0e-6))
    with torch.no_grad():
        val_features = arrays.source_val_features.to(device)
        source_probability = torch.sigmoid(discriminator(val_features)).clamp(eps, 1.0 - eps).cpu()

    n_source = int(arrays.source_train_features.size(0))
    n_target = int(arrays.target_features.size(0))
    weights = (float(n_source) / float(n_target)) * (1.0 - source_probability) / source_probability
    result = dev_control_variate_risk(
        arrays.source_val_losses,
        weights,
        variance_eps=float(dev_cfg.get("variance_eps", 1.0e-12)),
    )
    result.update(
        {
            "dev_n_source_train": float(n_source),
            "dev_n_source_val": float(arrays.source_val_features.size(0)),
            "dev_n_target": float(n_target),
            **domain_metrics,
        }
    )
    return result


def dev_control_variate_risk(
    validation_losses: torch.Tensor,
    importance_weights: torch.Tensor,
    variance_eps: float = 1.0e-12,
) -> dict[str, float]:
    """Compute R_DEV = mean(W*l) + eta*mean(W) - eta."""

    losses = validation_losses.detach().double().flatten()
    weights = importance_weights.detach().double().flatten()
    if losses.numel() == 0 or losses.numel() != weights.numel():
        raise ValueError("DEV requires equally sized, non-empty validation losses and importance weights.")
    weighted_losses = weights * losses
    centered_l = weighted_losses - weighted_losses.mean()
    centered_w = weights - weights.mean()
    covariance = (centered_l * centered_w).mean()
    variance = centered_w.square().mean()
    eta = -covariance / variance if float(variance) > float(variance_eps) else variance.new_zeros(())
    risk = weighted_losses.mean() + eta * weights.mean() - eta
    return {
        "dev_risk": float(risk),
        "dev_eta": float(eta),
        "dev_weight_mean": float(weights.mean()),
        "dev_weight_std": float(weights.std(unbiased=False)),
        "dev_weight_min": float(weights.min()),
        "dev_weight_max": float(weights.max()),
        "dev_weighted_loss_mean": float(weighted_losses.mean()),
        "dev_weight_variance": float(variance),
    }


@torch.no_grad()
def extract_dev_arrays(
    model: nn.Module,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_loader: DataLoader,
    device: torch.device,
) -> DEVArrays:
    was_training = model.training
    model.eval()
    source_features = _extract_features(model, source_train_loader, device)
    target_features = _extract_features(model, target_loader, device)

    val_features: list[torch.Tensor] = []
    val_losses: list[torch.Tensor] = []
    for batch in source_val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        features, logits, _ = model(x, return_logits=True)
        val_features.append(features.detach().cpu())
        val_losses.append(F.cross_entropy(logits, y, reduction="none").detach().cpu())
    if was_training:
        model.train()
    if not val_features:
        raise ValueError("DEV source validation loader is empty.")
    return DEVArrays(
        source_train_features=source_features,
        target_features=target_features,
        source_val_features=torch.cat(val_features),
        source_val_losses=torch.cat(val_losses),
    )


@torch.no_grad()
def _extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        features = model.extract_features(x.to(device))
        rows.append(features.detach().cpu())
    if not rows:
        raise ValueError("DEV feature loader is empty.")
    return torch.cat(rows)


def _fit_domain_discriminator(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[DEVDiscriminator, dict[str, float]]:
    features = torch.cat((source_features, target_features), dim=0).float()
    domains = torch.cat((torch.ones(len(source_features)), torch.zeros(len(target_features))), dim=0).float()
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(TensorDataset(features, domains), batch_size=batch_size, shuffle=True, generator=generator)
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        discriminator = DEVDiscriminator(features.size(1), hidden_dim).to(device)
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)
        final_loss = 0.0
        for _ in range(epochs):
            discriminator.train()
            loss_sum = 0.0
            sample_count = 0
            for feature_batch, domain_batch in loader:
                feature_batch = feature_batch.to(device)
                domain_batch = domain_batch.to(device)
                loss = F.binary_cross_entropy_with_logits(discriminator(feature_batch), domain_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(feature_batch)
                sample_count += len(feature_batch)
            final_loss = loss_sum / max(sample_count, 1)

    discriminator.eval()
    with torch.no_grad():
        logits = discriminator(features.to(device)).cpu()
        predictions = (logits >= 0).float()
        accuracy = float((predictions == domains).float().mean())
    return discriminator, {"dev_domain_loss": final_loss, "dev_domain_accuracy": accuracy}
