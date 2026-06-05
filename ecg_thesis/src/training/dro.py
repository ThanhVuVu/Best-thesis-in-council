from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ClassGroupDROLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        class_weights: torch.Tensor | None = None,
        eta: float = 0.1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.eta = float(eta)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.register_buffer("adv_probs", torch.ones(self.num_classes, dtype=torch.float32) / self.num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        losses = F.cross_entropy(logits, target, weight=self.class_weights, reduction="none")
        group_losses = []
        for cls in range(self.num_classes):
            mask = target == cls
            if mask.any():
                group_losses.append(losses[mask].mean())
            else:
                group_losses.append(torch.zeros((), dtype=losses.dtype, device=losses.device))
        group_loss_tensor = torch.stack(group_losses)
        present = torch.tensor(
            [bool((target == cls).any()) for cls in range(self.num_classes)],
            dtype=losses.dtype,
            device=losses.device,
        )
        with torch.no_grad():
            update = torch.exp(self.eta * group_loss_tensor.detach()) * present
            if float(update.sum().detach().cpu()) > 0:
                new_probs = self.adv_probs.to(losses.device) * torch.clamp(update, min=1e-12)
                if float(new_probs.sum().detach().cpu()) > 0:
                    new_probs = new_probs / new_probs.sum()
                    self.adv_probs.copy_(new_probs.detach().cpu())
        probs = self.adv_probs.to(losses.device)
        loss = torch.sum(probs * group_loss_tensor)
        stats = {f"group_loss_{idx}": float(group_loss_tensor[idx].detach().cpu()) for idx in range(self.num_classes)}
        stats.update({f"dro_adv_prob_{idx}": float(probs[idx].detach().cpu()) for idx in range(self.num_classes)})
        return loss, stats


def classifier_discrepancy(probs1: torch.Tensor, probs2: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(probs1 - probs2, ord=2, dim=1).mean()

