from __future__ import annotations

import torch
from torch import nn

from src.models.catnet1d import CATNet1D


class CATNetBiClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        time_feature_dim: int = 3,
        classifier_hidden_dim: int = 128,
        classifier_dropout: float = 0.2,
        **catnet_kwargs,
    ):
        super().__init__()
        self.feature_extractor = CATNet1D(num_classes=num_classes, **catnet_kwargs)
        self.embedding_dim = int(self.feature_extractor.embedding_dim)
        self.time_feature_dim = int(time_feature_dim)
        fused_dim = self.embedding_dim + self.time_feature_dim
        self.classifier1 = _ClassifierHead(fused_dim, classifier_hidden_dim, num_classes, classifier_dropout)
        self.classifier2 = _ClassifierHead(fused_dim, classifier_hidden_dim, num_classes, classifier_dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor.forward_features(x)

    def forward(
        self,
        x: torch.Tensor,
        time_features: torch.Tensor,
        return_embedding: bool = False,
        return_all: bool = False,
    ):
        embedding = self.forward_features(x)
        fused = torch.cat([embedding, time_features], dim=1)
        logits1 = self.classifier1(fused)
        logits2 = self.classifier2(fused)
        logits = 0.5 * (logits1 + logits2)
        if return_all:
            probs1 = torch.softmax(logits1, dim=1)
            probs2 = torch.softmax(logits2, dim=1)
            return {
                "logits": logits,
                "logits1": logits1,
                "logits2": logits2,
                "probabilities1": probs1,
                "probabilities2": probs2,
                "embedding": embedding,
                "fused": fused,
            }
        if return_embedding:
            return logits, embedding
        return logits


class _ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

