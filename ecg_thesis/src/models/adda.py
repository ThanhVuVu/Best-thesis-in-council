from __future__ import annotations

import copy

import torch
from torch import nn


class ADDAClassifier(nn.Module):
    def __init__(
        self,
        source_encoder: nn.Module,
        classifier: nn.Module,
        embedding_dim: int,
        discriminator_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.source_encoder = source_encoder
        self.target_encoder = copy.deepcopy(source_encoder)
        self.classifier = classifier
        self.embedding_dim = int(embedding_dim)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.embedding_dim, int(discriminator_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(discriminator_hidden_dim), 1),
        )
        self._freeze_source_modules()

    def train(self, mode: bool = True):
        super().train(mode)
        self.source_encoder.eval()
        self.classifier.eval()
        return self

    @torch.no_grad()
    def forward_source_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(x)

    def forward_target_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x)

    def forward_domain_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.domain_discriminator(features).squeeze(-1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.forward_target_features(x)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits

    def _freeze_source_modules(self) -> None:
        for module in (self.source_encoder, self.classifier):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
