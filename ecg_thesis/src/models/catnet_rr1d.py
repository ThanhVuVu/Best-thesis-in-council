from __future__ import annotations

import torch
from torch import nn

from src.models.catnet1d import CATNet1D


class CATNetRR1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        rr_feature_dim: int = 4,
        rr_embedding_dim: int = 32,
        d_model: int = 128,
        num_heads: int = 4,
        dff: int = 128,
        num_transformer_layers: int = 1,
        attention_reduction: int = 8,
        dropout: float = 0.2,
        max_len: int = 512,
    ):
        super().__init__()
        self.waveform_encoder = CATNet1D(
            num_classes=num_classes,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            num_transformer_layers=num_transformer_layers,
            attention_reduction=attention_reduction,
            dropout=dropout,
            max_len=max_len,
        )
        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_feature_dim, rr_embedding_dim),
            nn.BatchNorm1d(rr_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(rr_embedding_dim, rr_embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model + rr_embedding_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.embedding_dim = d_model
        self.classifier = nn.Linear(d_model, num_classes)

    def forward_features(self, x: torch.Tensor, rr_features: torch.Tensor) -> torch.Tensor:
        waveform_embedding = self.waveform_encoder.forward_features(x)
        rr_embedding = self.rr_encoder(rr_features)
        return self.fusion(torch.cat([waveform_embedding, rr_embedding], dim=1))

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor, return_embedding: bool = False):
        embedding = self.forward_features(x, rr_features)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits
