from __future__ import annotations

import torch
from torch import nn

from src.models import build_model
from src.models.grl import GradientReversalLayer


class DANNModel(nn.Module):
    def __init__(
        self,
        backbone: str = "inceptiontime1d",
        num_classes: int = 3,
        num_domains: int = 2,
        dropout: float = 0.3,
        backbone_kwargs: dict | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.feature_extractor = build_model(backbone, num_classes=num_classes, **(backbone_kwargs or {}))
        embedding_dim = self._infer_embedding_dim()
        self.label_classifier = nn.Linear(embedding_dim, num_classes)
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_domains),
        )

    def _infer_embedding_dim(self) -> int:
        if hasattr(self.feature_extractor, "embedding_dim"):
            return int(self.feature_extractor.embedding_dim)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 250)
            _, embedding = self.feature_extractor(dummy, return_embedding=True)
            return int(embedding.shape[1])

    def extract_features(self, *inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self.feature_extractor, "forward_features"):
            return self.feature_extractor.forward_features(*inputs)
        _, embedding = self.feature_extractor(*inputs, return_embedding=True)
        return embedding

    def forward(self, *inputs: torch.Tensor, return_embedding: bool = False):
        features = self.extract_features(*inputs)
        logits = self.label_classifier(features)
        if return_embedding:
            return logits, features
        return logits

    def forward_domain(self, *inputs: torch.Tensor, lambd: float) -> torch.Tensor:
        features = self.extract_features(*inputs)
        reversed_features = self.grl(features, lambd)
        return self.domain_classifier(reversed_features)

    def forward_all(self, *inputs: torch.Tensor, lambd: float):
        features = self.extract_features(*inputs)
        class_logits = self.label_classifier(features)
        domain_logits = self.domain_classifier(self.grl(features, lambd))
        return class_logits, domain_logits, features
