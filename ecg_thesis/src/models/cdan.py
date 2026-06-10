from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models import build_model
from src.models.grl import GradientReversalLayer


class ConditionalMap(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        conditioning: str = "auto",
        randomized_threshold: int = 4096,
        random_dim: int = 1024,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.random_dim = int(random_dim)
        product_dim = self.feature_dim * self.num_classes
        if conditioning == "auto":
            conditioning = "randomized" if product_dim > int(randomized_threshold) else "multilinear"
        if conditioning not in {"multilinear", "randomized"}:
            raise ValueError(f"Unsupported CDAN conditioning: {conditioning!r}")
        self.conditioning = conditioning
        if self.conditioning == "randomized":
            self.register_buffer("random_features", torch.randn(self.feature_dim, self.random_dim))
            self.register_buffer("random_predictions", torch.randn(self.num_classes, self.random_dim))
            self.output_dim = self.random_dim
        else:
            self.output_dim = product_dim

    def forward(self, features: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "randomized":
            f_proj = features @ self.random_features
            g_proj = probabilities @ self.random_predictions
            return (f_proj * g_proj) / math.sqrt(float(self.random_dim))
        op = torch.bmm(probabilities.unsqueeze(2), features.unsqueeze(1))
        return op.flatten(start_dim=1)


class CDANModel(nn.Module):
    def __init__(
        self,
        backbone: str = "clef_pretrained",
        num_classes: int = 3,
        dropout: float = 0.3,
        backbone_kwargs: dict | None = None,
        reuse_backbone_classifier: bool = False,
        conditioning: str = "auto",
        randomized_threshold: int = 4096,
        random_dim: int = 1024,
        domain_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.feature_extractor = build_model(backbone, num_classes=num_classes, **(backbone_kwargs or {}))
        embedding_dim = self._infer_embedding_dim()
        self.reuse_backbone_classifier = bool(reuse_backbone_classifier)
        if self.reuse_backbone_classifier:
            classifier = getattr(self.feature_extractor, "classifier", None)
            if classifier is None:
                raise ValueError(f"Backbone {backbone!r} has no classifier to reuse for CDAN")
            self.label_classifier = classifier
        else:
            self.label_classifier = nn.Linear(embedding_dim, num_classes)
        self.conditional_map = ConditionalMap(
            feature_dim=embedding_dim,
            num_classes=num_classes,
            conditioning=conditioning,
            randomized_threshold=randomized_threshold,
            random_dim=random_dim,
        )
        self.grl = GradientReversalLayer()
        hidden_dim = int(domain_hidden_dim or embedding_dim)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.conditional_map.output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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

    def conditional_features(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        detach_softmax: bool = True,
    ) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        if detach_softmax:
            probabilities = probabilities.detach()
        return self.conditional_map(features, probabilities)

    def forward_domain_from_features(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        lambd: float,
        detach_softmax: bool = True,
    ) -> torch.Tensor:
        conditional = self.conditional_features(features, logits, detach_softmax=detach_softmax)
        return self.domain_classifier(self.grl(conditional, lambd))

