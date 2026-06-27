from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.models.cdan import ConditionalMap
from src.models.daeac_paper import ClassifierH, DAEACFeatureExtractor, LateFusionClassifierH
from src.models.grl import GradientReversalLayer


class DAEACDANNModel(nn.Module):
    def __init__(
        self,
        feature_extractor: DAEACFeatureExtractor,
        classifier: ClassifierH,
        feature_dim: int,
        num_classes: int,
        num_domains: int = 2,
        domain_hidden_dim: int | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        hidden_dim = int(domain_hidden_dim or feature_dim)
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(num_domains)),
        )

    def extract_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        raw_features = self.extract_raw_features(x)
        return self.domain_features(raw_features)

    def domain_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        if isinstance(self.classifier, LateFusionClassifierH):
            return self.classifier.extract_morph_features(raw_features)
        return raw_features

    def class_logits(self, raw_features: torch.Tensor, rr_features: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.classifier, LateFusionClassifierH):
            _, logits, _ = self.classifier(raw_features, rr_features, return_logits=True)
            return logits
        logits, _ = self.classifier(raw_features, return_logits=True)
        return logits

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor | None = None, return_embedding: bool = False):
        raw_features = self.extract_raw_features(x)
        features = self.domain_features(raw_features)
        logits = self.class_logits(raw_features, rr_features)
        if return_embedding:
            return logits, features
        return logits

    def forward_domain(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        features = self.extract_features(x)
        return self.forward_domain_from_features(features, lambd)

    def forward_domain_from_features(self, features: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.domain_classifier(self.grl(features, lambd))


class DAEACCDANModel(nn.Module):
    def __init__(
        self,
        feature_extractor: DAEACFeatureExtractor,
        classifier: ClassifierH,
        feature_dim: int,
        num_classes: int,
        conditioning: str = "auto",
        randomized_threshold: int = 4096,
        random_dim: int = 1024,
        domain_hidden_dim: int | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.conditional_map = ConditionalMap(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            conditioning=conditioning,
            randomized_threshold=int(randomized_threshold),
            random_dim=int(random_dim),
        )
        self.grl = GradientReversalLayer()
        hidden_dim = int(domain_hidden_dim or feature_dim)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.conditional_map.output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def extract_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        raw_features = self.extract_raw_features(x)
        return self.domain_features(raw_features)

    def domain_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        if isinstance(self.classifier, LateFusionClassifierH):
            return self.classifier.extract_morph_features(raw_features)
        return raw_features

    def class_logits(self, raw_features: torch.Tensor, rr_features: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.classifier, LateFusionClassifierH):
            _, logits, _ = self.classifier(raw_features, rr_features, return_logits=True)
            return logits
        logits, _ = self.classifier(raw_features, return_logits=True)
        return logits

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor | None = None, return_embedding: bool = False):
        raw_features = self.extract_raw_features(x)
        features = self.domain_features(raw_features)
        logits = self.class_logits(raw_features, rr_features)
        if return_embedding:
            return logits, features
        return logits

    def conditional_features(self, features: torch.Tensor, logits: torch.Tensor, detach_softmax: bool = True) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        if detach_softmax:
            probs = probs.detach()
        return self.conditional_map(features, probs)

    def forward_domain_from_features(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        lambd: float,
        detach_softmax: bool = True,
    ) -> torch.Tensor:
        conditional = self.conditional_features(features, logits, detach_softmax=detach_softmax)
        return self.domain_classifier(self.grl(conditional, lambd))


class DAEACADDAModel(nn.Module):
    def __init__(
        self,
        source_encoder: DAEACFeatureExtractor,
        classifier: ClassifierH,
        feature_dim: int,
        discriminator_hidden_dims: list[int] | tuple[int, ...] | int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.source_encoder = source_encoder
        self.target_encoder = copy.deepcopy(source_encoder)
        self.classifier = classifier
        self.feature_dim = int(feature_dim)
        self.domain_discriminator = ADDADomainDiscriminator(
            embedding_dim=self.feature_dim,
            hidden_dims=discriminator_hidden_dims,
            dropout=float(dropout),
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

    def class_logits(self, features: torch.Tensor) -> torch.Tensor:
        logits, _ = self.classifier(features, return_logits=True)
        return logits

    def forward_domain_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.domain_discriminator(features)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        features = self.forward_target_features(x)
        logits = self.class_logits(features)
        if return_embedding:
            return logits, features
        return logits

    def _freeze_source_modules(self) -> None:
        for module in (self.source_encoder, self.classifier):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


class ADDADomainDiscriminator(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: list[int] | tuple[int, ...] | int = 256, dropout: float = 0.1):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims, hidden_dims]
        dims = [int(embedding_dim), *[int(dim) for dim in hidden_dims], 1]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def entropy(probabilities: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities + float(eps))).sum(dim=1)
