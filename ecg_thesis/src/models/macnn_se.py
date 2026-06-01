from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SqueezeExcite2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1).flatten(1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale)).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class ASPP2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: tuple[int, ...] = (1, 6, 12, 18)):
        super().__init__()
        branch_channels = out_channels // len(dilations)
        remainder = out_channels - branch_channels * len(dilations)
        branches = []
        for i, dilation in enumerate(dilations):
            channels = branch_channels + (remainder if i == 0 else 0)
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class MACNNSEBlock(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...], se_reduction: int):
        super().__init__()
        self.aspp = ASPP2D(channels, channels, dilations=dilations)
        self.se = SqueezeExcite2D(channels, reduction=se_reduction)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.aspp(x)
        out = self.se(out)
        return self.act(self.norm(out + residual))


class MACNN_SE(nn.Module):
    """MACNN_SE for DAEAC-style [B, 1, 3, 128] ECG inputs.

    The default forward follows the DAEAC-style convention requested in Phase 5:
    `features, logits = model(x)`. Use `return_embedding=True` for compatibility
    with existing thesis evaluators.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 1,
        channels: int = 256,
        embedding_dim: int = 256,
        se_reduction: int = 16,
        dilations: tuple[int, ...] = (1, 6, 12, 18),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = MACNNSEBlock(channels, dilations=dilations, se_reduction=se_reduction)
        self.block2 = MACNNSEBlock(channels, dilations=dilations, se_reduction=se_reduction)
        self.final = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ASPP2D(channels, channels, dilations=dilations),
            SqueezeExcite2D(channels, reduction=se_reduction),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channels, embedding_dim) if embedding_dim != channels else nn.Identity(),
            nn.ReLU(inplace=True) if embedding_dim != channels else nn.Identity(),
        )
        self.embedding_dim = int(embedding_dim)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.apply(self._init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.final(out)
        out = self.pool(out)
        return self.embedding(out)

    def forward(self, x: torch.Tensor, return_embedding: bool = False, logits_only: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(features)
        if logits_only:
            return logits
        if return_embedding:
            return logits, features
        return features, logits

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
