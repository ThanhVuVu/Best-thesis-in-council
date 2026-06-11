from __future__ import annotations

import torch
from torch import nn


class AtrousConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 3),
                padding=(0, dilation),
                dilation=(1, dilation),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SELayer2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale.expand_as(x)


class ASPP2D(nn.Module):
    """Four-branch ASPP used by the paper reference implementation.

    The paper figure is ambiguous about an avg-pool path. The provided DAEAC
    reference code uses only four atrous branches, so this faithful v1 follows
    that implementation.
    """

    def __init__(self, in_channels: int, branch_channels: int, dilations: tuple[int, ...] = (1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [AtrousConvBNReLU(in_channels, branch_channels, dilation=d) for d in dilations]
        )
        self.out_channels = int(branch_channels) * len(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.stride = int(stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            stride=(1, self.stride),
            padding=(0, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.shortcut = nn.AvgPool2d(kernel_size=(1, self.stride)) if self.stride > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + self.shortcut(x)


class ASPPSEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        se_reduction: int,
        dilations: tuple[int, ...] = (1, 6, 12, 18),
    ):
        super().__init__()
        self.aspp = ASPP2D(in_channels, branch_channels, dilations=dilations)
        self.se = SELayer2D(self.aspp.out_channels, reduction=se_reduction)

    @property
    def out_channels(self) -> int:
        return self.aspp.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.aspp(x))


class DAEACFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        initial_channels: int = 4,
        feature_dim: int = 256,
        dilations: tuple[int, ...] = (1, 6, 12, 18),
        se_reduction: int = 16,
    ):
        super().__init__()
        dila_num = len(dilations)
        c1 = initial_channels * dila_num
        c2 = initial_channels * dila_num * dila_num
        c3 = initial_channels * dila_num * dila_num * dila_num
        if c3 != feature_dim:
            raise ValueError(f"Expected final ASPP channels to equal feature_dim={feature_dim}, got {c3}.")

        self.input_conv = nn.Conv2d(input_channels, initial_channels, kernel_size=(3, 3), padding=(0, 1), bias=False)
        self.aspp_se_1 = ASPPSEBlock(initial_channels, initial_channels, se_reduction=4, dilations=dilations)
        self.residual_1 = ResidualConvBlock(c1, stride=1)
        self.aspp_se_2 = ASPPSEBlock(c1, c1, se_reduction=8, dilations=dilations)
        self.residual_2 = ResidualConvBlock(c2, stride=2)
        self.transition = nn.Sequential(nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.final_aspp_se = ASPPSEBlock(c2, c2, se_reduction=se_reduction, dilations=dilations)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = int(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.aspp_se_1(x)
        x = self.residual_1(x)
        x = self.aspp_se_2(x)
        x = self.residual_2(x)
        x = self.transition(x)
        x = self.final_aspp_se(x)
        x = self.gap(x)
        return torch.flatten(x, 1)


class ClassifierH(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor, return_logits: bool = False):
        logits = self.fc(self.dropout(features))
        probs = torch.softmax(logits, dim=1)
        if return_logits:
            return logits, probs
        return probs


class DAEACNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 1,
        initial_channels: int = 4,
        feature_dim: int = 256,
        dilations: tuple[int, ...] = (1, 6, 12, 18),
        se_reduction: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = DAEACFeatureExtractor(
            input_channels=input_channels,
            initial_channels=initial_channels,
            feature_dim=feature_dim,
            dilations=dilations,
            se_reduction=se_reduction,
        )
        self.classifier = ClassifierH(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.apply(self._init_weights)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        features = self.extract_features(x)
        logits, probs = self.classifier(features, return_logits=True)
        if return_logits:
            return features, logits, probs
        return features, probs

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
