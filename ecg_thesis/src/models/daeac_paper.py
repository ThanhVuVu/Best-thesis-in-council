from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
        layers = self.forward_layers(x)
        return layers["gap_embed"]

    def forward_layers(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_conv(x)
        x = self.aspp_se_1(x)
        x = self.residual_1(x)
        x = self.aspp_se_2(x)
        x = self.residual_2(x)
        x = self.transition(x)
        transition_gap = _gap_flatten_2d(x)
        x = self.final_aspp_se(x)
        final_aspp_gap = _gap_flatten_2d(x)
        gap_embed = torch.flatten(self.gap(x), 1)
        return {
            "transition_gap": transition_gap,
            "final_aspp_gap": final_aspp_gap,
            "gap_embed": gap_embed,
        }


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


class DualClassifierH(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feature_dim, num_classes)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor, return_logits: bool = False):
        logits_1, logits_2 = self.forward_head_logits(features)
        logits = 0.5 * (logits_1 + logits_2)
        probs = torch.softmax(logits, dim=1)
        if return_logits:
            return logits, probs
        return probs

    def forward_head_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dropped = self.dropout(features)
        return self.fc(dropped), self.fc2(dropped)


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
        adaptation_fc: bool = False,
        dual_head: bool = False,
    ):
        super().__init__()
        self.feature_extractor = DAEACFeatureExtractor(
            input_channels=input_channels,
            initial_channels=initial_channels,
            feature_dim=feature_dim,
            dilations=dilations,
            se_reduction=se_reduction,
        )
        classifier_cls = DualClassifierH if dual_head else ClassifierH
        self.classifier = classifier_cls(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
        self.adaptation_fc = nn.Linear(feature_dim, feature_dim) if adaptation_fc else nn.Identity()
        self.adaptation_fc_enabled = bool(adaptation_fc)
        self.dual_head_enabled = bool(dual_head)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.apply(self._init_weights)
        if isinstance(self.adaptation_fc, nn.Linear):
            nn.init.eye_(self.adaptation_fc.weight)
            nn.init.zeros_(self.adaptation_fc.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.adaptation_fc(self.feature_extractor(x))

    def extract_feature_layers(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        layers = self.feature_extractor.forward_layers(x)
        layers["pre_adaptation_gap"] = layers["gap_embed"]
        layers["dan_fc"] = self.adaptation_fc(layers["gap_embed"])
        layers["gap_embed"] = layers["dan_fc"]
        return layers

    def forward(self, x: torch.Tensor, return_logits: bool = False, return_dict: bool = False):
        if return_dict:
            layers = self.extract_feature_layers(x)
            features = layers["gap_embed"]
            if isinstance(self.classifier, DualClassifierH):
                logits_1, logits_2 = self.classifier.forward_head_logits(features)
                logits = 0.5 * (logits_1 + logits_2)
                probs = torch.softmax(logits, dim=1)
                return {
                    "features": features,
                    "logits": logits,
                    "probabilities": probs,
                    "feature_layers": layers,
                    "logits_1": logits_1,
                    "logits_2": logits_2,
                    "probabilities_1": torch.softmax(logits_1, dim=1),
                    "probabilities_2": torch.softmax(logits_2, dim=1),
                }
            logits, probs = self.classifier(features, return_logits=True)
            output = {
                "features": features,
                "logits": logits,
                "probabilities": probs,
                "feature_layers": layers,
            }
            return output
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


def _gap_flatten_2d(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, 1).flatten(1)
