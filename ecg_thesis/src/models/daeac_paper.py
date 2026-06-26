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


class FrequencyConvolutionBlockAttention2D(nn.Module):
    """FCBA attention adapted to DAEAC's [B, C, 1, T] feature maps."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        frequency_modes: int = 4,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        if frequency_modes <= 0:
            raise ValueError("frequency_modes must be positive.")
        if spatial_kernel_size <= 0 or spatial_kernel_size % 2 == 0:
            raise ValueError("spatial_kernel_size must be a positive odd integer.")
        hidden_channels = max(channels // reduction, 1)
        hidden_frequency = max(frequency_modes // reduction, 1)
        self.frequency_modes = int(frequency_modes)
        self.frequency_mlp = nn.Sequential(
            nn.Linear(self.frequency_modes, hidden_frequency, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_frequency, 1, bias=False),
            nn.Sigmoid(),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=(1, spatial_kernel_size),
            padding=(0, spatial_kernel_size // 2),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency_scale = self.frequency_mlp(self._dct_descriptor(x)).view(x.shape[0], x.shape[1], 1, 1)
        x = x * frequency_scale.expand_as(x)

        avg_scale = self.channel_mlp(self.avg_pool(x).flatten(1))
        max_scale = self.channel_mlp(self.max_pool(x).flatten(1))
        channel_scale = torch.sigmoid(avg_scale + max_scale).view(x.shape[0], x.shape[1], 1, 1)
        x = x * channel_scale.expand_as(x)

        spatial_avg = x.mean(dim=1, keepdim=True)
        spatial_max = x.amax(dim=1, keepdim=True)
        spatial_scale = torch.sigmoid(self.spatial(torch.cat([spatial_avg, spatial_max], dim=1)))
        return x * spatial_scale.expand_as(x)

    def _dct_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        flat = x.flatten(2)
        length = flat.shape[-1]
        modes = min(self.frequency_modes, length)
        positions = torch.arange(length, device=x.device, dtype=x.dtype)
        frequencies = torch.arange(modes, device=x.device, dtype=x.dtype).unsqueeze(1)
        basis = torch.cos(torch.pi * (positions + 0.5) * frequencies / float(length))
        scale = torch.full((modes,), (2.0 / float(length)) ** 0.5, device=x.device, dtype=x.dtype)
        scale[0] = (1.0 / float(length)) ** 0.5
        descriptor = torch.matmul(flat, (basis * scale.unsqueeze(1)).t())
        if modes < self.frequency_modes:
            descriptor = F.pad(descriptor, (0, self.frequency_modes - modes))
        return descriptor.view(b, c, self.frequency_modes)


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
        attention_type: str = "se",
        fcba_frequency_modes: int = 4,
        fcba_spatial_kernel_size: int = 7,
        fcba_reduction: int | None = None,
    ):
        super().__init__()
        self.aspp = ASPP2D(in_channels, branch_channels, dilations=dilations)
        attention_type = attention_type.lower()
        if attention_type == "se":
            self.se = SELayer2D(self.aspp.out_channels, reduction=se_reduction)
        elif attention_type == "fcba":
            self.se = FrequencyConvolutionBlockAttention2D(
                self.aspp.out_channels,
                reduction=fcba_reduction if fcba_reduction is not None else se_reduction,
                frequency_modes=fcba_frequency_modes,
                spatial_kernel_size=fcba_spatial_kernel_size,
            )
        else:
            raise ValueError(f"Unknown DAEAC attention_type={attention_type!r}; expected 'se' or 'fcba'.")

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
        attention_type: str = "se",
        fcba_frequency_modes: int = 4,
        fcba_spatial_kernel_size: int = 7,
        fcba_reduction: int | None = None,
        input_rows: int = 3,
    ):
        super().__init__()
        input_rows = int(input_rows)
        if input_rows not in {1, 3}:
            raise ValueError(f"DAEAC input_rows must be 1 or 3, got {input_rows}.")
        dila_num = len(dilations)
        c1 = initial_channels * dila_num
        c2 = initial_channels * dila_num * dila_num
        c3 = initial_channels * dila_num * dila_num * dila_num
        if c3 != feature_dim:
            raise ValueError(f"Expected final ASPP channels to equal feature_dim={feature_dim}, got {c3}.")

        self.input_conv = nn.Conv2d(
            input_channels,
            initial_channels,
            kernel_size=(input_rows, 3),
            padding=(0, 1),
            bias=False,
        )
        self.aspp_se_1 = ASPPSEBlock(
            initial_channels,
            initial_channels,
            se_reduction=4,
            dilations=dilations,
            attention_type=attention_type,
            fcba_frequency_modes=fcba_frequency_modes,
            fcba_spatial_kernel_size=fcba_spatial_kernel_size,
            fcba_reduction=fcba_reduction,
        )
        self.residual_1 = ResidualConvBlock(c1, stride=1)
        self.aspp_se_2 = ASPPSEBlock(
            c1,
            c1,
            se_reduction=8,
            dilations=dilations,
            attention_type=attention_type,
            fcba_frequency_modes=fcba_frequency_modes,
            fcba_spatial_kernel_size=fcba_spatial_kernel_size,
            fcba_reduction=fcba_reduction,
        )
        self.residual_2 = ResidualConvBlock(c2, stride=2)
        self.transition = nn.Sequential(nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.final_aspp_se = ASPPSEBlock(
            c2,
            c2,
            se_reduction=se_reduction,
            dilations=dilations,
            attention_type=attention_type,
            fcba_frequency_modes=fcba_frequency_modes,
            fcba_spatial_kernel_size=fcba_spatial_kernel_size,
            fcba_reduction=fcba_reduction,
        )
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


class LateFusionClassifierH(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 4,
        rr_dim: int = 7,
        fc1_dim: int = 128,
        fc2_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(feature_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim + rr_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, num_classes)
        self.feature_dim = int(fc1_dim)
        self.rr_dim = int(rr_dim)

    def extract_morph_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.fc1(features)))

    def forward(self, features: torch.Tensor, rr_features: torch.Tensor | None = None, return_logits: bool = False):
        if rr_features is None:
            raise ValueError("LateFusionClassifierH requires rr_features.")
        if rr_features.ndim != 2 or rr_features.shape[1] != self.rr_dim:
            raise ValueError(f"Expected rr_features shape [B, {self.rr_dim}], got {tuple(rr_features.shape)}.")
        morph_features = self.extract_morph_features(features)
        fused = torch.cat([morph_features, rr_features.to(device=features.device, dtype=features.dtype)], dim=1)
        hidden = self.dropout(F.relu(self.fc2(fused)))
        logits = self.fc3(hidden)
        probs = torch.softmax(logits, dim=1)
        if return_logits:
            return morph_features, logits, probs
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
        adaptation_fc: bool = False,
        dual_head: bool = False,
        attention_type: str = "se",
        fcba_frequency_modes: int = 4,
        fcba_spatial_kernel_size: int = 7,
        fcba_reduction: int | None = None,
        input_rows: int = 3,
        late_fusion: bool = False,
        rr_dim: int = 7,
        late_fusion_fc1_dim: int = 128,
        late_fusion_fc2_dim: int = 64,
    ):
        super().__init__()
        if late_fusion and dual_head:
            raise ValueError("DAEAC late fusion does not support dual_head=True in this phase.")
        self.feature_extractor = DAEACFeatureExtractor(
            input_channels=input_channels,
            initial_channels=initial_channels,
            feature_dim=feature_dim,
            dilations=dilations,
            se_reduction=se_reduction,
            attention_type=attention_type,
            fcba_frequency_modes=fcba_frequency_modes,
            fcba_spatial_kernel_size=fcba_spatial_kernel_size,
            fcba_reduction=fcba_reduction,
            input_rows=input_rows,
        )
        if late_fusion:
            self.classifier = LateFusionClassifierH(
                feature_dim=feature_dim,
                num_classes=num_classes,
                rr_dim=rr_dim,
                fc1_dim=late_fusion_fc1_dim,
                fc2_dim=late_fusion_fc2_dim,
                dropout=dropout,
            )
        else:
            classifier_cls = DualClassifierH if dual_head else ClassifierH
            self.classifier = classifier_cls(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
        self.adaptation_fc = nn.Linear(feature_dim, feature_dim) if adaptation_fc else nn.Identity()
        self.adaptation_fc_enabled = bool(adaptation_fc)
        self.dual_head_enabled = bool(dual_head)
        self.late_fusion_enabled = bool(late_fusion)
        self.feature_dim = int(self.classifier.feature_dim if isinstance(self.classifier, LateFusionClassifierH) else feature_dim)
        self.num_classes = int(num_classes)
        self.apply(self._init_weights)
        if isinstance(self.adaptation_fc, nn.Linear):
            nn.init.eye_(self.adaptation_fc.weight)
            nn.init.zeros_(self.adaptation_fc.bias)

    def extract_features(self, x: torch.Tensor, rr_features: torch.Tensor | None = None) -> torch.Tensor:
        features = self.adaptation_fc(self.feature_extractor(x))
        if isinstance(self.classifier, LateFusionClassifierH):
            return self.classifier.extract_morph_features(features)
        return features

    def extract_feature_layers(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        layers = self.feature_extractor.forward_layers(x)
        layers["pre_adaptation_gap"] = layers["gap_embed"]
        layers["dan_fc"] = self.adaptation_fc(layers["gap_embed"])
        if isinstance(self.classifier, LateFusionClassifierH):
            layers["pre_fusion_gap"] = layers["dan_fc"]
            layers["gap_embed"] = self.classifier.extract_morph_features(layers["dan_fc"])
        else:
            layers["gap_embed"] = layers["dan_fc"]
        return layers

    def forward(
        self,
        x: torch.Tensor,
        rr_features: torch.Tensor | None = None,
        return_logits: bool = False,
        return_dict: bool = False,
    ):
        if return_dict:
            layers = self.extract_feature_layers(x)
            features = layers["gap_embed"]
            if isinstance(self.classifier, LateFusionClassifierH):
                _, logits, probs = self.classifier(layers["pre_fusion_gap"], rr_features, return_logits=True)
                return {
                    "features": features,
                    "logits": logits,
                    "probabilities": probs,
                    "feature_layers": layers,
                }
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
        raw_features = self.adaptation_fc(self.feature_extractor(x))
        if isinstance(self.classifier, LateFusionClassifierH):
            features, logits, probs = self.classifier(raw_features, rr_features, return_logits=True)
            if return_logits:
                return features, logits, probs
            return features, probs
        features = raw_features
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
