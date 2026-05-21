from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int = 3, channels: list[int] | None = None):
        super().__init__()
        channels = channels or [32, 64, 128, 128]
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_channels = channels[0]
        for idx, out_channels in enumerate(channels):
            stride = 2 if idx > 0 else 1
            blocks.append(ResidualBlock1D(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        h = self.stem(x)
        h = self.blocks(h)
        embedding = self.pool(h).squeeze(-1)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits
