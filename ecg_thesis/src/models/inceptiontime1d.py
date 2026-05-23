from __future__ import annotations

import torch
from torch import nn


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        kernel_sizes: tuple[int, int, int] = (9, 19, 39),
        use_bottleneck: bool = True,
    ):
        super().__init__()
        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            branch_in = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            branch_in = in_channels

        self.branches = nn.ModuleList([
            nn.Conv1d(branch_in, out_channels, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU(inplace=True)

    @property
    def out_channels(self) -> int:
        return self.bn.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bottleneck(x)
        outputs = [branch(h) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        return self.relu(self.bn(torch.cat(outputs, dim=1)))


class InceptionTime1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        num_blocks: int = 6,
        branch_channels: int = 32,
        bottleneck_channels: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        blocks = []
        residuals = []
        channels = in_channels
        residual_channels = in_channels
        for block_idx in range(num_blocks):
            block = InceptionBlock1D(
                in_channels=channels,
                out_channels=branch_channels,
                bottleneck_channels=bottleneck_channels,
                use_bottleneck=True,
            )
            blocks.append(block)
            next_channels = block.out_channels
            if (block_idx + 1) % 3 == 0:
                residuals.append(nn.Sequential(
                    nn.Conv1d(residual_channels, next_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(next_channels),
                ))
                residual_channels = next_channels
            channels = next_channels

        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = channels
        self.classifier = nn.Linear(channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        residual_idx = 0
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if (idx + 1) % 3 == 0:
                x = self.relu(x + self.residuals[residual_idx](residual_input))
                residual_input = x
                residual_idx += 1
        embedding = self.pool(x).squeeze(-1)
        return self.dropout(embedding)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits
