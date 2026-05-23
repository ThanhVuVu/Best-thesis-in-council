from __future__ import annotations

import math

import torch
from torch import nn


class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        attention = self.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        return x * attention.unsqueeze(-1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)


class ConvAttentionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        attention_reduction: int = 8,
        pool: str | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = ChannelAttention1D(out_channels, reduction=attention_reduction)
        if pool == "max":
            self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        elif pool == "avg":
            self.pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.attention(x)
        return self.pool(x)


class CATNet1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        d_model: int = 128,
        num_heads: int = 4,
        dff: int = 128,
        num_transformer_layers: int = 1,
        attention_reduction: int = 8,
        dropout: float = 0.2,
        max_len: int = 512,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvAttentionBlock1D(in_channels, 16, kernel_size=21, attention_reduction=attention_reduction, pool="max"),
            ConvAttentionBlock1D(16, 32, kernel_size=23, attention_reduction=attention_reduction, pool="max"),
            ConvAttentionBlock1D(32, 64, kernel_size=25, attention_reduction=attention_reduction, pool="max"),
            ConvAttentionBlock1D(64, d_model, kernel_size=27, attention_reduction=attention_reduction, pool=None),
        )
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.embedding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.embedding_dim = d_model
        self.classifier = nn.Linear(d_model, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.transpose(1, 2)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.embedding(x)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits
