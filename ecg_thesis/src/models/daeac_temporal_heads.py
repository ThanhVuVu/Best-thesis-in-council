from __future__ import annotations

import torch
from torch import nn


class TemporalIdentityPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x).squeeze(-1)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.0, kernel_size: int = 3):
        super().__init__()
        padding = int(dilation) * (int(kernel_size) - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=int(kernel_size),
                padding=padding,
                dilation=int(dilation),
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class TCNAttentionHead(nn.Module):
    def __init__(
        self,
        channels: int,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.tcn = nn.Sequential(
            *[
                ResidualTCNBlock(
                    channels=int(channels),
                    dilation=int(dilation),
                    dropout=float(dropout),
                    kernel_size=int(kernel_size),
                )
                for dilation in dilations
            ]
        )
        self.attention = nn.Conv1d(int(channels), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)
        weights = torch.softmax(self.attention(h), dim=-1)
        return torch.sum(h * weights, dim=-1)


class RMSNormCompat(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _rms_norm(dim: int) -> nn.Module:
    rms_norm = getattr(nn, "RMSNorm", None)
    if rms_norm is not None:
        return rms_norm(int(dim))
    return RMSNormCompat(int(dim))


def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    values = torch.stack([*blocks, partial_block], dim=0)
    keys = norm(values)
    query = proj.weight.squeeze(0)
    logits = torch.einsum("d,nbtd->nbt", query, keys)
    weights = torch.softmax(logits, dim=0)
    return torch.einsum("nbt,nbtd->btd", weights, values)


class AttnResTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float,
        layer_number: int,
        layers_per_block: int,
    ):
        super().__init__()
        self.layer_number = int(layer_number)
        self.layers_per_block = int(layers_per_block)
        self.attn_res_proj = nn.Linear(int(d_model), 1, bias=False)
        self.attn_res_norm = _rms_norm(int(d_model))
        self.mlp_res_proj = nn.Linear(int(d_model), 1, bias=False)
        self.mlp_res_norm = _rms_norm(int(d_model))
        self.attn_norm = nn.LayerNorm(int(d_model))
        self.mlp_norm = nn.LayerNorm(int(d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(nhead),
            dropout=float(dropout),
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(int(d_model), int(ffn_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(ffn_dim), int(d_model)),
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.attn_res_proj.weight)
        nn.init.zeros_(self.mlp_res_proj.weight)
        self.attn_res_proj._daeac_keep_zero_init = True
        self.mlp_res_proj._daeac_keep_zero_init = True

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        h_attn = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
        if self.layer_number > 0 and self.layer_number % self.layers_per_block == 0:
            blocks = [*blocks, partial_block]
            partial_block = None

        h_norm = self.attn_norm(h_attn)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        attn_out = self.dropout(attn_out)
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        h_mlp = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        mlp_out = self.dropout(self.mlp(self.mlp_norm(h_mlp)))
        partial_block = partial_block + mlp_out
        return blocks, partial_block


class AttnResTransformerHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        nhead: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        block_layers: int = 2,
    ):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("AttnResTransformerHead requires num_layers >= 1.")
        if int(block_layers) < 1:
            raise ValueError("AttnResTransformerHead requires block_layers >= 1.")
        self.layers = nn.ModuleList(
            [
                AttnResTransformerLayer(
                    d_model=int(d_model),
                    nhead=int(nhead),
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                    layer_number=idx,
                    layers_per_block=int(block_layers),
                )
                for idx in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        blocks = [sequence]
        partial_block = sequence
        for layer in self.layers:
            blocks, partial_block = layer(blocks, partial_block)
        return partial_block.mean(dim=1)
