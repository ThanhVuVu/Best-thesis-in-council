from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MyConv1dPadSame(nn.Module):
    """Conv1d with TensorFlow-style SAME padding.

    Adapted from CLEF's Net1D implementation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = pad // 2
        pad_right = pad - pad_left
        return self.conv(F.pad(x, (pad_left, pad_right), "constant", 0))


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = max(0, self.kernel_size - 1)
        pad_left = pad // 2
        pad_right = pad - pad_left
        return self.max_pool(F.pad(x, (pad_left, pad_right), "constant", 0))


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        is_first_block: bool = False,
        use_bn: bool = False,
        use_do: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.downsample = bool(downsample)
        self.stride = int(stride) if self.downsample else 1
        self.is_first_block = bool(is_first_block)
        self.use_bn = bool(use_bn)
        self.use_do = bool(use_do)
        middle_channels = int(out_channels * ratio)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels, middle_channels, kernel_size=1, stride=1)

        self.bn2 = nn.BatchNorm1d(middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(middle_channels, middle_channels, kernel_size=kernel_size, stride=self.stride, groups=groups)

        self.bn3 = nn.BatchNorm1d(middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(middle_channels, out_channels, kernel_size=1, stride=1)

        reduction = 2
        self.se_fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.se_fc2 = nn.Linear(out_channels // reduction, out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out)

        se = out.mean(-1)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = torch.sigmoid(self.se_fc2(se))
        out = torch.einsum("abc,ab->abc", out, se)

        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        return out + identity


class BasicStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        stage_index: int,
        num_blocks: int,
        use_bn: bool = False,
        use_do: bool = False,
    ):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for block_index in range(num_blocks):
            downsample = block_index == 0
            blocks.append(
                BasicBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    ratio=ratio,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=groups,
                    downsample=downsample,
                    is_first_block=stage_index == 0 and block_index == 0,
                    use_bn=use_bn,
                    use_do=use_do,
                )
            )
            current_channels = out_channels
        self.block_list = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.block_list:
            out = block(out)
        return out


class Net1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        ratio: float,
        filter_list: list[int],
        m_blocks_list: list[int],
        kernel_size: int,
        stride: int,
        groups_width: int,
        n_classes: int,
        use_bn: bool = False,
        use_do: bool = False,
    ):
        super().__init__()
        self.first_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=kernel_size, stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        stages = []
        current_channels = base_filters
        for stage_index, out_channels in enumerate(filter_list):
            stages.append(
                BasicStage(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    ratio=ratio,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=out_channels // groups_width,
                    stage_index=stage_index,
                    num_blocks=m_blocks_list[stage_index],
                    use_bn=use_bn,
                    use_do=use_do,
                )
            )
            current_channels = out_channels
        self.stage_list = nn.ModuleList(stages)
        self.dense = nn.Linear(current_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_conv(x)
        out = self.first_activation(out)
        for stage in self.stage_list:
            out = stage(out)
        out = out.mean(-1)
        return self.dense(out)


SIZE_CONFIGS: dict[str, dict[str, Any]] = {
    "small": {
        "base_filters": 32,
        "ratio": 0.5,
        "filter_list": [32, 64, 64, 128, 128, 256],
        "m_blocks_list": [1, 1, 2, 2, 2, 2],
        "groups_width": 8,
        "feature_dim": 256,
    },
    "medium": {
        "base_filters": 64,
        "ratio": 1.0,
        "filter_list": [64, 160, 160, 400, 400, 1024, 1024],
        "m_blocks_list": [2, 2, 2, 3, 3, 4, 4],
        "groups_width": 16,
        "feature_dim": 1024,
    },
    "large": {
        "base_filters": 128,
        "ratio": 1.5,
        "filter_list": [128, 256, 256, 512, 512, 1024, 1024, 2048, 2048],
        "m_blocks_list": [2, 3, 3, 4, 4, 5, 5, 6, 6],
        "groups_width": 32,
        "feature_dim": 2048,
    },
}


def create_clef_encoder(model_size: str, checkpoint_path: str | Path, in_channels: int = 1) -> Net1D:
    if model_size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported CLEF model_size={model_size!r}. Expected one of {sorted(SIZE_CONFIGS)}")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"CLEF checkpoint not found: {checkpoint}. Download clef_small.ckpt from Zenodo or pass --clef-checkpoint."
        )
    cfg = SIZE_CONFIGS[model_size]
    model = Net1D(
        in_channels=in_channels,
        base_filters=cfg["base_filters"],
        ratio=cfg["ratio"],
        filter_list=cfg["filter_list"],
        m_blocks_list=cfg["m_blocks_list"],
        kernel_size=16,
        stride=2,
        groups_width=cfg["groups_width"],
        n_classes=1000,
        use_bn=False,
        use_do=False,
    )
    model.dense = nn.Identity()
    checkpoint_obj = _torch_load_checkpoint(checkpoint)
    state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
    clean_state = {}
    for key, value in state_dict.items():
        clean_state[key.replace("backbone.", "", 1)] = value
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(
        "Loaded CLEF encoder:",
        {
            "checkpoint": str(checkpoint),
            "model_size": model_size,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
        flush=True,
    )
    return model


def _torch_load_checkpoint(checkpoint: Path) -> Any:
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint, map_location="cpu")


class CLEFPretrainedClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        model_size: str = "small",
        clef_checkpoint_path: str | Path = "models/clef/clef_small.ckpt",
        in_channels: int = 1,
        freeze_encoder: bool = True,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        encoder_lr: float = 3e-5,
    ):
        super().__init__()
        self.model_size = str(model_size)
        self.freeze_encoder = bool(freeze_encoder)
        self.encoder_lr = float(encoder_lr)
        self.encoder = create_clef_encoder(self.model_size, clef_checkpoint_path, in_channels=in_channels)
        self.embedding_dim = int(SIZE_CONFIGS[self.model_size]["feature_dim"])
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, int(head_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_dim), num_classes),
        )
        self._set_encoder_trainable(not self.freeze_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits

    def optimizer_param_groups(self, lr: float, weight_decay: float) -> list[dict[str, Any]]:
        head_params = [param for param in self.classifier.parameters() if param.requires_grad]
        groups: list[dict[str, Any]] = [{"params": head_params, "lr": float(lr), "weight_decay": float(weight_decay)}]
        encoder_params = [param for param in self.encoder.parameters() if param.requires_grad]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self.encoder_lr, "weight_decay": float(weight_decay)})
        return groups

    def _set_encoder_trainable(self, trainable: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = bool(trainable)
        if not trainable:
            self.encoder.eval()
