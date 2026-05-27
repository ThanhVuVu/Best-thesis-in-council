from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


class LeadBridge1To12(nn.Module):
    def __init__(self, input_leads: int = 1, hidden_channels: int = 64, output_leads: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_leads, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, output_leads, kernel_size=1),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ECGFMEncoderWrapper(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None,
        fairseq_signals_path: str | None = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.fairseq_signals_path = fairseq_signals_path
        if fairseq_signals_path not in (None, "", "null", "None"):
            path = str(Path(fairseq_signals_path))
            if path not in sys.path:
                sys.path.insert(0, path)
        self.ecgfm = self._load_ecgfm(checkpoint_path)
        self.freeze = bool(freeze)
        if self.freeze:
            for param in self.ecgfm.parameters():
                param.requires_grad = False
            self.ecgfm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.ecgfm.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ecgfm(source=x)
        if isinstance(out, dict):
            features = None
            for key in ("encoder_out", "x", "features"):
                if key in out and torch.is_tensor(out[key]):
                    features = out[key]
                    break
            if features is None:
                keys = ", ".join(sorted(out.keys()))
                raise KeyError(
                    "ECG-FM output does not contain a supported feature tensor "
                    f"(encoder_out, x, or features). Available keys: {keys}"
                )
        else:
            features = out
        if not torch.is_tensor(features):
            raise TypeError(f"Expected ECG-FM encoder output tensor, got {type(features)!r}")
        if features.dim() != 3:
            raise ValueError(f"Expected ECG-FM encoder output [B, T, D], got shape {tuple(features.shape)}")
        if features.shape[0] != x.shape[0] and features.shape[1] == x.shape[0]:
            features = features.transpose(0, 1)
        return features

    @staticmethod
    def _load_ecgfm(checkpoint_path: str | None) -> nn.Module:
        if checkpoint_path in (None, "", "null", "None"):
            raise FileNotFoundError(
                "ECG-FM checkpoint path is not configured. Set model.ecgfm_checkpoint_path "
                "in configs/phase4a_ecgfm_leadbridge.yaml or pass it in model_kwargs."
            )
        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"ECG-FM checkpoint not found: {checkpoint}. On Kaggle, attach the ECG-FM "
                "weights as an input dataset and update model.ecgfm_checkpoint_path."
            )
        try:
            from fairseq_signals.models import build_model_from_checkpoint
        except ImportError as exc:
            raise ImportError(
                "Could not import fairseq_signals. Attach/install fairseq-signals and set "
                "model.fairseq_signals_path to its repository path."
            ) from exc
        return build_model_from_checkpoint(checkpoint_path=str(checkpoint))


class ECGFMLeadBridgeClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        input_leads: int = 1,
        bridge_out_leads: int = 12,
        bridge_hidden_channels: int = 64,
        hidden_dim: int = 768,
        head_hidden_dim: int = 256,
        dropout: float = 0.0,
        ecgfm_checkpoint_path: str | None = None,
        fairseq_signals_path: str | None = None,
        freeze_ecgfm: bool = True,
    ):
        super().__init__()
        self.lead_bridge = LeadBridge1To12(
            input_leads=input_leads,
            hidden_channels=bridge_hidden_channels,
            output_leads=bridge_out_leads,
        )
        self.encoder = ECGFMEncoderWrapper(
            checkpoint_path=ecgfm_checkpoint_path,
            fairseq_signals_path=fairseq_signals_path,
            freeze=freeze_ecgfm,
        )
        head_layers: list[nn.Module] = [
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(head_hidden_dim, num_classes))
        self.classifier = nn.Sequential(*head_layers)
        self.embedding_dim = hidden_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_12 = self.lead_bridge(x)
        features = self.encoder(x_12)
        return self._masked_mean_pool(features)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits

    @staticmethod
    def _masked_mean_pool(features: torch.Tensor) -> torch.Tensor:
        nonzero = features != 0
        denom = nonzero.sum(dim=1).clamp_min(1)
        return features.sum(dim=1) / denom
