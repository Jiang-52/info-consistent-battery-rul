"""Additional RUL baselines: MLP, GRU, Vanilla Transformer, direct dual-heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.patch_transformer import LitePatchTransformer
from src.models.tcn import TemporalBlock


class DualHeads(nn.Module):
    """Shared SOH multi-step + scalar RUL heads (matched with Patch-TF)."""

    def __init__(self, d_model: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.soh_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.soh_head(context), self.rul_head(context).squeeze(-1)


class MLPDualHead(nn.Module):
    """Flatten window → MLP → dual heads (architecture lower bound)."""

    def __init__(
        self,
        n_features: int = 6,
        window_size: int = 50,
        horizon: int = 5,
        d_model: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        in_dim = n_features * window_size
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = DualHeads(d_model, horizon, dropout)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = target, teacher_forcing_ratio
        flat = x.reshape(x.size(0), -1)
        return self.heads(self.backbone(flat))


class VanillaTransformer(nn.Module):
    """Per-cycle tokens (no patching) → TransformerEncoder → dual heads."""

    def __init__(
        self,
        n_features: int = 6,
        horizon: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.token_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = DualHeads(d_model, horizon, dropout)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = target, teacher_forcing_ratio
        tokens = self.token_proj(x)
        n = tokens.size(1)
        tokens = self.dropout(tokens + self.pos_encoding[:, :n, :])
        enc = self.encoder(tokens)
        context = self.pool(enc.transpose(1, 2)).squeeze(-1)
        return self.heads(context)


class GRUDirect(nn.Module):
    """GRU encoder + matched dual heads (no Seq2Seq decoder)."""

    def __init__(
        self,
        n_features: int = 6,
        horizon: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.GRU(
            n_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.heads = DualHeads(hidden_dim, horizon, dropout)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = target, teacher_forcing_ratio
        _, h = self.encoder(x)
        return self.heads(h[-1])


class LSTMDirect(nn.Module):
    """LSTM encoder + matched dual heads (no Seq2Seq decoder)."""

    def __init__(
        self,
        n_features: int = 6,
        horizon: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.heads = DualHeads(hidden_dim, horizon, dropout)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = target, teacher_forcing_ratio
        _, (h, _) = self.encoder(x)
        return self.heads(h[-1])


class TCNDirect(nn.Module):
    """TCN encoder + matched dual heads (no Seq2Seq decoder)."""

    def __init__(
        self,
        n_features: int = 6,
        horizon: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = (32, 32, hidden_dim)
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size=3, dilation=2**i, dropout=dropout)
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = DualHeads(hidden_dim, horizon, dropout)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = target, teacher_forcing_ratio
        y = self.network(x.transpose(1, 2))
        context = self.pool(y).squeeze(-1)
        return self.heads(context)


# Keep LitePatchTransformer import available for factory parity
__all__ = [
    "MLPDualHead",
    "VanillaTransformer",
    "GRUDirect",
    "LSTMDirect",
    "TCNDirect",
    "LitePatchTransformer",
]
