"""Lite Patch Transformer for multi-feature battery prognostics."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split temporal window into non-overlapping patches and project."""

    def __init__(self, n_features: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_features * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        b, t, f = x.shape
        p = self.patch_size
        if t % p != 0:
            pad = p - (t % p)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            t = t + pad
        n = t // p
        x = x.reshape(b, n, p * f)
        return self.proj(x)


class LitePatchTransformer(nn.Module):
    """
    Compact patch Transformer encoder with dual heads:
    multi-step SOH sequence + scalar RUL.
    """

    def __init__(
        self,
        n_features: int = 6,
        horizon: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        patch_size: int = 5,
        dropout: float = 0.1,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.patch_size = patch_size
        self.dropout_p = dropout

        self.patch_embed = PatchEmbedding(n_features, d_model, patch_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        n = tokens.size(1)
        tokens = tokens + self.pos_encoding[:, :n, :]
        tokens = self.dropout(tokens)
        enc = self.encoder(tokens)
        pooled = self.pool(enc.transpose(1, 2)).squeeze(-1)
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # target / teacher_forcing kept for API compatibility with Seq2Seq baselines
        _ = target, teacher_forcing_ratio
        context = self.encode(x)
        soh_pred = self.soh_head(context)
        rul_pred = self.rul_head(context).squeeze(-1)
        return soh_pred, rul_pred
