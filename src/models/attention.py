"""Attention modules for TCN-Attention Seq2Seq."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalPatternAttention(nn.Module):
    """
  Temporal Pattern Attention (TPA).

  Learns attention weights over encoder time steps and produces a context vector.
  """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, enc_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # enc_out: (B, T, H)
        scores = self.v(torch.tanh(self.proj(enc_out))).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)
        context = torch.sum(enc_out * weights.unsqueeze(-1), dim=1)  # (B, H)
        return context, weights


class ChannelAttention(nn.Module):
    """Squeeze-excitation style channel attention for 1D features."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        w = self.fc(x.mean(dim=1)).unsqueeze(1)
        return x * w
