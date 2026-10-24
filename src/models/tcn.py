"""Temporal Convolutional Network for battery SOH estimation.

Architecture follows Bai et al. (2018) dilated causal conv + residual blocks.
Adapted from open-source NASA SOH baselines (Bansilal01, EV-Degradation-Engine).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    """
    TCN encoder + global average pooling + linear head for SOH regression.

    Input:  (batch, seq_len, n_features)
    Output: (batch, 1) SOH in [0, 1]
    """

    def __init__(
        self,
        n_features: int = 3,
        num_channels: tuple[int, ...] = (32, 32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(num_channels):
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        y = self.network(x)
        return self.head(y)


class SimpleTCN(nn.Module):
    """
    Lightweight TCN from Bansilal01 reference — cycle-index only input.
    Input: (batch, 1, seq_len) cycle indices
    Output: (batch, seq_len, 1) normalized SOH curve
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._block(1, 64, 3, 1),
            self._block(64, 64, 3, 2),
            nn.Conv1d(64, 1, 1),
        )

    @staticmethod
    def _block(in_ch, out_ch, kernel_size, dilation):
        return _SimpleTCNBlock(in_ch, out_ch, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(1, 2)


class _SimpleTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv(x))
        res = self.downsample(x) if self.downsample else x
        return out[:, :, : x.size(2)] + res
