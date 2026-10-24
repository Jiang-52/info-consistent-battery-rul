"""TCN-Autoencoder for battery cycle anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.tcn import TemporalBlock


class TCNEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_channels: tuple[int, ...] = (32, 64, 64),
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
        self.net = nn.Sequential(*layers)
        self.out_dim = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2))


class TCNDecoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.expand = nn.Linear(hidden_dim, hidden_dim * seq_len)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, n_features, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, hidden)
        h = self.expand(z).view(z.size(0), -1, self.seq_len)
        return self.net(h).transpose(1, 2)


class TCNAutoencoder(nn.Module):
    """Encoder compresses sequence to latent vector; decoder reconstructs input."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TCNEncoder(n_features, (32, 64, 64), dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(64, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64)
        self.decoder = TCNDecoder(n_features, 64, seq_len, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        z = self.to_latent(self.pool(enc).squeeze(-1))
        recon = self.decoder(self.from_latent(z))
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2))
