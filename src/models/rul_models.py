"""RUL prediction models: TCN / TCN-Attention Seq2Seq / LSTM baselines."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import ChannelAttention, TemporalPatternAttention
from src.models.tcn import TemporalBlock


class TCNEncoder(nn.Module):
    def __init__(
        self,
        n_features: int = 1,
        num_channels: tuple[int, ...] = (32, 32, 64),
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
        self.hidden_dim = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T) -> (B, T, H)
        y = self.network(x.transpose(1, 2))
        return y.transpose(1, 2)


class Seq2SeqDecoder(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRUCell(hidden_dim + 1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch = context.size(0)
        h = context
        prev = torch.zeros(batch, 1, device=context.device)
        outputs = []
        for step in range(self.horizon):
            inp = torch.cat([h, prev], dim=-1)
            h = self.gru(inp, h)
            h = self.dropout(h)
            pred = self.fc(h)
            outputs.append(pred)
            if self.training and target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev = target[:, step : step + 1]
            else:
                prev = pred.detach()
        return torch.cat(outputs, dim=-1)


class TCNSeq2Seq(nn.Module):
    """TCN encoder + GRU decoder for multi-step SOH / RUL forecasting."""

    def __init__(
        self,
        n_features: int = 1,
        horizon: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TCNEncoder(n_features, (32, 32, hidden_dim), dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = Seq2SeqDecoder(hidden_dim, horizon, dropout)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)  # (B, T, H)
        pooled = self.pool(enc.transpose(1, 2)).squeeze(-1)
        return self.context_proj(pooled)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode(x)
        soh_pred = self.decoder(context, target, teacher_forcing_ratio)
        rul_pred = self.rul_head(context).squeeze(-1)
        return soh_pred, rul_pred


class TCNAttentionSeq2Seq(TCNSeq2Seq):
    """TCN encoder + optional Channel Attention / TPA + Seq2Seq decoder."""

    def __init__(
        self,
        n_features: int = 1,
        horizon: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_channel: bool = True,
        use_tpa: bool = True,
    ):
        super().__init__(n_features, horizon, hidden_dim, dropout)
        self.use_channel = use_channel
        self.use_tpa = use_tpa
        self.channel_attn = ChannelAttention(hidden_dim) if use_channel else None
        self.tpa = TemporalPatternAttention(hidden_dim) if use_tpa else None
        self.pool = nn.AdaptiveAvgPool1d(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        if self.channel_attn is not None:
            enc = self.channel_attn(enc)
        if self.tpa is not None:
            context, _ = self.tpa(enc)
        else:
            context = self.pool(enc.transpose(1, 2)).squeeze(-1)
        return self.context_proj(context)


class TCNBiGRUSeq2Seq(nn.Module):
    """TCN encoder + BiGRU context + GRU decoder (TCN-BiGRU hybrid baseline)."""

    def __init__(
        self,
        n_features: int = 1,
        horizon: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TCNEncoder(n_features, (32, 32, hidden_dim), dropout=dropout)
        self.bigru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = Seq2SeqDecoder(hidden_dim, horizon, dropout)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        out, _ = self.bigru(enc)
        context = out[:, -1, :]
        return self.context_proj(context)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode(x)
        soh_pred = self.decoder(context, target, teacher_forcing_ratio)
        rul_pred = self.rul_head(context).squeeze(-1)
        return soh_pred, rul_pred


class LSTMSeq2Seq(nn.Module):
    """LSTM baseline for multi-step SOH and RUL prediction."""

    def __init__(
        self,
        n_features: int = 1,
        horizon: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(
            n_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.decoder = Seq2SeqDecoder(hidden_dim, horizon, dropout)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.encoder(x)
        context = h[-1]
        soh_pred = self.decoder(context, target, teacher_forcing_ratio)
        rul_pred = self.rul_head(context).squeeze(-1)
        return soh_pred, rul_pred


def build_rul_model(
    name: str,
    horizon: int = 5,
    n_features: int = 1,
    d_model: int = 64,
    num_layers: int = 2,
    patch_size: int = 5,
    dropout: float = 0.1,
    window_size: int = 50,
) -> nn.Module:
    name = name.lower()
    if name == "tcn":
        return TCNSeq2Seq(n_features=n_features, horizon=horizon)
    if name in {"tcn-attn", "tcn_attention", "tcn-attention"}:
        return TCNAttentionSeq2Seq(
            n_features=n_features, horizon=horizon, use_channel=True, use_tpa=True
        )
    if name in {"tcn-attn-ch", "tcn_attn_ch", "tcn-channel"}:
        return TCNAttentionSeq2Seq(
            n_features=n_features, horizon=horizon, use_channel=True, use_tpa=False
        )
    if name in {"tcn-attn-tpa", "tcn_attn_tpa", "tcn-tpa"}:
        return TCNAttentionSeq2Seq(
            n_features=n_features, horizon=horizon, use_channel=False, use_tpa=True
        )
    if name in {"tcn-bigru", "tcn_bigru"}:
        return TCNBiGRUSeq2Seq(n_features=n_features, horizon=horizon)
    if name == "lstm":
        return LSTMSeq2Seq(n_features=n_features, horizon=horizon)
    if name in {"patch-tf", "patch_tf", "patchtransformer", "lite-patch-tf"}:
        from src.models.patch_transformer import LitePatchTransformer

        return LitePatchTransformer(
            n_features=n_features,
            horizon=horizon,
            d_model=d_model,
            num_layers=num_layers,
            patch_size=patch_size,
            dropout=dropout,
        )
    if name in {"vanilla-tf", "vanilla_tf", "transformer"}:
        from src.models.extra_baselines import VanillaTransformer

        return VanillaTransformer(
            n_features=n_features,
            horizon=horizon,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )
    if name == "mlp":
        from src.models.extra_baselines import MLPDualHead

        return MLPDualHead(
            n_features=n_features,
            window_size=window_size,
            horizon=horizon,
            d_model=d_model,
            dropout=dropout,
        )
    if name in {"gru", "gru-direct", "gru_direct"}:
        from src.models.extra_baselines import GRUDirect

        return GRUDirect(
            n_features=n_features, horizon=horizon, hidden_dim=d_model, num_layers=num_layers, dropout=dropout
        )
    if name in {"lstm-direct", "lstm_direct"}:
        from src.models.extra_baselines import LSTMDirect

        return LSTMDirect(
            n_features=n_features, horizon=horizon, hidden_dim=d_model, num_layers=num_layers, dropout=dropout
        )
    if name in {"tcn-direct", "tcn_direct"}:
        from src.models.extra_baselines import TCNDirect

        return TCNDirect(n_features=n_features, horizon=horizon, hidden_dim=d_model, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def iterative_multistep_predict(
    model: nn.Module,
    history,
    horizon: int,
    steps: int,
    device: torch.device,
):
    """Rolling multi-step prediction with iterative window correction."""
    import numpy as np

    window = np.asarray(history, dtype=np.float32).copy()
    preds = []
    model.eval()
    while len(preds) < steps:
        x = torch.tensor(window.reshape(1, *window.shape), dtype=torch.float32, device=device)
        soh_pred, _ = model(x, target=None, teacher_forcing_ratio=0.0)
        chunk = soh_pred.cpu().numpy().reshape(-1)
        for val in chunk:
            preds.append(float(val))
            window = np.roll(window, -1, axis=0)
            window[-1, 0] = val
            if len(preds) >= steps:
                break
    return np.array(preds[:steps], dtype=np.float32)
