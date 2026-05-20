"""DLinear: 序列分解 + 线性层 — 中频趋势/季节分离 (AAAI 2023)."""
import torch
import torch.nn as nn


class SeriesDecomp(nn.Module):
    """Moving-average series decomposition into trend + seasonal (residual)."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1,
                                       padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, S, C) → trend: (B, S, C), seasonal: (B, S, C)"""
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return trend, seasonal


class DLinearModel(nn.Module):
    """Decomposition-linear model for mid-frequency pattern extraction.

    Decomposes input into trend (MA-smoothed) and seasonal (residual) components,
    then applies independent linear projections and recombines.

    This captures medium-term cycles (minutes to hours) that sit between
    the low-frequency macro trends and high-frequency microstructure.
    """

    def __init__(self, seq_len: int = 120, pred_len: int = 1,
                 enc_in: int = 88, decomp_kernel: int = 25,
                 d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomp = SeriesDecomp(kernel_size=decomp_kernel)

        # Feature projection to d_model
        self.enc_trend = nn.Linear(enc_in, d_model)
        self.enc_seasonal = nn.Linear(enc_in, d_model)

        # Linear mapping from seq_len to pred_len (per-channel)
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Output projections
        self.proj_trend = nn.Linear(d_model, d_model)
        self.proj_seasonal = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, C) → out: (B, d_model) pooled representation"""
        # Decompose
        trend, seasonal = self.decomp(x)  # (B,S,C), (B,S,C)

        # Project features
        t = self.act(self.enc_trend(trend))         # (B,S,d_model)
        s = self.act(self.enc_seasonal(seasonal))   # (B,S,d_model)

        # Linear mapping across time
        t = self.trend_linear(t.transpose(1, 2)).transpose(1, 2)       # (B,pred_len,d_model)
        s = self.seasonal_linear(s.transpose(1, 2)).transpose(1, 2)    # (B,pred_len,d_model)

        # Project and combine
        t = self.dropout(self.proj_trend(t))
        s = self.dropout(self.proj_seasonal(s))
        out = self.norm(t + s)

        # Pool to single vector
        out = out.mean(dim=1)  # (B, d_model)
        return out
