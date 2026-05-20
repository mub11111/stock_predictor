"""FT-iTransformer: Time-Frequency Collaborative Transformer (2026).

Key innovations:
  - Frequency branch: FFT spectrum → learnable frequency filters → freq embedding
  - Time branch: inverted Transformer (attention across VARIABLES, not time steps)
  - Cross-modal fusion: cross-attention between time and frequency representations

Designed for non-stationary financial data with multi-scale patterns.
Reference: "FT-iTransformer: Time-Frequency Collaborative Forecasting" (2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FreqEncoder(nn.Module):
    """FFT-based frequency feature encoder with learnable frequency filters."""

    def __init__(self, seq_len: int, feature_dim: int, hidden: int = 64, top_k: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.top_k = top_k

        # FFT returns seq_len//2 + 1 frequency bins
        freq_bins = seq_len // 2 + 1
        self.freq_bins = freq_bins

        # Learnable frequency band filters (like learnable wavelets)
        self.freq_weights = nn.Parameter(torch.randn(freq_bins, feature_dim) * 0.02)

        # Magnitude → embedding
        self.mag_proj = nn.Sequential(
            nn.Linear(freq_bins, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden * 2, hidden),
        )

        # Phase → embedding
        self.phase_proj = nn.Sequential(
            nn.Linear(freq_bins, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden * 2, hidden),
        )

        # Fusion gate
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, feature_dim] → freq_feat: [B, hidden]"""
        B, S, D = x.shape

        # FFT along time dimension for each feature
        x_fft = torch.fft.rfft(x, dim=1)  # [B, freq_bins, D]

        # Apply learnable frequency weights
        x_fft = x_fft * self.freq_weights.unsqueeze(0)  # [B, freq_bins, D]

        # Magnitude and phase
        mag = torch.abs(x_fft)  # [B, freq_bins, D]
        phase = torch.angle(x_fft)  # [B, freq_bins, D]

        # Aggregate across features: [B, freq_bins, D] → [B, freq_bins]
        mag_agg = mag.mean(dim=-1)
        phase_agg = phase.mean(dim=-1)

        # Project to hidden
        mag_emb = self.mag_proj(mag_agg)  # [B, hidden]
        phase_emb = self.phase_proj(phase_agg)  # [B, hidden]

        # Gated fusion
        combined = torch.cat([mag_emb, phase_emb], dim=-1)  # [B, hidden*2]
        gate = self.gate(combined)  # [B, hidden]
        out = gate * mag_emb + (1 - gate) * phase_emb  # [B, hidden]

        return out


class InvertedTransformerEncoder(nn.Module):
    """iTransformer-style encoder: attention across VARIABLES (features), not time.

    Standard Transformer attends across time steps. iTransformer inverts this:
    each variable (feature) becomes a token, and attention captures cross-variable
    relationships. This is more efficient for many features and captures feature
    interactions explicitly.
    """

    def __init__(self, seq_len: int, feature_dim: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # Project each variable's time series to d_model
        self.var_projection = nn.Sequential(
            nn.Linear(seq_len, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Variable embedding (positional encoding for variables)
        self.var_pos = nn.Parameter(torch.randn(1, feature_dim, d_model) * 0.02)

        # Transformer layers for cross-variable attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: pool variables → single embedding
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, feature_dim] → time_feat: [B, d_model]"""
        B, S, D = x.shape

        # Invert: [B, seq_len, feature_dim] → [B, feature_dim, seq_len]
        x_inv = x.permute(0, 2, 1)  # [B, D, S]

        # Project each variable: [B, D, S] → [B, D, d_model]
        var_tokens = self.var_projection(x_inv)  # [B, D, d_model]

        # Add variable position embeddings
        var_tokens = var_tokens + self.var_pos[:, :D, :]

        # Cross-variable attention: [B, D, d_model] → [B, D, d_model]
        var_tokens = self.transformer(var_tokens)

        # Pool variables (mean + max)
        pooled = var_tokens.mean(dim=1) + var_tokens.max(dim=1).values  # [B, d_model]
        out = self.output_proj(pooled)  # [B, d_model]

        return out


class FT_iTransformer(nn.Module):
    """FT-iTransformer: Time-Frequency Collaborative Transformer.

    Combines frequency-domain features (FFT) with inverted time-domain
    attention for financial time series prediction.
    """

    def __init__(self, input_dim: int = 85, seq_len: int = 120,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dropout: float = 0.15, top_k_freq: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        # Frequency encoder
        self.freq_encoder = FreqEncoder(
            seq_len=seq_len, feature_dim=input_dim,
            hidden=d_model, top_k=top_k_freq
        )

        # Time encoder (inverted Transformer)
        self.time_encoder = InvertedTransformerEncoder(
            seq_len=seq_len, feature_dim=input_dim,
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )

        # Cross-modal attention: time attends to frequency
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # Layer norms
        self.ln_time = nn.LayerNorm(d_model)
        self.ln_freq = nn.LayerNorm(d_model)
        self.ln_fused = nn.LayerNorm(d_model)

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Output heads
        self.dropout = nn.Dropout(dropout)
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3), nn.Tanh()  # q10, q50, q90 quantiles
        )
        self.register_buffer('price_scale', torch.tensor(3.0))  # ±3% quantile delta range

        # Kendall uncertainty log-variance params
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract fused time-frequency representation before heads. (B, d_model)"""
        freq_feat = self.freq_encoder(x)
        freq_feat = self.ln_freq(freq_feat)
        time_feat = self.time_encoder(x)
        time_feat = self.ln_time(time_feat)
        time_q = time_feat.unsqueeze(1)
        freq_kv = freq_feat.unsqueeze(1)
        fused, _ = self.cross_attn(time_q, freq_kv, freq_kv)
        fused = fused.squeeze(1)
        fused = self.ln_fused(fused)
        gate = self.fusion_gate(torch.cat([time_feat, fused], dim=-1))
        final = gate * fused + (1 - gate) * time_feat
        return self.dropout(final)

    def forward(self, x: torch.Tensor):
        """x: [B, seq_len, input_dim] → (dir_logits, price_quantiles) where price_quantiles: (B, 3)"""
        final = self.forward_features(x)
        dir_logits = self.dir_head(final)
        price_quantiles = self.price_head(final) * self.price_scale
        return dir_logits, price_quantiles


# ── Compatibility wrapper for existing training code ──

class FT_iTransformerWrapper(nn.Module):
    """Wrapper matching the HybridModel interface for drop-in replacement."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        self.model = FT_iTransformer(
            input_dim=input_dim, seq_len=max_seq_len,
            d_model=d_model, nhead=nhead, num_layers=transformer_layers,
            dropout=dropout
        )
        self.log_sigma_dir = self.model.log_sigma_dir
        self.log_sigma_price = self.model.log_sigma_price

    def forward(self, x):
        return self.model(x)
