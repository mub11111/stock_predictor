"""Frequency-Decomposed Model Orchestrator.

Three-branch architecture with dynamic fusion:
  - Low-freq:  FT-iTransformer → macro trend anchoring
  - Mid-freq:  DLinear (trend/seasonal decomp) → medium cycles
  - High-freq: MicroStructure (CMF/MFI/OBV) → microstructure volatility

A learned gating network dynamically weights each branch's contribution
based on the input regime, producing fused direction + price quantile predictions.
"""

import torch
import torch.nn as nn

from model.ft_transformer import FT_iTransformer
from model.dlinear import DLinearModel
from model.micro_structure import MicroStructureModule


class FreqOrchestrator(nn.Module):
    """Three-frequency model with per-branch heads and dynamic fusion gate.

    Args:
        feature_group_indices: dict {"low": [...], "mid": [...], "high": [...]}
            mapping each frequency group to its global feature column indices.
            When None (backward compat), all input_dim features go to all branches.
    """

    def __init__(self, input_dim: int = 85, seq_len: int = 120,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dropout: float = 0.15, decomp_kernel: int = 25,
                 feature_group_indices: dict | None = None):
        super().__init__()
        self.d_model = d_model

        # Determine per-branch feature dimensions from group indices
        if feature_group_indices is not None:
            low_dim = len(feature_group_indices["low"])
            mid_dim = len(feature_group_indices["mid"])
            high_dim = len(feature_group_indices["high"])
            self.register_buffer('low_indices',
                torch.tensor(feature_group_indices["low"], dtype=torch.long))
            self.register_buffer('mid_indices',
                torch.tensor(feature_group_indices["mid"], dtype=torch.long))
            self.register_buffer('high_indices',
                torch.tensor(feature_group_indices["high"], dtype=torch.long))
        else:
            low_dim = mid_dim = input_dim
            high_dim = input_dim
            self.low_indices = None
            self.mid_indices = None
            self.high_indices = None

        # Low-freq: FT-iTransformer for macro trends
        self.low_freq = FT_iTransformer(
            input_dim=low_dim, seq_len=seq_len,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dropout=dropout
        )

        # Mid-freq: DLinear for trend/seasonal cycles
        self.mid_freq = DLinearModel(
            seq_len=seq_len, pred_len=1, enc_in=mid_dim,
            decomp_kernel=decomp_kernel, d_model=d_model, dropout=dropout
        )

        # High-freq: MicroStructure for order-flow / volatility
        micro_d = d_model // 2
        self.high_freq = MicroStructureModule(
            seq_len=seq_len, d_model=micro_d, dropout=dropout,
            n_micro=high_dim
        )
        self.micro_proj = nn.Linear(micro_d, d_model)  # project to common dim

        # Per-branch lightweight auxiliary heads (deep supervision)
        self.low_dir_head = nn.Linear(d_model, 2)
        self.low_price_head = nn.Sequential(nn.Linear(d_model, 3), nn.Tanh())
        self.mid_dir_head = nn.Linear(d_model, 2)
        self.mid_price_head = nn.Sequential(nn.Linear(d_model, 3), nn.Tanh())
        self.high_dir_head = nn.Linear(d_model, 2)
        self.high_price_head = nn.Sequential(nn.Linear(d_model, 3), nn.Tanh())

        # Dynamic fusion gate: input-regime → branch weights
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 3, 64), nn.GELU(),
            nn.Linear(64, 3), nn.Softmax(dim=-1)
        )

        # Final fused heads
        fused_dim = d_model * 3
        self.dir_head = nn.Sequential(
            nn.Linear(fused_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        self.price_head = nn.Sequential(
            nn.Linear(fused_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 3), nn.Tanh()
        )
        self.register_buffer('price_scale', torch.tensor(3.0))

        # Kendall uncertainty log-variances
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in [self.low_dir_head, self.mid_dir_head, self.high_dir_head,
                  self.dir_head]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """x: (B, S, C_full) → (dir_logits, price_quantiles) where price_quantiles: (B, 3)."""
        # Split features by frequency group (or pass full if no routing configured)
        if self.low_indices is not None:
            x_low = x[:, :, self.low_indices]
            x_mid = x[:, :, self.mid_indices]
            x_high = x[:, :, self.high_indices]
        else:
            x_low = x_mid = x_high = x

        # Low-freq: time-frequency collaborative features
        low_feat = self.low_freq.forward_features(x_low)  # (B, d_model)

        # Mid-freq: trend + seasonal decomposition features
        mid_feat = self.mid_freq(x_mid)  # (B, d_model)

        # High-freq: microstructure features + volatility
        high_feat_raw, _volatility = self.high_freq(x_high)  # (B, micro_d)
        high_feat = self.micro_proj(high_feat_raw)             # (B, d_model)

        # Per-branch auxiliary predictions (for deep supervision / analysis)
        self.low_dir = self.low_dir_head(low_feat)
        self.low_price = self.low_price_head(low_feat) * self.price_scale
        self.mid_dir = self.mid_dir_head(mid_feat)
        self.mid_price = self.mid_price_head(mid_feat) * self.price_scale
        self.high_dir = self.high_dir_head(high_feat)
        self.high_price = self.high_price_head(high_feat) * self.price_scale

        # Dynamic fusion weights
        concat = torch.cat([low_feat, mid_feat, high_feat], dim=-1)  # (B, 3*d_model)
        w = self.fusion_gate(concat)  # (B, 3)

        # Weighted concatenation
        fused = torch.cat([
            w[:, 0:1] * low_feat,
            w[:, 1:2] * mid_feat,
            w[:, 2:3] * high_feat,
        ], dim=-1)  # (B, 3*d_model)

        dir_logits = self.dir_head(fused)
        price_quantiles = self.price_head(fused) * self.price_scale

        return dir_logits, price_quantiles


class FreqOrchestratorWrapper(nn.Module):
    """Wrapper matching HybridModel interface for drop-in replacement."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5, feature_group_indices: dict | None = None):
        super().__init__()
        self.model = FreqOrchestrator(
            input_dim=input_dim, seq_len=max_seq_len,
            d_model=d_model, nhead=nhead, num_layers=transformer_layers,
            dropout=dropout, feature_group_indices=feature_group_indices
        )
        self.log_sigma_dir = self.model.log_sigma_dir
        self.log_sigma_price = self.model.log_sigma_price

    def forward(self, x):
        return self.model(x)
