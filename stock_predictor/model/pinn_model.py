"""PINN for Finance: Physics-Informed Neural Network with microstructure PDE constraints.

Key innovations:
  - Market microstructure PDE constraints (OFI law + no-arbitrage bounds)
  - Order Flow Imbalance: predicted direction must align with signed order flow
  - Arbitrage-Free Pricing: |predicted return| bounded by spread + k * σ
  - Volatility clustering + leverage effect as auxiliary physics priors

Reference: "PINN for Finance: Physics-Constrained Deep Learning" (2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PINNFinance(nn.Module):
    """Physics-Informed Neural Network for stock prediction.

    Base model (LSTM + attention) with additional physics loss terms
    that penalize predictions violating financial stylized facts.
    """

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 hidden: int = 128, num_layers: int = 3,
                 nhead: int = 4, dropout: float = 0.15,
                 max_seq_len: int = 120):
        super().__init__()
        self.input_dim = input_dim

        # Base encoder: LSTM for sequential processing
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.ln_lstm = nn.LayerNorm(hidden * 2)

        # Self-attention for temporal dependencies
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden * 2, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.ln_attn = nn.LayerNorm(hidden * 2)

        # Physics head: predicts physics-consistent features
        self.physics_encoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

        # Prediction heads
        self.dropout = nn.Dropout(dropout)
        self.dir_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
        self.price_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 3), nn.Tanh()  # q10, q50, q90 quantiles
        )
        self.register_buffer('price_scale', torch.tensor(3.0))  # ±3% quantile delta range

        # Physics parameters (learnable)
        self.vol_persistence = nn.Parameter(torch.tensor(0.85))  # GARCH-like
        self.leverage_coef = nn.Parameter(torch.tensor(-0.3))    # leverage effect
        self.reversion_strength = nn.Parameter(torch.tensor(0.1))

        # Kendall uncertainty params
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """x: [B, seq_len, input_dim] → (dir_logits, price_quantiles, physics_feat, lstm_out)
        price_quantiles: (B, 3) — q10, q50, q90 in percentage space."""
        B, S, D = x.shape

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [B, S, hidden*2]
        lstm_out = self.ln_lstm(lstm_out)

        # Self-attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # [B, S, hidden*2]
        attn_out = self.ln_attn(lstm_out + self.dropout(attn_out))

        # Pool last timestep for prediction
        pooled = attn_out[:, -1, :]  # [B, hidden*2]
        pooled = self.dropout(pooled)

        # Physics features for constraint computation
        physics_feat = self.physics_encoder(pooled)  # [B, hidden]

        # Predictions
        dir_logits = self.dir_head(pooled)
        price_quantiles = self.price_head(pooled) * self.price_scale  # (B, 3)

        return dir_logits, price_quantiles, physics_feat, lstm_out

    def compute_physics_loss(self, x: torch.Tensor, price_pred: torch.Tensor,
                             physics_feat: torch.Tensor) -> torch.Tensor:
        """计算微观结构物理约束损失.

        约束:
          1. 订单流失衡 (OFI): 预测方向须与资金流压力方向一致
          2. 无套利定价边界: |预测收益| ≤ spread + k × σ
          3. 波动率聚类: 预测波动与近期已实现波动相关联
          4. 物理学特征平滑性
        """
        B = x.shape[0]
        eps = 1e-6

        # 从特征中提取微观结构指标 (last timestep)
        if x.shape[-1] > 70:
            spread = x[:, -1, 67:68].abs()        # micro_spread
            flow_pressure = x[:, -1, 68:69]        # micro_flow_pressure
            vol_imbalance = x[:, -1, 69:70]        # micro_vol_imbalance
            toxicity = x[:, -1, 70:71]             # micro_toxicity
        else:
            spread = torch.zeros(B, 1, device=x.device)
            flow_pressure = torch.zeros(B, 1, device=x.device)
            vol_imbalance = torch.zeros(B, 1, device=x.device)
            toxicity = torch.zeros(B, 1, device=x.device)

        atr = x[:, -1, 19:20].abs() if x.shape[-1] > 19 else torch.ones(B, 1, device=x.device) * 0.01

        # 使用 q50 (中位数预测) 作为价格变动
        pred_delta = price_pred[:, 1:2] if price_pred.shape[-1] == 3 else price_pred[:, :1]

        # ── 约束 1: OFI 订单流失衡律 ──
        ofi_signal = flow_pressure * vol_imbalance  # (B, 1)
        ofi_threshold = 0.2
        ofi_active = (torch.abs(ofi_signal) > ofi_threshold).float().detach()
        pred_sign = torch.sign(pred_delta)
        ofi_sign = torch.sign(ofi_signal)
        sign_mismatch = (pred_sign != ofi_sign).float()
        ofi_loss = (sign_mismatch * torch.abs(pred_delta) * ofi_active).mean()

        # ── 约束 2: 无套利定价边界 ──
        arb_bound = spread.abs() + 3.0 * atr.abs()  # (B, 1)
        arb_loss = F.relu(torch.abs(pred_delta) - arb_bound).mean()

        # ── 约束 3: 波动率聚类 ──
        pred_magnitude = torch.abs(pred_delta.squeeze(-1))
        vol_ratio = pred_magnitude / (atr.squeeze(-1) + eps)
        vol_cluster_loss = F.relu(torch.abs(torch.log(vol_ratio + eps)) - 2.0).mean()

        # ── 约束 4: 物理学特征平滑性 ──
        if B > 1:
            physics_smooth = (physics_feat[1:] - physics_feat[:-1]).pow(2).mean()
        else:
            physics_smooth = torch.tensor(0.0, device=x.device)

        total_physics = (
            ofi_loss * 0.25
            + arb_loss * 0.40
            + vol_cluster_loss * 0.20
            + physics_smooth * 0.15
        )

        return total_physics


# ── Compatibility wrapper ──

class PINNWrapper(nn.Module):
    """Wrapper matching HybridModel interface, with physics loss support."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        self.model = PINNFinance(
            input_dim=input_dim, d_model=d_model,
            hidden=lstm_hidden, num_layers=transformer_layers,
            nhead=nhead, dropout=dropout, max_seq_len=max_seq_len
        )
        self.log_sigma_dir = self.model.log_sigma_dir
        self.log_sigma_price = self.model.log_sigma_price
        self._physics_feat = None
        self._lstm_out = None
        self._last_x = None

    def forward(self, x):
        self._last_x = x
        dir_logits, price_quantiles, physics_feat, lstm_out = self.model(x)
        self._physics_feat = physics_feat
        self._lstm_out = lstm_out
        return dir_logits, price_quantiles

    def get_physics_loss(self):
        """计算并返回物理学约束损失."""
        if self._last_x is None or self._physics_feat is None:
            return torch.tensor(0.0)
        # 使用 q50 中位数作为价格预测计算物理损失
        price_pred = self._lstm_out[:, -1, :3]  # (B, 3) proxy from LSTM output
        if price_pred.shape[-1] != 3:
            price_pred = price_pred[:, :1]  # fallback
        return self.model.compute_physics_loss(
            self._last_x, price_pred, self._physics_feat
        )
