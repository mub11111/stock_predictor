"""Hybrid model: ChannelMixer → PatchEmbed → BiLSTM(+residual) → Transformer → MHA Pool → Dual Head.
PatchTST-style patching + cross-channel mixing + multi-head attention pooling.
"""
from __future__ import annotations
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")


class ChannelMixer(nn.Module):
    """Lightweight FFN that learns cross-feature interactions before patching.

    Applied per-timestep: (B, S, C) → (B, S, C)
    Helps PatchEmbed see richer local patterns across correlated features.
    """
    def __init__(self, in_dim: int, expansion: int = 2):
        super().__init__()
        hidden = in_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim),
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class PatchEmbed(nn.Module):
    """Segment time series into patches and project to d_model.

    Like ViT for images / PatchTST for time series:
    raw (B, S_raw, C) → (B, S_raw//P, d_model)

    Patching reduces token count by factor P, which quadratically
    reduces Transformer attention cost while keeping more raw context.
    """
    def __init__(self, patch_len: int, in_dim: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, raw_seq_len, C)
        B, S, C = x.shape
        n_patches = S // self.patch_len
        x = x[:, :n_patches * self.patch_len, :]  # trim to multiple
        x = x.reshape(B, n_patches, self.patch_len * C)  # flatten each patch
        x = self.proj(x)
        x = self.norm(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification. gamma=2 down-weights easy examples."""
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


class HybridModel(nn.Module):
    """ChannelMixer → PatchEmbed → BiLSTM(+residual) → Transformer → MHA Pooling → Dual Head."""

    def __init__(self,
                 input_dim: int,
                 d_model: int = 64,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2,
                 transformer_layers: int = 2,
                 nhead: int = 4,
                 dropout: float = 0.1,
                 max_seq_len: int = 240,
                 patch_len: int = 5):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len

        # ChannelMixer: cross-feature interaction before patching
        self.channel_mixer = ChannelMixer(input_dim)

        # PatchEmbed: raw (B, S_raw, C) → (B, S_raw/P, d_model)
        self.stem = PatchEmbed(patch_len, input_dim, d_model)

        # Effective sequence length after patching
        effective_len = max_seq_len // patch_len

        lstm_out = lstm_hidden * 2  # bidirectional
        tf_dim = d_model * 2

        self.pos_encoding = nn.Parameter(torch.randn(1, effective_len, d_model) * 0.02)

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, bidirectional=True,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0
        )
        self.lstm_to_tf = nn.Sequential(
            nn.Linear(lstm_out, tf_dim),
            nn.LayerNorm(tf_dim)
        )

        # Residual projection: map d_model → tf_dim for skip connection
        self.skip_proj = nn.Linear(d_model, tf_dim)
        self.skip_norm = nn.LayerNorm(tf_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_dim, nhead=nhead, dim_feedforward=tf_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        # Multi-head attention pooling: cross-attention with learned query
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=tf_dim, num_heads=4, batch_first=True
        )
        self.attn_query = nn.Parameter(torch.randn(1, 1, tf_dim) * 0.01)

        # Kendall uncertainty: learnable task weights for multi-task loss
        self.log_sigma_dir = nn.Parameter(torch.tensor(0.0))   # direction classification
        self.log_sigma_price = nn.Parameter(torch.tensor(0.0))  # price regression

        self.dropout = nn.Dropout(dropout)
        self.direction_head = nn.Sequential(
            nn.Linear(tf_dim, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        self.price_head = nn.Sequential(
            nn.Linear(tf_dim, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 3), nn.Tanh()  # q10, q50, q90 quantiles (3 values)
        )
        self.register_buffer('price_scale', torch.tensor(3.0))  # ±3% quantile delta range
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (B, S_raw, input_dim)

        # [新增] 维度安全校验门
        expected_dim = self.channel_mixer.net[0].in_features
        if x.size(-1) != expected_dim:
            raise ValueError(
                f"🚨 维度致命冲突！模型期望的特征维度是 {expected_dim}，"
                f"但实际传入的数据 x 包含了 {x.size(-1)} 维特征。\n"
                f"请检查 train_test.py 或路由代码，确保实例化 HybridModel 时传入的 input_dim 等于数据真实的特征数！"
            )

        x = self.channel_mixer(x)  # cross-feature mixing per timestep
        x = self.stem(x)            # (B, S_raw/P, d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # LSTM with residual skip
        residual = self.skip_proj(x)  # (B, S_patch, tf_dim)
        lstm_out, _ = self.lstm(x)    # (B, S_patch, lstm_hidden*2)
        x = self.lstm_to_tf(lstm_out) # (B, S_patch, tf_dim)
        x = self.skip_norm(x + residual)

        x = self.transformer(x, src_key_padding_mask=mask)

        # Multi-head attention pooling
        B = x.size(0)
        query = self.attn_query.expand(B, -1, -1)  # (B, 1, tf_dim)
        x, _ = self.attn_pool(query, x, x)          # cross-attention: query attends to sequence
        x = x.squeeze(1)  # (B, tf_dim)

        x = self.dropout(x)
        direction = self.direction_head(x)           # (B, 2) logits
        price_quantiles = self.price_head(x) * self.price_scale  # (B, 3) q10,q50,q90
        return direction, price_quantiles
