"""Node Transformer: 截面资产图预测 + 情绪注入 (v2).

关键升级:
  - 图结构从"特征为节点 (B,1,d)" → "股票为节点 (1,B,d)" 的截面资产图
  - 多头注意力捕获股票间的板块轮动、产业链传导、资金溢出效应
  - 余弦相似度稀疏化邻接矩阵，仅保留高相关股票间的消息传递
  - 当 B=1（单股推理）时自动退化为自注意力精炼

Reference: "Node Transformer + BERT for Multi-Asset Financial Forecasting" (2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """截面图注意力: 残差 + 可选稀疏邻接掩码."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.15):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feats: torch.Tensor,
                attn_mask: torch.Tensor | None = None):
        """node_feats: [B, N, d_model] — B个图, 每图N个节点(股票)
        attn_mask: [N, N] or [B*N, N] — 邻接掩码, True=mask掉该边"""
        attn_out, _ = self.attn(node_feats, node_feats, node_feats,
                                attn_mask=attn_mask)
        node_feats = self.ln(node_feats + self.dropout(attn_out))
        return node_feats


class SentimentGate(nn.Module):
    """Gating: sentiment modulates feature channels via sigmoid gate + tanh bias."""

    def __init__(self, d_model: int, sentiment_dim: int = 6):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(sentiment_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.bias = nn.Sequential(
            nn.Linear(sentiment_dim, d_model),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, sentiment: torch.Tensor):
        gate = self.gate(sentiment)
        bias = self.bias(sentiment)
        return gate * x + (1 - gate) * bias


class TimeSeriesEncoder(nn.Module):
    """Conv1D → BiLSTM → temporal attention with residual skip."""

    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        # 1D conv pre-encoder: extract local patterns across time
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.conv_ln = nn.LayerNorm(hidden)

        self.lstm = nn.LSTM(hidden, hidden // 2, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Multi-head temporal attention
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        """x: [B, seq_len, input_dim] → [B, hidden]"""
        # Conv1D expects [B, C, L]
        x_t = x.transpose(1, 2)  # [B, input_dim, seq_len]
        conv_out = self.conv(x_t).transpose(1, 2)  # [B, seq_len, hidden]
        conv_out = self.conv_ln(conv_out)

        lstm_out, _ = self.lstm(conv_out)  # [B, seq_len, hidden]
        lstm_out = self.ln(lstm_out)

        attn_weights = self.temporal_attn(lstm_out)  # [B, seq_len, 1]
        pooled = (lstm_out * attn_weights).sum(dim=1)  # [B, hidden]

        return self.dropout(pooled)


class NodeTransformer(nn.Module):
    """Node Transformer v2: 截面资产图 + 情绪门控.

    架构:
      1. Conv1D → BiLSTM 时序编码
      2. 动态情绪注入 (最后 sentiment_dim 个特征)
      3. 情绪门控调制特征通道
      4. 截面图注意力: (B,) → (1, B, d) 以股票为节点
         余弦相似度稀疏化邻接矩阵
      5. 残差 FFN → 集成预测头
    """

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 nhead: int = 4, num_graph_layers: int = 2,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 sentiment_dim: int = 6, adj_threshold: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.sentiment_dim = min(sentiment_dim, input_dim)
        self.adj_threshold = adj_threshold

        # Input normalization
        self.input_ln = nn.LayerNorm(input_dim)

        # Time series encoder (per stock)
        self.ts_encoder = TimeSeriesEncoder(
            input_dim=input_dim, hidden=d_model, dropout=dropout
        )

        # Sentiment = last sentiment_dim features from each timestep (dynamic)
        self.sentiment_dim = min(sentiment_dim, input_dim)

        # Sentiment gating
        self.sentiment_gate = SentimentGate(d_model, self.sentiment_dim)

        # 截面图注意力层 — 以股票为节点
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, nhead, dropout)
            for _ in range(max(1, num_graph_layers))
        ])

        # Residual FFN
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Multi-head prediction ensemble: 3 small heads → average
        dir_heads = []
        price_heads = []
        for _ in range(3):
            dir_heads.extend([
                nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2)
            ])
            price_heads.extend([
                nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3), nn.Tanh()  # q10, q50, q90
            ])
        self.dir_heads = nn.ModuleList([
            nn.Sequential(*dir_heads[i*4:(i+1)*4]) for i in range(3)
        ])
        self.price_heads = nn.ModuleList([
            nn.Sequential(*price_heads[i*5:(i+1)*5]) for i in range(3)
        ])
        self.register_buffer('price_scale', torch.tensor(3.0))  # ±3% quantile delta range

        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))

    def _build_adjacency(self, node_feats: torch.Tensor) -> torch.Tensor | None:
        """基于余弦相似度构建稀疏邻接掩码.

        Args:
            node_feats: (B, d_model) — B 只股票的特征向量

        Returns:
            attn_mask: (B, B) — True 表示 MASK 掉无效边
            当 B < 2 时返回 None
        """
        B = node_feats.shape[0]
        if B < 2:
            return None  # 单股推理, 无邻接掩码
        # 余弦相似度矩阵
        node_norm = F.normalize(node_feats, dim=-1)  # (B, d)
        sim = node_norm @ node_norm.T  # (B, B), range [-1, 1]
        # 只保留高相关边, 低相关/负相关 edge 被 mask 掉
        adj = sim > self.adj_threshold  # (B, B) bool
        # 自己对自己的边始终保留 (self-loop)
        adj.fill_diagonal_(True)
        # MultiheadAttention 的 attn_mask: True = mask (ignore)
        return ~adj  # invert: True=masked

    def forward(self, x: torch.Tensor):
        """x: [B, seq_len, input_dim] → (dir_logits, price_quantiles)
        price_quantiles: (B, 3) — q10, q50, q90."""
        B = x.shape[0]

        # Input normalization
        x = self.input_ln(x)

        # 1. Time series encoding
        ts_emb = self.ts_encoder(x)  # [B, d_model]

        # 2. Dynamic sentiment: last sentiment_dim features of last timestep
        n_feat = x.shape[-1]
        if n_feat >= self.sentiment_dim:
            sentiment_feat = x[:, -1, -self.sentiment_dim:]  # [B, sentiment_dim]
        else:
            sentiment_feat = torch.zeros(B, self.sentiment_dim, device=x.device)

        # 3. Sentiment gating
        gated = self.sentiment_gate(ts_emb, sentiment_feat)  # [B, d_model]

        # 4. 截面资产图注意力 — 以股票为节点 (1 graph, B nodes)
        node_feats = gated.unsqueeze(0)  # [1, B, d_model]: 1个图, B个节点(股票)
        adj_mask = self._build_adjacency(gated)  # (B, B) or None
        for graph_layer in self.graph_layers:
            node_feats = graph_layer(node_feats, attn_mask=adj_mask)
        node_feats = node_feats.squeeze(0)  # [B, d_model]

        # 5. Residual FFN
        ffn_out = self.ffn2(F.gelu(self.ffn1(node_feats)))
        node_feats = self.ln_ffn(node_feats + self.dropout(ffn_out))
        node_feats = self.dropout(node_feats)

        # 6. Ensemble prediction heads (average 3 heads)
        dir_logits = sum(h(node_feats) for h in self.dir_heads) / len(self.dir_heads)
        price_quantiles = sum(h(node_feats) for h in self.price_heads) / len(self.price_heads) * self.price_scale

        return dir_logits, price_quantiles


# ── Compatibility wrapper ──

class NodeTransformerWrapper(nn.Module):
    """Wrapper matching the HybridModel interface (v2 architecture)."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        self.model = NodeTransformer(
            input_dim=input_dim, d_model=d_model,
            nhead=nhead, num_graph_layers=transformer_layers,
            dropout=dropout, max_seq_len=max_seq_len
        )
        self.log_sigma_dir = self.model.log_sigma_dir
        self.log_sigma_price = self.model.log_sigma_price

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════
# Legacy v1 architecture (for backward compatibility with old checkpoints)
# ═══════════════════════════════════════════════════════════════

class GraphAttentionLayerV1(nn.Module):
    """V1 graph attention: includes edge_bias parameter."""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.15):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.edge_bias = nn.Parameter(torch.zeros(1, nhead, 1, 1))

    def forward(self, node_feats: torch.Tensor):
        attn_out, _ = self.attn(node_feats, node_feats, node_feats)
        node_feats = self.ln(node_feats + self.dropout(attn_out))
        return node_feats


class TimeSeriesEncoderV1(nn.Module):
    """V1 encoder: BiLSTM → temporal attention (no Conv1D pre-encoder)."""

    def __init__(self, input_dim: int, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.ln = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        self.temporal_attn = nn.Sequential(
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        """x: [B, seq_len, input_dim] → [B, 256]"""
        lstm_out, _ = self.lstm(x)  # [B, seq_len, 256]
        lstm_out = self.ln(lstm_out)
        attn_weights = self.temporal_attn(lstm_out)  # [B, seq_len, 1]
        pooled = (lstm_out * attn_weights).sum(dim=1)  # [B, 256]
        return self.dropout(pooled)


class NodeTransformerV1(nn.Module):
    """V1 Node Transformer (original architecture before v2 rewrite).

    Architecture:
      1. BiLSTM temporal encoding (raw features → 256-dim)
      2. Linear projection 256 → d_model
      3. Sentiment gating (same as v2)
      4. Multi-layer graph attention with edge_bias
      5. Residual FFN → single prediction heads
    """

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 nhead: int = 4, num_graph_layers: int = 3,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 sentiment_dim: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.sentiment_dim = min(sentiment_dim, input_dim)

        self.ts_encoder = TimeSeriesEncoderV1(
            input_dim=input_dim, dropout=dropout
        )
        self.ts_proj = nn.Linear(256, d_model)

        self.sentiment_dim = min(sentiment_dim, input_dim)
        self.sentiment_gate = SentimentGate(d_model, self.sentiment_dim)

        self.graph_layers = nn.ModuleList([
            GraphAttentionLayerV1(d_model, nhead, dropout)
            for _ in range(max(1, num_graph_layers))
        ])

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.dir_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Tanh()
        )
        self.register_buffer('price_scale', torch.tensor(2.0))  # ±2% fixed (A股连续竞价基准价)

        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """x: [B, seq_len, input_dim] → (dir_logits, price_pred_pct)"""
        B = x.shape[0]

        ts_emb = self.ts_encoder(x)  # [B, 256]
        ts_emb = self.ts_proj(ts_emb)  # [B, d_model]

        n_feat = x.shape[-1]
        if n_feat >= self.sentiment_dim:
            sentiment_feat = x[:, -1, -self.sentiment_dim:]
        else:
            sentiment_feat = torch.zeros(B, self.sentiment_dim, device=x.device)

        gated = self.sentiment_gate(ts_emb, sentiment_feat)  # [B, d_model]

        node_feats = gated.unsqueeze(1)  # [B, 1, d_model]
        for graph_layer in self.graph_layers:
            node_feats = graph_layer(node_feats)
        node_feats = node_feats.squeeze(1)  # [B, d_model]

        ffn_out = self.ffn(node_feats)
        node_feats = self.ln_ffn(node_feats + self.dropout(ffn_out))
        node_feats = self.dropout(node_feats)

        dir_logits = self.dir_head(node_feats)
        price_pred = self.price_head(node_feats) * self.price_scale

        return dir_logits, price_pred


class NodeTransformerV1Wrapper(nn.Module):
    """Legacy wrapper for v1 checkpoints — matches HybridModel interface."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        self.model = NodeTransformerV1(
            input_dim=input_dim, d_model=d_model,
            nhead=nhead, num_graph_layers=transformer_layers,
            dropout=dropout, max_seq_len=max_seq_len
        )
        self.log_sigma_dir = self.model.log_sigma_dir
        self.log_sigma_price = self.model.log_sigma_price

    def forward(self, x):
        return self.model(x)
