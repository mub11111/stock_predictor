"""MicroStructure Module — 高频微观波动建模 (CMF/MFI/OBV/量比/短期收益)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 全局 88 维特征空间中的微观结构特征索引 (仅用于未预子集的向后兼容)
_MICRO_FEATURE_GLOBAL_INDICES = [0, 1, 2, 3, 4, 5, 6, 14, 15, 18, 45, 46, 47, 52]
# 特征名映射 (用于从 selected_features 动态计算本地索引)
_MICRO_FEATURE_NAMES = [
    "open", "high", "low", "close", "volume", "ret_1", "ret_5",
    "rsi6", "rsi14", "macd_hist",
    "cmf", "mfi", "obv", "vol_ratio",
]


class MicroStructureModule(nn.Module):
    """高频微观结构模块.

    从 OHLCV / 短期收益 / RSI / 资金流指标中提取
    局部微观模式与波动率估计.
    """

    def __init__(self, seq_len: int = 120, d_model: int = 64,
                 dropout: float = 0.1, n_micro: int | None = None,
                 micro_indices: list[int] | None = None):
        super().__init__()
        # 如果调用方提供了 micro_indices (本地索引), 直接使用
        # 否则如果给了 n_micro, 说明特征已预子集, 无需再索引
        # 否则回退到全局索引 (向后兼容)
        if micro_indices is not None:
            self.register_buffer('micro_indices',
                                 torch.tensor(micro_indices, dtype=torch.long))
            self._feature_count = len(micro_indices)
        elif n_micro is not None:
            # 调用方已预子集特征, 不需要内部索引
            self.micro_indices = None
            self._feature_count = n_micro
        else:
            self.register_buffer('micro_indices',
                                 torch.tensor(_MICRO_FEATURE_GLOBAL_INDICES, dtype=torch.long))
            self._feature_count = len(_MICRO_FEATURE_GLOBAL_INDICES)

        # Adaptive input projection (lazily created when input dim != _feature_count)
        self.input_proj: nn.Linear | None = None

        # Conv1d for local microstructure patterns
        self.conv1 = nn.Conv1d(self._feature_count, d_model // 2, kernel_size=5,
                               padding=2)
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=3,
                               padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Volatility head: predict short-term volatility from microstructure
        self.vol_head = nn.Sequential(
            nn.Linear(d_model, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Softplus()
        )

    @staticmethod
    def resolve_indices(selected_features: list[str]) -> list[int] | None:
        """从特征名列表动态计算微观结构特征的本地索引.

        给定 selected_features (如 MRMR 输出的 40 个特征名),
        返回其中属于微观结构子集的特征的本地位置索引.

        Args:
            selected_features: 特征名列表 (如 ["open", "high", ..., "rsi14", ...])

        Returns:
            本地索引列表 (如 [0, 1, 2, 3, 4, 5, 6, 10, 11, 15]) 或 None
        """
        indices = []
        for name in _MICRO_FEATURE_NAMES:
            try:
                indices.append(selected_features.index(name))
            except ValueError:
                pass  # 该特征未被选中, 跳过
        return indices if indices else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, S, C) → pooled: (B, d_model), vol: (B, 1)."""
        # 仅在所有全局索引都在范围内时才做子集化
        if self.micro_indices is not None and x.shape[-1] > max(self.micro_indices):
            x = x[:, :, self.micro_indices]

        # 自适应投影: 当输入通道数与预期的特征数不匹配时
        # (例如特征被MRMR裁剪但micro_indices未更新),
        # 将输入投影到期望的通道数, 避免Conv1d维度不匹配
        in_ch = x.shape[-1]
        if in_ch != self._feature_count:
            if self.input_proj is None or self.input_proj.in_features != in_ch:
                self.input_proj = nn.Linear(in_ch, self._feature_count, device=x.device)
            x = self.input_proj(x)

        # Conv over time
        h = self.conv1(x.transpose(1, 2))  # (B, d/2, S)
        h = self.act(h)
        h = self.conv2(h)                   # (B, d_model, S)
        h = self.dropout(h)
        h = h.transpose(1, 2)              # (B, S, d_model)

        # Global average pool over time
        pooled = h.mean(dim=1)              # (B, d_model)
        pooled = self.norm(pooled)

        # Volatility estimate
        volatility = self.vol_head(pooled)   # (B, 1)

        return pooled, volatility
