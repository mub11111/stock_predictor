"""PINN 物理约束损失 — 布林带约束 + 微观结构 PDE 约束 + Adaptive Huber 基座."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature column indices for microstructure PDE features in FEATURE_COLS
# micro_spread(67), micro_flow_pressure(68), micro_vol_imbalance(69),
# micro_toxicity(70), micro_arrival_impact(71), atr14(19)
_MICRO_FEATURE_MAP = {
    "spread": 67,        # bid-ask spread proxy: (high-low)/close
    "flow_pressure": 68,  # cumulative delta(close*volume)
    "vol_imbalance": 69,  # buy vs sell volume classification [-1,1]
    "toxicity": 70,       # VPIN-style flow toxicity [0,1]
    "arrival_impact": 71, # price impact per unit volume
    "atr14": 19,          # average true range
}


class AdaptiveHuberLoss(nn.Module):
    """自适应 Huber Loss — delta 根据批次波动率动态调整.

    高波动（突破行情）→ delta 自动增大 → 更像 MAE，抑制异常值对梯度的破坏。
    低波动（盘整行情）→ delta 自动缩小 → 更像 MSE，对微小的价格变化保持敏感。
    """

    def __init__(self, delta_min: float = 0.01, delta_max: float = 2.0,
                 delta_base: float = 0.5, vol_scale: float = 1.0):
        super().__init__()
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta_base = delta_base
        self.vol_scale = vol_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算自适应 Huber Loss.

        delta = clamp(delta_base + vol_scale * std(target), delta_min, delta_max)
        """
        vol = torch.std(target).detach()
        delta = self.delta_base + self.vol_scale * vol
        delta = float(torch.clamp(torch.tensor(delta), self.delta_min, self.delta_max))
        # delta < 1: more MSE-like; delta > 1: more MAE-like
        return F.smooth_l1_loss(pred, target, beta=delta)


class StableQuantPINNLoss(nn.Module):
    """稳健量化物理信息神经网络损失函数.

    基础损失使用 SmoothL1Loss (Huber Loss)，天生防梯度爆炸。
    边界惩罚使用 log1p(d) 平滑处理，避免平方项引起的数值爆炸。
    """

    def __init__(self, lambda_quant: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.base_loss_fn = AdaptiveHuberLoss(delta_base=beta)
        self.lambda_quant = lambda_quant

    def forward(self, pred_price: torch.Tensor, target_price: torch.Tensor,
                upper_band: torch.Tensor, lower_band: torch.Tensor) -> torch.Tensor:
        """计算带边界惩罚的总损失.

        Args:
            pred_price: (B, 1) 预测价格变化百分比 (百分点)
            target_price: (B, 1) 真实价格变化百分比 (百分点)
            upper_band: (B, 1) 布林上轨相对 close 的百分比 (百分点)
            lower_band: (B, 1) 布林下轨相对 close 的百分比 (百分点)
            lambda_quant: 量化惩罚权重，默认 0.1

        Returns:
            total_loss: 基础损失 + lambda_quant * 量化边界惩罚
        """
        # 1. 基础回归误差 — SmoothL1Loss 天生防梯度爆炸
        base_loss = self.base_loss_fn(pred_price, target_price)

        # 2. 计算违规溢出量 (越出上轨或跌破下轨)
        # 使用 F.relu 提取超出边界的部分，没超出就是 0
        violation_upper = F.relu(pred_price - upper_band)
        violation_lower = F.relu(lower_band - pred_price)
        total_violation = violation_upper + violation_lower

        # 3. 防爆惩罚计算 — Log1p: ln(1+x)，坚决避免平方项
        quant_penalty = torch.mean(torch.log1p(total_violation))

        # 4. 总 Loss
        return base_loss + self.lambda_quant * quant_penalty


def _safe_index(last: torch.Tensor, idx: int) -> torch.Tensor:
    """安全索引提取 — 索引越界时返回零张量."""
    C = last.shape[-1]
    if idx < C:
        return last[:, idx:idx + 1]
    return torch.zeros(last.shape[0], 1, device=last.device, dtype=last.dtype)


def extract_bb_bands(x: torch.Tensor, close_idx: int = 3,
                     upper_idx: int = 17, lower_idx: int = 19):
    """从标准化特征 batch 中提取布林带上下轨 (百分比空间).

    将 bb_upper/bb_lower 绝对价格转为相对于 close 的百分比，
    与模型输出的 price_delta 百分点对齐量纲。
    特征维度缩减时自动安全回退 (返回零张量).

    Args:
        x: (B, S, C) 标准化特征 batch
        close_idx: close 列索引 (FEATURE_COLS 中为 3)
        upper_idx: bb_upper 列索引 (FEATURE_COLS 中为 17)
        lower_idx: bb_lower 列索引 (FEATURE_COLS 中为 19)

    Returns:
        upper_pct: (B, 1) 上轨相对 close 的百分比 (百分点)
        lower_pct: (B, 1) 下轨相对 close 的百分比 (百分点)
    """
    close = _safe_index(x[:, -1, :], close_idx)
    bb_u = _safe_index(x[:, -1, :], upper_idx)
    bb_l = _safe_index(x[:, -1, :], lower_idx)

    eps = 1e-8
    upper_pct = (bb_u - close) / (close.abs() + eps) * 100.0
    lower_pct = (bb_l - close) / (close.abs() + eps) * 100.0

    upper_pct = torch.clamp(upper_pct, 0.0, 20.0)
    lower_pct = torch.clamp(lower_pct, -20.0, 0.0)

    return upper_pct, lower_pct


def extract_micro_features(x: torch.Tensor,
                           spread_idx: int = 67,
                           flow_idx: int = 68,
                           imbalance_idx: int = 69,
                           toxicity_idx: int = 70,
                           atr_idx: int = 19):
    """从标准化特征 batch 中提取微观结构指标 (百分比/比例空间).

    Args:
        x: (B, S, C) 标准化特征 batch
        spread_idx: micro_spread 列索引 (FEATURE_COLS 中为 67)
        flow_idx: micro_flow_pressure 列索引 (68)
        imbalance_idx: micro_vol_imbalance 列索引 (69)
        toxicity_idx: micro_toxicity 列索引 (70)
        atr_idx: atr14 列索引 (19)

    Returns:
        spread: (B, 1) 买卖价差代理 (比例)
        flow_pressure: (B, 1) 资金流压力 [-1, 1]
        vol_imbalance: (B, 1) 量失衡 [-1, 1]
        toxicity: (B, 1) 流动毒性 [0, 1]
        atr_pct: (B, 1) ATR相对价格百分比
    """
    last = x[:, -1, :]
    spread = _safe_index(last, spread_idx)
    flow_pressure = _safe_index(last, flow_idx)
    vol_imbalance = _safe_index(last, imbalance_idx)
    toxicity = _safe_index(last, toxicity_idx)
    atr_pct = _safe_index(last, atr_idx)
    return spread, flow_pressure, vol_imbalance, toxicity, atr_pct


class MicrostructurePDELoss(nn.Module):
    """市场微观结构 PDE 约束损失.

    基于真实市场微观结构方程:
      1. 订单流失衡律 (OFI, Kyle 1985):
         L_OFI = |Δp̂_i − sign(OFI_i) · λ · |OFI_i| · σ_i|
         当 |OFI| > threshold 时激活 — 预测方向须与资金流方向一致.

      2. 无套利定价边界:
         L_arb = relu(|Δp̂| − (spread + k · σ))
         预测幅度不得超过价差 + 波动率隐含的无套利边界.

    复合损失: L_total = base_loss + λ_ofi * L_OFI + λ_arb * L_arbitrage_free
    """

    def __init__(self, lambda_ofi: float = 0.15, lambda_arb: float = 0.10,
                 ofi_threshold: float = 0.2, arb_k: float = 3.0,
                 beta: float = 1.0):
        super().__init__()
        self.base_loss_fn = AdaptiveHuberLoss(delta_base=beta)
        self.lambda_ofi = lambda_ofi
        self.lambda_arb = lambda_arb
        self.ofi_threshold = ofi_threshold
        self.arb_k = arb_k

    def forward(self, pred_price: torch.Tensor, target_price: torch.Tensor,
                x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """计算微观结构PDE约束总损失.

        Args:
            pred_price: (B, 1) 预测价格变化百分比
            target_price: (B, 1) 真实价格变化百分比
            x: (B, S, C) 原始特征batch (用于提取微观结构指标)

        Returns:
            total_loss: 标量总损失
            loss_details: dict 含各分量用于日志
        """
        base_loss = self.base_loss_fn(pred_price, target_price)

        spread, flow_pressure, vol_imbalance, toxicity, atr_pct = \
            extract_micro_features(x)

        # ── 约束 1: 订单流失衡律 (OFI) ──
        # OFI 信号 = flow_pressure * vol_imbalance (合并压力+失衡)
        ofi_signal = flow_pressure * vol_imbalance  # (B, 1)
        ofi_active = (torch.abs(ofi_signal) > self.ofi_threshold).float()
        # Kyle lambda: price impact coefficient ≈ arrival_impact * toxicity
        impact_coef = toxicity * 0.5 + 0.1  # (B, 1), range [0.1, 0.6]
        # 预期价格变动方向应与 OFI 一致
        expected_sign = torch.sign(ofi_signal)
        pred_sign = torch.sign(pred_price)
        sign_mismatch = (pred_sign != expected_sign).float()
        # 方向不一致时惩罚：|Δp̂ − sign(OFI) · λ · |OFI| · σ|
        ofi_penalty = sign_mismatch * torch.abs(pred_price) * ofi_active
        L_ofi = (ofi_penalty * impact_coef).mean()

        # ── 约束 2: 无套利定价边界 ──
        # |Δp̂| ≤ spread_pct + arb_k × atr_pct
        bound = spread.abs() + self.arb_k * atr_pct.abs()  # (B, 1)
        L_arb = F.relu(torch.abs(pred_price) - bound).mean()

        total = base_loss + self.lambda_ofi * L_ofi + self.lambda_arb * L_arb

        details = {
            "base_loss": base_loss.item(),
            "L_ofi": L_ofi.item(),
            "L_arb": L_arb.item(),
            "ofi_active_ratio": ofi_active.mean().item(),
        }
        return total, details
