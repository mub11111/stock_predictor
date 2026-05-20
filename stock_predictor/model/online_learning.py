"""抗概念漂移在线学习 — 预测残差监控 + 轻量微调触发器.

当检测到市场风格切换 (Regime Shift) 时自动触发:
  1. Rank IC 显著衰减 (预测排序与实际排序相关性下降)
  2. q10-q90 置信带频繁被实际价格刺穿

触发后仅对模型最后一层或高频支路进行极少轮次微调,
更新本地权重以适应新市场状态。
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


class PredictionResidualMonitor:
    """实时预测残差监控管道.

    维护滚动窗口内的预测-实际对, 持续评估:
      - Rank IC (Spearman): 预测排序 vs 实际排序
      - 置信带穿透率: 实际价格落在 [q10, q90] 之外的频率

    当任一指标恶化超过阈值时发出触发信号.
    """

    def __init__(self, window_size: int = 60,
                 ic_threshold: float = 0.05,
                 breach_threshold: float = 0.35,
                 min_samples: int = 20):
        self.window_size = window_size
        self.ic_threshold = ic_threshold      # Rank IC 低于此值 → 触发
        self.breach_threshold = breach_threshold  # 穿透率高于此值 → 触发
        self.min_samples = min_samples          # 最少样本数才开始检测

        # 滚动缓冲区
        self._pred_q10: list[float] = []
        self._pred_q50: list[float] = []
        self._pred_q90: list[float] = []
        self._actual_deltas: list[float] = []   # 实际价格变动百分比
        self._actual_prices: list[float] = []
        self._base_prices: list[float] = []

        # 状态
        self.trigger_count: int = 0
        self.last_rank_ic: float = 0.0
        self.last_breach_ratio: float = 0.0
        self.total_samples: int = 0

    def record(self, q10_pct: float, q50_pct: float, q90_pct: float,
               actual_delta: float, base_price: float, actual_price: float):
        """记录一次预测-实际对.

        Args:
            q10_pct: 预测 q10 分位数 (比例, 如 0.01 = 1%)
            q50_pct: 预测中位数 (比例)
            q90_pct: 预测 q90 分位数 (比例)
            actual_delta: 实际价格变动 (比例)
            base_price: 基准价
            actual_price: 实际价格
        """
        self._pred_q10.append(q10_pct)
        self._pred_q50.append(q50_pct)
        self._pred_q90.append(q90_pct)
        self._actual_deltas.append(actual_delta)
        self._actual_prices.append(actual_price)
        self._base_prices.append(base_price)
        self.total_samples += 1

        # 保持窗口大小
        if len(self._pred_q50) > self.window_size * 2:
            for buf in [self._pred_q10, self._pred_q50, self._pred_q90,
                        self._actual_deltas, self._actual_prices, self._base_prices]:
                buf[:] = buf[-self.window_size:]

    def check_trigger(self) -> bool:
        """检测是否触发在线微调.

        Returns:
            True 如果检测到风格切换, 需要微调
        """
        n = len(self._pred_q50)
        if n < self.min_samples:
            return False

        # 使用最近 window_size 个样本
        w = min(n, self.window_size)
        preds = np.array(self._pred_q50[-w:], dtype=np.float64)
        actuals = np.array(self._actual_deltas[-w:], dtype=np.float64)
        q10s = np.array(self._pred_q10[-w:], dtype=np.float64)
        q90s = np.array(self._pred_q90[-w:], dtype=np.float64)

        # ── Rank IC (Spearman) ──
        rank_ic = _spearman_rank(preds, actuals)
        self.last_rank_ic = rank_ic

        # ── 置信带穿透率 ──
        # 实际值落在 [q10, q90] 之外
        below_lower = actuals < q10s
        above_upper = actuals > q90s
        breaches = np.sum(below_lower | above_upper)
        breach_ratio = breaches / w
        self.last_breach_ratio = breach_ratio

        # ── 触发条件 ──
        ic_triggered = rank_ic < self.ic_threshold
        breach_triggered = breach_ratio > self.breach_threshold

        if ic_triggered or breach_triggered:
            self.trigger_count += 1
            return True
        return False

    def get_recent_deltas(self, n: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """获取最近 n 个预测-实际对 (用于微调).

        Returns:
            (pred_q50_array, actual_delta_array) — 均为 (n,) float64
        """
        w = min(n, len(self._pred_q50))
        preds = np.array(self._pred_q50[-w:], dtype=np.float64)
        actuals = np.array(self._actual_deltas[-w:], dtype=np.float64)
        return preds, actuals

    def summary(self) -> dict:
        """返回监控状态摘要."""
        return {
            "samples": self.total_samples,
            "window": len(self._pred_q50),
            "last_rank_ic": round(self.last_rank_ic, 4),
            "last_breach_ratio": round(self.last_breach_ratio, 4),
            "trigger_count": self.trigger_count,
            "ic_threshold": self.ic_threshold,
            "breach_threshold": self.breach_threshold,
        }


def _spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关系数 (纯 numpy, 无 scipy 依赖).

    对 x, y 分别排序取秩, 再算 Pearson 相关.
    """
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        # 处理平局: 取平均秩
        uniq, inv = np.unique(arr, return_inverse=True)
        if len(uniq) < n:
            for u in uniq:
                mask = arr == u
                ranks[mask] = ranks[mask].mean()
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    num = ((rx - rx_mean) * (ry - ry_mean)).sum()
    den = np.sqrt(((rx - rx_mean) ** 2).sum() * ((ry - ry_mean) ** 2).sum())
    return float(num / (den + 1e-10))


class LightweightFineTuner:
    """轻量级在线微调 — 仅更新模型最后一层或高频支路.

    设计原则:
      - 最少轮次 (1-3 epochs), 极小学习率
      - 仅微调 price_head + dir_head (最后一层) 或高频支路
      - 自动快照 + 验证门控回滚
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 5e-6, max_epochs: int = 3):
        self.model = model
        self.device = device
        self.lr = lr
        self.max_epochs = max_epochs

    def _get_finetune_params(self, branch: str = "last_layer"):
        """获取需要微调的参数.

        Args:
            branch: "last_layer" — 仅最后的 price_head + dir_head
                    "high_freq" — FreqOrchestrator 的高频支路
                    "all" — 全部参数 (谨慎使用)
        """
        params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if branch == "last_layer":
                if any(k in name for k in ["price_head", "dir_head",
                                            "price_heads", "dir_heads",
                                            "log_sigma"]):
                    params.append(param)
            elif branch == "high_freq":
                if "high_freq" in name or "micro" in name.lower():
                    params.append(param)
            elif branch == "all":
                params.append(param)
        # Fallback: if no params matched (e.g. model doesn't have these specific names),
        # fine-tune all parameters with small learning rate
        if not params:
            for param in self.model.parameters():
                if param.requires_grad:
                    params.append(param)
        return params

    def finetune(self, features: np.ndarray, pred_deltas: np.ndarray,
                 actual_deltas: np.ndarray, branch: str = "last_layer") -> dict:
        """执行轻量在线微调.

        Args:
            features: (N, S, C) — 原始特征矩阵 (标准化后)
            pred_deltas: (N,) — 模型预测的 q50 中位数 (比例)
            actual_deltas: (N,) — 实际价格变动 (比例)
            branch: 微调范围

        Returns:
            result dict with fine-tuning metadata
        """
        params = self._get_finetune_params(branch)
        if not params:
            return {"success": False, "reason": "no_params_matched",
                    "branch": branch}

        # 快照 (仅保存被微调的参数)
        snapshot = {id(p): p.data.clone() for p in params}

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0)
        mse_loss = nn.MSELoss()

        # 准备数据
        x = torch.from_numpy(features).float().to(self.device)  # (N, S, C)
        target = torch.from_numpy(actual_deltas).float().to(self.device)  # (N,)
        # 转为百分点以匹配模型输出
        target_pct = target * 100.0  # 比例 → 百分点

        self.model.train()
        pre_loss = None
        post_loss = None

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            _, price_out = self.model(x)  # (N, 3) — q10, q50, q90
            q50_pred = price_out[:, 1]  # (N,) 百分点
            loss = mse_loss(q50_pred, target_pct)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()

            if epoch == 0:
                pre_loss = loss.item()
            post_loss = loss.item()

        self.model.eval()

        # ── 验证门控: 如果微调使 loss 增大则回滚 ──
        improved = post_loss is not None and (pre_loss is None or post_loss < pre_loss)
        if not improved and pre_loss is not None and post_loss is not None:
            for p in params:
                p.data.copy_(snapshot[id(p)])
            return {
                "success": False,
                "reason": "loss_increased",
                "pre_loss": round(pre_loss, 6),
                "post_loss": round(post_loss, 6),
                "branch": branch,
                "n_samples": len(features),
                "epochs": self.max_epochs,
            }

        return {
            "success": True,
            "pre_loss": round(pre_loss, 6) if pre_loss else None,
            "post_loss": round(post_loss, 6) if post_loss else None,
            "branch": branch,
            "n_params": sum(p.numel() for p in params),
            "n_samples": len(features),
            "epochs": self.max_epochs,
            "lr": self.lr,
        }


def online_finetune_if_needed(predictor, monitor: PredictionResidualMonitor,
                               recent_features: np.ndarray,
                               branch: str = "last_layer") -> dict | None:
    """便捷函数: 检查触发条件, 必要时执行在线微调.

    Args:
        predictor: Predictor 实例
        monitor: 预测残差监控器
        recent_features: (N, S, C) 最近的特征矩阵 (已标准化)
        branch: 微调范围

    Returns:
        微调结果 dict 或 None (未触发)
    """
    if not monitor.check_trigger():
        return None

    preds, actuals = monitor.get_recent_deltas(30)
    n_feat = min(len(recent_features), len(preds))
    features = recent_features[-n_feat:]
    preds = preds[-n_feat:]
    actuals = actuals[-n_feat:]

    tuner = LightweightFineTuner(predictor.model, predictor.device)
    return tuner.finetune(features, preds, actuals, branch=branch)
