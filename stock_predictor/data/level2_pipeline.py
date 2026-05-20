"""Level-2 / Tick 数据管道 — 订单簿快照与逐笔成交特征提取.

为 Freq Orchestrator 高频支路 (MicroStructure/Conv1d) 提供
更高维度的真实信息源，从 Level-1 (OHLCV) 衍生特征升级到
Level-2 市场微观结构特征。

设计原则:
  - 所有计算完全向量化 (numpy), 无 Python for 循环
  - 接口抽象, 支持未来接入真实 Level-2 数据源
  - 无缝集成现有 FEATURE_COLS 管线
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ── 数据结构定义 ──

@dataclass
class OrderBookSnapshot:
    """单笔订单簿快照 (Level-2 买卖十档).

    当真实数据不可用时，可从 Level-1 OHLCV 合成近似快照.
    """
    timestamp: float          # epoch 秒
    bid_prices: np.ndarray    # (10,) 买一~买十价
    bid_volumes: np.ndarray   # (10,) 买一~买十量
    ask_prices: np.ndarray    # (10,) 卖一~卖十价
    ask_volumes: np.ndarray   # (10,) 卖一~卖十量
    last_price: float         # 最新成交价
    last_volume: float        # 最新成交量

    @classmethod
    def from_ohlcv_bar(cls, row: np.ndarray, atr: float = 0.01) -> "OrderBookSnapshot":
        """从 Level-1 OHLCV 单根 K 线合成近似订单簿快照.

        基于对数正态假设重建虚拟的 10 档买卖盘:
          - 买卖价差 = max(high-low, atr * 0.3)
          - 各档量遵循指数衰减 vol_i = total_vol * decay^i
        """
        open_p, high, low, close, volume = row[0], row[1], row[2], row[3], row[4]
        spread = max(high - low, atr * close * 0.003)  # 至少 3bp
        half_spread = spread / 2.0
        mid = (high + low) / 2.0

        # 10 档价格: mid ± half_spread * 1, 1.3, 1.6, ..., 4.0
        ticks = np.arange(1, 11)
        mult = 1.0 + 0.3 * (ticks - 1)  # 1.0, 1.3, 1.6, ..., 3.7
        bid_prices = mid - half_spread * mult
        ask_prices = mid + half_spread * mult

        # 虚拟量: 总量按指数分配, 近端档位量更大
        decay = 0.6 ** ticks  # 0.6, 0.36, 0.22, ...
        decay = decay / decay.sum()
        bid_volumes = volume * decay * 0.5
        ask_volumes = volume * decay * 0.5

        return cls(
            timestamp=0.0,
            bid_prices=bid_prices.astype(np.float32),
            bid_volumes=bid_volumes.astype(np.float32),
            ask_prices=ask_prices.astype(np.float32),
            ask_volumes=ask_volumes.astype(np.float32),
            last_price=close,
            last_volume=volume,
        )


@dataclass
class TickRecord:
    """单笔逐笔成交记录."""
    timestamp: float   # epoch 秒
    price: float       # 成交价
    volume: float      # 成交量
    direction: int     # 1=主动买, -1=主动卖, 0=不确定


# ── Level-2 特征提取器 ──

class Level2FeatureExtractor:
    """从订单簿快照序列 & 逐笔成交序列中提取高频微观特征.

    所有方法均为静态/类方法, 无状态, 便于并行调用.
    """

    @staticmethod
    def compute_order_book_pressure(snapshots: list[OrderBookSnapshot]) -> np.ndarray:
        """买卖盘口压迫比 — bid/ask 量失衡 (向量化).

        Returns:
            (T, 2) — [pressure_ratio, pressure_imbalance]
        """
        T = len(snapshots)
        # 堆叠所有快照的买卖量 → (T, 10)
        bid_vols = np.stack([s.bid_volumes for s in snapshots])  # (T, 10)
        ask_vols = np.stack([s.ask_volumes for s in snapshots])
        bid_top5 = bid_vols[:, :5].sum(axis=1)  # (T,)
        ask_top5 = ask_vols[:, :5].sum(axis=1)
        denom = bid_top5 + ask_top5 + 1e-10
        pressure = (bid_top5 - ask_top5) / denom
        # 不平衡: log(bid/ask) 均值 across top5
        ratio = bid_vols[:, :5] / (ask_vols[:, :5] + 1e-10)
        imbalance = np.clip(np.log(ratio + 1e-6).mean(axis=1), -2, 2)
        return np.column_stack([pressure, imbalance]).astype(np.float32)

    @staticmethod
    def compute_passive_absorption(ticks: list[TickRecord],
                                    window: int = 50) -> np.ndarray:
        """大单被动吃货率 (向量化滚动窗口).

        Returns:
            (T_out, 3) — [absorption_rate, passive_buy_ratio, large_trade_intensity]
        """
        n = len(ticks)
        if n < window:
            return np.zeros((1, 3), dtype=np.float32)

        volumes = np.array([t.volume for t in ticks], dtype=np.float64)
        directions = np.array([t.direction for t in ticks], dtype=np.float32)

        large_threshold = np.percentile(volumes, 85) if n > 10 else np.inf
        is_large = (volumes >= large_threshold).astype(np.float32)
        is_passive = (directions == 0).astype(np.float32)
        is_buy = (directions >= 0).astype(np.float32)

        # 滚动窗口和 via cumsum
        cum_large = np.cumsum(np.pad(is_large, (1, 0), mode='constant')[:-1])
        cum_passive = np.cumsum(np.pad(is_large * is_passive, (1, 0), mode='constant')[:-1])
        cum_passive_buy = np.cumsum(np.pad(is_large * is_passive * is_buy, (1, 0), mode='constant')[:-1])
        cum_vol = np.cumsum(np.pad(volumes, (1, 0), mode='constant')[:-1])
        cum_large_vol = np.cumsum(np.pad(is_large * volumes, (1, 0), mode='constant')[:-1])

        n_out = n - window + 1
        # 窗口差值
        w_large = cum_large[window:] - cum_large[:n_out]  # (n_out,)
        w_passive = cum_passive[window:] - cum_passive[:n_out]
        w_passive_buy = cum_passive_buy[window:] - cum_passive_buy[:n_out]
        w_vol = cum_vol[window:] - cum_vol[:n_out]
        w_large_vol = cum_large_vol[window:] - cum_large_vol[:n_out]

        absorption = np.where(w_large > 0, w_passive / w_large, 0.0)
        passive_buy = np.where(w_large > 0, w_passive_buy / w_large, 0.0)
        intensity = w_large_vol / (w_vol + 1e-10)

        return np.column_stack([absorption, passive_buy, intensity]).astype(np.float32)

    @staticmethod
    def compute_tick_features(ticks: list[TickRecord],
                               interval_seconds: float = 60.0) -> np.ndarray:
        """从逐笔成交聚合为分钟级高频特征 (向量化).

        Returns:
            (T, 4) — [trade_intensity, vol_clustering, price_impact, flow_toxicity]
        """
        n = len(ticks)
        if n < 2:
            return np.zeros((1, 4), dtype=np.float32)

        prices = np.array([t.price for t in ticks], dtype=np.float64)
        volumes = np.array([t.volume for t in ticks], dtype=np.float64)
        directions = np.array([t.direction for t in ticks], dtype=np.float32)
        eps = 1e-10

        vol_series = volumes / (volumes.mean() + eps)
        # 向量化 vol_cluster: shift-1 乘积
        vol_cluster = np.zeros(n, dtype=np.float32)
        vol_cluster[1:] = vol_series[1:] * vol_series[:-1]

        rets = np.zeros(n, dtype=np.float64)
        rets[0] = 0
        rets[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + eps)
        price_impact = np.abs(rets) / (volumes + eps) * 1e6

        toxicity = directions * np.sign(rets)

        return np.column_stack([
            vol_series,
            np.clip(vol_cluster, 0, 10),
            np.clip(price_impact, 0, 1),
            np.clip(toxicity, -1, 1),
        ]).astype(np.float32)

    @staticmethod
    def synth_from_ohlcv(ohlcv: np.ndarray) -> np.ndarray:
        """从 Level-1 OHLCV bar 矩阵合成 Level-2 高频特征.

        这是无真实 Level-2 数据时的降级方案, 直接从 OHLCV
        中近似高频微观结构特征.

        Args:
            ohlcv: (T, 5+) — open, high, low, close, volume, ...

        Returns:
            (T, 8) — 8 个增量高频特征列:
              [l2_pressure, l2_imbalance, l2_absorption,
               l2_passive_buy, l2_large_intensity,
               l2_trade_intensity, l2_price_impact, l2_toxicity]
        """
        T = ohlcv.shape[0]
        open_p, high, low, close, volume = (
            ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        )
        eps = 1e-10

        # --- 1. 虚拟盘口压迫比 ---
        spread_pct = (high - low) / (close + eps)
        # 假设买卖压力与 K 线实体方向相关
        body_dir = np.sign(close - open_p)  # +1 阳线, -1 阴线
        # 上影线 vs 下影线比例反映买卖力量
        upper_wick = high - np.maximum(open_p, close)
        lower_wick = np.minimum(open_p, close) - low
        wick_imbalance = (lower_wick - upper_wick) / (high - low + eps)
        l2_pressure = np.clip(wick_imbalance + body_dir * 0.3, -1, 1)

        # --- 2. 虚拟量失衡 ---
        # 价格上涨时量放大 = 主动买; 下跌时量放大 = 主动卖
        ret_1 = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + eps)
        vol_dir = np.sign(ret_1) * volume / (volume.mean() + eps)
        l2_imbalance = np.clip(vol_dir, -1, 1)

        # --- 3. 被动吃货率 (代理) ---
        # 十字星/小实体 = 多空均未主动打破平衡, 可能存在被动吸筹
        body_ratio = np.abs(close - open_p) / (high - low + eps)
        is_doji = (body_ratio < 0.3).astype(np.float32)
        # 在 doji 上的成交量占滚动窗口成交量的比例
        vol_ma20 = np.convolve(volume, np.ones(20) / 20, mode='same')
        doji_vol_ratio = np.where(vol_ma20 > 0, is_doji * volume / (vol_ma20 + eps), 0)
        l2_absorption = np.clip(doji_vol_ratio, 0, 3)

        # --- 4. 被动买入比 ---
        # 阳十字星 = 买方挂单被消化, 价格未大涨
        bullish_doji = is_doji * (body_dir > 0).astype(np.float32)
        l2_passive_buy = np.clip(bullish_doji * volume / (vol_ma20 + eps), 0, 2)

        # --- 5. 大单强度 ---
        vol_95 = np.percentile(volume, 85) if T > 20 else volume.max()
        is_large = (volume >= vol_95).astype(np.float32)
        large_vol = np.convolve(is_large * volume, np.ones(5) / 5, mode='same')
        l2_large_intensity = np.clip(large_vol / (vol_ma20 + eps), 0, 3)

        # --- 6. 成交强度 ---
        l2_trade_intensity = np.clip(volume / (vol_ma20 + eps), 0, 5)

        # --- 7. 价格冲击 (Amihud) ---
        price_impact = np.abs(ret_1) / (volume + eps) * 1e8
        l2_price_impact = np.clip(price_impact / (price_impact.mean() + eps), 0, 5)

        # --- 8. 流动毒性 ---
        # VPIN 代理: 量失衡的滚动绝对值 / 总成交量
        abs_imb = np.abs(l2_imbalance * volume)
        abs_imb_ma = np.convolve(abs_imb, np.ones(20) / 20, mode='same')
        l2_toxicity = np.clip(abs_imb_ma / (vol_ma20 + eps) * 5, 0, 1)

        return np.column_stack([
            l2_pressure, l2_imbalance, l2_absorption,
            l2_passive_buy, l2_large_intensity,
            l2_trade_intensity, l2_price_impact, l2_toxicity,
        ]).astype(np.float32)


# ── 预处理器集成 ──

# 新增的 8 个 Level-2 高频特征列名
L2_FEATURE_COLS = [
    "l2_pressure",         # 盘口压迫比 [-1,1]
    "l2_imbalance",        # 量失衡 [-1,1]
    "l2_absorption",       # 被动吃货率 [0,3]
    "l2_passive_buy",      # 被动买入比 [0,2]
    "l2_large_intensity",  # 大单强度 [0,3]
    "l2_trade_intensity",  # 成交强度 [0,5]
    "l2_price_impact",     # 价格冲击系数 [0,5]
    "l2_toxicity",         # 流动毒性 [0,1]
]


def compute_level2_features(df) -> np.ndarray:
    """预处理器集成入口 — 从 DataFrame 计算 Level-2 增量特征.

    Args:
        df: pandas DataFrame, 至少包含 open/high/low/close/volume 列

    Returns:
        (T, 8) float32 array, 可直接与现有特征矩阵水平拼接
    """
    ohlcv = df[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    return Level2FeatureExtractor.synth_from_ohlcv(ohlcv)
