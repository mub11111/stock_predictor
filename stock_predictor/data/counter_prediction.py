"""Counter-prediction features: detect trader fingerprints to anticipate their moves.

Models 4 market participant types from OHLCV data:
  - Smart money / Institutions: follow their flow
  - Retail crowd: fade them (contrarian)
  - Quant/Algo: anticipate mean-reversion & momentum patterns
  - High-frequency/speculators: volume spike patterns

References:
  - "Order Flow Imbalance & Institutional Activity" (Biais et al. 1995)
  - "The Behavior of Individual Investors" (Barber & Odean 2011)
  - "Man vs Machine: Quant Trading in Chinese A-Share Market"
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _safe_div(a, b, fill=0.0):
    return np.where(np.abs(b) > 1e-10, a / (b + 1e-10), fill)


def compute_mfi(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Money Flow Index: volume-weighted RSI. Smart money leaves traces in MFI divergences."""
    high, low, close, vol = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    typical = (high + low + close) / 3.0
    raw_flow = typical * vol
    delta = np.diff(typical, prepend=typical[0])
    pos_flow = np.where(delta > 0, raw_flow, 0)
    neg_flow = np.where(delta < 0, raw_flow, 0)
    # Rolling sum via pandas for simplicity
    pos_sum = pd.Series(pos_flow).rolling(period, min_periods=1).sum().values
    neg_sum = pd.Series(neg_flow).rolling(period, min_periods=1).sum().values
    mfi = 100.0 - 100.0 / (1.0 + _safe_div(pos_sum, neg_sum, fill=1.0))
    return np.nan_to_num(mfi, nan=50.0)


def compute_chaikin_ad(df: pd.DataFrame) -> np.ndarray:
    """Chaikin Accumulation/Distribution line — cumulative smart money flow."""
    high, low, close, vol = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    hl_range = high - low
    clv = _safe_div((close - low) - (high - close), hl_range)  # -1 to +1
    ad = np.cumsum(clv * vol)
    return ad


def compute_vpt(df: pd.DataFrame) -> np.ndarray:
    """Volume Price Trend — cumulative volume * return."""
    close, vol = df["close"].values, df["volume"].values
    ret = np.diff(close) / (close[:-1] + 1e-10)
    ret = np.append(ret, 0)
    vpt = np.cumsum(vol * ret)
    return vpt


def compute_obv_divergence(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """OBV vs price divergence: positive = OBV leading price up (bullish divergence)."""
    close, vol = df["close"].values, df["volume"].values
    direction = np.sign(np.diff(close, prepend=close[0]))
    obv = np.cumsum(vol * direction)
    # Rolling correlation proxy: OBV momentum minus price momentum
    obv_roc = pd.Series(obv).pct_change(lookback).fillna(0).values
    price_roc = pd.Series(close).pct_change(lookback).fillna(0).values
    divergence = obv_roc - price_roc  # >0: OBV stronger than price (bullish)
    return np.nan_to_num(divergence, nan=0)


def compute_vwap_series(df: pd.DataFrame) -> np.ndarray:
    """Cumulative VWAP — institutional participation reference price."""
    high, low, close, vol = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    typical = (high + low + close) / 3.0
    cum_pv = np.cumsum(typical * vol)
    cum_vol = np.maximum(np.cumsum(vol), 1.0)
    return cum_pv / cum_vol


def counter_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trader fingerprint features from OHLCV data.

    Returns DataFrame with 14 columns usable as model input features.
    All values are scaled/normalized for direct neural network input.
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"].values.astype(float)
    vol = df["volume"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_p = df["open"].values.astype(float)

    vol_ma5 = pd.Series(vol).rolling(5, min_periods=1).mean().values
    vol_ma20 = pd.Series(vol).rolling(20, min_periods=1).mean().values
    price_impact = np.abs(close - np.roll(close, 1)) / (np.roll(close, 1) + 1e-10)

    # ========== Smart Money Features (跟随) ==========

    # 1. MFI — volume-weighted momentum
    result["mfi"] = compute_mfi(df)
    result["mfi"] = result["mfi"].fillna(50.0) / 100.0  # normalize 0-1

    # 2. Smart money flow — A/D line 5-period change, normalized by volume
    ad = compute_chaikin_ad(df)
    ad_delta = pd.Series(ad).diff(5).fillna(0).values
    result["sm_flow"] = np.clip(_safe_div(ad_delta, vol_ma20 * 5), -5, 5)

    # 3. VPT ratio — smart money flow trend
    vpt = compute_vpt(df)
    vpt_ma20 = pd.Series(vpt).rolling(20, min_periods=1).mean().values
    result["vpt_ratio"] = np.clip(_safe_div(vpt, vpt_ma20, fill=1.0) - 1.0, -1, 1)

    # 4. OBV divergence — when smart money moves opposite to price
    result["obv_div"] = np.clip(compute_obv_divergence(df), -0.5, 0.5)

    # 5. Large lot ratio — volume 2x average + price impact > 0.3% → institution active
    large_lot = ((vol > 2.0 * vol_ma20) & (price_impact > 0.003)).astype(float)
    result["large_lot"] = pd.Series(large_lot).rolling(10, min_periods=1).mean().values

    # 6. Smart money composite score (-1 to 1)
    result["sm_score"] = np.clip(
        (result["mfi"].values - 0.5) * 0.25
        + result["sm_flow"].values * 0.25
        + result["vpt_ratio"].values * 0.25
        + result["obv_div"].values * 0.25,
        -1, 1,
    )

    # ========== Retail Crowd Features (逆向) ==========

    # 7. Retail intensity — high volume + low price impact = noise traders
    raw_retail = _safe_div(vol, vol_ma20, fill=1.0) * (1.0 - np.clip(price_impact * 200, 0, 1))
    result["retail_intensity"] = pd.Series(raw_retail).rolling(5, min_periods=1).mean().values

    # 8. FOMO (fear of missing out) — chasing after sustained up move
    ma10 = pd.Series(close).rolling(10, min_periods=1).mean().values
    pct5 = pd.Series(close).pct_change(5).fillna(0).values
    fomo = ((close > ma10) & (pct5 > 0.005)).astype(float)
    result["fomo_score"] = pd.Series(fomo).rolling(8, min_periods=1).mean().values

    # 9. Panic selling — dumping after sustained down move
    panic = ((close < ma10) & (pct5 < -0.005)).astype(float)
    result["panic_sel"] = pd.Series(panic).rolling(8, min_periods=1).mean().values

    # 10. Crowd sentiment — greed minus fear (-1 to 1)
    result["crowd_sent"] = np.clip(result["fomo_score"].values * 0.7 - result["panic_sel"].values * 0.7, -1, 1)

    # ========== Quant/Algo Features (利用规律) ==========

    # 11. Mean reversion z-score — distance from Bollinger center in std units
    bb_ma = pd.Series(close).rolling(20, min_periods=1).mean().values
    bb_std = pd.Series(close).rolling(20, min_periods=1).std().values
    result["rev_zscore"] = np.clip(_safe_div(close - bb_ma, bb_std, fill=0), -4, 4)

    # 12. VWAP gravity — how far price is from cumulative VWAP (algos anchor to VWAP)
    vwap = compute_vwap_series(df)
    result["vwap_pull"] = np.clip(_safe_div(close - vwap, close), -0.05, 0.05) * 20  # scaled -1 to 1

    # 13. Momentum signal — for trend-following algos
    ma5 = pd.Series(close).rolling(5, min_periods=1).mean().values
    result["momentum"] = np.clip(_safe_div(ma5 - ma10, ma10, fill=0) * 50, -1, 1)

    # 14. Vol regime — expanding (>1) vs contracting (<1) volatility
    vol5_std = pd.Series(close).pct_change().rolling(5).std().values
    vol20_std = pd.Series(close).pct_change().rolling(20).std().values
    result["vol_regime"] = np.clip(_safe_div(vol5_std, vol20_std, fill=1.0), 0.2, 5.0)

    # ========== Meta Counter-Trade Signal ==========

    # 15. Composite signal: consensus of all trader type signals
    # Positive = bullish: follow smart money, fade retail, mean-revert at lows
    result["meta_signal"] = np.clip(
        result["sm_score"].values * 0.30  # follow smart money
        - result["crowd_sent"].values * 0.25  # contrarian to retail
        - result["rev_zscore"].values / 4.0 * 0.25  # mean reversion (fade extremes)
        + result["momentum"].values * 0.20,  # short-term momentum
        -1, 1,
    )

    return result
