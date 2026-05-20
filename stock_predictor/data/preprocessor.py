"""Feature preprocessing and target construction with microstructure + sentiment features."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def sanitize_scaler(scaler: StandardScaler) -> StandardScaler:
    """Replace near-zero scale_ values with 1.0 to prevent overflow in transform()."""
    tiny = np.abs(scaler.scale_) < 1e-10
    if tiny.any():
        scaler.scale_[tiny] = 1.0
    return scaler


FEATURE_COLS = [
    # OHLCV raw (5)
    "open", "high", "low", "close", "volume",
    # Returns (3)
    "ret_1", "ret_5", "ret_15",
    # MA (4)
    "ma5", "ma10", "ma20", "ma60",
    # RSI (2)
    "rsi6", "rsi14",
    # MACD (3)
    "macd", "macd_signal", "macd_hist",
    # Bollinger (3)
    "bb_upper", "bb_middle", "bb_lower",
    # ATR (1)
    "atr14",
    # Volume (2)
    "vol_ratio", "vol_ma5",
    # Price relative to MAs (3)
    "pct_ma5", "pct_ma10", "pct_ma20",
    # Intraday (2)
    "hl_ratio", "oc_ratio",
    # Time features (4)
    "minute_sin", "minute_cos",
    "day_sin", "day_cos",
    # Counter-prediction: trader fingerprints (15)
    "mfi", "sm_flow", "vpt_ratio", "obv_div", "large_lot",
    "sm_score", "retail_intensity", "fomo_score", "panic_sel",
    "crowd_sent", "rev_zscore", "vwap_pull", "momentum",
    "vol_regime", "meta_signal",
    # Expanded indicators (19): KDJ, CCI, Williams %R, Donchian, ROC, OBV, etc.
    "kdj_k", "kdj_d", "kdj_j",
    "cci14", "willr14",
    "dc_upper", "dc_mid", "dc_lower",
    "roc5", "roc10",
    "obv", "chaikin_osc", "vol_roc5",
    "close_location", "ma20_slope", "ma60_slope",
    "gap_ratio", "up_down_vol", "intraday_intensity",
    # Microstructure features (8)
    "micro_spread",
    "micro_flow_pressure",
    "micro_vol_imbalance",
    "micro_trade_intensity",
    "micro_arrival_impact",
    "micro_order_depth",
    "micro_toxicity",
    "micro_bid_ask_bounce",
    # News sentiment proxy (6)
    "news_gap_signal",
    "news_vol_spike",
    "news_extreme_moves",
    "news_momentum_decay",
    "news_sentiment_proxy",
    "news_event_strength",
    # Volatility features (5)
    "vol_5bar_realized",
    "vol_expanding",
    "vol_skew",
    "vol_persistence",
    "vol_hl_ratio",
    # Quant features (3) — 金融学金牌指标注入
    "bb_pos",
    "rsi6_scaled",
    "rsi14_scaled",
]


def _safe_div(a, b, fill=0.0):
    return np.where(np.abs(b) > 1e-10, a / (b + 1e-10), fill)


def _compute_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """Extract microstructure/order-flow features from OHLCV bars."""
    out = pd.DataFrame(index=df.index)
    high, low, close, volume, open_p = (
        df["high"].values, df["low"].values, df["close"].values,
        df["volume"].values, df["open"].values
    )
    ret_1 = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + 1e-10)

    # 1. Spread proxy: (high-low)/close — wider spread = lower liquidity
    out["micro_spread"] = (high - low) / (close + 1e-10)

    # 2. Flow pressure: cumulative delta(close * volume) normalized
    price_vol = close * volume
    delta_pv = np.diff(price_vol, prepend=price_vol[0])
    cum_delta = np.cumsum(delta_pv)
    cum_vol = np.maximum(np.cumsum(volume), 1.0)
    out["micro_flow_pressure"] = np.clip(cum_delta / (cum_vol * close.mean() + 1e-10), -5, 5)

    # 3. Volume imbalance (tick test proxy): classify each bar as buy/sell via close vs open
    buy_vol = np.where(close > open_p, volume, np.where(close >= open_p, volume * 0.5, 0))
    sell_vol = np.where(close < open_p, volume, np.where(close > open_p, 0, volume * 0.5))
    buy_roll = pd.Series(buy_vol).rolling(10, min_periods=1).sum().values
    sell_roll = pd.Series(sell_vol).rolling(10, min_periods=1).sum().values
    out["micro_vol_imbalance"] = np.clip(_safe_div(buy_roll - sell_roll, buy_roll + sell_roll + 1), -1, 1)

    # 4. Trade intensity: volume per unit of |price change|
    abs_ret = np.abs(ret_1)
    out["micro_trade_intensity"] = np.clip(_safe_div(volume, abs_ret * close * 100 + 1e-10) / 1000, 0, 10)

    # 5. Arrival impact: |close-open| / volume — price impact per volume unit
    out["micro_arrival_impact"] = np.clip(_safe_div(np.abs(close - open_p) / (close + 1e-10),
                                                     volume / 1e6, fill=0), 0, 5)

    # 6. Order depth proxy: volume / (high-low) — deeper market absorbs more volume
    vol_per_range = _safe_div(volume, (high - low) / (close + 1e-10) + 1e-10)
    vol_per_range_ma = pd.Series(vol_per_range).rolling(20, min_periods=1).mean().values
    out["micro_order_depth"] = np.clip(_safe_div(vol_per_range, vol_per_range_ma + 1), 0, 5)

    # 7. Toxicity (VPIN-style): rolling volume imbalance / total volume
    vol_imbal_abs = np.abs(buy_roll - sell_roll)
    out["micro_toxicity"] = np.clip(_safe_div(vol_imbal_abs, buy_roll + sell_roll + 1), 0, 1)

    # 8. Bid-ask bounce: autocorrelation of 1-bar returns (neg = liquidity taking causes bounce)
    ret_series = pd.Series(ret_1)
    out["micro_bid_ask_bounce"] = ret_series.rolling(20, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 3 else 0, raw=False
    ).fillna(0).values

    return out


def _compute_news_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute news sentiment proxy features from price/volume patterns.

    News events leave statistical footprints: gaps, volume spikes, extreme returns.
    These are STANDARD features not dependent on actual news data — effective for
    any stock where news drives price discovery.
    """
    out = pd.DataFrame(index=df.index)
    close, volume, open_p, high, low = (
        df["close"].values, df["volume"].values, df["open"].values,
        df["high"].values, df["low"].values
    )
    ret_1 = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + 1e-10)
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    ret_std_20 = pd.Series(ret_1).rolling(20, min_periods=1).std().values

    # 1. Gap signal: overnight/opening gap magnitude and direction
    prev_close = np.roll(close, 1)
    gap = (open_p - prev_close) / (prev_close + 1e-10)
    out["news_gap_signal"] = np.clip(gap * 100, -5, 5)  # scaled to -5..5

    # 2. Volume spike: sudden volume surge suggests news-driven trading
    vol_ratio_inst = volume / (vol_ma20 + 1)
    out["news_vol_spike"] = np.clip(vol_ratio_inst / 3.0, 0, 5)  # >3x avg = spike

    # 3. Extreme moves: |ret| > 3σ indicates potential news event
    z_ret = np.abs(ret_1) / (ret_std_20 + 1e-10)
    out["news_extreme_moves"] = np.clip(z_ret / 3.0, 0, 5)  # normalized to 3σ threshold

    # 4. Momentum decay: after a news spike, how fast does ret autocorrelation decay?
    # Fast decay = news-driven (one-time shock); slow decay = trend
    abs_ret_ma3 = pd.Series(np.abs(ret_1)).rolling(3).mean().values
    abs_ret_ma10 = pd.Series(np.abs(ret_1)).rolling(10).mean().values
    out["news_momentum_decay"] = np.clip(_safe_div(abs_ret_ma3, abs_ret_ma10 + 1e-10), 0, 5)

    # 5. Composite sentiment proxy: weighted combination of all signals
    # Gap dominates (most news-like), spike + extreme confirm
    gap_sign = np.sign(gap)
    composite = (
        gap_sign * out["news_gap_signal"].values * 0.40  # gap direction × magnitude
        + out["news_vol_spike"].values * 0.25 * gap_sign  # vol spike with gap sign
        + out["news_momentum_decay"].values * 0.15 * gap_sign
        - out["news_extreme_moves"].values * 0.20  # extreme moves = uncertainty (negative)
    )
    out["news_sentiment_proxy"] = np.clip(composite, -3, 3)

    # 6. Event strength: binary-like signal for significant news events
    event_score = (
        (out["news_vol_spike"].values > 1.0).astype(float) * 0.5  # vol > 3x avg
        + (out["news_extreme_moves"].values > 1.0).astype(float) * 0.5  # |ret| > 3σ
    )
    out["news_event_strength"] = np.clip(
        pd.Series(event_score).rolling(5, min_periods=1).mean().values, 0, 1
    )

    return out


def _compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional volatility features beyond what counter_prediction provides."""
    out = pd.DataFrame(index=df.index)
    close = df["close"].values
    high, low = df["high"].values, df["low"].values
    ret_1 = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + 1e-10)

    # 1. 5-bar realized volatility
    out["vol_5bar_realized"] = pd.Series(ret_1).rolling(5).std().fillna(0).values * np.sqrt(5)

    # 2. Vol expanding vs contracting
    vol_5 = pd.Series(ret_1).rolling(5).std().fillna(0).values
    vol_20 = pd.Series(ret_1).rolling(20).std().fillna(1e-10).values
    out["vol_expanding"] = np.clip(_safe_div(vol_5, vol_20), 0.1, 10)

    # 3. Vol skew: up-vol vs down-vol
    up_ret = np.where(ret_1 > 0, ret_1, 0)
    down_ret = np.where(ret_1 < 0, np.abs(ret_1), 0)
    up_vol = pd.Series(up_ret).rolling(10).std().fillna(0).values
    down_vol = pd.Series(down_ret).rolling(10).std().fillna(1e-10).values
    out["vol_skew"] = np.clip(_safe_div(up_vol, down_vol + 1e-10), 0.1, 10)

    # 4. Vol persistence: autocorrelation of |ret| (GARCH-like)
    abs_ret = pd.Series(np.abs(ret_1))
    out["vol_persistence"] = abs_ret.rolling(20, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 5 else 0.5, raw=False
    ).fillna(0.5).values

    # 5. Parkinson vol proxy: ln(high/low)^2 / (4*ln2) — more efficient than close-close
    parkinson = np.log((high + 1e-10) / (low + 1e-10)) ** 2 / (4 * np.log(2))
    out["vol_hl_ratio"] = pd.Series(parkinson).rolling(10).mean().fillna(0).values

    return out


def preprocess(df: pd.DataFrame, fit_scaler: bool = True, scaler: StandardScaler = None) -> tuple[np.ndarray, StandardScaler]:
    """Transform raw minute OHLCV DataFrame into normalized feature array with all features."""
    data = df.copy()
    close = data["close"].values

    # Returns
    data["ret_1"] = np.pad(np.diff(close) / (close[:-1] + 1e-10), (1, 0))
    data["ret_5"] = np.pad(np.diff(close, 5) / (close[:-5] + 1e-10), (5, 0))
    data["ret_15"] = np.pad(np.diff(close, 15) / (close[:-15] + 1e-10), (15, 0))

    # Price relative to MAs
    for w in [5, 10, 20]:
        ma_col = f"ma{w}"
        if ma_col in data.columns:
            data[f"pct_{ma_col}"] = (data["close"] - data[ma_col]) / (data[ma_col] + 1e-10)

    # Time features
    if "trade_time" in data.columns:
        times = pd.to_datetime(data["trade_time"])
        minutes = times.dt.hour * 60 + times.dt.minute
        data["minute_sin"] = np.sin(2 * np.pi * minutes / 240)
        data["minute_cos"] = np.cos(2 * np.pi * minutes / 240)
        data["day_sin"] = np.sin(2 * np.pi * times.dt.dayofweek / 5)
        data["day_cos"] = np.cos(2 * np.pi * times.dt.dayofweek / 5)
    else:
        for col in ["minute_sin", "minute_cos", "day_sin", "day_cos"]:
            data[col] = 0

    # Intraday features
    if all(c in data.columns for c in ["high", "low", "close"]):
        data["hl_ratio"] = (data["high"] - data["low"]) / (data["close"] + 1e-10)
    if all(c in data.columns for c in ["open", "close"]):
        data["oc_ratio"] = (data["close"] - data["open"]) / (data["open"] + 1e-10)
    if "volume" in data.columns:
        data["vol_ma5"] = data["volume"] / (data["volume"].rolling(5).mean() + 1)

    # Microstructure features
    micro = _compute_microstructure(df)
    for col in micro.columns:
        data[col] = micro[col].values

    # News sentiment proxy features
    news = _compute_news_proxy(df)
    for col in news.columns:
        data[col] = news[col].values

    # Volatility features
    vol_feats = _compute_volatility_features(df)
    for col in vol_feats.columns:
        data[col] = vol_feats[col].values

    # Select available features, fill missing with zeros
    for col in FEATURE_COLS:
        if col not in data.columns:
            data[col] = 0.0
    arr = data[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
    # Clip to float32 range to avoid overflow during cast
    f32_max = np.finfo(np.float32).max
    f32_min = np.finfo(np.float32).min
    arr = np.clip(arr, f32_min, f32_max).astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(arr)
        sanitize_scaler(scaler)
        arr = scaler.transform(arr)
    elif scaler is not None:
        sanitize_scaler(scaler)
        arr = scaler.transform(arr)

    return arr, scaler


def build_targets(close: np.ndarray, horizon: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Build direction labels (0=down, 1=up) and price change targets (percentage).

    price_change is scaled to percentage (e.g., 0.5 = 0.5%) for better numerical stability.
    Kept for backward compatibility. See build_targets_v2 for new targets.
    """
    n = len(close)
    direction = np.zeros(n, dtype=np.int64)
    price_change = np.zeros(n, dtype=np.float32)
    threshold = 0.0005

    for i in range(n - horizon):
        future = close[i + horizon]
        change_pct = (future - close[i]) / (close[i] + 1e-10)
        price_change[i] = change_pct * 100.0  # percentage form
        if change_pct > threshold:
            direction[i] = 1
        else:
            direction[i] = 0
    return direction, price_change


# ── New prediction targets ──

def build_targets_v2(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                     volume: np.ndarray, horizon: int = 15) -> dict:
    """Build enhanced prediction targets that are more predictable than raw direction.

    Returns dict with keys:
      - large_move_up:    P(|return| > threshold AND ret > 0) — binary
      - large_move_down:  P(|return| > threshold AND ret < 0) — binary
      - vol_regime:       categorical (0=low vol, 1=med vol, 2=high vol) — 3-class
      - magnitude:        actual |return| normalized by ATR — regression
      - direction:        legacy direction (0=down, 1=up) for comparison
      - price_change:     raw return for backward compat
    """
    n = len(close)
    out = {
        "large_move_up": np.zeros(n, dtype=np.int64),
        "large_move_down": np.zeros(n, dtype=np.int64),
        "large_move_any": np.zeros(n, dtype=np.int64),  # combined: either direction large move
        "vol_regime": np.zeros(n, dtype=np.int64),
        "magnitude": np.zeros(n, dtype=np.float32),
        "direction": np.zeros(n, dtype=np.int64),
        "price_change": np.zeros(n, dtype=np.float32),
    }

    # Compute ATR for adaptive threshold
    tr_arr = np.zeros(n)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    for i in range(n):
        tr_arr[i] = max(
            high[i] - low[i],
            abs(high[i] - prev_close[i]),
            abs(low[i] - prev_close[i])
        )
    atr = pd.Series(tr_arr).ewm(alpha=1/14, adjust=False).mean().values

    # Rolling return std for vol regime
    ret_1 = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + 1e-10)
    roll_vol_10 = pd.Series(ret_1).rolling(10).std().fillna(0).values
    roll_vol_50 = pd.Series(ret_1).rolling(50).std().fillna(0).values

    for i in range(n - horizon):
        future = close[i + horizon]
        change_pct = (future - close[i]) / (close[i] + 1e-10)
        abs_change = abs(change_pct)

        # Adaptive threshold: 1.5x ATR over horizon (in pct terms)
        atr_pct = (atr[i] * np.sqrt(horizon)) / (close[i] + 1e-10)
        threshold = max(atr_pct * 1.5, 0.003)  # at least 0.3%

        out["price_change"][i] = change_pct * 100.0  # percentage form
        out["direction"][i] = 1 if change_pct > 0.0005 else 0
        out["magnitude"][i] = abs_change / (atr_pct + 1e-10)  # normalized by expected vol

        if change_pct > threshold:
            out["large_move_up"][i] = 1
            out["large_move_any"][i] = 1
        elif change_pct < -threshold:
            out["large_move_down"][i] = 1
            out["large_move_any"][i] = 1
        # else all 0 = no large move

        # Vol regime: predict volatility at horizon vs current
        future_vol = roll_vol_10[i + horizon] if i + horizon < n else roll_vol_10[-1]
        current_vol = roll_vol_50[i] if roll_vol_50[i] > 0 else roll_vol_10[i]
        vol_ratio = future_vol / (current_vol + 1e-10)
        if vol_ratio > 1.5:
            out["vol_regime"][i] = 2  # high vol
        elif vol_ratio > 0.7:
            out["vol_regime"][i] = 1  # medium vol
        else:
            out["vol_regime"][i] = 0  # low vol

    return out
