import numpy as np
import pandas as pd


def compute_ma(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    result = {}
    for w in windows:
        result[f"ma{w}"] = series.rolling(w).mean()
    return pd.DataFrame(result)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    middle = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    return upper, middle, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_kdj(df: pd.DataFrame, period: int = 9, k_period: int = 3, d_period: int = 3):
    """Compute KDJ indicator. Returns (K, D, J) series."""
    high, low, close = df["high"], df["low"], df["close"]
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    k = rsv.ewm(alpha=1/k_period, adjust=False).mean()
    d = k.ewm(alpha=1/d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def compute_cci(df: pd.DataFrame, period: int = 14):
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)


def compute_williams_r(df: pd.DataFrame, period: int = 14):
    """Williams %R — overbought/oversold indicator."""
    high, low, close = df["high"], df["low"], df["close"]
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (hh - close) / (hh - ll + 1e-10) * -100


def compute_donchian(df: pd.DataFrame, period: int = 20):
    """Donchian Channel: upper, middle, lower."""
    high, low = df["high"], df["low"]
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def compute_obv(df: pd.DataFrame):
    """On-Balance Volume."""
    close, vol = df["close"], df.get("vol", df.get("volume"))
    sign = np.sign(close.diff()).fillna(0)
    return (sign * vol).cumsum()


def compute_chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10):
    """Chaikin Oscillator: difference of ADL EMAs."""
    high, low, close, vol = df["high"], df["low"], df["close"], df.get("vol", df.get("volume"))
    hl_range = high - low
    mfm = ((close - low) - (high - close)) / (hl_range + 1e-10)
    mfv = mfm * vol
    adl = mfv.cumsum()
    return adl.ewm(span=fast, adjust=False).mean() - adl.ewm(span=slow, adjust=False).mean()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full technical indicator suite + counter-prediction features from OHLCV DataFrame."""
    close = df["close"]
    vol = df.get("vol", df.get("volume", pd.Series([0] * len(df))))
    result = pd.DataFrame(index=df.index)
    result["ts_code"] = df.get("ts_code", "")
    result["trade_date"] = df.get("trade_date", df.get("trade_time", ""))

    # Moving averages
    mas = compute_ma(close, [5, 10, 20, 60])
    for col in mas.columns:
        result[col] = mas[col]

    # RSI
    result["rsi6"] = compute_rsi(close, 6)
    result["rsi14"] = compute_rsi(close, 14)

    # MACD
    macd, sig, hist = compute_macd(close)
    result["macd"] = macd
    result["macd_signal"] = sig
    result["macd_hist"] = hist

    # Bollinger Bands
    bb_u, bb_m, bb_l = compute_bollinger(close)
    result["bb_upper"] = bb_u
    result["bb_middle"] = bb_m
    result["bb_lower"] = bb_l

    # ATR
    if all(c in df.columns for c in ["high", "low"]):
        result["atr14"] = compute_atr(df)

    # Volume ratio
    result["vol_ratio"] = vol / vol.rolling(20).mean().replace(0, 1)

    # KDJ
    if all(c in df.columns for c in ["high", "low", "close"]):
        k, d, j = compute_kdj(df)
        result["kdj_k"] = k
        result["kdj_d"] = d
        result["kdj_j"] = j

    # CCI (Commodity Channel Index)
    if all(c in df.columns for c in ["high", "low", "close"]):
        result["cci14"] = compute_cci(df, period=14)

    # Williams %R
    if all(c in df.columns for c in ["high", "low", "close"]):
        result["willr14"] = compute_williams_r(df, period=14)

    # Donchian Channels
    if all(c in df.columns for c in ["high", "low"]):
        dc_u, dc_m, dc_l = compute_donchian(df, period=20)
        result["dc_upper"] = dc_u
        result["dc_mid"] = dc_m
        result["dc_lower"] = dc_l

    # Rate of Change
    result["roc5"] = close.pct_change(5)
    result["roc10"] = close.pct_change(10)

    # OBV divergence
    result["obv"] = compute_obv(df)

    # Chaikin Oscillator
    if all(c in df.columns for c in ["high", "low", "close"]):
        result["chaikin_osc"] = compute_chaikin_oscillator(df)

    # Volume ROC
    result["vol_roc5"] = vol.pct_change(5)

    # Close location within high-low range
    if all(c in df.columns for c in ["high", "low"]):
        result["close_location"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

    # MA slopes (trend direction)
    for w in [20, 60]:
        ma_col = f"ma{w}"
        if ma_col in result.columns:
            ma_s = result[ma_col]
            result[f"ma{w}_slope"] = (ma_s - ma_s.shift(5)) / (ma_s.shift(5) + 1e-10)

    # Gap ratio (open vs prev close)
    result["gap_ratio"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-10)

    # Up/down volume ratio (last 5 bars)
    up_vol = vol.where(close > close.shift(1), 0).rolling(5).sum()
    down_vol = vol.where(close < close.shift(1), 0).rolling(5).sum()
    result["up_down_vol"] = up_vol / (down_vol + 1)

    # Intraday intensity (volatility × volume)
    if all(c in df.columns for c in ["high", "low"]):
        result["intraday_intensity"] = (df["high"] - df["low"]) / (df["close"] + 1e-10) * vol / vol.rolling(20).mean().replace(0, 1)

    # Counter-prediction: trader fingerprint features
    from data.counter_prediction import counter_trade_features
    cpf = counter_trade_features(df)
    for col in cpf.columns:
        result[col] = cpf[col].values

    # ── 量化特征注入 (Quant Feature Injection) ──
    # BB_Pos: 价格在布林带内的相对位置 [0,1]，反映超买超卖
    result["bb_pos"] = (close - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"] + 1e-8)

    # RSI 缩放到 [0,1] 区间，与 BB_Pos 对齐量纲
    result["rsi6_scaled"] = result["rsi6"] / 100.0
    result["rsi14_scaled"] = result["rsi14"] / 100.0

    # 极端 NaN/Inf 清洗 — 防止 PyTorch 梯度爆炸
    # limit=5 限制填充深度：不会跨越午休(>5根K线缺口)或跨日缺口
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result = result.ffill(limit=5).bfill(limit=5).fillna(0.0)

    return result
