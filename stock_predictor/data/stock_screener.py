"""Dynamic stock screening: filters A-stocks by liquidity, trend, volatility, momentum.

Runs in a background thread (ScreeningWorker). Scoring formula:
    score = 0.35 × momentum + 0.30 × liquidity + 0.20 × volatility + 0.15 × trend_strength
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger("stock_pred")

MIN_DAILY_AMOUNT = 5_000_0000
MIN_LIST_DAYS = 60
MIN_ATR_RATIO = 0.02
TOP_N = 50
DAYS_LOOKBACK = 90


def _code_to_ts(code6: str) -> str:
    """Convert '600519' → '600519.SH'."""
    return f"{code6}.{'SH' if code6.startswith(('6', '9')) else 'SZ'}"


def _fetch_all_stocks() -> list[dict]:
    """Fetch all A-stock codes, filtering out ST, *ST."""
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    stocks = []
    for _, row in df.iterrows():
        code = str(row["代码"])
        name = str(row["名称"])
        if "ST" in name:
            continue
        stocks.append({"code": code, "name": name})
    return stocks


def _compute_metrics(df: pd.DataFrame) -> dict | None:
    if df.empty or len(df) < 20:
        return None

    df = df.sort_values("trade_date").tail(DAYS_LOOKBACK)
    close = df["close"].values.astype(float)
    amount = df.get("amount", pd.Series([0] * len(df))).values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)

    avg_amount = np.mean(amount[-20:]) if len(amount) >= 20 else np.mean(amount)
    if avg_amount < MIN_DAILY_AMOUNT:
        return None
    liquidity_score = min(np.log1p(avg_amount / 1e6) / 10, 1.0)

    ma20 = np.mean(close[-20:]) if n >= 20 else np.mean(close)
    ma60 = np.mean(close[-60:]) if n >= 60 else np.mean(close)
    trend_up = 1.0 if ma20 > ma60 else 0.3 if ma20 > ma60 * 0.95 else 0.0
    trend_strength = min((ma20 / max(ma60, 1e-10) - 1) * 10, 1.0)

    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])
    atr20 = np.mean(tr[-20:]) if len(tr) >= 20 else np.mean(tr)
    atr_ratio = atr20 / (close[-1] + 1e-10)
    if atr_ratio < MIN_ATR_RATIO:
        return None
    volatility_score = min(atr_ratio * 15, 1.0)

    ret5 = (close[-1] / max(close[-6], 1e-10) - 1) if n >= 6 else 0
    ret10 = (close[-1] / max(close[-11], 1e-10) - 1) if n >= 11 else 0
    ret20 = (close[-1] / max(close[-21], 1e-10) - 1) if n >= 21 else 0
    momentum = 0.4 * ret5 + 0.35 * ret10 + 0.25 * ret20
    momentum_score = min(max(momentum * 3 + 0.5, 0), 1.0)

    score = (
        0.35 * momentum_score +
        0.30 * liquidity_score +
        0.20 * volatility_score +
        0.15 * min(trend_strength * trend_up, 1.0)
    )

    ts_code = str(df["ts_code"].iloc[0])
    return {
        "ts_code": ts_code if "." in ts_code else _code_to_ts(ts_code),
        "name": df.get("name", pd.Series([""])).iloc[0] if "name" in df.columns else "",
        "score": round(score, 4),
        "momentum": round(momentum_score, 3),
        "liquidity": round(avg_amount / 1e8, 2),
        "volatility": round(atr_ratio * 100, 2),
        "trend": "up" if trend_up > 0.5 else "weak",
        "avg_amount": round(avg_amount / 1e8, 2),
        "atr_pct": round(atr_ratio * 100, 2),
    }


def screen_stocks(fetcher=None, top_n: int = TOP_N,
                  on_progress=None, on_phase=None) -> pd.DataFrame:
    def _emit_phase(msg):
        if on_phase:
            on_phase(msg)

    def _emit_progress(cur, total, df):
        if on_progress:
            on_progress(cur, total, df)

    _emit_phase("获取A股列表...")
    all_stocks = _fetch_all_stocks()
    _emit_phase(f"共 {len(all_stocks)} 只股票待筛选")

    import akshare as ak

    results: list[dict] = []
    total = len(all_stocks)
    batch: list[dict] = []
    start_time = time.time()

    for i, stock in enumerate(all_stocks):
        try:
            code = stock["code"]
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=DAYS_LOOKBACK + 20)).strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                     start_date=start_date, end_date=end_date, adjust="qfq")
            if df.empty or len(df) < 20:
                continue

            df = df.rename(columns={"日期": "trade_date", "开盘": "open", "最高": "high",
                                     "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["ts_code"] = code

            metrics = _compute_metrics(df)
            if metrics is not None:
                metrics["name"] = stock["name"]
                batch.append(metrics)

            if (i + 1) % 50 == 0:
                time.sleep(0.3)
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 1)
            eta = (total - i - 1) / max(rate, 0.01)
            _emit_phase(
                f"筛选进度: {i + 1}/{total} ({rate:.0f}只/秒, ETA {eta:.0f}秒) "
                f"| 已入选: {len(batch)}"
            )
            if batch:
                df_batch = pd.DataFrame(batch).sort_values("score", ascending=False).head(top_n * 2)
                _emit_progress(i + 1, total, df_batch.head(top_n))

    if not batch:
        _emit_phase("筛选完成：无符合条件的股票")
        return pd.DataFrame()

    result = pd.DataFrame(batch).sort_values("score", ascending=False).head(top_n)
    result["industry"] = "A股"
    result["is_screened"] = True
    result["list_date"] = None

    elapsed = time.time() - start_time
    _emit_phase(f"筛选完成: {len(result)}/{len(all_stocks)} 只入选 (耗时 {elapsed:.0f}秒)")
    _emit_progress(total, total, result)

    return result.reset_index(drop=True)
