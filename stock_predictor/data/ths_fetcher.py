"""同花顺/东方财富数据源（通过 akshare）。"""
from __future__ import annotations
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("stock_pred")

FREQ_MAP = {"1min": "1", "5min": "5", "15min": "15", "30min": "30", "60min": "60"}


def _code6(ts_code: str) -> str:
    return ts_code.split(".")[0]


def _to_ts(code6: str) -> str:
    return f"{code6}.{'SH' if code6.startswith(('6','9')) else 'SZ'}"


class THSFetcher:
    def __init__(self):
        self._ok: bool | None = None

    def _has_ak(self) -> bool:
        if self._ok is None:
            try:
                import akshare  # noqa: F401
                self._ok = True
            except ImportError:
                self._ok = False
        return self._ok

    @property
    def source_name(self) -> str:
        return "同花顺/东方财富 (akshare)" if self._has_ak() else "无可用数据源"

    def get_daily(self, ts_code: str, start_date: str | None = None,
                  end_date: str | None = None) -> pd.DataFrame:
        if not self._has_ak():
            return pd.DataFrame()
        try:
            import akshare as ak
            sd = (start_date or (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")).replace("-", "")
            ed = (end_date or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
            df = ak.stock_zh_a_hist(symbol=_code6(ts_code), period="daily",
                                     start_date=sd, end_date=ed, adjust="qfq")
            if df.empty:
                return df
            df = df.rename(columns={"日期": "trade_date", "开盘": "open", "最高": "high",
                                     "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["ts_code"] = ts_code
            return df[["trade_date", "open", "high", "low", "close", "volume", "amount", "ts_code"]]
        except Exception as e:
            logger.warning(f"同花顺日线失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_minutes(self, ts_code: str, freq: str = "5min", days: int = 60) -> pd.DataFrame:
        if not self._has_ak():
            return pd.DataFrame()
        try:
            import akshare as ak
            period = FREQ_MAP.get(freq, "5")
            now = datetime.now()
            start = (now - timedelta(days=days + 5)).strftime("%Y-%m-%d 09:30:00")
            end = now.strftime("%Y-%m-%d 15:00:00")
            df = ak.stock_zh_a_hist_min_em(symbol=_code6(ts_code), period=period,
                                            start_date=start, end_date=end, adjust="qfq")
            if df.empty:
                return df
            df = df.rename(columns={"时间": "trade_time", "开盘": "open", "最高": "high",
                                     "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"})
            df["trade_time"] = pd.to_datetime(df["trade_time"])
            df = df[df["trade_time"] <= now]
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["ts_code"] = ts_code
            df["freq"] = freq
            return df[["trade_time", "open", "high", "low", "close", "volume", "amount", "ts_code", "freq"]]
        except Exception as e:
            logger.warning(f"同花顺分钟线失败 {ts_code}: {e}")
            return pd.DataFrame()

    def fetch_recent_mins(self, ts_code: str, days: int = 60, freq: str = "5min") -> pd.DataFrame:
        return self.get_minutes(ts_code, freq=freq, days=days)
