"""Free stock data fetcher using baostock (TCP direct, no proxy issues)."""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import re
import os
import sys

logger = logging.getLogger("stock_pred")

FREQ_MAP = {"5min": "5", "15min": "15", "30min": "30", "60min": "60"}

_BS_CODE_RE = re.compile(r"^(sh|sz)\.\d{6}$")


class EFinanceFetcher:
    """Free stock data fetcher using baostock. No API key needed."""

    def __init__(self):
        self._call_times: list[float] = []
        self._logged_in: bool = False

    def _ensure_login(self):
        if self._logged_in:
            return
        import baostock as bs
        bs.login()
        self._logged_in = True

    def _rate_limit(self, min_interval: float = 1.0):
        now = time.time()
        if self._call_times:
            elapsed = now - self._call_times[-1]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._call_times.append(time.time())

    def _code_to_baostock(self, ts_code: str) -> str:
        code = ts_code.split(".")[0]
        if "." in ts_code:
            suffix = ts_code.split(".")[1].lower()
        else:
            if code.startswith(("6", "9")):
                suffix = "sh"
            else:
                suffix = "sz"
        return f"{suffix}.{code}"

    @staticmethod
    def _suppress_bs_query(fn, *args, **kwargs):
        """Suppress baostock's noisy stdout during query."""
        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, "w")
            return fn(*args, **kwargs)
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr

    def get_daily(self, ts_code: str, start_date: str | None = None,
                  end_date: str | None = None) -> pd.DataFrame:
        import baostock as bs
        self._ensure_login()
        self._rate_limit()
        bs_code = self._code_to_baostock(ts_code)
        if not _BS_CODE_RE.match(bs_code):
            return pd.DataFrame()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")

        rs = self._suppress_bs_query(
            bs.query_history_k_data_plus,
            bs_code, "date,code,open,high,low,close,volume,amount",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )

        if rs.error_code != "0":
            return pd.DataFrame()

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        df = df.rename(columns={"date": "trade_date", "volume": "vol"})
        df = df.drop(columns=["code"], errors="ignore")
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        for col in ["open", "high", "low", "close", "vol", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts_code"] = ts_code
        return df.sort_values("trade_date").reset_index(drop=True)

    def get_minutes(self, ts_code: str, freq: str = "5min",
                    days: int = 60) -> pd.DataFrame:
        import baostock as bs
        self._ensure_login()
        self._rate_limit()
        bs_code = self._code_to_baostock(ts_code)
        if not _BS_CODE_RE.match(bs_code):
            return pd.DataFrame()

        bs_freq = FREQ_MAP.get(freq, "5")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")

        rs = self._suppress_bs_query(
            bs.query_history_k_data_plus,
            bs_code, "date,time,code,open,high,low,close,volume,amount",
            start_date=start_date, end_date=end_date,
            frequency=bs_freq, adjustflag="2"
        )

        if rs.error_code != "0":
            return pd.DataFrame()

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        # baostock time field format: YYYYMMDDHHMMSSmmm → take first 14 chars
        df["trade_time"] = pd.to_datetime(
            df["time"].astype(str).str[:14],
            format="%Y%m%d%H%M%S"
        )
        df = df.drop(columns=["date", "time", "code"], errors="ignore")
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts_code"] = ts_code
        df["freq"] = freq
        return df.sort_values("trade_time").reset_index(drop=True)

    def fetch_recent_mins(self, ts_code: str, days: int = 60,
                          freq: str = "5min") -> pd.DataFrame:
        return self.get_minutes(ts_code, freq=freq, days=days)
