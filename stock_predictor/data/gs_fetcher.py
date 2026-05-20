"""金太阳 (Golden Sun) / 通达信 TDX 实时行情数据接口。

金太阳是国信证券的交易平台，底层使用通达信协议。
支持两种连接方式：
  1. 本地金太阳终端 — 127.0.0.1:7709
  2. 公共 TDX 行情服务器 — 119.147.212.81:7709 (上海), 119.147.171.206:7709 (深圳)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os
import time

logger = logging.getLogger("stock_pred")

# TDX freq → category mapping
TDX_CATEGORY = {"1min": 8, "5min": 0, "15min": 1, "30min": 2, "60min": 3, "daily": 4}

# 热门 A 股 — 启动时预加载
POPULAR_STOCKS = [
    # 沪深300 权重+成交量热门
    ("000001.SZ", "平安银行"), ("000002.SZ", "万科A"), ("000063.SZ", "中兴通讯"),
    ("000333.SZ", "美的集团"), ("000568.SZ", "泸州老窖"), ("000651.SZ", "格力电器"),
    ("000725.SZ", "京东方A"), ("000858.SZ", "五粮液"), ("002142.SZ", "宁波银行"),
    ("002230.SZ", "科大讯飞"), ("002415.SZ", "海康威视"), ("002594.SZ", "比亚迪"),
    ("002714.SZ", "牧原股份"), ("300015.SZ", "爱尔眼科"), ("300059.SZ", "东方财富"),
    ("300124.SZ", "汇川技术"), ("300274.SZ", "阳光电源"), ("300750.SZ", "宁德时代"),
    ("600000.SH", "浦发银行"), ("600009.SH", "上海机场"), ("600016.SH", "民生银行"),
    ("600028.SH", "中国石化"), ("600030.SH", "中信证券"), ("600031.SH", "三一重工"),
    ("600036.SH", "招商银行"), ("600050.SH", "中国联通"), ("600085.SH", "同仁堂"),
    ("600104.SH", "上汽集团"), ("600276.SH", "恒瑞医药"), ("600309.SH", "万华化学"),
    ("600519.SH", "贵州茅台"), ("600547.SH", "山东黄金"), ("600570.SH", "恒生电子"),
    ("600585.SH", "海螺水泥"), ("600588.SH", "用友网络"), ("600690.SH", "海尔智家"),
    ("600809.SH", "山西汾酒"), ("600837.SH", "海通证券"), ("600887.SH", "伊利股份"),
    ("600900.SH", "长江电力"), ("601012.SH", "隆基绿能"), ("601088.SH", "中国神华"),
    ("601111.SH", "中国国航"), ("601166.SH", "兴业银行"), ("601288.SH", "农业银行"),
    ("601318.SH", "中国平安"), ("601328.SH", "交通银行"), ("601398.SH", "工商银行"),
    ("601601.SH", "中国太保"), ("601628.SH", "中国人寿"), ("601668.SH", "中国建筑"),
    ("601857.SH", "中国石油"), ("601888.SH", "中国中免"), ("601899.SH", "紫金矿业"),
    ("601919.SH", "中远海控"), ("601988.SH", "中国银行"), ("603259.SH", "药明康德"),
    ("603288.SH", "海天味业"), ("603993.SH", "洛阳钼业"), ("688981.SH", "中芯国际"),
    ("688111.SH", "金山办公"), ("688256.SH", "寒武纪"),
]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
CACHE_FILE = os.path.join(CACHE_DIR, "tdx_stock_cache.json")


def _load_stock_cache() -> list[dict]:
    """加载本地缓存的完整股票列表。"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_stock_cache(stocks: list[tuple[str, str]]):
    """保存完整股票列表到本地缓存。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump([{"code": c, "name": n} for c, n in stocks], f, ensure_ascii=False)
    except Exception:
        pass


# Market codes: 0=深圳 1=上海
def _to_tdx_market(ts_code: str) -> int:
    suffix = ts_code.split(".")[-1].upper() if "." in ts_code else ("SH" if ts_code.startswith(("6", "9")) else "SZ")
    return 0 if suffix == "SZ" else 1

def _to_tdx_code(ts_code: str) -> str:
    return ts_code.split(".")[0]


class GoldenSunFetcher:
    """实时行情数据接口 — 优先金太阳本地服务器，回退到公共 TDX 服务器。"""

    # TDX 行情服务器 (来自金太阳 connect.cfg)
    PUBLIC_SERVERS = [
        ("109.244.35.28", 7709),    # 腾讯云广州
        ("101.133.231.193", 7709),  # 阿里云
        ("120.79.210.76", 7709),    # 阿里云
        ("109.244.73.13", 7709),    # 腾讯云
        ("162.14.135.116", 7709),   # 腾讯云广州2
        ("120.234.57.15", 7709),    # 东莞
        ("175.6.43.87", 7709),      # 长沙
        ("218.6.170.91", 7709),     # 成都
        ("123.125.108.9", 7709),    # 北京联通
        ("182.118.8.6", 7709),      # 郑州联通
        ("182.131.3.228", 7709),    # 成都联通
    ]

    def __init__(self, host: str = "127.0.0.1", port: int = 7709,
                 use_public: bool = False):
        self._host = host
        self._port = port
        self._use_public = use_public
        self._api = None
        self._connected = False
        self._current_ip = None
        self._call_times: list[float] = []

    def _rate_limit(self, min_interval: float = 0.3):
        now = time.time()
        if self._call_times:
            elapsed = now - self._call_times[-1]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._call_times.append(time.time())

    def _connect(self) -> bool:
        """连接 TDX 服务器，优先本地再公共。"""
        if self._connected and self._api is not None:
            return True

        try:
            from pytdx.hq import TdxHq_API
        except ImportError:
            logger.warning("pytdx 未安装，金太阳不可用。请运行: pip install pytdx")
            return False
        api = TdxHq_API()

        # Try local server first
        if not self._use_public:
            try:
                if api.connect(self._host, self._port):
                    self._api = api
                    self._connected = True
                    self._current_ip = f"{self._host}:{self._port}"
                    logger.info(f"金太阳已连接: {self._current_ip}")
                    return True
            except Exception:
                pass

        # Try public servers
        for ip, port in self.PUBLIC_SERVERS:
            try:
                # Recreate api for each attempt
                if api is None:
                    api = TdxHq_API()
                if api.connect(ip, port):
                    self._api = api
                    self._connected = True
                    self._current_ip = f"{ip}:{port}"
                    logger.info(f"TDX 公共服务器已连接: {self._current_ip}")
                    return True
            except Exception:
                if api is not None:
                    try:
                        api.disconnect()
                    except Exception:
                        pass
                api = None
                continue

        logger.warning("无法连接金太阳或 TDX 公共服务器")
        return False

    def _disconnect(self):
        if self._api:
            try:
                self._api.disconnect()
            except Exception:
                pass
        self._api = None
        self._connected = False
        self._current_ip = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_realtime_quote(self, ts_code: str) -> dict | None:
        """获取单只股票的实时行情 (Level-1)。

        返回字典包含: open, high, low, price (最新价), volume, amount,
        bid1-5, ask1-5, bid_vol1-5, ask_vol1-5
        """
        if not self._connect():
            return None
        self._rate_limit(0.2)
        try:
            market = _to_tdx_market(ts_code)
            code = _to_tdx_code(ts_code)
            quotes = self._api.get_security_quotes([(market, code)])
            if not quotes:
                return None
            q = quotes[0]
            return {
                "ts_code": ts_code,
                "open": float(q.get("open", 0)),
                "high": float(q.get("high", 0)),
                "low": float(q.get("low", 0)),
                "price": float(q.get("price", 0)),
                "volume": float(q.get("cur_vol", 0) or q.get("volume", 0)),
                "amount": float(q.get("amount", 0)),
                "pre_close": float(q.get("last_close", 0)),
                "bid1": float(q.get("bid1", 0)),
                "bid2": float(q.get("bid2", 0)),
                "bid3": float(q.get("bid3", 0)),
                "bid4": float(q.get("bid4", 0)),
                "bid5": float(q.get("bid5", 0)),
                "ask1": float(q.get("ask1", 0)),
                "ask2": float(q.get("ask2", 0)),
                "ask3": float(q.get("ask3", 0)),
                "ask4": float(q.get("ask4", 0)),
                "ask5": float(q.get("ask5", 0)),
                "bid_vol1": float(q.get("bid_vol1", 0)),
                "bid_vol2": float(q.get("bid_vol2", 0)),
                "bid_vol3": float(q.get("bid_vol3", 0)),
                "bid_vol4": float(q.get("bid_vol4", 0)),
                "bid_vol5": float(q.get("bid_vol5", 0)),
                "ask_vol1": float(q.get("ask_vol1", 0)),
                "ask_vol2": float(q.get("ask_vol2", 0)),
                "ask_vol3": float(q.get("ask_vol3", 0)),
                "ask_vol4": float(q.get("ask_vol4", 0)),
                "ask_vol5": float(q.get("ask_vol5", 0)),
            }
        except Exception as e:
            logger.warning(f"获取实时行情失败 {ts_code}: {e}")
            return None

    def get_batch_quotes(self, ts_codes: list[str]) -> list[dict]:
        """批量获取实时行情 (一次请求多只股票)。"""
        if not self._connect():
            return []
        self._rate_limit(0.3)
        results = []
        try:
            tdx_codes = [(_to_tdx_market(c), _to_tdx_code(c)) for c in ts_codes]
            quotes = self._api.get_security_quotes(tdx_codes)
            if not quotes:
                return []
            for i, q in enumerate(quotes):
                ts_code = ts_codes[i] if i < len(ts_codes) else ""
                results.append({
                    "ts_code": ts_code,
                    "open": float(q.get("open", 0)),
                    "high": float(q.get("high", 0)),
                    "low": float(q.get("low", 0)),
                    "price": float(q.get("price", 0)),
                    "volume": float(q.get("cur_vol", 0) or q.get("volume", 0)),
                    "amount": float(q.get("amount", 0)),
                    "pre_close": float(q.get("last_close", 0)),
                    "bid1": float(q.get("bid1", 0)),
                    "ask1": float(q.get("ask1", 0)),
                })
            return results
        except Exception as e:
            logger.warning(f"批量获取行情失败: {e}")
            return []

    def get_minutes(self, ts_code: str, freq: str = "5min",
                    days: int = 60, count: int = 800) -> pd.DataFrame:
        """获取历史分钟 K 线。

        TDX 支持最多 800 根最新 K 线。对于 5min 频率 800 根约 67 小时。
        """
        if not self._connect():
            return pd.DataFrame()
        self._rate_limit(0.3)

        market = _to_tdx_market(ts_code)
        code = _to_tdx_code(ts_code)
        category = TDX_CATEGORY.get(freq, 0)

        try:
            bars = self._api.get_security_bars(category, market, code, 0, min(count, 800))
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            df = df.rename(columns={
                "year": "year", "month": "month", "day": "day",
                "hour": "hour", "minute": "minute",
            })
            # Build trade_time from year/month/day/hour/minute
            def _build_time(row):
                try:
                    return pd.Timestamp(
                        year=int(row["year"]), month=int(row["month"]),
                        day=int(row["day"]), hour=int(row["hour"]),
                        minute=int(row.get("minute", 0))
                    )
                except Exception:
                    return pd.NaT
            df["trade_time"] = df.apply(_build_time, axis=1)
            # Keep only valid rows
            df = df.dropna(subset=["trade_time"])
            # Filter: drop future timestamps and keep only data within requested days
            now = datetime.now()
            cutoff = now - timedelta(days=days + 5)
            df = df[(df["trade_time"] >= cutoff) & (df["trade_time"] <= now)]

            df = df.rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "vol": "volume", "amount": "amount",
            })
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["ts_code"] = ts_code
            df["freq"] = freq
            df = df.drop(columns=["year", "month", "day", "hour", "minute"], errors="ignore")
            return df.sort_values("trade_time").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"获取分钟数据失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_daily(self, ts_code: str, start_date: str | None = None,
                  end_date: str | None = None, count: int = 800) -> pd.DataFrame:
        """获取日线 K 线数据。"""
        if not self._connect():
            return pd.DataFrame()
        self._rate_limit(0.3)

        market = _to_tdx_market(ts_code)
        code = _to_tdx_code(ts_code)

        try:
            bars = self._api.get_security_bars(4, market, code, 0, min(count, 800))
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            def _build_date(row):
                try:
                    return f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}"
                except Exception:
                    return None
            df["trade_date"] = df.apply(_build_date, axis=1)
            df = df.dropna(subset=["trade_date"])
            if start_date:
                df = df[df["trade_date"] >= start_date]
            if end_date:
                df = df[df["trade_date"] <= end_date]

            df = df.rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "amount": "amount",
            })
            for col in ["open", "high", "low", "close", "vol", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["ts_code"] = ts_code
            df = df.drop(columns=["year", "month", "day", "hour", "minute"], errors="ignore")
            return df.sort_values("trade_date").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"获取日线数据失败 {ts_code}: {e}")
            return pd.DataFrame()

    def fetch_recent_mins(self, ts_code: str, days: int = 60,
                          freq: str = "5min") -> pd.DataFrame:
        """便捷方法：获取近期分钟数据。"""
        return self.get_minutes(ts_code, freq=freq, days=days)

    def get_index_bars(self, ts_code: str, freq: str = "5min",
                       count: int = 800) -> pd.DataFrame:
        """获取指数 K 线 (上证/深证等)。"""
        if not self._connect():
            return pd.DataFrame()
        self._rate_limit(0.3)

        market = _to_tdx_market(ts_code)
        code = _to_tdx_code(ts_code)
        category = TDX_CATEGORY.get(freq, 0)

        try:
            bars = self._api.get_index_bars(category, market, code, 0, min(count, 800))
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            def _build_time(row):
                try:
                    return pd.Timestamp(
                        year=int(row["year"]), month=int(row["month"]),
                        day=int(row["day"]), hour=int(row["hour"]),
                        minute=int(row.get("minute", 0))
                    )
                except Exception:
                    return pd.NaT
            df["trade_time"] = df.apply(_build_time, axis=1)
            df = df.dropna(subset=["trade_time"])
            for col in ["open", "high", "low", "close", "vol", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.rename(columns={"vol": "volume"})
            df["ts_code"] = ts_code
            df = df.drop(columns=["year", "month", "day", "hour", "minute"], errors="ignore")
            return df.sort_values("trade_time").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"获取指数数据失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_all_stocks(self) -> list[tuple[str, str]]:
        """获取全部 A 股列表 (ts_code, name)。结果缓存到本地 JSON。"""
        if not self._connect():
            return []
        self._rate_limit(0.05)

        a_sz_prefixes = ("000", "001", "002", "003")
        a_cy_prefixes = ("300", "301")
        a_sh_prefixes = ("600", "601", "603", "605")
        a_kcb_prefixes = ("688", "689")

        results = []
        for mkt in (0, 1):
            total = self._api.get_security_count(mkt)
            for start in range(0, total, 1000):
                try:
                    batch = self._api.get_security_list(mkt, start)
                except Exception:
                    continue
                if not batch:
                    continue  # skip empty batches (some TDX servers return None for index 0)
                for s in batch:
                    code = s.get("code", "")
                    name = s.get("name", "")
                    if not name or not code:
                        continue
                    if mkt == 0:
                        if code.startswith(a_sz_prefixes) or code.startswith(a_cy_prefixes):
                            results.append((f"{code}.SZ", name))
                    else:
                        if code.startswith(a_sh_prefixes) or code.startswith(a_kcb_prefixes):
                            results.append((f"{code}.SH", name))
        if results:
            _save_stock_cache(results)
        return results

    def ensure_stock_cache(self):
        """确保本地缓存有完整股票列表（首次调用时从 TDX 拉取全量）。"""
        cached = _load_stock_cache()
        if len(cached) >= 1000:
            return cached
        stocks = self.get_all_stocks()
        return _load_stock_cache()

    def search_stock(self, keyword: str) -> tuple[str, str] | None:
        """智能搜索股票：精确代码 > 代码前缀 > 名称开头 > 名称包含。
        如缓存未就绪则先构建。返回 (ts_code, name) 或 None。"""
        if not keyword:
            return None
        kw = keyword.strip().upper()
        cached = self.ensure_stock_cache()

        # Priority tiers
        exact_code = []       # code == kw (e.g. "600760.SH")
        code_prefix = []      # code starts with kw (e.g. "60076")
        name_starts = []      # name starts with kw
        name_contains = []    # name contains kw
        code_contains = []    # code contains kw (e.g. "007" in "000007.SZ")

        for entry in cached:
            code = entry.get("code", "")
            name = entry.get("name", "")
            code_u = code.upper()
            name_u = name.upper()

            if code_u == kw:
                exact_code.append(entry)
            elif code_u.startswith(kw):
                code_prefix.append(entry)
            elif name_u.startswith(kw):
                name_starts.append(entry)
            elif kw in name_u:
                name_contains.append(entry)
            elif kw in code_u:
                code_contains.append(entry)

        for bucket in (exact_code, code_prefix, name_starts, name_contains, code_contains):
            if bucket:
                # Prefer main-board A stocks over others within same bucket
                bucket.sort(key=lambda e: (
                    0 if e["code"][:3] in ("600", "601", "603", "000", "001", "002", "300", "688") else 1
                ))
                entry = bucket[0]
                return (entry["code"], entry["name"])
        return None

    def __del__(self):
        self._disconnect()
