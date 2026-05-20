"""多市场交易规则：A股/美股/港股 — 涨跌停板、临停机制、熔断、冷静期。

规则来源：
  A股 — 上交所/深交所/北交所 2026 年现行规则 + 2026-07-06 规则变更
  美股 — SEC Reg NMS (LULD), S&P 500 Circuit Breaker (Rule 80B)
  港股 — HKEX VCM (Volatility Control Mechanism)

规则索引:
  规则1:  涨跌停板 — A股主板±10%/科创±20%/创业±20%/北交±30%/ST±5%(7月6日后主板ST±10%)
  规则2:  T+1交割 — 当日买入次日方可卖出
  规则3:  交易时间 — 集合竞价9:15-9:25/连续竞价9:30-11:30,13:00-15:00
  规则4:  最小交易单位 — 100股(1手)
  规则5:  临时停牌 — 无涨跌幅限制股票首次达开盘±30%或±60%临停10分钟
  规则6:  有效申报范围 — 连续竞价买入≤基准价102%/卖出≥基准价98%
  规则7:  新股首日 — 无涨跌幅限制，临停机制适用
  规则8:  规则变更(2026-07-06) — ST±10%、盘后交易扩展至全市场
  规则9:  价格优先时间优先 — 撮合原则
  规则10: 委托价格边界 — 不得超过涨跌停范围
  规则11: 美股LULD熔断 — Tier1±5%/Tier2±10%，暂停5分钟
  规则12: 美股大盘熔断 — S&P500跌7%/13%/20%，暂停15分钟或休市
  规则13: 港股VCM冷静期 — 5分钟内波动>±10%(恒指)/±15%(其他)，5分钟限定区间
"""
from __future__ import annotations
from datetime import datetime, time, timedelta, date
from enum import Enum
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 交易时段
# ═══════════════════════════════════════════════════════════════

class TradingSession(Enum):
    CALL_AUCTION = "集合竞价"         # 9:15-9:25
    CONT_AUCTION = "连续竞价"          # 9:30-11:30, 13:00-14:57
    CLOSING_AUCTION = "收盘集合竞价"    # 14:57-15:00
    LUNCH_BREAK = "午间休市"           # 11:30-13:00
    AFTER_HOURS = "盘后固定价格"       # 15:05-15:30 (科创/创业板)
    CLOSED = "闭市"                    # 其他时间


def get_trading_session(now: datetime | None = None) -> TradingSession:
    """根据当前时间返回交易时段。"""
    if now is None:
        now = datetime.now()
    t = now.time()
    wd = now.weekday()  # 0=Monday

    # 周末闭市
    if wd >= 5:
        return TradingSession.CLOSED

    # 集合竞价 9:15-9:25
    if time(9, 15) <= t < time(9, 25):
        return TradingSession.CALL_AUCTION

    # 上午连续竞价 9:30-11:30
    if time(9, 30) <= t < time(11, 30):
        return TradingSession.CONT_AUCTION

    # 午休 11:30-13:00
    if time(11, 30) <= t < time(13, 0):
        return TradingSession.LUNCH_BREAK

    # 下午连续竞价 13:00-14:57
    if time(13, 0) <= t < time(14, 57):
        return TradingSession.CONT_AUCTION

    # 收盘集合竞价 14:57-15:00
    if time(14, 57) <= t < time(15, 0):
        return TradingSession.CLOSING_AUCTION

    # 盘后固定价格 15:05-15:30
    if time(15, 5) <= t < time(15, 30):
        return TradingSession.AFTER_HOURS

    return TradingSession.CLOSED


def is_trading_time(now: datetime | None = None) -> bool:
    """连续竞价时段（适合下单交易）。不包含集合竞价和盘后。"""
    return get_trading_session(now) == TradingSession.CONT_AUCTION


def is_valid_trading_time(dt: datetime) -> bool:
    """Check if a datetime falls within valid A-share continuous auction hours.

    Valid hours (Beijing time):
      - 9:30-11:30 (morning session)
      - 13:00-14:57 (afternoon continuous auction)
      - 14:57-15:00 (closing auction — included as valid)
    Excludes: weekends, lunch break (11:30-13:00), pre-market, after-hours.
    """
    if dt.weekday() >= 5:  # Saturday/Sunday
        return False
    t = dt.time()
    # Morning session: 9:30-11:30 (11:30 inclusive — last bar of the morning)
    if time(9, 30) <= t <= time(11, 30):
        return True
    # Afternoon session: 13:00-15:00
    if time(13, 0) <= t <= time(15, 0):
        return True
    return False


def filter_trading_hours(df: "pd.DataFrame", time_col: str = "trade_time") -> "pd.DataFrame":
    """Remove rows whose timestamp falls outside valid A-share trading hours.

    This strips lunch-break bars, pre-market noise, after-hours artifacts,
    and weekend data that may leak from upstream APIs or data merges.
    """
    if df.empty or time_col not in df.columns:
        return df
    times = pd.to_datetime(df[time_col])
    mask = times.apply(is_valid_trading_time)
    return df[mask.values].copy()


def get_session_description(now: datetime | None = None) -> str:
    """返回当前时段的详细中文描述。"""
    session = get_trading_session(now)
    descriptions = {
        TradingSession.CALL_AUCTION: "集合竞价时段 (9:15-9:25)，9:20前可撤单，9:20-9:25不可撤单",
        TradingSession.CONT_AUCTION: "连续竞价时段，可正常下单交易",
        TradingSession.CLOSING_AUCTION: "收盘集合竞价 (14:57-15:00)，以收盘价成交",
        TradingSession.LUNCH_BREAK: "午间休市，13:00恢复连续竞价",
        TradingSession.AFTER_HOURS: "盘后固定价格交易 (15:05-15:30)，仅科创板/创业板",
        TradingSession.CLOSED: "闭市时段，请等待下一交易日开盘",
    }
    return descriptions[session]


# ═══════════════════════════════════════════════════════════════
# 交易制度
# ═══════════════════════════════════════════════════════════════

def is_t_plus_one() -> bool:
    """A股实行T+1交割制度：当日买入的股票，次日才能卖出。"""
    return True


def get_min_lot_size() -> int:
    """最小交易单位：100股（1手）。"""
    return 100


def round_to_lot(shares: int) -> int:
    """将股数圆整到1手（100股）的整数倍。"""
    lot = get_min_lot_size()
    return max(shares // lot * lot, lot)


def get_auction_times() -> dict:
    """集合竞价规则。"""
    return {
        "morning_start": "9:15",
        "morning_cancel_deadline": "9:20",   # 9:20后不可撤单
        "morning_end": "9:25",
        "description": "9:15-9:20可申报可撤单，9:20-9:25只可申报不可撤单，9:25产生开盘价",
    }


# ═══════════════════════════════════════════════════════════════
# 涨跌停板
# ═══════════════════════════════════════════════════════════════

def get_price_limit_pct(ts_code: str, as_of: date | None = None) -> float:
    """返回该股票的日涨跌幅限制比例。

    规则1: A股涨跌停板制度 (含2026-07-06规则变更)
      主板(600/601/603/000/001/002): ±10%
      科创板(688): ±20%
      创业板(300/301): ±20%
      北交所(8xx): ±30%
      ST/*ST:
        - 2026-07-06前: ±5%
        - 2026-07-06起 主板ST: ±10%（规则8）
        - 创业板/科创板ST: 维持±20%
    """
    code = ts_code.split(".")[0] if "." in ts_code else ts_code

    if "ST" in ts_code.upper() or "st" in ts_code:
        if as_of is None:
            as_of = date.today()
        # 2026-07-06 规则变更：主板ST涨跌幅由±5%放宽至±10%
        rule_change_date = date(2026, 7, 6)
        if as_of >= rule_change_date and not code.startswith("300") and not code.startswith("301") and not code.startswith("688"):
            return 0.10  # 主板ST放宽后
        return 0.05  # 其他ST或变更前

    if code.startswith("8"):
        return 0.30
    if code.startswith("688"):
        return 0.20
    if code.startswith("300") or code.startswith("301"):
        return 0.20
    return 0.10


def get_effective_price_range(current_price: float, ts_code: str) -> dict:
    """规则6: 连续竞价阶段有效申报价格范围。

    科创板/创业板(注册制): 买入≤基准价×102%, 卖出≥基准价×98%
    主板: 无2%限制, 按涨跌停板范围
    北交所: 同主板
    """
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    is_kechuang = code.startswith("688")
    is_chuangye = code.startswith("300") or code.startswith("301")

    if is_kechuang or is_chuangye:
        return {
            "buy_upper": round(current_price * 1.02, 2),
            "sell_lower": round(current_price * 0.98, 2),
            "rule": "规则6: 注册制板块连续竞价有效申报范围 — 买入≤基准价102%，卖出≥基准价98%",
        }
    return {
        "buy_upper": round(current_price * (1 + get_price_limit_pct(ts_code)), 2),
        "sell_lower": round(current_price * (1 - get_price_limit_pct(ts_code)), 2),
        "rule": "规则1: 涨跌停板范围",
    }


def get_temporary_halting_rules(ts_code: str) -> dict:
    """规则5: A股临时停牌机制。

    适用于无涨跌幅限制的股票（IPO前5日、并购重组复牌首日等）:
      - 盘中价较当日开盘价首次涨跌≥30%，临停10分钟
      - 盘中价较当日开盘价首次涨跌≥60%，临停10分钟
    创业板/科创板注册制新股首5日适用；主板新股首日适用。
    """
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    is_kechuang = code.startswith("688")
    is_chuangye = code.startswith("300") or code.startswith("301")
    is_beijiao = code.startswith("8")

    if is_kechuang or is_chuangye:
        return {
            "has_halting": True,
            "thresholds": [0.30, 0.60],
            "duration_minutes": 10,
            "rule": "规则5: 无涨跌幅限制股票 — 较开盘价涨跌≥30%或≥60%时，临停10分钟（注册制板块）",
        }
    if is_beijiao:
        return {
            "has_halting": True,
            "thresholds": [0.30, 0.60],
            "duration_minutes": 10,
            "rule": "规则5: 北交所新股首日 — 较开盘价涨跌≥30%或≥60%时，临停10分钟",
        }
    return {
        "has_halting": True,  # 主板新股首日也有临停
        "thresholds": [0.30, 0.60],
        "duration_minutes": 10,
        "rule": "规则5: 无涨跌幅限制股票 — 较开盘价涨跌≥30%或≥60%时，临停10分钟",
    }


# ═══════════════════════════════════════════════════════════════
# 美股交易规则
# ═══════════════════════════════════════════════════════════════

def get_us_luld_rules(stock_price: float, tier: int | None = None) -> dict:
    """规则11: 美股 LULD (Limit Up/Limit Down) 个股熔断机制。

    SEC Reg NMS: 根据股价和流动性分档
      Tier 1 (S&P500/Russell1000): ±5% 触发，暂停5分钟
      Tier 2 (其他): ±10% 触发，暂停5分钟
    交易时段：9:30-9:45及15:35-16:00放宽至双倍阈值
    盘前/盘后不适用
    """
    if tier is None:
        tier = 1 if stock_price > 3.0 else 2

    if tier == 1:
        pct = 0.05
        desc = "Tier1(S&P500/R1000成分股) ±5%"
    else:
        pct = 0.10
        desc = "Tier2(其他) ±10%"

    return {
        "market": "美股",
        "mechanism": "LULD (Limit Up/Limit Down)",
        "tier": tier,
        "threshold_pct": pct,
        "pause_minutes": 5,
        "upper_limit": round(stock_price * (1 + pct), 2),
        "lower_limit": round(stock_price * (1 - pct), 2),
        "rule": f"规则11: 美股LULD — {desc}，超限暂停5分钟",
    }


def get_us_market_circuit_breaker(sp500_change_pct: float) -> dict:
    """规则12: 美股大盘熔断 (Market-Wide Circuit Breaker, Rule 80B)。

    以S&P500为基准，触发阈值:
      Level 1: S&P500跌7%  → 暂停交易15分钟
      Level 2: S&P500跌13% → 暂停交易15分钟
      Level 3: S&P500跌20% → 当日剩余时间休市
    交易结束前35分钟内触发Level1/2不暂停。
    """
    abs_change = abs(sp500_change_pct)
    if sp500_change_pct <= -0.20:
        return {
            "level": 3,
            "trigger": "-20%",
            "action": "当日休市",
            "rule": "规则12: S&P500跌20% — 触发三级熔断，当日剩余时间休市",
        }
    elif sp500_change_pct <= -0.13:
        return {
            "level": 2,
            "trigger": "-13%",
            "action": "暂停15分钟",
            "rule": "规则12: S&P500跌13% — 触发二级熔断，暂停交易15分钟",
        }
    elif sp500_change_pct <= -0.07:
        return {
            "level": 1,
            "trigger": "-7%",
            "action": "暂停15分钟",
            "rule": "规则12: S&P500跌7% — 触发一级熔断，暂停交易15分钟",
        }
    else:
        return {
            "level": 0,
            "trigger": "无",
            "action": "正常交易",
            "rule": "规则12: 未触发熔断",
        }


# ═══════════════════════════════════════════════════════════════
# 港股交易规则
# ═══════════════════════════════════════════════════════════════

def get_hk_vcm_rules(stock_price: float, is_hsi_constituent: bool = False) -> dict:
    """规则13: 港股 VCM (Volatility Control Mechanism) 冷静期机制。

    触发条件: 5分钟内价格波动超过阈值
      恒生指数成分股: ±10%
      其他股票: ±15%
    触发后: 进入5分钟冷静期，价格限定在触发价±阈值区间内
    冷静期内: 交易继续但价格不得超出限定区间
    每天每个方向最多触发2次(早市/午市各1次)
    """
    if is_hsi_constituent:
        pct = 0.10
        desc = "恒指成分股 ±10%"
    else:
        pct = 0.15
        desc = "非恒指成分股 ±15%"

    return {
        "market": "港股",
        "mechanism": "VCM (冷静期)",
        "is_hsi": is_hsi_constituent,
        "threshold_pct": pct,
        "cooling_minutes": 5,
        "upper_limit": round(stock_price * (1 + pct), 2),
        "lower_limit": round(stock_price * (1 - pct), 2),
        "rule": f"规则13: 港股VCM — 5分钟内波动>{desc}触发5分钟冷静期，价格限定在参考价±{pct:.0%}区间",
    }


def check_hk_vcm_trigger(reference_price: float, current_price: float,
                         is_hsi: bool = False) -> dict:
    """检查港股是否触发VCM冷静期。

    reference_price: 5分钟前的参考价
    current_price: 当前价格
    返回是否触发及限定区间
    """
    rules = get_hk_vcm_rules(reference_price, is_hsi)
    change_pct = (current_price - reference_price) / (reference_price + 1e-10)

    if abs(change_pct) > rules["threshold_pct"]:
        direction = "上涨" if change_pct > 0 else "下跌"
        return {
            "triggered": True,
            "direction": direction,
            "change_pct": round(change_pct * 100, 2),
            "allowed_range": [
                round(reference_price * (1 - rules["threshold_pct"]), 2),
                round(reference_price * (1 + rules["threshold_pct"]), 2),
            ],
            "remaining_minutes": rules["cooling_minutes"],
            "rule": rules["rule"],
        }
    return {
        "triggered": False,
        "change_pct": round(change_pct * 100, 2),
        "rule": "港股VCM未触发",
    }


def is_new_stock_listing_day(ts_code: str, listing_date: str | None = None) -> bool:
    """规则7: 判断是否为新股上市首日（首日无涨跌幅限制）。

    新股上市首日不设涨跌幅限制，但有临时停牌机制。
    实际判断需要数据库中的上市日期 — 此处提供接口。
    """
    if listing_date is None:
        return False
    try:
        ld = datetime.strptime(listing_date, "%Y-%m-%d")
        return ld.date() == datetime.now().date()
    except (ValueError, TypeError):
        return False


def is_post_market_available(ts_code: str, as_of: date | None = None) -> bool:
    """规则8: 盘后固定价格交易是否可用于该股票。

    2026-07-06前: 仅科创板(688)/创业板(300/301)
    2026-07-06起: 扩展至全市场A股及ETF
    """
    if as_of is None:
        as_of = date.today()
    rule_change_date = date(2026, 7, 6)
    if as_of >= rule_change_date:
        return True  # 全市场可用
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    return code.startswith("688") or code.startswith("300") or code.startswith("301")


def get_board_name(ts_code: str) -> str:
    """返回股票所属板块中文名称。"""
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    if code.startswith("688"):
        return "科创板"
    if code.startswith("300") or code.startswith("301"):
        return "创业板"
    if code.startswith("8"):
        return "北交所"
    if code.startswith("6"):
        return "沪市主板"
    if code.startswith("00") or code.startswith("001") or code.startswith("002"):
        return "深市主板"
    return "主板"


# ═══════════════════════════════════════════════════════════════
# 价格裁剪（带规则引用）
# ═══════════════════════════════════════════════════════════════

def clip_price_delta(price_delta: float, ts_code: str) -> float:
    """规则6+规则1: 裁剪预测变动幅度到有效申报范围(2%)，再受涨跌停限制。

    优先使用连续竞价有效申报范围（2%），防止预测单笔拉到涨跌停。
    """
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    # 注册制板块(科创/创业): 连续竞价有效申报±2%
    is_registration = code.startswith("688") or code.startswith("300") or code.startswith("301")
    effective_limit = 0.02 if is_registration else get_price_limit_pct(ts_code)
    price_limit = get_price_limit_pct(ts_code)
    # 取更严格的那个
    limit = min(effective_limit, price_limit)
    return max(-limit, min(limit, price_delta))


def clip_price_to_limit(current_price: float, target_price: float, ts_code: str) -> dict:
    """规则1: 将预测目标价裁剪到该股涨跌停板范围内，并记录规则引用。"""
    limit = get_price_limit_pct(ts_code)
    upper = current_price * (1 + limit)
    lower = current_price * (1 - limit)
    original = target_price
    clipped = max(lower, min(upper, target_price))
    was_clipped = abs(clipped - original) > 0.01
    board = get_board_name(ts_code)

    return {
        "adjusted": round(clipped, 2),
        "original": round(original, 2),
        "limit_pct": limit,
        "upper_bound": round(upper, 2),
        "lower_bound": round(lower, 2),
        "was_clipped": was_clipped,
        "board": board,
        "rule": f"规则1: {board}涨跌停板为±{limit:.0%}" + (" — 预测价已裁剪至合理范围" if was_clipped else " — 预测价在合理范围内"),
    }


# ═══════════════════════════════════════════════════════════════
# 完整的规则摘要（供 AI 分析和 GUI 展示）
# ═══════════════════════════════════════════════════════════════

def get_all_rules_summary(ts_code: str, market: str = "A") -> dict:
    """返回该股票适用的全部规则摘要，供 AI 分析和 GUI 展示。

    market: "A"=A股, "US"=美股, "HK"=港股
    """
    board = get_board_name(ts_code)
    limit_pct = get_price_limit_pct(ts_code)
    session = get_trading_session()
    halting = get_temporary_halting_rules(ts_code)
    price_range = get_effective_price_range(100.0, ts_code)

    base = {
        "ts_code": ts_code,
        "board": board,
        "market": "A股" if market == "A" else ("美股" if market == "US" else "港股"),
        "price_limit": f"规则1: {board}涨跌停板 ±{limit_pct:.0%}",
        "halting": halting["rule"],
        "price_range": price_range["rule"],
        "trading_session": get_session_description(session),
        "session_enum": session.value,
    }

    if market == "A":
        base["t_plus_one"] = "规则2: T+1交割 — 当日买入次日方可卖出"
        base["min_lot"] = f"规则4: 最小交易单位 — {get_min_lot_size()}股（1手）"
    elif market == "US":
        us_rules = get_us_luld_rules(100.0)
        base["luld"] = us_rules["rule"]
        base["circuit_breaker"] = "规则12: 美股大盘熔断 — S&P500跌7%/13%/20%触发暂停或休市"
    elif market == "HK":
        hk_rules = get_hk_vcm_rules(100.0)
        base["vcm"] = hk_rules["rule"]
        base["min_lot"] = "规则4: 港股每手股数因股票而异 (100/200/500/1000等)"

    return base
