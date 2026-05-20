"""News impact analyzer: classify impact, estimate magnitude and duration.

Keyword-based classification for Chinese A-share news without heavy NLP deps.
"""

from __future__ import annotations
from datetime import datetime, timedelta

# ── Direction × Magnitude keyword scoring ──
# (keyword, base_magnitude, duration_minutes, direction)

STRONG_POSITIVE = [
    ("重大利好", 1.0, 480), ("业绩大幅增长", 0.9, 240), ("中标", 0.8, 120),
    ("获得订单", 0.7, 120), ("政策扶持", 0.9, 480), ("增持", 0.6, 240),
    ("回购", 0.7, 240), ("高分红", 0.5, 240), ("突破", 0.6, 120),
    ("涨停", 0.7, 60), ("创新药获批", 1.0, 480), ("新产品上市", 0.7, 240),
    ("签订合同", 0.7, 120), ("专利", 0.5, 240), ("扩产", 0.5, 240),
    ("扭亏为盈", 0.8, 240), ("业绩预增", 0.6, 120), ("减税", 0.7, 480),
    ("降准", 0.8, 480), ("降息", 0.8, 480), ("营收增长", 0.6, 240),
    ("净利润增长", 0.7, 240), ("研发成功", 0.7, 240),
]

MODERATE_POSITIVE = [
    ("利润增长", 0.4, 120), ("合作", 0.4, 240), ("签订", 0.4, 120),
    ("新品发布", 0.5, 120), ("获批", 0.5, 240), ("入选", 0.3, 120),
    ("利好", 0.4, 120), ("上涨", 0.3, 30), ("反弹", 0.3, 30),
    ("看好", 0.3, 60), ("增资", 0.4, 120), ("转型", 0.4, 240),
    ("拓展", 0.4, 120), ("布局", 0.3, 120), ("增长", 0.3, 60),
    ("战略合作", 0.5, 240), ("机构调研", 0.3, 60),
]

WEAK_POSITIVE = [
    ("预期改善", 0.2, 30), ("企稳", 0.15, 30), ("资金流入", 0.2, 30),
    ("买入评级", 0.25, 60), ("推荐", 0.15, 30), ("加仓", 0.2, 30),
    ("低估值", 0.15, 60), ("有望", 0.1, 30),
]

STRONG_NEGATIVE = [
    ("重大利空", 1.0, 480), ("业绩大幅下滑", 0.9, 240), ("亏损", 0.7, 240),
    ("退市风险", 1.0, 480), ("ST", 0.9, 960), ("*ST", 0.95, 960),
    ("减持", 0.7, 240), ("立案调查", 1.0, 480), ("处罚", 0.8, 480),
    ("债务违约", 1.0, 480), ("停产", 0.9, 240), ("跌停", 0.7, 60),
    ("商誉减值", 0.8, 240), ("资产减值", 0.7, 240), ("诉讼", 0.6, 240),
    ("被警示", 0.7, 240), ("违规", 0.7, 240), ("财务造假", 1.0, 960),
    ("重组失败", 0.8, 240), ("业绩预降", 0.6, 120),
    ("净利润下滑", 0.7, 240), ("营收下滑", 0.6, 240),
]

MODERATE_NEGATIVE = [
    ("利润下滑", 0.5, 120), ("股东减持", 0.5, 120), ("解禁", 0.5, 240),
    ("下调评级", 0.5, 120), ("下跌", 0.3, 30), ("调整", 0.3, 30),
    ("承压", 0.3, 60), ("风险提示", 0.4, 60), ("监管", 0.5, 120),
    ("立案", 0.7, 240), ("处罚", 0.8, 480), ("问询", 0.5, 120),
    ("冻结", 0.6, 240), ("质押", 0.4, 120), ("下滑", 0.4, 120),
]

WEAK_NEGATIVE = [
    ("回落", 0.2, 30), ("资金流出", 0.2, 30), ("卖出评级", 0.25, 60),
    ("减持计划", 0.3, 60), ("低迷", 0.15, 60), ("放缓", 0.2, 60),
    ("不及预期", 0.3, 60), ("警惕", 0.2, 30),
]

ALL_CATEGORIES = [
    (STRONG_POSITIVE, "positive", "强利好"),
    (MODERATE_POSITIVE, "positive", "中利好"),
    (WEAK_POSITIVE, "positive", "弱利好"),
    (STRONG_NEGATIVE, "negative", "强利空"),
    (MODERATE_NEGATIVE, "negative", "中利空"),
    (WEAK_NEGATIVE, "negative", "弱利空"),
]


def analyze_single_news(news_item: dict) -> dict | None:
    """Classify a single news item. Returns impact dict or None if no impact detected."""
    title = news_item.get("title", "")
    content = news_item.get("content", "")
    text = f"{title} {content[:200]}"  # check title + first 200 chars of content

    best_match = None
    best_magnitude = 0

    for category, direction, level in ALL_CATEGORIES:
        for keyword, magnitude, duration in category:
            if keyword in text:
                if magnitude > best_magnitude:
                    best_magnitude = magnitude
                    best_match = {
                        "direction": direction,
                        "magnitude": magnitude,
                        "duration_min": duration,
                        "level": level,
                        "keyword": keyword,
                        "title": title,
                    }

    if best_match:
        best_match["time"] = news_item.get("time", "")
        best_match["url"] = news_item.get("url", "")
        return best_match
    return None


def analyze_news_impacts(news_list: list[dict]) -> list[dict]:
    """Analyze all news and return list of impact dicts, sorted by magnitude desc."""
    if not news_list:
        return []

    impacts = []
    for n in news_list:
        impact = analyze_single_news(n)
        if impact:
            impacts.append(impact)

    impacts.sort(key=lambda x: x["magnitude"], reverse=True)
    return impacts


def compute_news_bias(impacts: list[dict], current_time: datetime) -> dict:
    """
    Compute net news bias at current_time with temporal decay.

    Decay: linear from magnitude → 0 over duration_min.
    Expired impacts are dropped.

    Returns dict with:
      - bias: -1.0 to +1.0 net directional bias
      - magnitude: 0-1 combined magnitude
      - active_count: number of still-active impacts
      - active_impacts: list of active impact details
    """
    active = []
    total_bias = 0.0
    total_magnitude = 0.0

    for imp in impacts:
        try:
            news_time = datetime.strptime(imp.get("time", ""), "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            try:
                news_time = datetime.strptime(imp.get("time", ""), "%Y-%m-%d")
            except (ValueError, TypeError):
                # If can't parse time, assume recent
                news_time = current_time - timedelta(minutes=5)

        duration = imp.get("duration_min", 30)
        elapsed = (current_time - news_time).total_seconds() / 60.0

        if elapsed >= duration:
            continue  # expired

        # Linear decay
        decay_factor = 1.0 - (elapsed / duration)
        effective_magnitude = imp.get("magnitude", 0) * decay_factor

        sign = 1.0 if imp.get("direction") == "positive" else -1.0
        total_bias += sign * effective_magnitude
        total_magnitude += effective_magnitude

        active.append({
            "title": imp.get("title", ""),
            "keyword": imp.get("keyword", ""),
            "direction": imp.get("direction"),
            "level": imp.get("level", ""),
            "magnitude": round(imp.get("magnitude", 0), 2),
            "effective": round(effective_magnitude, 3),
            "decay": round(decay_factor, 2),
            "remaining_min": round(max(0, duration - elapsed), 1),
        })

    return {
        "bias": round(max(-1.0, min(1.0, total_bias)), 4),
        "magnitude": round(total_magnitude, 3),
        "active_count": len(active),
        "active_impacts": active,
    }


def has_news_impact(news_list: list[dict]) -> bool:
    """Quick check: does any news item contain an impact signal?"""
    if not news_list:
        return False
    for n in news_list:
        if analyze_single_news(n):
            return True
    return False


def apply_news_adjustment(target_price: float, price_delta: float,
                          news_bias: dict, max_adj_pct: float = 0.02) -> dict:
    """
    Adjust predicted target price based on news bias.

    max_adj_pct: maximum price adjustment as fraction (2% default).

    Returns adjusted prediction updates:
      - adjusted_target: new target price
      - adjustment_pct: how much was adjusted
      - news_bias_applied: the bias used
    """
    bias = news_bias.get("bias", 0)
    if abs(bias) < 0.01 or news_bias.get("active_count", 0) == 0:
        return {
            "adjusted_target": target_price,
            "adjustment_pct": 0,
            "news_bias_applied": 0,
        }

    # Scale adjustment: bias * magnitude * max_adj_pct
    adjustment_pct = bias * min(news_bias.get("magnitude", 0), 1.0) * max_adj_pct * 100
    adjusted_target = target_price * (1 + adjustment_pct / 100)
    adjusted_delta = (adjusted_target - target_price) / target_price

    return {
        "adjusted_target": round(adjusted_target, 2),
        "original_target": round(target_price, 2),
        "adjustment_pct": round(adjustment_pct, 4),
        "adjusted_delta": round(adjusted_delta, 4),
        "news_bias_applied": bias,
        "active_count": news_bias.get("active_count", 0),
    }
