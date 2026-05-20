"""Rule engine: loads structured A-share trading rules and provides context for AI analysis,
prediction validation, and trading advice.

Integrates:
  1. market_rules.py — programmatic rule functions (price limits, sessions, etc.)
  2. 股票规则.txt — structured knowledge injection (stock profiles, glossary, rule changes)
  3. Multi-stock rule validation — cross-check rules across batch predictions
"""
from __future__ import annotations
from datetime import datetime, date
from pathlib import Path
import re
from data.market_rules import (
    get_price_limit_pct, get_board_name, get_trading_session,
    get_session_description, is_trading_time, get_all_rules_summary,
    get_auction_times, get_min_lot_size, is_t_plus_one,
    clip_price_delta, get_effective_price_range, get_temporary_halting_rules,
    is_post_market_available,
)

RULES_FILE = Path("D:/AI/参考资料/股票规则/股票规则.txt")

# ── Cached parsed knowledge ──
_stock_profiles: dict[str, dict] = {}
_glossary: dict[str, str] = {}
_rule_changes: list[dict] = []
_constraints: list[dict] = []
_parsed = False


def _parse_rules_file():
    """Parse YAML-like structured sections from 股票规则.txt."""
    global _stock_profiles, _glossary, _rule_changes, _constraints, _parsed
    if _parsed:
        return
    _parsed = True
    if not RULES_FILE.exists():
        return
    text = RULES_FILE.read_text(encoding="utf-8", errors="ignore")

    # Parse stock profiles from module_id lines
    profile_pattern = re.compile(
        r'module_id:\s*"stock_(\w+)_profile".*?'
        r'entity:\s*"([^"]*)"\s*'
        r'ticker:\s*"([^"]*)"\s*'
        r'full_name:\s*"([^"]*)"\s*'
        r'market:\s*"([^"]*)"\s*'
        r'board:\s*"([^"]*)"\s*'
        r'listing_date:\s*"([^"]*)"\s*',
        re.DOTALL
    )
    for m in profile_pattern.finditer(text):
        ticker = m.group(3)
        _stock_profiles[f"{ticker}.SZ"] = {
            "name": m.group(2),
            "full_name": m.group(4),
            "market": m.group(5),
            "board": m.group(6),
            "listing_date": m.group(7),
        }
        # Also store with just the code
        _stock_profiles[ticker] = _stock_profiles[f"{ticker}.SZ"]

    # Parse glossary
    glossary_section = re.search(r'module_id:\s*"术语映射".*?glossary:(.*?)(?=query_handling|\Z)', text, re.DOTALL)
    if glossary_section:
        term_pattern = re.compile(r'term:\s*"([^"]*)"\s*\n\s*definition:\s*"([^"]*)"')
        for m in term_pattern.finditer(glossary_section.group(1)):
            _glossary[m.group(1)] = m.group(2)

    # Parse rule changes
    changes_section = re.search(r'module_id:\s*"2026.*?规则变更".*?changes:(.*?)(?=\n\s*- module_id|\Z)', text, re.DOTALL)
    if changes_section:
        change_pattern = re.compile(r'change:\s*"([^"]*)"(?:\s*\n\s*scope:\s*"([^"]*)")?')
        for m in change_pattern.finditer(changes_section.group(1)):
            _rule_changes.append({"change": m.group(1), "scope": m.group(2) or ""})

    # Parse constraints
    constraints_section = re.search(r'module_id:\s*"风险约束".*?constraints:(.*?)(?=\n\s*- module_id|\Z)', text, re.DOTALL)
    if constraints_section:
        cons_pattern = re.compile(r'type:\s*"([^"]*)"\s*\n\s*description:\s*"([^"]*)"')
        for m in cons_pattern.finditer(constraints_section.group(1)):
            _constraints.append({"type": m.group(1), "description": m.group(2)})


def load_rules_text() -> str:
    """Load the raw 股票规则.txt content."""
    if RULES_FILE.exists():
        return RULES_FILE.read_text(encoding="utf-8", errors="ignore")
    return ""


def get_stock_profile(ts_code: str) -> dict:
    """Get stock-specific profile from parsed rules knowledge."""
    _parse_rules_file()
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    key = f"{code}.SZ" if not ts_code.endswith(".SH") else ts_code
    return _stock_profiles.get(key, _stock_profiles.get(code, {}))


def get_glossary() -> dict[str, str]:
    """Get terminology mappings (e.g., T+1, ST, 集合竞价)."""
    _parse_rules_file()
    return dict(_glossary)


def get_rule_changes() -> list[dict]:
    """Get upcoming rule changes (2026-07-06)."""
    _parse_rules_file()
    return list(_rule_changes)


def get_constraints() -> list[dict]:
    """Get trading constraints (ST limits, price bounds, suspension, disclaimer)."""
    _parse_rules_file()
    return list(_constraints)


def get_all_stock_profiles() -> dict[str, dict]:
    """Get all parsed stock profiles."""
    _parse_rules_file()
    return dict(_stock_profiles)


def build_rules_context(ts_code: str, current_price: float = 0.0) -> str:
    """Build a condensed rules context string for AI analysis prompt."""
    _parse_rules_file()
    board = get_board_name(ts_code)
    limit_pct = get_price_limit_pct(ts_code)
    session_desc = get_session_description()
    halting = get_temporary_halting_rules(ts_code)
    price_range = get_effective_price_range(current_price or 100.0, ts_code)
    is_t0 = ts_code.startswith("5") and not ts_code.startswith("51")

    lines = [
        f"【{ts_code} 交易规则上下文】",
        f"板块: {board} | 涨跌停: ±{limit_pct:.0%}",
        f"交割制度: {'T+0 (ETF)' if is_t0 else 'T+1 — 当日买入次日方可卖出，卖出资金当日可继续买入'}",
        f"最小交易单位: {get_min_lot_size()}股（1手）",
        f"当前时段: {session_desc}",
        f"价格申报范围: {price_range['rule']}",
        f"临时停牌: {halting['rule']}",
        f"集合竞价: 9:15-9:20可撤单，9:20-9:25不可撤单，9:25产生开盘价",
    ]

    today = date.today()
    if today < date(2026, 7, 6):
        lines.append("【即将实施的规则变更 (2026-07-06)】")
        for rc in _rule_changes:
            lines.append(f"  - {rc['change']}" + (f"（{rc['scope']}）" if rc.get('scope') else ""))

    if current_price > 0:
        limit_up = round(current_price * (1 + limit_pct), 2)
        limit_down = round(current_price * (1 - limit_pct), 2)
        lines.append(f"今日涨跌停价: 涨停 ¥{limit_up} / 跌停 ¥{limit_down}")

    if not is_trading_time():
        lines.append("⚠ 当前非连续竞价时段，无法正常下单交易")

    # Constraints
    if _constraints:
        lines.append("【交易约束】")
        for c in _constraints:
            lines.append(f"  - {c['type']}: {c['description']}")

    return "\n".join(lines)


def validate_prediction(pred: dict, ts_code: str) -> dict:
    """Validate a single prediction against market rules."""
    warnings = []
    target = pred.get("target_price", 0)

    if not is_trading_time():
        warnings.append({
            "rule": "规则3",
            "severity": "warning",
            "message": "当前非连续竞价时段，预测仅供参考，无法实际下单",
        })

    limit_pct = get_price_limit_pct(ts_code)
    if target > 0:
        # Use target_price as an approximation of current price for delta check
        implied_delta = abs(pred.get("price_delta", 0))
        if implied_delta > limit_pct:
            warnings.append({
                "rule": "规则1",
                "severity": "error",
                "message": f"预测价格变动{implied_delta:.1%}超出{get_board_name(ts_code)}涨跌停限制±{limit_pct:.0%}",
            })

    if not (ts_code.startswith("5") and not ts_code.startswith("51")):
        warnings.append({
            "rule": "规则2",
            "severity": "info",
            "message": "T+1交割：今日买入需明日（下一交易日）方可卖出",
        })

    now = datetime.now()
    if now.weekday() >= 5:
        warnings.append({
            "rule": "规则3",
            "severity": "warning",
            "message": "当前为周末，市场休市。预测基于历史数据，仅供参考",
        })

    return {
        "has_warnings": len([w for w in warnings if w["severity"] != "info"]) > 0,
        "has_errors": len([w for w in warnings if w["severity"] == "error"]) > 0,
        "warnings": warnings,
    }


def validate_multi_stock(predictions: list[dict]) -> dict:
    """Cross-validate predictions across multiple stocks for rule consistency.

    Checks:
      1. Are all predictions within their respective price limits?
      2. Are there timing conflicts (all signaling in same direction)?
      3. Is current session appropriate for all stocks?
    """
    result = {
        "total": len(predictions),
        "errors": 0,
        "warnings": 0,
        "by_rule": {},
        "consensus_direction": None,
        "consensus_count": 0,
    }

    dir_counts = {"up": 0, "down": 0}
    per_stock = {}

    for pred in predictions:
        ts_code = pred.get("ts_code", "000001.SZ")
        check = validate_prediction(pred, ts_code)
        per_stock[ts_code] = check

        if check["has_errors"]:
            result["errors"] += 1
        if check["has_warnings"]:
            result["warnings"] += 1

        for w in check["warnings"]:
            rule = w["rule"]
            if rule not in result["by_rule"]:
                result["by_rule"][rule] = 0
            result["by_rule"][rule] += 1

        d = pred.get("direction", "flat")
        if d in dir_counts:
            dir_counts[d] += 1

    # Consensus direction
    total_with_dir = dir_counts["up"] + dir_counts["down"]
    if total_with_dir > 0:
        dominant = "up" if dir_counts["up"] >= dir_counts["down"] else "down"
        result["consensus_direction"] = dominant
        result["consensus_count"] = dir_counts[dominant]
        result["consensus_pct"] = round(dir_counts[dominant] / total_with_dir * 100, 1)

    result["per_stock"] = per_stock
    return result


def build_ai_system_prompt(ts_code: str, current_price: float = 0.0) -> str:
    """Build a comprehensive system prompt for AI analysis with full rule context."""
    _parse_rules_file()
    board = get_board_name(ts_code)
    limit_pct = get_price_limit_pct(ts_code)
    profile = get_stock_profile(ts_code)
    halting = get_temporary_halting_rules(ts_code)

    profile_text = ""
    if profile:
        profile_text = f"""
个股信息:
  名称: {profile.get('name', '未知')} ({profile.get('full_name', '')})
  板块: {profile.get('board', board)}
  上市日期: {profile.get('listing_date', '未知')}
  做空机制: {'无' if not profile.get('short_selling') else '有'}"""

    constraints_text = ""
    if _constraints:
        constraints_text = "\n".join(f"  - {c['type']}: {c['description']}" for c in _constraints)

    prompt = f"""你是专业的A股量化分析师，精通以下规则和指标：

【A股核心交易规则】
1. 涨跌停板: {board}{limit_pct:.0%}，超过此范围无法成交
2. T+1交割: 当日买入次日方可卖出，卖出资金当日可继续买入（ETF支持T+0）
3. 交易时间: 集合竞价9:15-9:25 / 连续竞价9:30-11:30,13:00-15:00 / 盘后15:05-15:30
4. 最小交易单位: 100股（1手），须为100股整数倍
5. 临时停牌: {halting['rule']}
6. 价格优先、时间优先: 较高买价优先、较低卖价优先、同价早报优先
7. 委托价格边界: 不得超过涨跌停范围，否则系统拒单

【2026年7月6日规则变更（即将实施）】
- ST涨跌幅由±5%放宽至±10%（主板），科创/创业板ST维持±20%
- 盘后固定价格交易扩展至全市场A股及ETF
- 创业板大宗交易确认时间延长至15:30{profile_text}

【交易约束】
{constraints_text}

【分析要求】
1. 结合规则评估预测是否在合理范围（涨跌停、T+1影响、时段可行性）
2. 给出技术关键位时，引用具体规则编号（规则1-规则7）
3. 风险提示中考虑当前交易时段对下单的可行性影响
4. 回答简洁专业，不超过300字，引用规则时标注编号"""

    return prompt
