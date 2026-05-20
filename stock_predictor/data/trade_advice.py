"""Trading advice: position sizing, stop-loss, and Plan B recovery.

References:
- Kelly Criterion (Kelly 1956)
- ATR-based position sizing (Van Tharp)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from data.market_rules import get_price_limit_pct, get_board_name, get_trading_session, get_session_description, is_trading_time, get_min_lot_size


def compute_atr(high, low, close, period=14):
    """Compute Average True Range."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values[-1]


def generate_trade_advice(pred: dict, df: pd.DataFrame,
                          total_capital: float = 100000,
                          ts_code: str = "000001.SZ",
                          min_conf: float = 0.75) -> dict:
    """
    Generate trading advice based on model prediction and recent data.

    Returns dict with:
      - position_pct: recommended position size as % of capital
      - position_shares: suggested number of shares (100-share lots)
      - stop_loss: stop-loss price level
      - take_profit: take-profit price level
      - risk_reward: risk/reward ratio
      - plan_b: recovery plan if prediction is wrong
      - risk_warning: specific risk notes
    """
    if df is None or df.empty or len(df) < 20:
        return _empty_advice()

    limit_pct = get_price_limit_pct(ts_code)
    board = get_board_name(ts_code)

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float) if "high" in df.columns else close * 1.01
    low = df["low"].values.astype(float) if "low" in df.columns else close * 0.99
    current_price = close[-1]
    atr_val = compute_atr(high, low, close)

    direction = pred.get("direction", "flat")
    conf = pred.get("direction_conf", 0.5)
    target = pred.get("target_price", current_price)

    if direction == "flat" or conf < min_conf:
        return {
            "action": "观望",
            "reason": f"预测方向不明确或置信度不足 (需>={min_conf:.0%})",
            "position_pct": 0,
            "position_shares": 0,
            "stop_loss": round(current_price * 0.97, 2),
            "take_profit": round(current_price * 1.03, 2),
            "risk_reward": 0,
            "plan_b": "等待明确信号后再入场",
            "risk_warning": "当前不适合交易，建议观望",
            "rules": ["规则1: 涨跌停板限制", "规则2: T+1交割"],
        }

    # Expected price movement
    price_move_pct = abs(target - current_price) / current_price

    # Kelly fraction: f = p - q / (win/loss_ratio)
    win_prob = conf
    loss_prob = 1 - conf
    win_amount = price_move_pct if direction == "up" else price_move_pct
    loss_amount = max(atr_val / current_price, 0.005)

    if loss_amount < 0.001:
        loss_amount = 0.005

    win_loss_ratio = win_amount / loss_amount if loss_amount > 0 else 1.0
    kelly_raw = win_prob - loss_prob / max(win_loss_ratio, 0.1)
    # Half-Kelly for safety
    kelly_safe = max(0, kelly_raw) * 0.5
    # Cap at 25% position
    position_pct = min(kelly_safe, 0.25)

    # Position sizing
    position_amount = total_capital * position_pct
    position_shares = int(position_amount / current_price / 100) * 100  # round to lots

    # Stop-loss: 1.5 ATR below entry (for long) or above (for short)
    if direction == "up":
        stop_loss = round(current_price - 1.5 * atr_val, 2)
        take_profit = round(target, 2)
    else:
        stop_loss = round(current_price + 1.5 * atr_val, 2)
        take_profit = round(target, 2)

    # Risk/Reward
    risk = abs(current_price - stop_loss)
    reward = abs(target - current_price)
    risk_reward = round(reward / risk, 2) if risk > 0 else 0

    # Plan B
    if direction == "up":
        plan_b = (
            f"若跌破止损位 ¥{stop_loss}，立即平仓止损。"
            f"观察是否形成支撑：若在 ¥{current_price * 0.98:.2f} 附近放量企稳，"
            f"可考虑小仓位回补。若继续下跌超3%，转为观望等待右侧信号。"
        )
    else:
        plan_b = (
            f"若突破止损位 ¥{stop_loss}，立即平仓止损。"
            f"观察是否形成阻力：若在 ¥{current_price * 1.02:.2f} 附近承压回落，"
            f"可考虑重新入场。若继续上涨超3%，转为观望等待右侧信号。"
        )

    # Risk warning
    vol_recent = pd.Series(close).pct_change().tail(20).std()
    limit_up = current_price * (1 + limit_pct)
    limit_down = current_price * (1 - limit_pct)
    if vol_recent > 0.03:
        risk_warning = "当前波动率较高，建议减半仓位或等待波动收敛后再入场"
    elif conf < (min_conf + 0.05):
        risk_warning = "置信度偏低，建议轻仓操作（不超过10%仓位）"
    elif abs(target - current_price) / current_price > limit_pct * 0.8:
        risk_warning = f"预测幅度接近{board}{limit_pct:.0%}涨跌停限制，注意封板风险"
    else:
        risk_warning = "注意设置止损，盘中密切跟踪"

    # Clip take-profit to price limit
    if direction == "up":
        take_profit = min(take_profit, limit_up)
    else:
        take_profit = max(take_profit, limit_down)

    action = "买入" if direction == "up" else "卖出"

    # Build applicable rules list
    applicable_rules = [
        f"规则1: {board}涨跌停±{limit_pct:.0%}",
        "规则2: T+1交割 — 当日买入次日方可卖出",
        f"规则4: 最小交易单位 {get_min_lot_size()}股",
    ]
    if not is_trading_time():
        applicable_rules.append(f"规则3: 当前{get_session_description()}，非连续竞价时段")
    if abs(target - current_price) / current_price > limit_pct * 0.8:
        applicable_rules.append("规则10: 委托价不得超过涨跌停范围，注意封板风险")

    return {
        "action": action,
        "direction": direction,
        "position_pct": round(position_pct * 100, 1),
        "position_shares": position_shares,
        "position_amount": round(position_amount, 0),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
        "plan_b": plan_b,
        "risk_warning": risk_warning,
        "kelly_raw": round(kelly_raw, 3),
        "board": board,
        "limit_pct": f"{limit_pct:.0%}",
        "limit_up": round(limit_up, 2),
        "limit_down": round(limit_down, 2),
        "rules": applicable_rules,
        "session": get_session_description(),
        "is_trading": is_trading_time(),
    }


def _empty_advice() -> dict:
    return {
        "action": "数据不足",
        "position_pct": 0, "position_shares": 0,
        "stop_loss": 0, "take_profit": 0,
        "risk_reward": 0, "plan_b": "", "risk_warning": "数据不足无法生成建议",
    }
