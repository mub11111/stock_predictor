"""Shareholder group analysis & capital flow detection.

Identifies institutional (主力/机构), large retail (大散户), and small retail
(小散户) participation using OHLCV bar data. Extends to daily/weekly/monthly
aggregation when sufficient data is available.

Methodology references:
  - "Measuring Institutional Trading Costs" (Keim & Madhavan 1998)
    → VWAP as institutional execution benchmark
  - "All That Glitters" (Barber & Odean 2008)
    → Retail attention & herding patterns
  - Chaikin Money Flow (CMF): accumulation/distribution indicator
  - Money Flow Index (MFI): volume-weighted RSI variant
  - OBV divergence for smart-money detection
  - Volume surge + price impact → institutional footprint proxy
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def analyze_trader_behavior(df: pd.DataFrame) -> dict:
    """Full shareholder analysis: participant breakdown + capital flow."""
    if df.empty or len(df) < 30:
        return _empty_result()

    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)
    open_p = df["open"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)

    # ═══════════════════════════════════════════════
    # 1. Core indicators
    # ═══════════════════════════════════════════════

    # Returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.append(returns, returns[-1])

    # Volume relative to moving average
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=5).mean().values
    vol_ratio = volume / (vol_ma20 + 1)

    # Typical Price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0

    # Raw Money Flow = TP × Volume
    raw_money_flow = typical_price * volume

    # Money Flow direction (1 if TP rises, -1 if falls, 0 flat)
    tp_diff = np.diff(typical_price, prepend=typical_price[0])
    flow_sign = np.sign(tp_diff)

    # Positive / Negative Money Flow (rolling N-period sum)
    pos_flow = np.zeros(n)
    neg_flow = np.zeros(n)
    for i in range(n):
        pos_flow[i] = raw_money_flow[i] if flow_sign[i] > 0 else 0
        neg_flow[i] = raw_money_flow[i] if flow_sign[i] < 0 else 0

    # ═══════════════════════════════════════════════
    # 2. Chaikin Money Flow (CMF) — 20-period
    #    CMF = Σ(Mult × Vol) / Σ(Vol)
    #    Mult = ((C-L) - (H-C)) / (H-L)
    # ═══════════════════════════════════════════════
    hl_range = high - low
    hl_range = np.where(hl_range < 1e-10, 1e-10, hl_range)
    cmf_mult = ((close - low) - (high - close)) / hl_range
    cmf_mult = np.clip(cmf_mult, -1, 1)
    cmf_num = pd.Series(cmf_mult * volume).rolling(20, min_periods=5).sum().values
    cmf_den = pd.Series(volume).rolling(20, min_periods=5).sum().values
    cmf = cmf_num / (cmf_den + 1)

    # ═══════════════════════════════════════════════
    # 3. Money Flow Index (MFI) — 14-period
    # ═══════════════════════════════════════════════
    pos_mf_14 = pd.Series(pos_flow).rolling(14, min_periods=5).sum().values
    neg_mf_14 = pd.Series(neg_flow).rolling(14, min_periods=5).sum().values
    mf_ratio = pos_mf_14 / (neg_mf_14 + 1)
    mfi = 100.0 - 100.0 / (1.0 + mf_ratio)

    # ═══════════════════════════════════════════════
    # 4. VWAP deviation (institutional benchmark)
    # ═══════════════════════════════════════════════
    cum_vol = np.maximum(np.cumsum(volume), 1)
    vwap = np.cumsum(close * volume) / cum_vol
    vwap_dev = (close - vwap) / (vwap + 1e-10)

    # ═══════════════════════════════════════════════
    # 5. OBV for accumulation / distribution
    # ═══════════════════════════════════════════════
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    obv_ma20 = pd.Series(obv).rolling(20, min_periods=5).mean().values
    obv_dev = (obv - obv_ma20) / (obv_ma20 + 1)

    # ═══════════════════════════════════════════════
    # 6. Participant classification (per bar)
    #
    # Methodology (based on authoritative research):
    #   - Dynamic percentile thresholds on volume & bar amount replace fixed
    #     cutoffs — each stock's own distribution determines what is "large"
    #     (海通证券 2024.09 KMedian clustering; 广发证券 2024 mean+N*std).
    #   - Impact score = vol_ratio × |return| captures the Amihud-style
    #     price-impact footprint that distinguishes informed from uninformed
    #     flow (Keim & Madhavan 1998; 主力资金异象, 金融研究 2025).
    #   - VWAP deviation serves as institutional execution benchmark.
    #   - OBV / CMF / MFI provide accumulation-distribution confirmation.
    #
    # Tier allocation (approximate, adapts per stock):
    #   主力  ~5-8%  of bars — top percentile impact, multi-signal
    #   机构  ~15-20% — upper-quartile activity, significant impact
    #   大户  ~25-35% — above-median activity, moderate impact
    #   小散户 ~40-50% — below-median, passive / noise trading
    # ═══════════════════════════════════════════════

    # ── 6a. Per-bar metrics ──
    bar_return = (close - open_p) / (open_p + 1e-10)
    abs_ret = np.abs(bar_return)

    # Impact = volume surge × price movement (Amihud illiquidity proxy)
    impact = vol_ratio * abs_ret

    # Transaction amount (成交额) — the classic size classifier
    bar_amount = volume * typical_price
    amount_ma20 = pd.Series(bar_amount).rolling(20, min_periods=5).mean().values
    amount_ratio = bar_amount / (amount_ma20 + 1)

    # Rolling percentile ranks (20-bar window, min 10 bars)
    vol_pct = np.full(n, 0.5)
    amt_pct = np.full(n, 0.5)
    for i in range(20, n):
        win_s = i - 19
        vol_win = volume[win_s:i + 1]
        amt_win = bar_amount[win_s:i + 1]
        vol_pct[i] = (vol_win < volume[i]).mean()
        amt_pct[i] = (amt_win < bar_amount[i]).mean()

    # ── 6b. Buy/sell volume estimation (all bars, separate from classification) ──
    bar_buy_vol = np.zeros(n)
    bar_sell_vol = np.zeros(n)

    for i in range(n):
        if close[i] > open_p[i]:
            buy_pct = 0.5 + 0.5 * abs_ret[i] / (abs_ret[i] + 0.005)
        elif close[i] < open_p[i]:
            buy_pct = 0.5 - 0.5 * abs_ret[i] / (abs_ret[i] + 0.005)
        else:
            buy_pct = 0.5
        buy_pct = np.clip(buy_pct, 0.15, 0.85)
        bar_buy_vol[i] = volume[i] * buy_pct
        bar_sell_vol[i] = volume[i] * (1 - buy_pct)

    # ── 6c. Participant classification (bars 20+, with rolling context) ──
    participant = np.full(n, "小散户换手", dtype=object)
    # Classify bars 0-19 with simple direction heuristic
    for i in range(min(20, n)):
        if close[i] > open_p[i]:
            participant[i] = "小散户买入"
        elif close[i] < open_p[i]:
            participant[i] = "小散户卖出"
        else:
            participant[i] = "小散户换手"

    for i in range(20, n):
        direction_up = close[i] > open_p[i]
        direction_down = close[i] < open_p[i]

        # ── 主力: top ~8% by amount AND elevated impact ──
        # Requires: high amount percentile + meaningful price impact +
        # at least one confirming signal (CMF/OBV/VWAP/MFI)
        inst_sig_count = 0
        if cmf[i] > 0.03 or cmf[i] < -0.03:
            inst_sig_count += 1
        if obv_dev[i] > 0.03 or obv_dev[i] < -0.03:
            inst_sig_count += 1
        if abs(vwap_dev[i]) > 0.002:
            inst_sig_count += 1
        if mfi[i] > 80 or mfi[i] < 20:
            inst_sig_count += 1

        is_high_impact = impact[i] > 0.0008  # e.g. 1.6x vol * 0.05% ret
        is_top_amount = amt_pct[i] > 0.88
        is_top_volume = vol_pct[i] > 0.85

        if (is_top_amount or is_top_volume) and is_high_impact and inst_sig_count >= 2:
            if direction_up:
                participant[i] = "主力买入"
            elif direction_down:
                participant[i] = "主力卖出"
            elif cmf[i] > 0.03:
                participant[i] = "主力买入"
            else:
                participant[i] = "主力卖出"
            continue

        # ── 机构: top ~25% by activity OR strong impact ──
        is_upper_quartile = amt_pct[i] > 0.72 or vol_pct[i] > 0.70
        is_moderate_impact = impact[i] > 0.0003

        if is_upper_quartile or (is_moderate_impact and inst_sig_count >= 1):
            if direction_up:
                participant[i] = "机构买入"
            elif direction_down:
                participant[i] = "机构卖出"
            elif mfi[i] > 55:
                participant[i] = "机构买入"
            elif mfi[i] < 45:
                participant[i] = "机构卖出"
            elif cmf[i] > 0.01:
                participant[i] = "机构买入"
            else:
                participant[i] = "机构卖出"
            continue

        # ── 大户: above-median activity ──
        # Captures elevated-but-not-institutional bars (active retail,
        # medium-sized participants, trend followers).
        is_above_median = amt_pct[i] > 0.45 or vol_pct[i] > 0.48
        is_elevated_vol = vol_ratio[i] > 1.10
        is_notable_ret = abs_ret[i] > 0.0010  # 0.1% move

        if is_above_median or is_elevated_vol or is_notable_ret:
            if direction_up:
                participant[i] = "大户买入"
            elif direction_down:
                participant[i] = "大户卖出"
            elif is_elevated_vol:
                participant[i] = "大户换手"
            else:
                participant[i] = "大户换手"
            continue

        # ── 小散户: below-median, quiet / passive trading ──
        if direction_up:
            participant[i] = "小散户买入"
        elif direction_down:
            participant[i] = "小散户卖出"
        else:
            participant[i] = "小散户换手"

    # ═══════════════════════════════════════════════
    # 7. Aggregate by trading-day windows
    # ═══════════════════════════════════════════════

    # Count actual trading dates from trade_time column
    trade_dates = None
    if "trade_time" in df.columns:
        trade_dates = pd.to_datetime(df["trade_time"]).dt.date
        unique_dates = sorted(trade_dates.unique())
        num_trading_days = len(unique_dates)
    else:
        num_trading_days = max(1, n // 48)  # fallback: assume ~48 bars/day (5min)

    # "当日" = last 1 trading day (all bars of the most recent date)
    if trade_dates is not None and num_trading_days >= 1:
        last_date = unique_dates[-1]
        today_mask = trade_dates == last_date
        today_idx = np.where(today_mask.values)[0]
        daily = _aggregate_participant(
            participant[today_idx], bar_buy_vol[today_idx],
            bar_sell_vol[today_idx], volume[today_idx], "当日"
        )
    else:
        daily = _aggregate_participant(
            participant[-12:], bar_buy_vol[-12:],
            bar_sell_vol[-12:], volume[-12:], "当日"
        )

    # "本周" = last 5 trading days (or all if fewer)
    if trade_dates is not None and num_trading_days >= 2:
        week_start_date = unique_dates[max(0, num_trading_days - 5)]
        week_mask = trade_dates >= week_start_date
        week_idx = np.where(week_mask.values)[0]
        weekly = _aggregate_participant(
            participant[week_idx], bar_buy_vol[week_idx],
            bar_sell_vol[week_idx], volume[week_idx], "本周"
        )
    else:
        weekly = None

    # "30日" = last 30 trading days (or all available if fewer, min 2 days)
    if trade_dates is not None and num_trading_days >= 2:
        month_start_date = unique_dates[max(0, num_trading_days - 30)]
        month_mask = trade_dates >= month_start_date
        month_idx = np.where(month_mask.values)[0]
        monthly = _aggregate_participant(
            participant[month_idx], bar_buy_vol[month_idx],
            bar_sell_vol[month_idx], volume[month_idx], "30日"
        )
    else:
        monthly = None

    # ═══════════════════════════════════════════════
    # 8. Smart money & sentiment indices
    # ═══════════════════════════════════════════════

    # Net institutional flow
    inst_buy = daily.get("主力买入量", 0) + daily.get("机构买入量", 0)
    inst_sell = daily.get("主力卖出量", 0) + daily.get("机构卖出量", 0)
    inst_net = (inst_buy - inst_sell) / (inst_buy + inst_sell + 1e-10)

    retail_buy = daily.get("大户买入量", 0) + daily.get("小散户买入量", 0)
    retail_sell = daily.get("大户卖出量", 0) + daily.get("小散户卖出量", 0)
    retail_net = (retail_buy - retail_sell) / (retail_buy + retail_sell + 1e-10)

    # Herding index (Christie & Huang 1995)
    ret_std_short = pd.Series(returns).rolling(10, min_periods=3).std().values
    ret_std_long = pd.Series(returns).rolling(30, min_periods=10).std().values
    herding = np.clip(1.0 - (ret_std_short / (ret_std_long + 1e-10)), 0, 1)

    # Panic / Greed
    recent = min(20, n)
    r_ret = returns[-recent:]
    r_vol = vol_ratio[-recent:]
    r_drop = r_ret[r_ret < -0.002]
    r_rise = r_ret[r_ret > 0.002]

    panic = np.clip(
        np.mean(r_vol[r_ret < -0.002]) * abs(np.mean(r_drop)) * 400
        if len(r_drop) > 0 else 0, 0, 100
    )
    greed = np.clip(
        np.mean(r_vol[r_ret > 0.002]) * abs(np.mean(r_rise)) * 400
        if len(r_rise) > 0 else 0, 0, 100
    )
    sentiment = np.clip(greed - panic, -100, 100)
    herding_now = float(np.mean(herding[-recent:]))

    # ═══════════════════════════════════════════════
    # 9. Dominant force & regime
    # ═══════════════════════════════════════════════

    if inst_net > 0.15:
        regime = "主力吸筹 — 机构资金持续流入"
    elif inst_net < -0.15:
        regime = "主力派发 — 机构资金持续流出"
    elif herding_now > 0.5:
        regime = "羊群效应 — 市场高度一致，注意拐点"
    elif sentiment > 30:
        regime = "贪婪主导 — 散户追涨情绪浓厚"
    elif sentiment < -30:
        regime = "恐慌主导 — 抛售压力大"
    elif abs(sentiment) < 10 and abs(inst_net) < 0.1:
        regime = "横盘整理 — 方向不明，等待突破"
    else:
        regime = "正常波动"

    # Dominant participant
    flow_types = {}
    for k, v in daily.items():
        if k.endswith("量") and "总" not in k:
            flow_types[k] = v
    if flow_types:
        dominant_flow = max(flow_types, key=flow_types.get)
        dominant_name = dominant_flow.replace("买入量", "买入").replace("卖出量", "卖出")
    else:
        dominant_name = "数据不足"

    # ═══════════════════════════════════════════════
    # 10. Build display profile
    # ═══════════════════════════════════════════════

    profile_parts = []

    # ── Capital flow summary ──
    profile_parts.append("═══ 持股人群分析 · 资金流向 ═══")
    profile_parts.append("")

    # Daily flow
    profile_parts.append(_format_flow_block(daily, "当日"))
    if weekly:
        profile_parts.append(_format_flow_block(weekly, "本周"))
    if monthly:
        profile_parts.append(_format_flow_block(monthly, "30日"))

    # ── Participant balance ──
    profile_parts.append("── 买卖力量对比 ──")
    inst_total = inst_buy + inst_sell
    retail_total = retail_buy + retail_sell
    total_vol = inst_total + retail_total
    if total_vol > 0:
        inst_pct = inst_total / total_vol
        retail_pct = retail_total / total_vol
        profile_parts.append(
            f"  机构/主力: {inst_pct:.0%}  |  散户: {retail_pct:.0%}"
        )
        profile_parts.append(
            f"  机构净流向: {inst_net:+.1%}  |  散户净流向: {retail_net:+.1%}"
        )

    # ── Sentiment ──
    profile_parts.append("── 情绪指标 ──")
    profile_parts.append(f"  恐慌: {panic:.0f}/100  |  贪婪: {greed:.0f}/100")
    profile_parts.append(f"  羊群效应: {herding_now:.2f}  |  综合情绪: {sentiment:+.0f}")

    # ── Regime ──
    profile_parts.append(f"── 市场状态 ──\n  {regime}")
    profile_parts.append(f"  主导力量: {dominant_name}")

    # ── CMF / MFI ──
    profile_parts.append("── 技术面资金指标 ──")
    profile_parts.append(
        f"  CMF: {cmf[-1]:+.3f} {'(吸筹)' if cmf[-1] > 0.05 else '(派发)' if cmf[-1] < -0.05 else '(中性)'}"
    )
    profile_parts.append(
        f"  MFI: {mfi[-1]:.0f} "
        f"{'(超买)' if mfi[-1] > 80 else '(超卖)' if mfi[-1] < 20 else '(正常)'}"
    )
    profile_parts.append(
        f"  VWAP偏离: {vwap_dev[-1]:+.2%} "
        f"{'(高于均价)' if vwap_dev[-1] > 0.002 else '(低于均价)' if vwap_dev[-1] < -0.002 else '(均价附近)'}"
    )

    profile = "\n".join(profile_parts)

    # ═══════════════════════════════════════════════
    # Return full result dict
    # ═══════════════════════════════════════════════
    return {
        # Daily flow breakdown
        **daily,
        # Weekly / monthly (None if insufficient data)
        "weekly": weekly,
        "monthly": monthly,
        # Net flows
        "inst_net": float(inst_net),
        "retail_net": float(retail_net),
        "inst_buy_total": float(inst_buy),
        "inst_sell_total": float(inst_sell),
        "retail_buy_total": float(retail_buy),
        "retail_sell_total": float(retail_sell),
        # Sentiment
        "panic_index": float(panic),
        "greed_index": float(greed),
        "sentiment": float(sentiment),
        "herding_index": float(herding_now),
        # Indicators
        "cmf": float(cmf[-1]) if n > 0 else 0,
        "mfi": float(mfi[-1]) if n > 0 else 50,
        "vwap_dev": float(vwap_dev[-1]) if n > 0 else 0,
        "obv_dev": float(obv_dev[-1]) if n > 0 else 0,
        # Meta
        "dominant_type": dominant_name,
        "regime": regime,
        "trader_profile": profile,
        "data_sufficient": True,
    }


def _aggregate_participant(
    participant: np.ndarray,
    buy_vol: np.ndarray,
    sell_vol: np.ndarray,
    total_vol: np.ndarray,
    label: str,
) -> dict:
    """Aggregate buy/sell volume by participant type for a time window."""
    result = {}
    categories = [
        "主力买入", "主力卖出", "机构买入", "机构卖出",
        "大户买入", "大户卖出", "大户换手",
        "小散户买入", "小散户卖出", "小散户换手",
    ]
    for cat in categories:
        mask = participant == cat
        if "买入" in cat or "换手" in cat:
            result[f"{label}{cat}量"] = float(buy_vol[mask].sum())
        elif "卖出" in cat:
            result[f"{label}{cat}量"] = float(sell_vol[mask].sum())

    # Net institutional (主力+机构)
    inst_buy = sum(
        result.get(f"{label}{c}量", 0)
        for c in ["主力买入", "机构买入"]
    )
    inst_sell = sum(
        result.get(f"{label}{c}量", 0)
        for c in ["主力卖出", "机构卖出"]
    )
    result[f"{label}机构净买量"] = float(inst_buy - inst_sell)
    result[f"{label}总成交量"] = float(total_vol.sum())

    return result


def _format_flow_block(data: dict, label: str) -> str:
    """Format a flow data block for display, with Chinese-unit scaled amounts."""
    total = data.get(f"{label}总成交量", 1)
    if total < 1:
        total = 1

    def _pct(key: str) -> str:
        v = data.get(f"{label}{key}量", 0)
        return f"{v / total:.1%}"

    def _vol(key: str) -> str:
        v = data.get(f"{label}{key}量", 0)
        return _fmt_vol(v)

    lines = [f"【{label}资金流向】"]
    lines.append(f"  主力买入: {_vol('主力买入')} ({_pct('主力买入')})")
    lines.append(f"  主力卖出: {_vol('主力卖出')} ({_pct('主力卖出')})")
    lines.append(f"  机构买入: {_vol('机构买入')} ({_pct('机构买入')})")
    lines.append(f"  机构卖出: {_vol('机构卖出')} ({_pct('机构卖出')})")
    lines.append(f"  大户买入: {_vol('大户买入')} ({_pct('大户买入')})")
    lines.append(f"  大户卖出: {_vol('大户卖出')} ({_pct('大户卖出')})")
    lines.append(f"  小散买入: {_vol('小散户买入')} ({_pct('小散户买入')})")
    lines.append(f"  小散卖出: {_vol('小散户卖出')} ({_pct('小散户卖出')})")
    net = data.get(f"{label}机构净买量", 0)
    direction = "流入" if net > 0 else "流出" if net < 0 else "平衡"
    lines.append(f"  机构净{'买' if net >= 0 else '卖'}: {_fmt_vol(abs(net))} ({direction})")
    return "\n".join(lines)


def _fmt_vol(v: float) -> str:
    """Format volume to human-readable Chinese units (手)."""
    if v >= 1e8:
        return f"{v / 1e8:.2f}亿手"
    elif v >= 1e4:
        return f"{v / 1e4:.0f}万手"
    elif v >= 1e3:
        return f"{v / 1e3:.1f}千手"
    else:
        return f"{v:.0f}手"


def _empty_result() -> dict:
    return {
        "当日机构净买量": 0, "当日总成交量": 0,
        "inst_net": 0, "retail_net": 0,
        "inst_buy_total": 0, "inst_sell_total": 0,
        "retail_buy_total": 0, "retail_sell_total": 0,
        "panic_index": 0, "greed_index": 0, "sentiment": 0,
        "herding_index": 0, "cmf": 0, "mfi": 50, "vwap_dev": 0, "obv_dev": 0,
        "dominant_type": "数据不足", "regime": "数据不足",
        "trader_profile": "数据不足，请先刷新分钟数据（至少需要30根K线）",
        "weekly": None, "monthly": None, "data_sufficient": False,
    }
