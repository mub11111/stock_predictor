"""Per-stock adaptive feature selection using MRMR (Max-Relevance Min-Redundancy).

Correlation-based greedy selection: picks top-K features per stock that are
most predictive of future returns while minimizing mutual redundancy.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_POOL = [
    # Same pool as preprocessor.FEATURE_COLS — keep in sync manually

    "open", "high", "low", "close", "volume",
    "ret_1", "ret_5", "ret_15",
    "ma5", "ma10", "ma20", "ma60",
    "rsi6", "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower",
    "atr14", "vol_ratio", "vol_ma5",
    "pct_ma5", "pct_ma10", "pct_ma20",
    "hl_ratio", "oc_ratio",
    "minute_sin", "minute_cos", "day_sin", "day_cos",
    "mfi", "sm_flow", "vpt_ratio", "obv_div", "large_lot",
    "sm_score", "retail_intensity", "fomo_score", "panic_sel",
    "crowd_sent", "rev_zscore", "vwap_pull", "momentum",
    "vol_regime", "meta_signal",
    "kdj_k", "kdj_d", "kdj_j",
    "cci14", "willr14",
    "dc_upper", "dc_mid", "dc_lower",
    "roc5", "roc10",
    "obv", "chaikin_osc", "vol_roc5",
    "close_location", "ma20_slope", "ma60_slope",
    "gap_ratio", "up_down_vol", "intraday_intensity",
    # Microstructure
    "micro_spread", "micro_flow_pressure", "micro_vol_imbalance",
    "micro_trade_intensity", "micro_arrival_impact", "micro_order_depth",
    "micro_toxicity", "micro_bid_ask_bounce",
    # News sentiment proxy
    "news_gap_signal", "news_vol_spike", "news_extreme_moves",
    "news_momentum_decay", "news_sentiment_proxy", "news_event_strength",
    # Volatility
    "vol_5bar_realized", "vol_expanding", "vol_skew",
    "vol_persistence", "vol_hl_ratio",
    # Quant features — 金融学金牌指标注入 (keep in sync with preprocessor.FEATURE_COLS)
    "bb_pos",
    "rsi6_scaled",
    "rsi14_scaled",
]

# Frequency-group routing tags for Freq Orchestrator branches
# "low"  → FT-iTransformer (macro trend anchoring)
# "mid"  → DLinear (momentum / cyclical patterns)
# "high" → MicroStructure (order-flow / microstructure noise)
FEATURE_GROUP_MAP: dict[str, str] = {
    # ── Low-freq: trend / macro ──
    "open": "low", "high": "low", "low": "low", "close": "low", "volume": "low",
    "ma5": "low", "ma10": "low", "ma20": "low", "ma60": "low",
    "pct_ma5": "low", "pct_ma10": "low", "pct_ma20": "low",
    "ma20_slope": "low", "ma60_slope": "low",
    "bb_upper": "low", "bb_middle": "low", "bb_lower": "low", "bb_pos": "low",
    "atr14": "low",
    "dc_upper": "low", "dc_mid": "low", "dc_lower": "low",
    "kdj_k": "low", "kdj_d": "low", "kdj_j": "low",
    "cci14": "low", "willr14": "low",
    "macd": "low", "macd_signal": "low", "macd_hist": "low",
    "roc5": "low", "roc10": "low",
    "close_location": "low",
    "gap_ratio": "low",
    # ── Mid-freq: momentum / cyclical ──
    "rsi6": "mid", "rsi14": "mid",
    "rsi6_scaled": "mid", "rsi14_scaled": "mid",
    "ret_1": "mid", "ret_5": "mid", "ret_15": "mid",
    "vol_ratio": "mid", "vol_ma5": "mid", "vol_roc5": "mid",
    "hl_ratio": "mid", "oc_ratio": "mid",
    "minute_sin": "mid", "minute_cos": "mid",
    "day_sin": "mid", "day_cos": "mid",
    "momentum": "mid", "rev_zscore": "mid",
    "vwap_pull": "mid", "meta_signal": "mid",
    "crowd_sent": "mid", "vol_regime": "mid",
    "up_down_vol": "mid", "intraday_intensity": "mid",
    # ── High-freq: microstructure / order-flow / vol / news ──
    "mfi": "high", "sm_flow": "high", "vpt_ratio": "high", "obv_div": "high",
    "large_lot": "high", "sm_score": "high",
    "retail_intensity": "high", "fomo_score": "high", "panic_sel": "high",
    "obv": "high", "chaikin_osc": "high",
    "micro_spread": "high", "micro_flow_pressure": "high",
    "micro_vol_imbalance": "high", "micro_trade_intensity": "high",
    "micro_arrival_impact": "high", "micro_order_depth": "high",
    "micro_toxicity": "high", "micro_bid_ask_bounce": "high",
    "news_gap_signal": "high", "news_vol_spike": "high",
    "news_extreme_moves": "high", "news_momentum_decay": "high",
    "news_sentiment_proxy": "high", "news_event_strength": "high",
    "vol_5bar_realized": "high", "vol_expanding": "high", "vol_skew": "high",
    "vol_persistence": "high", "vol_hl_ratio": "high",
}


def select_features_mrmr(df: pd.DataFrame, k: int | None = None,
                         horizon: int = 10, alpha: float = 0.5,
                         min_k: int = 20, max_k: int = 55,
                         relevance_threshold: float = 0.85) -> list[str]:
    """
    Select top-K features per stock using MRMR.

    If k is None (adaptive mode), automatically determines the number of
    features based on cumulative relevance: keeps features until the
    cumulative relevance reaches `relevance_threshold` of total relevance.

    Args:
        df: DataFrame with feature columns + 'close' price
        k: number of features to select (None = adaptive)
        horizon: forecast horizon (bars ahead) for computing target
        alpha: redundancy penalty weight (0.5 = equal weight)
        min_k: minimum features in adaptive mode
        max_k: maximum features in adaptive mode
        relevance_threshold: cumulative relevance ratio for adaptive k

    Returns:
        list of selected feature names, ordered by importance
    """
    available = [f for f in FEATURE_POOL if f in df.columns]

    # Target: future return at horizon
    close = df["close"].values
    future_ret = np.zeros(len(close))
    future_ret[:-horizon] = (close[horizon:] - close[:-horizon]) / (close[:-horizon] + 1e-10)

    # Extract feature matrix
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
    n_samples, n_feats = X.shape

    if n_feats == 0:
        return []

    # Compute relevance: abs(correlation with target)
    relevance = np.zeros(n_feats)
    for j in range(n_feats):
        col_data = X[:, j]
        if np.std(col_data) < 1e-10:
            relevance[j] = 0
        else:
            corr = np.corrcoef(col_data, future_ret)[0, 1]
            relevance[j] = abs(corr) if not np.isnan(corr) else 0

    # Adaptive k: keep features until cumulative relevance meets threshold
    if k is None:
        total_rel = np.sum(relevance)
        if total_rel < 1e-10:
            k = min_k  # fallback when no features are relevant
        else:
            sorted_rel = np.sort(relevance)[::-1]
            cumsum = np.cumsum(sorted_rel)
            # Find index where cumulative relevance reaches threshold_pct of total
            threshold_idx = np.searchsorted(cumsum, relevance_threshold * total_rel)
            k = max(min_k, min(max_k, int(threshold_idx) + 1))

    if len(available) <= k:
        return available

    # Precompute pairwise correlation matrix for redundancy
    corr_matrix = np.abs(np.corrcoef(X.T))
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_matrix, 0)

    # Greedy MRMR selection
    selected_idx = []
    remaining_idx = list(range(n_feats))

    # First: pick the feature with highest relevance
    best_first = int(np.argmax(relevance))
    selected_idx.append(best_first)
    remaining_idx.remove(best_first)

    # Iterate until k features selected
    while len(selected_idx) < k and remaining_idx:
        scores = np.zeros(len(remaining_idx))
        for i, feat_idx in enumerate(remaining_idx):
            rel = relevance[feat_idx]
            red = np.mean([corr_matrix[feat_idx, s] for s in selected_idx])
            scores[i] = rel - alpha * red
        best_local = int(np.argmax(scores))
        selected_idx.append(remaining_idx.pop(best_local))

    return [available[i] for i in selected_idx]


def adaptive_feature_count(df: pd.DataFrame, horizon: int = 10,
                           min_k: int = 20, max_k: int = 55,
                           threshold: float = 0.85) -> int:
    """Quick estimate of adaptive k without running full MRMR."""
    available = [f for f in FEATURE_POOL if f in df.columns]
    close = df["close"].values
    future_ret = np.zeros(len(close))
    future_ret[:-horizon] = (close[horizon:] - close[:-horizon]) / (close[:-horizon] + 1e-10)
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float64)
    n_feats = X.shape[1]
    relevance = np.zeros(n_feats)
    for j in range(n_feats):
        col_data = X[:, j]
        if np.std(col_data) < 1e-10:
            relevance[j] = 0
        else:
            corr = np.corrcoef(col_data, future_ret)[0, 1]
            relevance[j] = abs(corr) if not np.isnan(corr) else 0
    total_rel = np.sum(relevance)
    if total_rel < 1e-10:
        return min_k
    sorted_rel = np.sort(relevance)[::-1]
    cumsum = np.cumsum(sorted_rel)
    threshold_idx = np.searchsorted(cumsum, threshold * total_rel)
    return max(min_k, min(max_k, int(threshold_idx) + 1))


def get_feature_mask(selected: list[str]) -> np.ndarray:
    """Convert selected feature names to a boolean mask over FEATURE_POOL."""
    mask = np.zeros(len(FEATURE_POOL), dtype=bool)
    for i, name in enumerate(FEATURE_POOL):
        if name in selected:
            mask[i] = True
    return mask


def get_feature_group_indices(selected_features: list[str] | None = None
                              ) -> dict[str, list[int]]:
    """Return local column indices for each frequency group.

    When selected_features is provided, indices are local positions within
    that list (0 .. K-1). When None, indices are global positions within
    the full FEATURE_POOL (0 .. 87).

    Args:
        selected_features: if provided, indices are local to this list.
                           If None, uses the full FEATURE_POOL.

    Returns:
        {"low": [0, 1, ...], "mid": [...], "high": [...]}
    """
    # When a subset is provided, sort by FEATURE_POOL order so local indices
    # match the actual column order produced by get_feature_mask / apply_feature_mask.
    if selected_features:
        pool = sorted(selected_features, key=lambda f: FEATURE_POOL.index(f) if f in FEATURE_POOL else 999)
    else:
        pool = FEATURE_POOL
    groups: dict[str, list[int]] = {"low": [], "mid": [], "high": []}
    for local_idx, feat_name in enumerate(pool):
        group = FEATURE_GROUP_MAP.get(feat_name, "mid")
        groups[group].append(local_idx)
    for g in groups:
        groups[g] = sorted(set(groups[g]))
    return groups


def select_features_on_split(raw_features: np.ndarray, raw_close: np.ndarray,
                              train_indices: list[int],
                              k: int | None = None, horizon: int = 10
                              ) -> tuple[list[str], np.ndarray]:
    """Run MRMR on training split only — no data leakage to validation.

    Args:
        raw_features: (N, len(FEATURE_POOL)) full feature array before selection
        raw_close: (N,) raw close prices for target computation
        train_indices: sample indices belonging to the training fold
        k: feature count (None = adaptive via cumulative relevance)

    Returns:
        (selected_feature_names, boolean_mask over FEATURE_POOL)
    """
    import pandas as pd
    if len(train_indices) < 50:
        return list(FEATURE_POOL), np.ones(len(FEATURE_POOL), dtype=bool)
    train_feat = raw_features[train_indices]
    train_close = raw_close[train_indices]
    df = pd.DataFrame(train_feat, columns=FEATURE_POOL)
    df["close"] = train_close
    selected = select_features_mrmr(df, k=k, horizon=horizon)
    mask = get_feature_mask(selected)
    return selected, mask


def save_feature_selection(selected: list[str], path: str | Path):
    """Save selected feature names to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"selected_features": selected, "feature_pool_version": len(FEATURE_POOL)}, f)


def load_feature_selection(path: str | Path) -> list[str] | None:
    """Load selected feature names from JSON, or None if file missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("selected_features", None)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def apply_feature_mask(arr: np.ndarray, selected: list[str]) -> np.ndarray:
    """
    Given a full feature array [N, len(FEATURE_POOL)] and selected feature names,
    return array with only selected columns, maintaining order.
    """
    mask = get_feature_mask(selected)
    return arr[:, mask]


_ILLEGAL_FILENAME_CHARS = __import__('re').compile(r'[<>:"/\\|?*]')

def _sanitize_filename(name: str) -> str:
    return _ILLEGAL_FILENAME_CHARS.sub('_', name)


def select_and_save(df: pd.DataFrame, ts_code: str, checkpoint_dir: str | Path,
                    k: int | None = None, horizon: int = 10) -> list[str]:
    """Run MRMR selection (adaptive k if k=None) and save to checkpoint_dir/{ts_code}_features.json."""
    selected = select_features_mrmr(df, k=k, horizon=horizon)
    path = Path(checkpoint_dir) / f"{_sanitize_filename(ts_code)}_features.json"
    save_feature_selection(selected, path)
    return selected
