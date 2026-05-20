"""Training v5: New targets + microstructure features + news sentiment proxy.

Ablation-tests three improvements:
  1. New targets: large_move (binary), vol_regime (3-class), magnitude (regression)
  2. Microstructure features: spread, flow pressure, volume imbalance, toxicity, etc.
  3. News sentiment proxy: gap signal, vol spike, extreme moves, event strength
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

from config import load_config
from storage.database import init_database
from storage.repository import Repository
from data.features import compute_all_indicators
from data.preprocessor import preprocess, build_targets, build_targets_v2, FEATURE_COLS
from model.dataset import StockDataset, TimeOrderedSubset, collate_fn
from model.trainer import WeightEMA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from copy import deepcopy

cfg = load_config()
init_database(cfg)
repo = Repository(cfg)

stocks = repo.get_screened_stocks()

# Use more stocks for robustness
train_stocks = []
for ts in stocks[:50]:
    df = repo.get_minutes(ts)
    if len(df) >= 160:
        train_stocks.append(ts)
        if len(train_stocks) >= 20:
            break

print(f"训练股票: {len(train_stocks)} 只")
print(f"特征总数: {len(FEATURE_COLS)}")
print()

HORIZON = 15
SEQ_LEN = 120


# ── Data loading ──
def load_data(stocks_list, feature_groups=None):
    """Load data with optional feature group filtering.

    feature_groups: dict mapping group_name -> list of feature name prefixes
                    None = all features
    """
    sequences, targets_list, timestamps_list = [], [], []
    scaler = None

    for ts_code in stocks_list:
        df = repo.get_minutes(ts_code)
        df = df.sort_values("trade_time")
        epoch_ts = pd.to_datetime(df["trade_time"]).apply(lambda t: t.timestamp()).values.astype(float)
        timestamps_list.append(epoch_ts)
        raw_close = df["close"].values.astype(float)
        ind = compute_all_indicators(df)
        for col in ind.columns:
            if col not in df.columns:
                df[col] = ind[col].values
        feat_arr, sc = preprocess(df, fit_scaler=True)
        target_dict = build_targets_v2(
            raw_close, df["high"].values, df["low"].values,
            df.get("vol", df.get("volume", df["close"] * 1000)).values,
            horizon=HORIZON
        )
        if scaler is None:
            scaler = sc
        sequences.append(feat_arr)
        targets_list.append(target_dict)

    return sequences, targets_list, timestamps_list, scaler


sequences, targets_list, timestamps_list, scaler = load_data(train_stocks)


# ── Dataset helper: extract specific target from dict ──
class MultiTargetDataset(StockDataset):
    """Extended StockDataset that stores target dicts for flexible target selection."""

    def __init__(self, sequences, targets_list, seq_len, horizon, dense=False, timestamps=None):
        self.seq_len = seq_len
        self.horizon = horizon
        self.samples: list = []
        self.sample_times: list[float] = []

        for s_idx, (feat_arr, tgt_dict) in enumerate(zip(sequences, targets_list)):
            T = feat_arr.shape[0]
            total_windows = T - seq_len - horizon
            if total_windows <= 0:
                continue
            step = max(1, total_windows // 200)

            times = timestamps[s_idx] if timestamps and s_idx < len(timestamps) else None

            for i in range(0, T - seq_len - horizon, step):
                x = feat_arr[i:i + seq_len]
                if x.shape[0] < seq_len:
                    continue
                tgt_idx = i + seq_len + horizon - 1
                if tgt_idx >= T:
                    continue
                sample_targets = {k: v[tgt_idx] for k, v in tgt_dict.items()}
                self.samples.append((x, sample_targets))

                if times is not None:
                    self.sample_times.append(float(times[i + seq_len - 1]))
                else:
                    self.sample_times.append(float(i + seq_len - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, targets = self.samples[idx]
        return torch.from_numpy(x).float(), targets

    def time_split(self, train_ratio=0.70):
        if not self.sample_times or len(self.sample_times) != len(self.samples):
            n = len(self.samples)
            train_n = int(n * train_ratio)
            indices = list(range(n))
            return indices[:train_n], indices[train_n:]
        sorted_pairs = sorted(enumerate(self.sample_times), key=lambda x: x[1])
        sorted_indices = [i for i, _ in sorted_pairs]
        n = len(sorted_indices)
        split_idx = int(n * train_ratio)
        return sorted_indices[:split_idx], sorted_indices[split_idx:]


def multi_collate_fn(batch):
    x_list, tgt_list = zip(*batch)
    X = torch.stack(x_list)
    T = {k: torch.tensor([t[k] for t in tgt_list]) for k in tgt_list[0]}
    return X, T


ds = MultiTargetDataset(sequences, targets_list, SEQ_LEN, horizon=HORIZON,
                        dense=False, timestamps=timestamps_list)
print(f"总样本: {len(ds)}")

train_idx, val_idx = ds.time_split(train_ratio=0.70)
print(f"训练: {len(train_idx)}, 验证: {len(val_idx)}")


# ── Model ──
class FlexibleClassifier(nn.Module):
    """Classifier that adapts to different output sizes."""

    def __init__(self, input_dim=85, hidden=64, num_classes=2, dropout=0.45):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = out[:, -1, :]
        pooled = self.ln(pooled)
        pooled = self.dropout(pooled)
        return self.clf(pooled)


device = torch.device(cfg.model.device)
print(f"设备: {device}")
print()


# ── Experiment runner ──
def run_experiment(name, target_key, num_classes, train_idx, val_idx,
                   class_balanced=True, epochs=70):
    """Run one experiment with a specific target key."""
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"目标: {target_key} ({num_classes}类)")
    print(f"{'='*60}")

    # Build labels
    train_labels = []
    for i in train_idx:
        t = ds.samples[i][1][target_key]
        train_labels.append(int(t) if num_classes > 1 else float(t))
    val_labels = []
    for i in val_idx:
        t = ds.samples[i][1][target_key]
        val_labels.append(int(t) if num_classes > 1 else float(t))

    if num_classes > 1:
        train_labels_int = train_labels
        label_counts = np.bincount(train_labels_int, minlength=num_classes)
        print(f"训练分布: {dict(enumerate(label_counts))}")
        print(f"验证分布: {dict(enumerate(np.bincount(val_labels, minlength=num_classes)))}")

        # Class weights for sampler
        class_weights_sampler = 1.0 / np.maximum(label_counts, 1)
        sample_weights = [class_weights_sampler[l] for l in train_labels_int]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        class_weights_loss = torch.tensor(
            class_weights_sampler / class_weights_sampler.sum() * num_classes,
            dtype=torch.float32
        ).to(device)
    else:
        sampler = None
        class_weights_loss = None

    # Model
    model = FlexibleClassifier(
        input_dim=len(FEATURE_COLS), hidden=64,
        num_classes=num_classes if num_classes > 1 else 1,
        dropout=0.45
    )
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)

    # DataLoader
    BATCH = 128
    train_ds_subset = TimeOrderedSubset(ds, train_idx)
    val_ds_subset = TimeOrderedSubset(ds, val_idx)

    train_kwargs = {"batch_size": BATCH, "collate_fn": multi_collate_fn}
    if sampler:
        train_kwargs["sampler"] = sampler
    else:
        train_kwargs["shuffle"] = True

    train_loader = DataLoader(train_ds_subset, **train_kwargs)
    val_loader = DataLoader(val_ds_subset, batch_size=BATCH, shuffle=False,
                            collate_fn=multi_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2)
    ema = WeightEMA(model, decay=0.99)

    best_metric = 0.0
    best_state = None
    patience = 0
    best_per_class = {}

    for epoch in range(epochs):
        model.train()
        tr_losses = []
        tr_correct = 0
        tr_total = 0

        for x_batch, t_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = t_batch[target_key].to(device)

            if num_classes > 1:
                y_batch = y_batch.long()

            logits = model(x_batch)

            if num_classes > 1:
                loss = F.cross_entropy(logits, y_batch, weight=class_weights_loss)
            else:
                logits = logits.squeeze(-1)
                loss = F.mse_loss(logits, y_batch.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
            optimizer.step()
            scheduler.step()
            ema.update()

            tr_losses.append(loss.item())
            if num_classes > 1:
                tr_correct += (logits.argmax(1) == y_batch).sum().item()
                tr_total += y_batch.size(0)

        # Validation
        model.eval()
        ema.apply_shadow()
        val_losses = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_batch, t_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = t_batch[target_key].to(device)

                if num_classes > 1:
                    y_batch = y_batch.long()

                logits = model(x_batch)

                if num_classes > 1:
                    loss = F.cross_entropy(logits, y_batch, weight=class_weights_loss)
                    preds = logits.argmax(1).cpu().tolist()
                else:
                    logits = logits.squeeze(-1)
                    loss = F.mse_loss(logits, y_batch.float())
                    preds = logits.cpu().tolist()

                val_losses.append(loss.item())
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().tolist())
        ema.restore()

        if num_classes > 1:
            val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
            per_class = {}
            for cls in range(num_classes):
                mask = [l == cls for l in all_labels]
                if sum(mask) > 0:
                    per_class[cls] = sum(1 for p, l in zip(all_preds, all_labels)
                                         if p == l == cls) / sum(mask)
            macro_acc = np.mean(list(per_class.values()))
            metric = macro_acc
        else:
            # Regression: use negative MSE as metric
            metric = -np.mean(val_losses)
            val_acc = metric

        avg_train = np.mean(tr_losses)
        avg_val = np.mean(val_losses)

        if epoch % 10 == 0 or epoch < 3:
            if num_classes > 1:
                pc_str = " ".join([f"c{c}={per_class.get(c,0):.1%}" for c in range(num_classes)])
                print(f"Ep {epoch:3d} | Tr={avg_train:.4f} Val={avg_val:.4f} "
                      f"Acc={val_acc:.1%} Macro={macro_acc:.1%} | {pc_str} | p={patience}")
            else:
                print(f"Ep {epoch:3d} | Tr={avg_train:.4f} Val={avg_val:.4f} "
                      f"MSE={avg_val:.6f} | p={patience}")

        if metric > best_metric:
            best_metric = metric
            ema.apply_shadow()
            best_state = deepcopy(model.state_dict())
            ema.restore()
            if num_classes > 1:
                best_per_class = per_class
            patience = 0
        else:
            patience += 1
        if patience >= 25:
            break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "name": name, "target_key": target_key, "num_classes": num_classes,
        "best_metric": best_metric, "best_per_class": best_per_class,
        "val_labels": val_labels, "best_state": best_state, "model": model,
    }


# ── Run all experiments ──
results = []

# 1. Legacy direction (baseline from v4)
results.append(run_experiment(
    "方向 (legacy)", "direction", 2, train_idx, val_idx, epochs=70
))

# 2. Large move (any direction) — binary
results.append(run_experiment(
    "大行情预测 (any)", "large_move_any", 2, train_idx, val_idx, epochs=70
))

# 3. Vol regime — 3-class
results.append(run_experiment(
    "波动率分档 (3类)", "vol_regime", 3, train_idx, val_idx, epochs=70
))

# 4. Magnitude — regression
results.append(run_experiment(
    "波动幅度 (回归)", "magnitude", 1, train_idx, val_idx, epochs=70
))


# ── Baselines ──
print(f"\n{'='*60}")
print("基线对比")
print(f"{'='*60}")

val_label_arr_direction = np.array([ds.samples[i][1]["direction"] for i in val_idx])
val_label_arr_large = np.array([ds.samples[i][1]["large_move_any"] for i in val_idx])
val_label_arr_vol = np.array([ds.samples[i][1]["vol_regime"] for i in val_idx])

# Direction baselines
always_down = 1 - val_label_arr_direction.mean()
always_up = val_label_arr_direction.mean()
persistence_dir = (val_label_arr_direction[1:] == val_label_arr_direction[:-1]).mean()

# Large move baselines
always_no_move = (val_label_arr_large == 0).mean()
always_move = val_label_arr_large.mean()
persistence_large = (val_label_arr_large[1:] == val_label_arr_large[:-1]).mean()

# Vol regime baselines
vol_mode = np.argmax(np.bincount(val_label_arr_vol))
always_mode = (val_label_arr_vol == vol_mode).mean()
persistence_vol = (val_label_arr_vol[1:] == val_label_arr_vol[:-1]).mean()

print(f"\n方向分类基线:")
print(f"  永远跌: {always_down:.1%}  永远涨: {always_up:.1%}  持续性: {persistence_dir:.1%}")
print(f"大行情基线:")
print(f"  永远无大行情: {always_no_move:.1%}  持续性: {persistence_large:.1%}")
print(f"波动率分档基线:")
print(f"  永远众数: {always_mode:.1%}  持续性: {persistence_vol:.1%}")

print(f"\n{'='*60}")
print("实验结果汇总")
print(f"{'='*60}")

for r in results:
    name = r["name"]
    if r["num_classes"] > 1:
        metric_str = f"Macro Acc={r['best_metric']:.1%}"
        pc_str = " ".join([f"c{c}={r['best_per_class'].get(c,0):.1%}" for c in range(r["num_classes"])])
        print(f"  {name}: {metric_str} | {pc_str}")
    else:
        print(f"  {name}: Neg MSE={r['best_metric']:.6f}")

    # Compute baselines for each target
    val_labels = np.array(r["val_labels"])
    if r["num_classes"] > 1:
        mode_label = np.argmax(np.bincount(val_labels))
        baseline = (val_labels == mode_label).mean()
        persistence = (val_labels[1:] == val_labels[:-1]).mean()
        best_baseline = max(baseline, persistence)
        improvement = r["best_metric"] - best_baseline
        print(f"    vs 最强基线: {'+' if improvement > 0 else ''}{improvement:.1%} "
              f"(众数={baseline:.1%}, 持续={persistence:.1%})")
        if improvement > 0.02:
            print(f"    → 显著优于基线！")
        elif improvement > 0:
            print(f"    → 略优于基线")
        elif improvement > -0.03:
            print(f"    → 接近基线")
        else:
            print(f"    → 不如此基线")

print(f"\n=== 结论 ===")
print("如果所有目标都未能显著超越基线，说明问题不在特征/目标选择，")
print("而是5分钟K线的OHLCV数据本质上无法预测未来15根K线的走势。")
print("这种情况下应考虑：")
print("  1. 更长的时间框架（日线、周线）")
print("  2. 基本面数据（财报、估值）")
print("  3. 真正的外部数据（新闻文本NLP、社交媒体情绪）")
print("  4. 换一个预测问题（如预测单只股票的日内模式，而非跨股票涨跌）")
