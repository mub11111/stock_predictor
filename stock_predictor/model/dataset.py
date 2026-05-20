import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Slide window over multi-stock minute data to produce training samples.

    Each sample stores its timestamp (time of the last feature bar) so that
    train/val splits can be done chronologically — never leaking future data.
    """

    def __init__(self, sequences: list[np.ndarray], targets: list[np.ndarray],
                 seq_len: int = 240, horizon: int = 5, dense: bool = False,
                 timestamps: list[np.ndarray] | None = None):
        """
        sequences: list of (T_i, feature_dim) arrays, one per stock
        targets:   list of (T_i, 2) arrays [direction, price_change]
        timestamps: optional list of (T_i,) arrays of epoch seconds per stock
        dense: if True, use max overlap (step=1) for per-stock training
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.samples: list[tuple[np.ndarray, int, float]] = []
        self.sample_times: list[float] = []  # epoch seconds for each sample

        for s_idx, (feat_arr, tgt_arr) in enumerate(zip(sequences, targets)):
            T = feat_arr.shape[0]
            if dense:
                step = 1
            else:
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
                direction = int(tgt_arr[tgt_idx, 0])
                price_change = float(tgt_arr[tgt_idx, 1])
                self.samples.append((x, direction, price_change))

                if times is not None:
                    self.sample_times.append(float(times[i + seq_len - 1]))
                else:
                    self.sample_times.append(float(i + seq_len - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, direction, price_change = self.samples[idx]
        return (
            torch.from_numpy(x).float(),
            torch.tensor(direction, dtype=torch.long),
            torch.tensor(price_change, dtype=torch.float32)
        )

    def apply_feature_mask(self, mask: np.ndarray):
        """Apply boolean feature mask to all stored samples (in-place). Idempotent.

        Args:
            mask: bool array of shape (feature_dim,) — True = keep column.
        """
        if len(self.samples) == 0:
            return
        current_dim = self.samples[0][0].shape[1]
        if current_dim == int(mask.sum()):
            return  # already masked
        for i in range(len(self.samples)):
            x, direction, price_change = self.samples[i]
            self.samples[i] = (x[:, mask], direction, price_change)

    def time_split(self, train_ratio: float = 0.70, purge_window: int = 1200):
        """Split dataset chronologically with purging.

        Purging removes training samples whose timestamps fall within
        purge_window seconds before the first validation sample. This
        prevents data leakage from overlapping observation windows and
        matches real-world prediction conditions where the model cannot
        see data from the prediction horizon.

        Args:
            train_ratio: fraction of data for training (0 < train_ratio < 1)
            purge_window: gap in seconds between train and val (default 20 min)
        """
        if not self.sample_times or len(self.sample_times) != len(self.samples):
            n = len(self.samples)
            train_n = int(n * train_ratio)
            indices = list(range(n))
            return indices[:train_n], indices[train_n:]

        # Sort indices by timestamp
        sorted_pairs = sorted(enumerate(self.sample_times), key=lambda x: x[1])
        sorted_indices = [i for i, _ in sorted_pairs]
        sorted_times = [self.sample_times[i] for i in sorted_indices]

        n = len(sorted_indices)
        split_idx = int(n * train_ratio)

        # Purge: remove training samples within purge_window of first val sample
        if purge_window > 0 and 0 < split_idx < n:
            first_val_time = sorted_times[split_idx]
            purge_cutoff = first_val_time - purge_window
            # Walk back the split point to enforce the gap
            while split_idx > 0 and sorted_times[split_idx - 1] > purge_cutoff:
                split_idx -= 1

        if split_idx == 0:
            split_idx = max(1, int(n * 0.5))

        return sorted_indices[:split_idx], sorted_indices[split_idx:]


class TimeOrderedSubset(Dataset):
    """Subset of a StockDataset selected by time-ordered indices."""

    def __init__(self, dataset: StockDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def collate_fn(batch):
    x, direction, price = zip(*batch)
    return torch.stack(x), torch.stack(direction), torch.stack(price)
