"""Training loop with focal loss, Kendall dynamic task weighting, Spearman rank loss, weight EMA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from model.lstm_transformer import HybridModel, FocalLoss
from model.dataset import StockDataset, TimeOrderedSubset, collate_fn
from model.pinn_loss import StableQuantPINNLoss, extract_bb_bands
from model.model_registry import sanitize_filename


class WeightEMA:
    """Exponential Moving Average of model weights for better generalization.

    Delays tracking until ema_start to avoid unstable early weights,
    then linearly ramps decay toward decay_end over remaining epochs.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999,
                 decay_end: float = 0.9999, total_epochs: int = 100,
                 ema_start: int = 0):
        self.model = model
        self.decay = decay
        self.decay_end = decay_end
        self.total_epochs = total_epochs
        self.ema_start = ema_start
        self._epoch = 0
        self._current_decay = decay
        self.shadow = deepcopy(model.state_dict())
        self._enabled = False
        self._tracking = False

    def step(self):
        """Advance epoch and ramp decay linearly from decay to decay_end."""
        self._epoch += 1
        if self._epoch < self.ema_start:
            self._tracking = False
            self._current_decay = self.decay
        else:
            if not self._tracking:
                self._tracking = True
                self.shadow = deepcopy(self.model.state_dict())
            remaining = max(1, self.total_epochs - self.ema_start)
            progress = min(1.0, (self._epoch - self.ema_start) / remaining)
            self._current_decay = self.decay + (self.decay_end - self.decay) * progress

    def update(self):
        if not self._tracking:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self._current_decay).add_(param.data, alpha=1 - self._current_decay)

    def apply_shadow(self):
        self._enabled = True
        self._backup = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)

    def restore(self):
        if self._enabled:
            self.model.load_state_dict(self._backup)
            self._enabled = False


def _compute_class_weights(dataset, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced data."""
    counts = np.zeros(num_classes)
    for i in range(len(dataset)):
        _, direction, _ = dataset[i]
        counts[direction] += 1
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def pearson_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient (differentiable). Batch-level."""
    pred = pred - pred.mean()
    target = target - target.mean()
    pred = pred / (pred.std(unbiased=False) + 1e-8)
    target = target / (target.std(unbiased=False) + 1e-8)
    return (pred * target).mean()


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    """Pinball (quantile) loss for uncertainty quantification.

    L(y, ŷ_q) = max(q*(y-ŷ_q), (q-1)*(y-ŷ_q))

    q=0.10: penalizes over-prediction more → lower bound
    q=0.50: symmetric → median (equivalent to MAE/2)
    q=0.90: penalizes under-prediction more → upper bound
    """
    err = target - pred
    return torch.mean(torch.maximum(quantile * err, (quantile - 1) * err))


def kendall_loss(loss_dir: torch.Tensor, loss_price: torch.Tensor,
                 log_sigma_dir: torch.Tensor, log_sigma_price: torch.Tensor,
                 corr_loss: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-task loss with homoscedastic uncertainty weighting (Kendall et al. 2018).

    total = loss_dir * exp(-2σ₁) + σ₁ + loss_price * exp(-2σ₂) + σ₂ + α * corr_loss
    """
    precision_dir = torch.exp(-2 * log_sigma_dir)
    precision_price = torch.exp(-2 * log_sigma_price)
    weighted = loss_dir * precision_dir + log_sigma_dir + loss_price * precision_price + log_sigma_price
    if corr_loss is not None:
        weighted = weighted + 0.1 * corr_loss
    return weighted


def train_model(model: HybridModel, dataset: StockDataset, config,
                checkpoint_dir: str = "pretrained", scaler=None,
                progress_callback=None, should_stop=None,
                ts_code: str = "default",
                selected_features: list[str] | None = None,
                epochs: int | None = None,
                model_type: str = "Node Transformer (图注意力)",
                save_ckpt: bool = True,
                feature_group_indices: dict | None = None,
                raw_features: np.ndarray | None = None,
                raw_close: np.ndarray | None = None) -> dict:
    """Train with focal loss, Kendall dynamic weighting, Spearman rank loss, weight EMA."""
    device = torch.device(config.model.device)
    model = model.to(device)

    # Purged walk-forward split: 70% train, gap, 30% val (no leakage)
    train_indices, val_indices = dataset.time_split(train_ratio=0.70, purge_window=1200)

    # Run MRMR on training fold only — prevents data leakage from validation set
    if selected_features is None and raw_features is not None and raw_close is not None:
        from data.feature_selector import select_features_on_split, save_feature_selection, get_feature_group_indices
        selected_features, feature_mask = select_features_on_split(
            raw_features, raw_close, train_indices, k=None, horizon=10
        )
        if ts_code and checkpoint_dir:
            save_feature_selection(selected_features,
                                   f"{checkpoint_dir}/{sanitize_filename(ts_code)}_features.json")
        dataset.apply_feature_mask(feature_mask)
        # Rebuild grouped indices from newly selected features
        feature_group_indices = get_feature_group_indices(selected_features)
    elif selected_features is not None:
        from data.feature_selector import get_feature_mask
        mask = get_feature_mask(selected_features)
        dataset.apply_feature_mask(mask)
    val_n = len(val_indices)
    if val_n > 0:
        # Further split val into val (first 2/3) and test (last 1/3) chronologically
        val_split = int(val_n * 0.67)
        val_idx = val_indices[:val_split]
        test_idx = val_indices[val_split:]
    else:
        val_idx = val_indices
        test_idx = []

    train_ds = TimeOrderedSubset(dataset, train_indices)
    val_ds = TimeOrderedSubset(dataset, val_idx)

    batch_size = min(config.model.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=min(config.model.batch_size, max(1, len(val_ds))),
                            shuffle=False, collate_fn=collate_fn, drop_last=False)

    class_weights = _compute_class_weights(train_ds, num_classes=2)
    class_weights = class_weights.to(device)

    n_epochs = epochs if epochs is not None else config.model.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.model.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.model.lr * 3, epochs=n_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.2, anneal_strategy="cos"
    )
    ce_loss = FocalLoss(gamma=2.0, alpha=class_weights)
    mse_loss = nn.MSELoss()  # 保留 MSE 用于基础回归
    lambda_quant = getattr(config.model, 'lambda_quant', 0.1)
    use_pinn = getattr(config.model, 'use_pinn_loss', True)
    pinn_loss = StableQuantPINNLoss(lambda_quant=lambda_quant) if use_pinn else None
    ema_start = max(1, n_epochs // 2)
    ema = WeightEMA(model, decay=0.999, decay_end=0.9999,
                     total_epochs=n_epochs, ema_start=ema_start)

    best_val = float("inf")
    best_acc = 0.0
    patience_counter = 0
    noise_std = 0.008

    for epoch in range(n_epochs):
        if should_stop and should_stop():
            break

        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for x, direction, price in train_loader:
            if should_stop and should_stop():
                break
            x, direction, price = x.to(device), direction.to(device), price.to(device)
            x = x + torch.randn_like(x) * noise_std
            dir_out, price_out = model(x)  # price_out: (B, 3) q10, q50, q90

            loss_dir = ce_loss(dir_out, direction)
            # Pinball quantile loss for price (3 quantiles)
            loss_q10 = pinball_loss(price_out[:, 0], price, 0.10)
            loss_q50 = pinball_loss(price_out[:, 1], price, 0.50)
            loss_q90 = pinball_loss(price_out[:, 2], price, 0.90)
            loss_price = (loss_q10 + loss_q50 + loss_q90) / 3.0
            # Bollinger band soft constraint on outer quantiles
            if pinn_loss is not None:
                upper_pct, lower_pct = extract_bb_bands(x)
                # Penalize q90 > upper_band or q10 < lower_band
                bb_penalty = (F.relu(price_out[:, 2].unsqueeze(-1) - upper_pct) +
                              F.relu(lower_pct - price_out[:, 0].unsqueeze(-1))).mean()
                loss_price = loss_price + 0.1 * bb_penalty
            # Ranking loss uses q50 (median prediction)
            corr = pearson_corr(price_out[:, 1], price)
            loss_rank = 1.0 - corr
            loss = kendall_loss(loss_dir, loss_price,
                               model.log_sigma_dir, model.log_sigma_price, loss_rank)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update()
            train_losses.append(loss.item())
            train_correct += (dir_out.argmax(1) == direction).sum().item()
            train_total += direction.size(0)

        model.eval()
        ema.apply_shadow()
        val_losses = []
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []
        all_price_preds = []   # (N,) predicted price deltas
        all_price_actuals = []  # (N,) actual price changes
        with torch.no_grad():
            for x, direction, price in val_loader:
                x, direction, price = x.to(device), direction.to(device), price.to(device)
                dir_out, price_out = model(x)  # price_out: (B, 3)
                loss_dir_v = ce_loss(dir_out, direction)
                loss_q10_v = pinball_loss(price_out[:, 0], price, 0.10)
                loss_q50_v = pinball_loss(price_out[:, 1], price, 0.50)
                loss_q90_v = pinball_loss(price_out[:, 2], price, 0.90)
                loss_price_v = (loss_q10_v + loss_q50_v + loss_q90_v) / 3.0
                if pinn_loss is not None:
                    upper_pct, lower_pct = extract_bb_bands(x)
                    bb_penalty = (F.relu(price_out[:, 2].unsqueeze(-1) - upper_pct) +
                                  F.relu(lower_pct - price_out[:, 0].unsqueeze(-1))).mean()
                    loss_price_v = loss_price_v + 0.1 * bb_penalty
                corr_v = pearson_corr(price_out[:, 1], price)  # q50 median
                loss_rank_v = 1.0 - corr_v
                loss = kendall_loss(loss_dir_v, loss_price_v,
                                   model.log_sigma_dir, model.log_sigma_price, loss_rank_v)
                val_losses.append(loss.item())
                preds = dir_out.argmax(1)
                val_correct += (preds == direction).sum().item()
                val_total += direction.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(direction.cpu().tolist())
                all_price_preds.extend(price_out[:, 1].cpu().tolist())  # q50 for IC
                all_price_actuals.extend(price.cpu().tolist())
        ema.restore()
        ema.step()

        avg_train = np.mean(train_losses) if train_losses else 0
        avg_val = np.mean(val_losses) if val_losses else 0
        val_acc = val_correct / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)

        per_class = {}
        for cls in range(2):
            mask = [l == cls for l in all_labels]
            if sum(mask) > 0:
                per_class[cls] = sum(1 for p, l in zip(all_preds, all_labels) if p == l == cls) / sum(mask)
            else:
                per_class[cls] = 0.0
        macro_acc = np.mean(list(per_class.values()))

        # IC / Rank IC — correlation between predicted price change and actual
        ic = 0.0
        rank_ic = 0.0
        if len(all_price_preds) > 1:
            pp = np.array(all_price_preds, dtype=np.float64)
            pa = np.array(all_price_actuals, dtype=np.float64)
            # Pearson IC
            pp_std = np.std(pp); pa_std = np.std(pa)
            if pp_std > 0 and pa_std > 0:
                ic = float(np.corrcoef(pp, pa)[0, 1])
            # Spearman Rank IC = Pearson on ranks
            try:
                from scipy.stats import spearmanr as _spearmanr
                rank_ic = float(_spearmanr(pp, pa)[0])
            except ImportError:
                # Manual Spearman: Pearson on ranks
                def _rank(arr):
                    order = np.argsort(arr)
                    ranks = np.empty_like(order, dtype=np.float64)
                    ranks[order] = np.arange(1, len(arr) + 1)
                    return ranks
                rp, ra = _rank(pp), _rank(pa)
                rank_ic = float(np.corrcoef(rp, ra)[0, 1])

        # Log IC metrics
        if epoch % max(1, n_epochs // 5) == 0 or macro_acc >= best_acc:
            print(f"[IC] epoch {epoch:3d} | IC={ic:+.4f}  RankIC={rank_ic:+.4f}  "
                  f"macro_acc={macro_acc:.3f}  val_loss={avg_val:.4f}")

        if progress_callback:
            progress_callback(epoch, avg_train, avg_val, val_acc)

        if macro_acc >= 0.95:
            if progress_callback:
                progress_callback(epoch, avg_train, avg_val, val_acc)
            ema.apply_shadow()
            checkpoint = {
                "model_state_dict": deepcopy(model.state_dict()),
                "epoch": epoch, "val_loss": avg_val, "val_acc": val_acc,
                "model_type": model_type,
                "ic": ic, "rank_ic": rank_ic,
            }
            ema.restore()
            if scaler is not None:
                checkpoint["scaler"] = scaler
            if selected_features is not None:
                checkpoint["selected_features"] = selected_features
            if feature_group_indices is not None:
                checkpoint["feature_group_indices"] = feature_group_indices
            if save_ckpt:
                torch.save(checkpoint, f"{checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt")
                _update_index(checkpoint_dir, ts_code, model_type, val_acc, macro_acc, selected_features, ic, rank_ic)
            break

        if macro_acc > best_acc:
            best_acc = macro_acc
            best_val = avg_val
            patience_counter = 0
            ema.apply_shadow()
            checkpoint = {
                "model_state_dict": deepcopy(model.state_dict()),
                "epoch": epoch, "val_loss": avg_val, "val_acc": val_acc,
                "model_type": model_type,
                "ic": ic, "rank_ic": rank_ic,
            }
            ema.restore()
            if scaler is not None:
                checkpoint["scaler"] = scaler
            if selected_features is not None:
                checkpoint["selected_features"] = selected_features
            if feature_group_indices is not None:
                checkpoint["feature_group_indices"] = feature_group_indices
            if save_ckpt:
                torch.save(checkpoint, f"{checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt")
                _update_index(checkpoint_dir, ts_code, model_type, val_acc, macro_acc, selected_features, ic, rank_ic)
        else:
            patience_counter += 1

        if patience_counter >= config.model.early_stop_patience:
            break

    return {"train_loss": avg_train, "val_loss": avg_val, "val_acc": val_acc,
            "macro_acc": macro_acc, "per_class": per_class, "epoch": epoch,
            "ic": ic, "rank_ic": rank_ic}


def _update_index(checkpoint_dir, ts_code, model_type, val_acc, macro_acc,
                  selected_features, ic=0.0, rank_ic=0.0):
    from model.model_registry import update_model_index
    update_model_index(checkpoint_dir, ts_code, model_type,
                        val_acc=val_acc, macro_acc=macro_acc,
                        ic=ic, rank_ic=rank_ic,
                        selected_features=selected_features)
