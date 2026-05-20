from PyQt6.QtCore import QThread, pyqtSignal
from data.ths_fetcher import THSFetcher
from data.gs_fetcher import GoldenSunFetcher
from data.features import compute_all_indicators
from data.preprocessor import preprocess, build_targets
from data.stock_screener import screen_stocks
from data.market_rules import get_price_limit_pct
from storage.repository import Repository
from model.predictor import Predictor
from model.model_registry import sanitize_filename
from mcp.claude_client import AIAnalyzer
import numpy as np
import pandas as pd
import torch


def predict_rolling(df: pd.DataFrame, ts_code: str, predictor: Predictor,
                    steps: int = 6, freq_min: int = 20) -> list[dict]:
    """Shared iterative rolling prediction with momentum decay and mean-reversion.

    Used by both MainWindow._predict_rolling and QuickPredictWorker for consistent results.
    """
    from data.features import compute_all_indicators
    from data.preprocessor import preprocess

    predictions = []
    df_ext = df.copy()
    indicator_cols: list[str] = []

    hist_close = df_ext["close"].values.astype(float)
    hist_mean = float(np.mean(hist_close[-60:])) if len(hist_close) >= 20 else float(hist_close[-1])
    hist_std = float(np.std(hist_close[-60:])) + 1e-10
    first_price = float(hist_close[-1])

    for step in range(steps):
        if indicator_cols:
            existing = [c for c in indicator_cols if c in df_ext.columns]
            if existing:
                df_ext = df_ext.drop(columns=existing)
            indicator_cols = []

        ind = compute_all_indicators(df_ext)
        indicator_cols = list(ind.columns)
        for col in indicator_cols:
            df_ext[col] = ind[col].values

        feat_arr, _ = preprocess(df_ext, fit_scaler=False)
        current_price = float(df_ext["close"].iloc[-1])

        pred = predictor.predict_one(feat_arr, current_price, ts_code, mc_samples=30)

        # ── Momentum decay ──
        momentum_decay = max(0.25, 0.92 ** step)

        # ── Mean-reversion pull ──
        deviation = (current_price - hist_mean) / hist_std
        z_score_sq = deviation ** 2
        reversion_strength = max(0.0, min(0.25, z_score_sq * 0.02))
        if deviation > 0:
            reversion_delta = -reversion_strength * 0.0008
        else:
            reversion_delta = reversion_strength * 0.0008

        # ── Noise: scale to volatility ──
        price_std = pred.get("price_std", 0.0005)
        if price_std and price_std > 0:
            noise = np.random.normal(0, price_std * 0.3)
        else:
            noise = np.random.normal(0, 0.0003)

        # ── Build adjusted delta ──
        base_delta = pred.get("price_delta", 0.0)
        adjusted_delta = base_delta * momentum_decay + reversion_delta + noise
        adjusted_delta = float(adjusted_delta)

        target = current_price * (1 + adjusted_delta)

        # Clip to daily price limit
        limit_pct = get_price_limit_pct(ts_code)
        limit_up = first_price * (1 + limit_pct)
        limit_down = first_price * (1 - limit_pct)
        target = max(limit_down, min(limit_up, target))

        pred["price_delta"] = round(adjusted_delta, 6)
        pred["target_price"] = round(target, 2)
        pred["price_lower"] = round(max(limit_down, target * 0.97), 2)
        pred["price_upper"] = round(min(limit_up, target * 1.03), 2)
        pred["step"] = step + 1
        pred["ts_code"] = ts_code
        pred["created_at"] = pd.Timestamp.now()
        predictions.append(pred)

        if step < steps - 1:
            avg_vol = float(df_ext["volume"].iloc[-20:].mean()) if len(df_ext) >= 20 else float(df_ext["volume"].iloc[-1])

            blended_close = current_price * (1 + adjusted_delta * 0.4)
            blended_close = max(limit_down, min(limit_up, blended_close))
            move = abs(adjusted_delta * current_price)

            new_high = max(blended_close, current_price) + move * np.random.uniform(0.05, 0.2)
            new_low = min(blended_close, current_price) - move * np.random.uniform(0.05, 0.2)
            new_row_data = {
                "open": current_price,
                "close": blended_close,
                "high": min(limit_up, new_high),
                "low": max(limit_down, new_low),
                "volume": avg_vol * (1 + np.random.uniform(-0.15, 0.15)),
            }

            if "trade_time" in df_ext.columns:
                last_time = pd.to_datetime(df_ext["trade_time"].iloc[-1])
                new_row_data["trade_time"] = last_time + pd.Timedelta(minutes=freq_min)
            for c in df_ext.columns:
                if c not in indicator_cols and c not in new_row_data:
                    new_row_data[c] = df_ext[c].iloc[-1] if len(df_ext) > 0 else 0
            new_row = pd.DataFrame([new_row_data])
            df_ext = pd.concat([df_ext, new_row], ignore_index=True)

    return predictions


class DataFetchWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, fetcher: THSFetcher, repo: Repository, stocks: list[str],
                 gs: GoldenSunFetcher | None = None):
        super().__init__()
        self.fetcher = fetcher
        self.gs = gs
        self.repo = repo
        self.stocks = stocks

    def _fetch(self, ts_code: str) -> pd.DataFrame:
        """Incremental fetch: only get new data, merge with existing. Try 金太阳 first."""
        existing = self.repo.get_minutes(ts_code, freq="5min")
        if not existing.empty:
            latest = pd.to_datetime(existing["trade_time"].max())
            from datetime import datetime
            if (datetime.now() - latest).days < 1 and len(existing) >= 100:
                return existing.sort_values("trade_time").reset_index(drop=True)

        new_df = pd.DataFrame()
        if self.gs:
            try:
                new_df = self.gs.fetch_recent_mins(ts_code, days=30, freq="5min")
            except Exception:
                pass
        if new_df.empty:
            new_df = self.fetcher.fetch_recent_mins(ts_code, days=30, freq="5min")
        if new_df.empty:
            return existing if not existing.empty else new_df
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["trade_time"], keep="last")
            return combined.sort_values("trade_time").reset_index(drop=True)
        return new_df.sort_values("trade_time").reset_index(drop=True)

    def run(self):
        total = len(self.stocks)
        for i, ts_code in enumerate(self.stocks):
            try:
                df = self._fetch(ts_code)
                if df is not None and not df.empty:
                    self.repo.insert_minutes(df)
            except Exception as e:
                self.error.emit(str(e))
            self.progress.emit(i + 1, total)
        self.repo.purge_old_minutes(90)
        self.finished.emit()


class TrainingWorker(QThread):
    epoch_update = pyqtSignal(int, float, float, float)
    phase_update = pyqtSignal(str)       # status messages for UI
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model_or_none, dataset_or_repo, config, checkpoint_dir: str, scaler=None,
                 ts_code: str = "default", selected_features: list[str] | None = None,
                 epochs_override: int | None = None, model_type: str = "Node Transformer (图注意力)",
                 # New lazy-init params (when dataset_or_repo is a Repository)
                 repo=None, selected_stock: str | None = None,
                 model_name: str = "", fetcher=None, gs=None):
        super().__init__()
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.scaler = scaler
        self.ts_code = ts_code
        self.selected_features = selected_features
        self.epochs_override = epochs_override
        self.model_type = model_type
        self._stop = False

        # If repo is provided, we do lazy data prep inside run()
        self._repo = repo
        self._selected_stock = selected_stock
        self._model_name = model_name or model_type
        self._lazy_model = model_or_none          # pre-built model (or None for lazy)
        self._lazy_dataset = dataset_or_repo      # pre-built dataset (or repo for lazy)
        self._fetcher = fetcher
        self._gs = gs

    def _prepare_data(self):
        """Run data prep in worker thread — keeps UI responsive."""
        if self._repo is None or self._selected_stock is None:
            # Legacy path: dataset already built, just use it
            self.model = self._lazy_model
            self.dataset = self._lazy_dataset
            return True

        from data.preprocessor import preprocess, build_targets
        from data.features import compute_all_indicators
        from model.dataset import StockDataset
        from datetime import datetime, timedelta
        import pandas as pd
        import numpy as np

        self.phase_update.emit("正在准备训练...")
        self.phase_update.emit("正在构建训练数据集...")
        one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        stocks = [self._selected_stock]
        sequences, targets_list, timestamps_list = [], [], []
        scaler = None

        for ts_code in stocks:
            for freq in ["1min", "5min"]:
                df = self._repo.get_minutes(ts_code, freq=freq, start=one_month_ago)
                if df is not None and len(df) >= self.config.model.seq_len + 10:
                    break
            # Auto-fetch if insufficient — try 金太阳 first, then 同花顺
            if df is None or len(df) < self.config.model.seq_len + 10:
                self.phase_update.emit(f"正在获取 {ts_code} 30天数据...")
                new_df = pd.DataFrame()
                if self._gs:
                    try:
                        new_df = self._gs.fetch_recent_mins(ts_code, days=30, freq="5min")
                    except Exception:
                        pass
                if new_df.empty and self._fetcher:
                    try:
                        new_df = self._fetcher.fetch_recent_mins(ts_code, days=30, freq="5min")
                    except Exception:
                        pass
                if not new_df.empty:
                    self._repo.insert_minutes(new_df)
                    df = new_df
            if df is None or len(df) < self.config.model.seq_len + 10:
                continue
            df = df.sort_values("trade_time")
            epoch_ts = pd.to_datetime(df["trade_time"]).apply(
                lambda t: t.timestamp()).values.astype(float)
            timestamps_list.append(epoch_ts)
            raw_close = df["close"].values.astype(float)
            ind = compute_all_indicators(df)
            for col in ind.columns:
                if col not in df.columns:
                    df[col] = ind[col].values
            feat_arr, sc = preprocess(df, fit_scaler=True)
            if scaler is None:
                scaler = sc
            # Save raw feature array + close for MRMR inside train_model (no leakage)
            self._raw_close = raw_close
            self._raw_features = feat_arr.copy()
            direction, price_change = build_targets(raw_close, horizon=10)
            tgts = np.stack([direction, price_change], axis=1)
            sequences.append(feat_arr)
            targets_list.append(tgts)

        if not sequences:
            self.error.emit("数据不足，无法训练")
            return False

        self.scaler = scaler
        self.dataset = StockDataset(sequences, targets_list, self.config.model.seq_len,
                                    horizon=10, dense=True, timestamps=timestamps_list)

        # Run MRMR on training fold only — no data leakage
        from data.feature_selector import select_features_on_split, get_feature_group_indices
        train_indices, _ = self.dataset.time_split(train_ratio=0.70, purge_window=1200)
        selected_features, _feature_mask = select_features_on_split(
            self._raw_features, self._raw_close, train_indices, k=None, horizon=10
        )
        self.selected_features = selected_features
        self.dataset.apply_feature_mask(_feature_mask)

        input_dim = len(selected_features)
        self.phase_update.emit(f"样本数: {len(self.dataset)} | 特征维度: {input_dim}")

        # Build feature group routing from selected features (local indices)
        self.feature_group_indices = get_feature_group_indices(selected_features)

        # Build model
        from gui.dialogs import MODEL_REGISTRY
        model_cls = MODEL_REGISTRY.get(self._model_name)
        if model_cls is None:
            self.error.emit(f"未知模型: {self._model_name}")
            return False
        # Only FreqOrchestrator-based models accept feature_group_indices
        from model.freq_orchestrator import FreqOrchestratorWrapper
        kwargs = dict(
            input_dim=input_dim,
            d_model=self.config.model.d_model,
            lstm_hidden=self.config.model.lstm_hidden,
            lstm_layers=self.config.model.lstm_layers,
            transformer_layers=self.config.model.transformer_layers,
            nhead=self.config.model.nhead,
            dropout=self.config.model.dropout,
            max_seq_len=self.config.model.seq_len,
            patch_len=self.config.model.patch_len,
        )
        if issubclass(model_cls, FreqOrchestratorWrapper):
            kwargs["feature_group_indices"] = self.feature_group_indices
        self.model = model_cls(**kwargs)
        self.model_type = self._model_name
        self.phase_update.emit(f"模型架构: {self.model_type}")
        return True

    def _run_ensemble_training(self):
        """Train ensemble: train all base models, then create ensemble checkpoint."""
        from model.ensemble import AllModelsWrapper

        if isinstance(self.model, AllModelsWrapper):
            return self._run_all_models_training()

        from model.trainer import train_model
        base_names = ["FT-iTransformer (时频协同)", "PINN (物理约束)"]
        all_metrics = {}

        for i, (base_model, name) in enumerate(zip(self.model.base_models, base_names)):
            if self._stop:
                break
            self.epoch_update.emit(0, 0, 0, 0)  # signal phase start
            metrics = train_model(
                base_model, self.dataset, self.config, self.checkpoint_dir,
                scaler=self.scaler,
                progress_callback=lambda e, tl, vl, va, ts=name:
                    self.epoch_update.emit(e, tl, vl, va),
                should_stop=lambda: self._stop,
                ts_code=f"{self.ts_code}_base{i}",
                selected_features=self.selected_features,
                epochs=self.epochs_override,
                model_type=name,
                save_ckpt=False,
                feature_group_indices=self.feature_group_indices,
                raw_features=self._raw_features if hasattr(self, '_raw_features') else None,
                raw_close=self._raw_close if hasattr(self, '_raw_close') else None,
            )
            all_metrics[name] = metrics

        # Train meta-learner if stacking
        if hasattr(self.model, 'meta') and not self._stop:
            self._train_stacking_meta()

        # Save ensemble checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "base_metrics": all_metrics,
        }
        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler
        if self.selected_features is not None:
            checkpoint["selected_features"] = self.selected_features
        if self.feature_group_indices is not None:
            checkpoint["feature_group_indices"] = self.feature_group_indices
        avg_acc = sum(m.get("val_acc", 0) for m in all_metrics.values()) / max(len(all_metrics), 1)

        torch.save(checkpoint, f"{self.checkpoint_dir}/{sanitize_filename(self.ts_code)}_best_model.pt")
        from model.model_registry import update_model_index
        update_model_index(self.checkpoint_dir, self.ts_code, self.model_type,
                            val_acc=avg_acc, macro_acc=avg_acc,
                            selected_features=self.selected_features)

        return {"val_acc": avg_acc, "val_loss": 0, "macro_acc": avg_acc,
                "per_class": {}, "epoch": 0, "ensemble": True, "base_metrics": all_metrics}

    def _run_all_models_training(self):
        """Train all 7 models independently, evaluate each, pick the best by accuracy."""
        from model.trainer import train_model
        from model.ensemble import ALL_MODEL_NAMES

        all_metrics = {}
        best_acc = 0.0
        best_idx = 0

        for i, model in enumerate(self.model.base_models):
            if self._stop:
                break
            name = ALL_MODEL_NAMES[i]
            self.epoch_update.emit(0, 0, 0, 0)

            is_sub_ensemble = hasattr(model, 'base_models') and isinstance(model.base_models, torch.nn.ModuleList)

            if is_sub_ensemble:
                # Ensemble wrapper: train its 3 base models first, then post-process
                sub_names = ["FT-iTransformer (时频协同)", "PINN (物理约束)"]
                sub_metrics = {}
                for j, (sub_model, sub_name) in enumerate(zip(model.base_models, sub_names)):
                    if self._stop:
                        break
                    sm = train_model(
                        sub_model, self.dataset, self.config, self.checkpoint_dir,
                        scaler=self.scaler,
                        progress_callback=lambda e, tl, vl, va, ts=f"{name}/{sub_name}":
                            self.epoch_update.emit(e, tl, vl, va),
                        should_stop=lambda: self._stop,
                        ts_code=f"{self.ts_code}_all_{i}_base{j}",
                        selected_features=self.selected_features,
                        epochs=self.epochs_override,
                        model_type=sub_name,
                        save_ckpt=False,
                        feature_group_indices=self.feature_group_indices,
                        raw_features=self._raw_features if hasattr(self, '_raw_features') else None,
                        raw_close=self._raw_close if hasattr(self, '_raw_close') else None,
                    )
                    sub_metrics[sub_name] = sm

                if hasattr(model, 'meta') and not self._stop:
                    self._train_stacking_meta_for(model, i)

                avg_acc = sum(m.get("macro_acc", m.get("val_acc", 0))
                              for m in sub_metrics.values()) / max(len(sub_metrics), 1)
                metrics = {"val_acc": avg_acc, "val_loss": 0, "macro_acc": avg_acc,
                           "per_class": {}, "epoch": 0, "ensemble": True, "base_metrics": sub_metrics}
            else:
                # Single model
                metrics = train_model(
                    model, self.dataset, self.config, self.checkpoint_dir,
                    scaler=self.scaler,
                    progress_callback=lambda e, tl, vl, va, ts=name:
                        self.epoch_update.emit(e, tl, vl, va),
                    should_stop=lambda: self._stop,
                    ts_code=f"{self.ts_code}_all_{i}",
                    selected_features=self.selected_features,
                    epochs=self.epochs_override,
                    model_type=name,
                    save_ckpt=False,
                    feature_group_indices=self.feature_group_indices,
                    raw_features=self._raw_features if hasattr(self, '_raw_features') else None,
                    raw_close=self._raw_close if hasattr(self, '_raw_close') else None,
                )

            all_metrics[name] = metrics
            acc = metrics.get("macro_acc", metrics.get("val_acc", 0))
            if acc > best_acc:
                best_acc = acc
                best_idx = i

        self.model.best_model_idx = best_idx
        self.model.best_model_name = ALL_MODEL_NAMES[best_idx]
        self.model.all_accuracies = {
            n: m.get("macro_acc", m.get("val_acc", 0)) for n, m in all_metrics.items()
        }

        best_model = self.model.base_models[best_idx]
        best_model_type = ALL_MODEL_NAMES[best_idx]
        checkpoint = {
            "model_state_dict": best_model.state_dict(),
            "model_type": best_model_type,
            "val_acc": best_acc,
            "macro_acc": best_acc,
            "all_accuracies": self.model.all_accuracies,
            "base_metrics": all_metrics,
        }
        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler
        if self.selected_features is not None:
            checkpoint["selected_features"] = self.selected_features
        if self.feature_group_indices is not None:
            checkpoint["feature_group_indices"] = self.feature_group_indices
        torch.save(checkpoint, f"{self.checkpoint_dir}/{sanitize_filename(self.ts_code)}_best_model.pt")
        from model.model_registry import update_model_index
        update_model_index(self.checkpoint_dir, self.ts_code, best_model_type,
                            val_acc=best_acc, macro_acc=best_acc,
                            selected_features=self.selected_features,
                            best_model_name=best_model_type,
                            all_accuracies=self.model.all_accuracies)

        return {"val_acc": best_acc, "val_loss": 0, "macro_acc": best_acc,
                "per_class": {}, "epoch": 0, "ensemble": True, "all_models": True,
                "best_model_name": best_model_type, "base_metrics": all_metrics}

    def _train_stacking_meta_for(self, stacking_ensemble, model_idx: int):
        """Train stacking meta-learner for a sub-ensemble within AllModelsWrapper."""
        from model.dataset import TimeOrderedSubset
        from torch.utils.data import DataLoader

        train_idx, val_idx = self.dataset.time_split(train_ratio=0.70, purge_window=1200)
        val_n = len(val_idx)
        if val_n > 0:
            val_split = int(val_n * 0.67)
            meta_val_idx = val_idx[val_split:]
        else:
            meta_val_idx = val_idx

        if len(meta_val_idx) < 10:
            return

        meta_ds = TimeOrderedSubset(self.dataset, meta_val_idx[:min(500, len(meta_val_idx))])
        meta_loader = DataLoader(meta_ds, batch_size=32, shuffle=False,
                                 collate_fn=self._meta_collate_fn)

        device = next(self.model.parameters()).device
        stacking_ensemble.to(device)
        stacking_ensemble.meta.to(device)
        opt = torch.optim.AdamW(stacking_ensemble.meta.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(20):
            if self._stop:
                break
            for x_batch, t_batch in meta_loader:
                x_batch = x_batch.to(device)
                y_dir = t_batch["direction"].to(device).long()
                y_price = t_batch["price_change"].to(device).float()

                base_outputs = []
                with torch.no_grad():
                    for m in stacking_ensemble.base_models:
                        m.to(device)
                        m.eval()
                        d, p = m(x_batch)
                        base_outputs.append((d, p))

                ctx = x_batch[:, -1, 73:74] if x_batch.shape[-1] > 73 else \
                      torch.zeros(x_batch.shape[0], 1, device=device)

                dir_out, price_out = stacking_ensemble.meta(base_outputs, ctx)
                loss = torch.nn.functional.cross_entropy(dir_out, y_dir) \
                       + 0.3 * torch.nn.functional.mse_loss(price_out.squeeze(-1), y_price)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _train_stacking_meta(self):
        """Train the stacking meta-learner on validation set predictions."""
        from model.dataset import TimeOrderedSubset
        from torch.utils.data import DataLoader

        # Use a small subset of validation data for meta-learner training
        train_idx, val_idx = self.dataset.time_split(train_ratio=0.70, purge_window=1200)
        val_n = len(val_idx)
        if val_n > 0:
            val_split = int(val_n * 0.67)
            meta_val_idx = val_idx[val_split:]  # test portion for meta training
        else:
            meta_val_idx = val_idx

        if len(meta_val_idx) < 10:
            return

        meta_ds = TimeOrderedSubset(self.dataset, meta_val_idx[:min(500, len(meta_val_idx))])
        meta_loader = DataLoader(meta_ds, batch_size=32, shuffle=False,
                                 collate_fn=self._meta_collate_fn)

        device = next(self.model.parameters()).device
        self.model.to(device)
        self.model.meta.to(device)
        opt = torch.optim.AdamW(self.model.meta.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(20):
            if self._stop:
                break
            total_loss = 0
            for x_batch, t_batch in meta_loader:
                x_batch = x_batch.to(device)
                y_dir = t_batch["direction"].to(device).long()
                y_price = t_batch["price_change"].to(device).float()

                # Get base model predictions
                base_outputs = []
                with torch.no_grad():
                    for m in self.model.base_models:
                        m.to(device)
                        m.eval()
                        d, p = m(x_batch)
                        base_outputs.append((d, p))

                # Market context
                ctx = x_batch[:, -1, 73:74] if x_batch.shape[-1] > 73 else \
                      torch.zeros(x_batch.shape[0], 1, device=device)

                dir_out, price_out = self.model.meta(base_outputs, ctx)
                loss = torch.nn.functional.cross_entropy(dir_out, y_dir) \
                       + 0.3 * torch.nn.functional.mse_loss(price_out.squeeze(-1), y_price)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

    @staticmethod
    def _meta_collate_fn(batch):
        x, direction, price = zip(*batch)
        X = torch.stack(x)
        T = {
            "direction": torch.stack(direction),
            "price_change": torch.stack(price),
        }
        return X, T

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # Phase 1: data prep (if not pre-built)
            if self._repo is not None and not self._prepare_data():
                return

            from model.trainer import train_model
            is_ensemble = hasattr(self.model, 'base_models') and isinstance(self.model.base_models, torch.nn.ModuleList)

            if is_ensemble:
                metrics = self._run_ensemble_training()
            else:
                metrics = train_model(
                    self.model, self.dataset, self.config, self.checkpoint_dir,
                    scaler=self.scaler,
                    progress_callback=lambda e, tl, vl, va: self.epoch_update.emit(e, tl, vl, va),
                    should_stop=lambda: self._stop,
                    ts_code=self.ts_code,
                    selected_features=self.selected_features,
                    epochs=self.epochs_override,
                    model_type=self.model_type,
                    feature_group_indices=self.feature_group_indices,
                    raw_features=self._raw_features if hasattr(self, '_raw_features') else None,
                    raw_close=self._raw_close if hasattr(self, '_raw_close') else None,
                )
            if self._stop:
                self.finished.emit("训练已取消")
            elif metrics.get("all_models"):
                best_name = metrics.get("best_model_name", "未知")
                self.finished.emit(
                    f"训练完成 | Val Acc: {metrics['val_acc']:.2%}"
                    f" | 最佳模型: {best_name} --为最高的那个模型"
                )
            else:
                self.finished.emit(f"训练完成 | Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['val_acc']:.2%}")
        except Exception as e:
            self.error.emit(str(e))


class PredictionWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, repo: Repository, checkpoint_dir: str, config, stocks: list[str]):
        super().__init__()
        self.repo = repo
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.stocks = stocks

    def run(self):
        import os
        results = []
        total = len(self.stocks)
        for i, ts_code in enumerate(self.stocks):
            try:
                ckpt = f"{self.checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt"
                if not os.path.exists(ckpt):
                    self.progress.emit(i + 1, total)
                    continue
                predictor = Predictor(ckpt, self.config)
                df = self.repo.get_minutes(ts_code)
                if df.empty:
                    self.progress.emit(i + 1, total)
                    continue
                df = df.sort_values("trade_time")
                ind = compute_all_indicators(df)
                for col in ind.columns:
                    if col not in df.columns:
                        df[col] = ind[col].values
                feat_arr, _ = preprocess(df, fit_scaler=True)
                current_price = float(df["close"].iloc[-1])
                pred = predictor.predict_one(feat_arr, current_price, ts_code)
                pred["ts_code"] = ts_code
                pred["created_at"] = pd.Timestamp.now()
                results.append(pred)
            except Exception as e:
                self.error.emit(str(e))
            self.progress.emit(i + 1, total)
        if results:
            self.repo.insert_predictions(pd.DataFrame(results))
        self.finished.emit()


class QuickPredictWorker(QThread):
    """Run prediction in background thread — keeps UI responsive."""
    result_ready = pyqtSignal(dict, list)  # final_pred, rolling_preds
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, predictor, repo, ts_code: str, freq_min: int = 20):
        super().__init__()
        self.predictor = predictor
        self.repo = repo
        self.ts_code = ts_code
        self.freq_min = freq_min

    def run(self):
        try:
            df = self.repo.get_minutes(self.ts_code)
            if df.empty:
                self.error.emit(f"{self.ts_code} 无分钟数据")
                return
            df = df.sort_values("trade_time")

            rolling_preds = predict_rolling(df, self.ts_code, self.predictor,
                                            steps=6, freq_min=self.freq_min)

            final_pred = rolling_preds[-1] if rolling_preds else None
            if final_pred:
                final_pred["ts_code"] = self.ts_code
                final_pred["created_at"] = pd.Timestamp.now()
                self.repo.insert_predictions(pd.DataFrame([final_pred]))

            self.result_ready.emit(final_pred, rolling_preds)
        except Exception as e:
            self.error.emit(str(e))
        self.finished.emit()


class AIAnalysisWorker(QThread):
    analysis_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, ai: AIAnalyzer, prediction: dict, behavior: dict | None = None,
                 news_text: str = ""):
        super().__init__()
        self.ai = ai
        self.prediction = prediction
        self.behavior = behavior
        self.news_text = news_text

    def run(self):
        try:
            result = self.ai.analyze_prediction(self.prediction, self.behavior,
                                                 self.news_text)
            self.analysis_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class OptimizationWorker(QThread):
    result_ready = pyqtSignal(dict)  # emits optimization suggestions
    error = pyqtSignal(str)

    def __init__(self, ai: AIAnalyzer, ts_code: str, pred: dict,
                 model_config, repo, model_registry):
        super().__init__()
        self.ai = ai
        self.ts_code = ts_code
        self.pred = pred
        self.model_config = model_config
        self.repo = repo
        self.model_registry = model_registry

    def run(self):
        try:
            from model.auto_optimizer import AutoOptimizer
            optimizer = AutoOptimizer(self.ai)
            model_context = self._gather_model_context()
            df_info = self._gather_data_info()
            result = optimizer.analyze_and_suggest(
                self.ts_code, self.pred, self.model_config,
                model_context, df_info,
            )
            result["ts_code"] = self.ts_code
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _gather_model_context(self) -> dict | None:
        entries = self.model_registry.get_models_for_stock(self.ts_code)
        valid = [e for e in entries if e.model_type != "未知"]
        if not valid:
            return None
        best = max(valid, key=lambda e: e.macro_acc)
        return {
            "model_type": best.model_type,
            "accuracy": best.macro_acc,
            "val_accuracy": best.val_acc,
        }

    def _gather_data_info(self) -> dict:
        df = self.repo.get_minutes(self.ts_code)
        if df is None or df.empty:
            return {"samples": 0}
        return {
            "samples": len(df),
            "date_range": (
                f"{df['trade_time'].min()} to {df['trade_time'].max()}"
                if "trade_time" in df.columns else "unknown"
            ),
            "frequencies_available": "5min",
        }


class ScreeningWorker(QThread):
    progress = pyqtSignal(int, int, pd.DataFrame)
    phase = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, top_n: int = 50):
        super().__init__()
        self.top_n = top_n

    def run(self):
        try:
            result = screen_stocks(None, self.top_n,
                                   on_progress=lambda cur, tot, df: self.progress.emit(cur, tot, df),
                                   on_phase=lambda msg: self.phase.emit(msg))
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
