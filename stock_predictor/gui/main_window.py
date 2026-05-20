from __future__ import annotations
from PyQt6.QtWidgets import (
    QMainWindow, QMenuBar, QToolBar, QStatusBar, QSplitter, QMessageBox, QInputDialog, QApplication, QLabel
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from config import AppConfig
from gui.stock_list_panel import StockListPanel
from gui.chart_canvas import ChartCanvas
from gui.prediction_dashboard import PredictionDashboard
from gui.dialogs import SettingsDialog, TrainDialog
from model.model_registry import ModelRegistry, sanitize_filename
from gui.workers import DataFetchWorker, TrainingWorker, PredictionWorker, QuickPredictWorker, AIAnalysisWorker, OptimizationWorker, ScreeningWorker
from data.ths_fetcher import THSFetcher
from data.efinance_fetcher import EFinanceFetcher
from data.gs_fetcher import GoldenSunFetcher
from data.behavior import analyze_trader_behavior
from data.news_fetcher import fetch_stock_news
from data.news_analyzer import analyze_news_impacts, compute_news_bias, apply_news_adjustment, has_news_impact
from data.trade_advice import generate_trade_advice
from data.market_rules import get_trading_session, get_session_description, is_trading_time, get_price_limit_pct, filter_trading_hours
from storage.repository import Repository
from model.predictor import Predictor, predict_batch
from data.preprocessor import preprocess, build_targets
from model.dataset import StockDataset
from model.lstm_transformer import HybridModel
from model.trainer import train_model
from mcp.claude_client import AIAnalyzer
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import deque


class MainWindow(QMainWindow):
    def closeEvent(self, event):
        """Clear all prediction records on exit, keep training models."""
        try:
            self.repo.delete_all_predictions()
        except Exception:
            pass
        self._prediction_cache.clear()
        super().closeEvent(event)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.repo = Repository(cfg)
        self.fetcher: THSFetcher = THSFetcher()
        self.ef_fetcher: EFinanceFetcher = EFinanceFetcher()  # baostock, no proxy issues
        self.gs: GoldenSunFetcher | None = None
        if cfg.data.gs_enabled:
            try:
                self.gs = GoldenSunFetcher(
                    host=cfg.data.gs_host, port=cfg.data.gs_port,
                    use_public=cfg.data.gs_use_public,
                )
                self.gs._connect()
            except Exception:
                self.gs = None
        self.predictors: dict[str, Predictor] = {}  # per-stock predictor cache
        self.model_registry = ModelRegistry(cfg.model.checkpoint_dir)
        # Cache: (final_pred, rolling_preds) per stock — persists across stock/timeframe switches
        self._prediction_cache: dict[str, tuple[dict, list[dict]]] = {}
        self.ai: AIAnalyzer | None = None
        self._scaler = None
        self._optimize_cooldown: dict[str, datetime] = {}  # prevent repeated optimization
        self._opt_worker: OptimizationWorker | None = None

        self.setWindowTitle("A股智能预测系统")
        self.resize(1400, 900)

        # Auto-predict timer
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._auto_predict_tick)
        self._auto_freq = "1min"
        self._auto_active = False
        self._auto_stocks: list[str] = []  # user-selected stocks (max 5) for real-time prediction
        self._last_predicted_price: float | None = None  # for online correction
        self._last_predict_time = None  # timestamp of last prediction
        self._correction_log: list[dict] = []  # correction history
        self._active_news_impacts: list[dict] = []  # current active news impacts with decay

        # Stock selection debounce — avoids cascade on rapid clicks/keyboard nav
        self._stock_debounce = QTimer(self)
        self._stock_debounce.setSingleShot(True)
        self._stock_debounce.setInterval(200)
        self._stock_debounce.timeout.connect(self._do_stock_load)
        self._pending_stock: str | None = None

        # CRC validation buffer: stores (feature_array, actual_price_delta) for recent bars
        self._val_buffer: deque[tuple[np.ndarray, float]] = deque(maxlen=12)

        # Training: single worker at a time, queued
        self._bg_workers: list[tuple] = []  # [(worker, ts_code), ...]
        self._train_queue: list[str] = []
        self._train_model_name: str = ""
        self._train_epochs: int | None = None
        self._active_train_count: int = 0
        self._active_train_codes: set[str] = set()

        # Animated prediction: 3s per point, grows prediction path live
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)
        self._anim_df_ext: pd.DataFrame | None = None
        self._anim_preds: list[dict] = []
        self._anim_step: int = 0
        self._anim_max_steps: int = 0
        self._anim_ts_code: str | None = None
        self._anim_indicator_cols: list[str] = []
        self._anim_freq_min: float = 0.05  # 3 seconds per bar

        self._setup_menus()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()
        if self.gs and self.gs.is_connected:
            self._gs_action.setChecked(True)

        if cfg.mcp.deepseek_api_key:
            self.ai = AIAnalyzer(cfg.mcp.deepseek_api_key, cfg.mcp.deepseek_model)

        self.showMaximized()
        QTimer.singleShot(500, self._sync_trained_stocks)
        QTimer.singleShot(600, self._load_cached_predictions)
        QTimer.singleShot(800, self._auto_sync_gs_stocks)
        QTimer.singleShot(2000, self._build_gs_cache)

    def _setup_menus(self):
        mb = self.menuBar()
        file_m = mb.addMenu("文件")
        act = QAction("设置", self)
        act.triggered.connect(self._open_settings)
        file_m.addAction(act)
        file_m.addAction("退出", self.close)

        data_m = mb.addMenu("数据")
        data_m.addAction("从金太阳同步股票列表", self._sync_gs_stock_list)
        data_m.addAction("刷新选股", self._refresh_screening)
        data_m.addAction("刷新分钟数据", self._refresh_minute_data)
        self._gs_action = QAction("连接金太阳实时行情", self)
        self._gs_action.setCheckable(True)
        self._gs_action.triggered.connect(self._toggle_gs)
        data_m.addAction(self._gs_action)

        model_m = mb.addMenu("模型")
        model_m.addAction("训练模型", self._open_train_dialog)
        model_m.addAction("预测全部", self._predict_all)
        model_m.addSeparator()
        model_m.addAction("清除所有模型", self._clear_all_models)

    def _setup_toolbar(self):
        tb = self.addToolBar("主工具栏")
        tb.addAction("刷新数据", self._refresh_minute_data)
        tb.addAction("选股", self._refresh_screening)
        tb.addAction("训练", self._open_train_dialog)
        tb.addAction("预测", self._quick_predict)
        self._auto_btn = QAction("▶ 实时预测", self)
        self._auto_btn.setCheckable(True)
        self._auto_btn.triggered.connect(self._toggle_auto_predict)
        tb.addAction(self._auto_btn)

    def _setup_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)

        self.stock_list = StockListPanel(self.repo)
        self.stock_list.setMinimumWidth(180)

        self.chart = ChartCanvas()
        self.chart.setMinimumWidth(400)

        self.dashboard = PredictionDashboard()
        self.dashboard.setMinimumWidth(280)
        self.dashboard.setMaximumWidth(420)

        splitter.addWidget(self.stock_list)
        splitter.addWidget(self.chart)
        splitter.addWidget(self.dashboard)

        # Proportional: left 15%, center 55%, right 30%
        total = 1280
        splitter.setSizes([int(total * 0.15), int(total * 0.55), int(total * 0.30)])
        splitter.setStretchFactor(0, 0)  # stock list: fixed
        splitter.setStretchFactor(1, 1)  # chart: stretches
        splitter.setStretchFactor(2, 0)  # dashboard: fixed

        self.setCentralWidget(splitter)

        self.stock_list.stock_selected.connect(self._on_stock_selected)
        self.stock_list.batch_train_requested.connect(self._batch_train_stocks)
        self.stock_list.batch_predict_requested.connect(self._batch_predict_stocks)
        self.stock_list.cloud_search_requested.connect(self._on_cloud_search)
        self.stock_list.view_prediction_requested.connect(self._on_view_prediction)
        self.chart.timeframe_changed.connect(self._on_timeframe_changed)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Training status is shown inline with the main status message
        self._base_status = ""
        self._train_text = ""

        # Timer to clear "训练完成" message after 5s
        self._train_status_clear_timer = QTimer(self)
        self._train_status_clear_timer.setSingleShot(True)
        self._train_status_clear_timer.timeout.connect(self._clear_train_status)

        session = get_trading_session()
        session_text = "交易中" if is_trading_time() else session.value
        device_label = "GPU" if self.cfg.model.device == "cuda" else "CPU"
        if self.gs and self.gs.is_connected:
            src = "金太阳/TDX 实时"
        else:
            src = self.fetcher.source_name
        self._base_status = f"就绪 | {session_text} | 计算: {device_label} | 数据源: {src} | 规则: T+1"
        self._refresh_status()

    def _refresh_status(self):
        """Rebuild status bar message from base + training text."""
        if self._train_text:
            self.status_bar.showMessage(f"{self._base_status}  {self._train_text}")
        else:
            self.status_bar.showMessage(self._base_status)

    def _set_train_status(self, text: str, color: str = "#ffd700", auto_clear_s: int = 0):
        """Update the training indicator at bottom-left, inline after main status."""
        self._train_text = text
        self._refresh_status()
        self._train_status_clear_timer.stop()
        if auto_clear_s > 0:
            self._train_status_clear_timer.start(auto_clear_s * 1000)

    def _clear_train_status(self):
        self._train_text = ""
        self._refresh_status()

    # ── Per-stock predictor helpers ──
    def _predict_remote(self, ts_code: str) -> dict | None:
        """使用 DeepSeek API 远程预测，附带本地训练模型上下文和本地预测结果。

        先运行本地模型获取预测，再将本地模型参数 + 本地预测结果一并上传至云端，
        作为 DeepSeek 推理的辅助参考。
        """
        if self.ai is None:
            return None
        try:
            df_min = self._fetch_minutes(ts_code, freq="1min", days=90)
            if df_min.empty:
                return None
            current_price = float(df_min["close"].iloc[-1])
            df_daily = self.repo.get_daily(ts_code, start=(datetime.now() - timedelta(days=100)).strftime("%Y%m%d"))
            model_context = self._build_model_context(ts_code, df_min)
            return self.ai.predict_remote(ts_code, df_min, df_daily, current_price, model_context)
        except Exception:
            return None

    def _extract_model_params(self, ts_code: str) -> dict | None:
        """Extract full model parameter statistics from the best checkpoint.

        Loads the checkpoint's state_dict, computes per-layer mean/std/min/max,
        extracts training metadata, and returns a structured dict for cloud context.
        Token budget: ~2000 chars max for the serialized form.
        """
        import os
        ckpt_path = f"{self.cfg.model.checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt"
        if not os.path.exists(ckpt_path):
            return None

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            return None

        state_dict = ckpt.get("model_state_dict", {})
        if not state_dict:
            return None

        # Per-layer weight statistics (skip buffers like price_scale)
        layer_stats = []
        total_params = 0
        for name, tensor in state_dict.items():
            if name.endswith("price_scale") or tensor.numel() == 0:
                continue
            t = tensor.float()
            n = t.numel()
            total_params += n
            stat = {
                "name": name,
                "shape": list(tensor.shape),
                "params": n,
                "mean": round(float(t.mean()), 6),
                "std": round(float(t.std(unbiased=False) if n < 2 else t.std()), 6),
                "min": round(float(t.min()), 6),
                "max": round(float(t.max()), 6),
            }
            # Quantiles for large layers
            if n > 10000:
                qs = [float(t.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
                stat["quantiles"] = [round(v, 6) for v in qs]
            layer_stats.append(stat)

        # Architecture summary
        arch = {
            "model_type": ckpt.get("model_type", "未知"),
            "total_params": total_params,
            "num_layers": len(layer_stats),
            "epoch": ckpt.get("epoch", 0),
            "val_loss": round(float(ckpt.get("val_loss", 0)), 4),
            "val_acc": round(float(ckpt.get("val_acc", 0)), 4),
        }

        # Feature list
        features = ckpt.get("selected_features")
        if features is None:
            if self.model_registry.has_models(ts_code):
                features = self.model_registry.get_features_for_stock(ts_code)

        return {
            "architecture": arch,
            "layer_params": layer_stats,
            "feature_list": features,
        }

    def _build_model_context(self, ts_code: str, df_min: pd.DataFrame | None = None) -> dict | None:
        """Gather local trained model metadata, full parameter stats, AND local prediction.

        Returns a dict with:
          - model_type, accuracy, best_model_name, all_models (model metadata)
          - model_params: {architecture, layer_params, feature_list} (weight statistics)
          - local_prediction: {direction, confidence, target_price} (local model output)
        """
        entries = self.model_registry.get_models_for_stock(ts_code)
        valid = [e for e in entries if e.model_type != "未知"]
        if not valid:
            return None
        best = max(valid, key=lambda e: e.macro_acc)
        ctx = {
            "model_type": best.model_type,
            "accuracy": best.macro_acc,
            "val_accuracy": best.val_acc,
            "best_model_name": best.best_model_name,
            "all_models": [
                {"type": e.model_type, "acc": e.macro_acc}
                for e in valid
            ],
        }

        # Extract full model parameter statistics from checkpoint
        model_params = self._extract_model_params(ts_code)
        if model_params:
            ctx["model_params"] = model_params

        # Run local model to get its own prediction as context for cloud
        predictor = self._get_predictor(ts_code)
        if predictor and df_min is not None and len(df_min) >= self.cfg.model.seq_len:
            try:
                from data.features import compute_all_indicators
                from data.preprocessor import preprocess
                df = df_min.sort_values("trade_time").copy()
                ind = compute_all_indicators(df)
                for col in ind.columns:
                    if col not in df.columns:
                        df[col] = ind[col].values
                feat_arr, _ = preprocess(df, fit_scaler=False)
                current_price = float(df["close"].iloc[-1])
                local_pred = predictor.predict_one(feat_arr, current_price, ts_code)
                if local_pred.get("direction_conf", 0) > 0:
                    ctx["local_prediction"] = {
                        "direction": local_pred.get("direction", "flat"),
                        "confidence": round(float(local_pred.get("direction_conf", 0)), 4),
                        "target_price": round(float(local_pred.get("target_price", current_price)), 2),
                    }
            except Exception:
                ctx["local_prediction"] = None
        else:
            ctx["local_prediction"] = None

        return ctx

    def _get_predictor(self, ts_code: str) -> Predictor | None:
        """Return the trained predictor for a specific stock, loading from disk if needed."""
        if ts_code in self.predictors:
            return self.predictors[ts_code]
        ckpt = f"{self.cfg.model.checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt"
        import os
        if os.path.exists(ckpt):
            p = Predictor(ckpt, self.cfg)
            self.predictors[ts_code] = p
            return p
        return None

    def _has_predictor(self, ts_code: str) -> bool:
        """Check if a valid model checkpoint file exists for the given stock."""
        if ts_code in self.predictors:
            return True
        import os
        ckpt_path = f"{self.cfg.model.checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt"
        return os.path.exists(ckpt_path)

    # ── Slots ──
    def _on_stock_selected(self, ts_code: str):
        """Debounced stock selection. Actual load happens in _do_stock_load."""
        self._pending_stock = ts_code
        self._stock_debounce.start()

    def _do_stock_load(self):
        """Perform the actual stock data loading (debounced)."""
        ts_code = self._pending_stock
        if not ts_code:
            return

        cached_entry = self._prediction_cache.get(ts_code)
        cached_pred, rolling_preds = cached_entry if cached_entry else (None, None)

        self._load_chart(ts_code, self._auto_freq,
                         show_prediction=cached_pred is not None,
                         cached_prediction=cached_pred,
                         cached_rolling_preds=rolling_preds)
        self._load_model_info(ts_code)
        self._load_behavior(ts_code)
        self._load_news(ts_code)
        self._refresh_realtime_quote(ts_code)
        has_model = self._has_predictor(ts_code)
        if has_model and self._auto_active:
            self._auto_predict_tick()
        elif cached_pred:
            self._show_prediction(cached_pred)
            df = self.repo.get_minutes(ts_code)
            if not df.empty:
                advice = generate_trade_advice(cached_pred, df, ts_code=ts_code)
                self.dashboard.show_trade_advice(advice)
            self.status_bar.showMessage(f"{ts_code} — 已显示预测结果")
        else:
            self.chart.clear_prediction_strip()
            self.dashboard.clear_trade_advice()
            self.dashboard.clear_rolling_prediction()
            self.status_bar.showMessage(f"{ts_code} — 原始数据（请先训练模型）")

    def _update_stock_colors(self):
        """Notify stock list panel which stocks have cached predictions."""
        predicted = set(self._prediction_cache.keys())
        self.stock_list.set_predicted_stocks(predicted)

    def _sync_trained_stocks(self):
        """Sync trained-stock markers to the stock list panel."""
        self.model_registry.refresh()
        trained = set(self.model_registry.list_trained_stocks())
        self.stock_list.set_trained_stocks(trained)

    def _load_cached_predictions(self):
        """Load latest predictions from SQLite into in-memory cache on startup."""
        try:
            df = self.repo.get_latest_predictions()
            if df.empty:
                return
            count = 0
            for _, row in df.iterrows():
                code = row["ts_code"]
                if code in self._prediction_cache:
                    continue
                def _safe_float(v, default=0.0):
                    try:
                        f = float(v)
                        return f if f == f else default  # NaN check
                    except (ValueError, TypeError):
                        return default

                pred = {
                    "direction": row["direction"] or "flat",
                    "direction_conf": _safe_float(row.get("direction_conf"), 0.5),
                    "target_price": _safe_float(row.get("target_price"), 0.0),
                    "price_lower": _safe_float(row.get("price_lower"), 0.0),
                    "price_upper": _safe_float(row.get("price_upper"), 0.0),
                    "ts_code": code,
                    "created_at": row["created_at"],
                }
                # Cached predictions store single point; rolling regenerated on demand
                self._prediction_cache[code] = (pred, [])
                count += 1
            if count > 0:
                self._update_stock_colors()
                self.status_bar.showMessage(
                    f"{self._base_status}  |  已加载 {count} 条历史预测")
        except Exception:
            pass  # DB not ready or empty — silently skip

    def _on_view_prediction(self, ts_code: str):
        """Handle '查看' button click in stock list — select stock and show prediction."""
        # Select the stock in the list
        self.stock_list.select_stock(ts_code)
        # Trigger stock load to show prediction in dashboard
        self._on_stock_selected(ts_code)

    def _fetch_minutes(self, ts_code: str, freq: str = "5min", days: int = 30) -> pd.DataFrame:
        """Fetch minute data incrementally: only get new data, merge with existing.

        所有时间戳均以北京时间为准。交易时段外的数据（午休、盘后）会被自动剔除。
        """
        now = datetime.now()
        # +3 天缓冲确保能覆盖周末/节假日后的前一个交易日
        cutoff = now - timedelta(days=days + 3)

        latest_ts = None
        # Check existing data first
        existing = self.repo.get_minutes(ts_code, freq=freq)
        if not existing.empty:
            # Filter: keep only data within date range and not in the future
            existing["trade_time"] = pd.to_datetime(existing["trade_time"])
            existing = existing[(existing["trade_time"] >= cutoff) & (existing["trade_time"] <= now)]
            existing = filter_trading_hours(existing)
            latest_ts = existing["trade_time"].max() if not existing.empty else None
            if latest_ts is not None and (now - latest_ts).days < 1 and len(existing) >= 100:
                # In trading hours, new bars appear every minute — don't short-circuit
                from data.market_rules import is_trading_time
                if not is_trading_time():
                    return self._trim_trading_days(existing, days)

        # Try 金太阳 / TDX first
        new_df = pd.DataFrame()
        if self.gs:
            try:
                new_df = self.gs.fetch_recent_mins(ts_code, days=days, freq=freq)
            except Exception:
                pass
        if new_df.empty:
            new_df = self.fetcher.fetch_recent_mins(ts_code, days=days, freq=freq)
        if new_df.empty:
            new_df = self.ef_fetcher.fetch_recent_mins(ts_code, days=days, freq=freq)
        if new_df.empty:
            result = existing if not existing.empty else new_df
            return self._trim_trading_days(result, days)

        # Filter new data: drop non-trading-hour timestamps, future timestamps, apply days cutoff
        if not new_df.empty:
            new_df["trade_time"] = pd.to_datetime(new_df["trade_time"])
            new_df = filter_trading_hours(new_df)
            new_df = new_df[(new_df["trade_time"] >= cutoff) & (new_df["trade_time"] <= now)]
            # Only keep bars newer than what we already have (incremental)
            if latest_ts is not None:
                new_df = new_df[new_df["trade_time"] > latest_ts]

        # If no truly new data, return existing
        if new_df.empty:
            return self._trim_trading_days(existing, days)

        # Merge existing + new, deduplicate by trade_time
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["trade_time"], keep="last")
            combined = combined.sort_values("trade_time").reset_index(drop=True)
            return self._trim_trading_days(combined, days)
        return self._trim_trading_days(new_df.sort_values("trade_time").reset_index(drop=True), days)

    @staticmethod
    def _trim_trading_days(df: pd.DataFrame, max_days: int) -> pd.DataFrame:
        """Keep only the last max_days trading days of data."""
        if df.empty or "trade_time" not in df.columns:
            return df
        dates = pd.to_datetime(df["trade_time"]).dt.date
        unique_dates = sorted(dates.unique())
        if len(unique_dates) <= max_days:
            return df
        keep_from = unique_dates[-max_days]
        return df[dates >= keep_from]

    def _trim_for_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim df to display days: 2 for 1min, 30 otherwise."""
        days = 2 if self._auto_freq == "1min" else 30
        return self._trim_trading_days(df, days)

    def _load_chart(self, ts_code: str, freq: str = "1min", show_prediction: bool = False,
                    cached_prediction: dict | None = None,
                    cached_rolling_preds: list[dict] | None = None):
        if freq == "daily":
            # 取90个日历日以确保覆盖至少30个交易日（跳过周末和节假日）
            cutoff = (datetime.now() - timedelta(days=90)).date()
            df = self.repo.get_daily(ts_code, start=cutoff)
            if df.empty:
                self.status_bar.showMessage(f"正在获取 {ts_code} 日线...")
                df = self.fetcher.get_daily(ts_code)
                if df.empty:
                    df = self.ef_fetcher.get_daily(ts_code)  # baostock fallback
                if not df.empty:
                    self.repo.insert_daily(df)
                    df = df[df["trade_date"] >= cutoff]
            # 截取最近30个交易日
            if not df.empty and len(df) > 30:
                df = df.iloc[-30:]
            if not df.empty:
                pred = cached_prediction if show_prediction else None
                if show_prediction and not pred:
                    pred = self._get_latest_pred(ts_code)
                self.chart.plot_kline(df, ts_code, freq, prediction=pred,
                                      rolling_preds=cached_rolling_preds)
                self.status_bar.showMessage(f"已加载 {ts_code} 日线")
            else:
                self.status_bar.showMessage(f"无 {ts_code} 日线数据")
            return

        # 分钟线取当日+上一日，其他频率取30天
        days = 2 if freq == "1min" else 30
        df_min = self._fetch_minutes(ts_code, freq=freq, days=days)
        if df_min.empty:
            self.status_bar.showMessage(f"无 {ts_code} {freq} 数据")
            return
        # Trim to exactly N trading days (calendar cutoff in _fetch_minutes has buffer)
        if "trade_time" in df_min.columns:
            dates = pd.to_datetime(df_min["trade_time"]).dt.date
            unique_dates = sorted(dates.unique())
            if len(unique_dates) > days:
                keep_from = unique_dates[-days]
                df_min = df_min[dates >= keep_from]
        # Store any new data back to DB
        try:
            self.repo.insert_minutes(df_min)
        except Exception:
            pass
        pred = cached_prediction if show_prediction else None
        if show_prediction and not pred:
            pred = self._get_latest_pred(ts_code)
        freq_label = {"1min": "分钟线", "daily": "日线"}.get(freq, freq)
        self.chart.plot_kline(df_min, ts_code, freq, prediction=pred,
                              rolling_preds=cached_rolling_preds)
        self.status_bar.showMessage(f"已加载 {ts_code} {freq_label}")

    def _get_latest_pred(self, ts_code: str) -> dict | None:
        """Get latest cached prediction for a stock, or None."""
        preds = self.repo.get_latest_predictions()
        if not preds.empty:
            row = preds[preds["ts_code"] == ts_code]
            if not row.empty:
                return row.iloc[0].to_dict()
        return None

    def _refresh_realtime_quote(self, ts_code: str):
        """Fetch real-time Level-1 quote from 金太阳 and update dashboard."""
        if not self.gs or not self.gs.is_connected:
            self.dashboard.show_realtime_quote(None)
            return
        try:
            quote = self.gs.get_realtime_quote(ts_code)
            self.dashboard.show_realtime_quote(quote)
        except Exception:
            self.dashboard.show_realtime_quote(None)

    def _load_model_info(self, ts_code: str):
        """Show trained model accuracy in right dashboard."""
        entries = self.model_registry.get_models_for_stock(ts_code)
        self.dashboard.show_model_info(entries if entries else None)

    def _show_prediction(self, pred: dict):
        """Show prediction in chart strip (moved from right dashboard)."""
        try:
            direction = pred.get("direction", "flat") or "flat"
            conf = pred.get("direction_conf", 0)
            target = pred.get("target_price", "--")
            # Sanitize NaN / non-numeric
            try:
                conf = float(conf)
                if conf != conf:
                    conf = 0.0
            except (TypeError, ValueError):
                conf = 0.0
            try:
                target = float(target) if target != "--" else 0.0
                if target != target:
                    target = 0.0
            except (TypeError, ValueError):
                target = 0.0
            self.chart.set_prediction_strip(direction, conf, target)
        except Exception:
            self.chart.clear_prediction_strip()

    def _load_prediction(self, ts_code: str):
        preds = self.repo.get_latest_predictions()
        if not preds.empty:
            row = preds[preds["ts_code"] == ts_code]
            if not row.empty:
                self._show_prediction(row.iloc[0].to_dict())

    def _load_behavior(self, ts_code: str):
        freq = self._auto_freq
        df = self.repo.get_minutes(ts_code, freq=freq)
        if df.empty:
            df = self._fetch_minutes(ts_code, freq=freq, days=60)
        if df.empty:
            return
        result = analyze_trader_behavior(df)
        self.dashboard.show_behavior(result)

    def _load_news(self, ts_code: str, freq: str = "5min"):
        """Fetch recent news for the selected stock, filtered to chart data time range."""
        try:
            news = fetch_stock_news(ts_code)
            df = self.repo.get_minutes(ts_code, freq=freq)
            if not df.empty and "trade_time" in df.columns:
                oldest = pd.to_datetime(df["trade_time"].min())
                newest = pd.to_datetime(df["trade_time"].max())
                filtered = []
                for n in news:
                    try:
                        nt = pd.to_datetime(n.get("time", ""))
                        if oldest <= nt <= newest + pd.Timedelta(days=1):
                            filtered.append(n)
                    except Exception:
                        filtered.append(n)  # keep if can't parse time
                news = filtered if filtered else news  # fallback: show all if none match
            self.dashboard.show_news(news)
            # Analyze news impacts and store for prediction adjustment
            if has_news_impact(news):
                self._active_news_impacts = analyze_news_impacts(news)
                self.dashboard.show_news_impact(self._active_news_impacts)
            else:
                self._active_news_impacts = []
                self.dashboard.show_news_impact([])
        except Exception:
            self.dashboard.show_news([])
            self._active_news_impacts = []
            self.dashboard.show_news_impact([])

    def _open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self.cfg.mcp.deepseek_api_key = dlg.deepseek_key_input.text()
            if self.cfg.mcp.deepseek_api_key:
                self.ai = AIAnalyzer(self.cfg.mcp.deepseek_api_key, self.cfg.mcp.deepseek_model)

    def _refresh_screening(self):
        self.status_bar.showMessage("正在加载股票列表...")
        self.screening_worker = ScreeningWorker(self.cfg.data.screened_count)
        self.screening_worker.phase.connect(lambda msg: self.status_bar.showMessage(msg))
        self.screening_worker.progress.connect(self._on_screening_progress)
        self.screening_worker.finished.connect(self._on_screening_done)
        self.screening_worker.error.connect(lambda e: self.status_bar.showMessage(f"筛选失败: {e}"))
        self.screening_worker.start()

    def _on_screening_progress(self, cur: int, total: int, accumulated: pd.DataFrame):
        if not accumulated.empty:
            self.repo.upsert_stocks(accumulated)
            self.stock_list.refresh(self.repo)
            self._sync_trained_stocks()
        self.status_bar.showMessage(f"筛选评分中... {cur}/{total}")

    def _on_screening_done(self, result: pd.DataFrame):
        if not result.empty:
            self.repo.upsert_stocks(result)
            self.stock_list.refresh(self.repo)
            self._sync_trained_stocks()
            self.status_bar.showMessage(f"筛选完成 | 共 {len(result)} 只股票")
        else:
            self.status_bar.showMessage("筛选失败或无符合条件的股票")

    def _build_gs_cache(self):
        """后台构建完整股票列表缓存（不阻塞 UI，不写入 DB）。"""
        if self.gs is None:
            return
        try:
            self.status_bar.showMessage("正在缓存完整股票列表...")
            QApplication.processEvents()
            self.gs.ensure_stock_cache()
            self.status_bar.showMessage("股票缓存已就绪")
        except Exception:
            pass

    def _auto_import_30d_data(self):
        """启动时后台导入所有热门股票最近 30 天的分钟数据。"""
        stocks = self.repo.get_screened_stocks()
        if not stocks:
            return
        self._import_worker = DataFetchWorker(self.fetcher, self.repo, stocks, gs=self.gs)
        self._import_worker.progress.connect(
            lambda n, t: self.status_bar.showMessage(f"后台导入30天数据... {n}/{t}"))
        self._import_worker.finished.connect(
            lambda: self.status_bar.showMessage("30天数据导入完成"))
        self._import_worker.start()

    def _auto_sync_gs_stocks(self):
        """启动时自动同步热门股票列表（仅在本地无数据时）。"""
        try:
            existing = self.repo.get_all_stocks()
            if not existing.empty:
                return  # Already have stocks
            from data.gs_fetcher import POPULAR_STOCKS
            self.repo.sync_stocks_from_gs(list(POPULAR_STOCKS))
            self.stock_list.refresh(self.repo)
            self._sync_trained_stocks()
        except Exception:
            pass

    def _sync_gs_stock_list(self):
        """同步热门 A 股列表到数据库（覆盖已有数据）。"""
        from data.gs_fetcher import POPULAR_STOCKS
        self.repo.sync_stocks_from_gs(list(POPULAR_STOCKS))
        self.stock_list.refresh(self.repo)
        self._sync_trained_stocks()
        self.status_bar.showMessage(f"已同步 {len(POPULAR_STOCKS)} 只热门 A 股")

    def _on_cloud_search(self, keyword: str):
        """Handle cloud search request from stock list panel."""
        self.status_bar.showMessage(f"🔍 云端搜索: {keyword}...")
        QApplication.processEvents()
        try:
            ts_code = self._cloud_search_stock(keyword)
            if ts_code:
                # Keep search text so the new stock appears immediately in filtered view
                self.stock_list._cached_df = None
                self.stock_list._do_filter()
                self.stock_list.select_stock(ts_code)
                self.status_bar.showMessage(f"✅ 已从云端添加: {ts_code} — 已永久记录")
            else:
                self.status_bar.showMessage(f"❌ 云端未找到: {keyword}")
        except Exception as e:
            self.status_bar.showMessage(f"云端搜索失败: {e}")

    def _cloud_search_stock(self, keyword: str) -> str | None:
        """云端搜索股票：从 TDX 缓存/服务器查找 stock 并补充到数据库。返回 ts_code 或 None。"""
        from data.gs_fetcher import GoldenSunFetcher
        if self.gs is None:
            self.gs = GoldenSunFetcher(
                host=self.cfg.data.gs_host, port=self.cfg.data.gs_port,
                use_public=self.cfg.data.gs_use_public,
            )
        # Try exact/partial match in cache first, fall back to TDX live
        result = self.gs.search_stock(keyword)
        if not result and len(keyword) >= 2:
            # Fuzzy: try without market suffix
            result = self.gs.search_stock(keyword.replace(".SZ", "").replace(".SH", ""))
        if result:
            ts_code, name = result
            self.repo.sync_stocks_from_gs([(ts_code, name)])
            self._sync_trained_stocks()
            return ts_code
        return None

    def _toggle_gs(self, checked: bool):
        """Toggle 金太阳 real-time data connection on/off."""
        if checked:
            if self.gs is None:
                from data.gs_fetcher import GoldenSunFetcher
                self.gs = GoldenSunFetcher(
                    host=self.cfg.data.gs_host, port=self.cfg.data.gs_port,
                    use_public=self.cfg.data.gs_use_public,
                )
            if self.gs.is_connected:
                self._gs_action.setChecked(True)
                self.status_bar.showMessage(f"金太阳实时行情已连接: {self.gs._current_ip}")
            else:
                self._gs_action.setChecked(False)
                self.status_bar.showMessage(f"金太阳连接失败，将使用 {self.fetcher.source_name} 回退")
        else:
            if self.gs:
                self.gs._disconnect()
                self.gs = None
            self._gs_action.setChecked(False)
            self.status_bar.showMessage(f"金太阳已断开，使用 {self.fetcher.source_name}")

    def _refresh_minute_data(self):
        """实时接收当前选中股票的数据并刷新图表。"""
        ts_code = self.stock_list.current_stock()
        if not ts_code:
            QMessageBox.warning(self, "提示", "请先双击选择一只股票")
            return

        self.status_bar.showMessage(f"正在接收 {ts_code} 实时数据...")
        QApplication.processEvents()

        # 1. 获取实时行情报价
        self._refresh_realtime_quote(ts_code)

        # 2. 增量拉取最新分钟数据
        df = self._fetch_minutes(ts_code, freq=self._auto_freq, days=2)
        if df.empty:
            self.status_bar.showMessage(f"{ts_code} 暂无新数据")
            return

        try:
            self.repo.insert_minutes(df)
        except Exception:
            pass

        # 3. 刷新图表
        cached_entry = self._prediction_cache.get(ts_code)
        cached_pred, rolling_preds = cached_entry if cached_entry else (None, None)
        self._load_chart(ts_code, self._auto_freq,
                         show_prediction=cached_pred is not None,
                         cached_prediction=cached_pred,
                         cached_rolling_preds=rolling_preds)

        # 4. 刷新行为分析和新闻
        self._load_behavior(ts_code)
        self._load_news(ts_code)

        self.status_bar.showMessage(f"✓ {ts_code} 实时数据已刷新")

    def _open_train_dialog(self):
        ts = self.stock_list.current_stock()
        selected = self.stock_list.get_selected_stocks()
        multi_mode = len(selected) > 1

        # ── Multi-select batch mode ──
        if multi_mode:
            self._train_dlg = TrainDialog(
                self.cfg, self.repo, self, selected_stocks=selected,
                model_registry=self.model_registry,
                fetcher=self.fetcher, gs=self.gs,
            )
            self._train_dlg.batch_started.connect(self._batch_train_stocks)
            # Also connect single-stock signals for when dialog is reused
            self._train_dlg.training_done.connect(self._on_training_done)
            self._train_dlg.background_requested.connect(self._on_background_training)
            self._train_dlg.show()
            return

        # ── Single-stock mode ──
        # Check if a background worker is already training this stock
        for w, t in list(self._bg_workers):
            if t == ts and w.isRunning():
                self._train_dlg = TrainDialog(self.cfg, self.repo, self, selected_stock=ts,
                                               model_registry=self.model_registry,
                                               fetcher=self.fetcher, gs=self.gs)
                self._train_dlg.training_done.connect(self._on_training_done)
                self._train_dlg.background_requested.connect(self._on_background_training)
                self._train_dlg.attach_existing_worker(w, ts)
                # Clean up from bg list when training finishes or is stopped
                w.finished.connect(lambda msg, t=ts: self._remove_bg_worker(t))
                w.finished.connect(lambda msg: self._train_dlg.close() if self._train_dlg and self._train_dlg.isVisible() else None)
                self._train_dlg.show()
                return

        if ts and self.model_registry.has_models(ts):
            # Model already exists — show what's trained, ask if re-train
            from PyQt6.QtWidgets import QMessageBox
            entries = self.model_registry.get_models_for_stock(ts)
            entry_info = "\n".join(
                f"  • {e.model_type} (准确率 {e.macro_acc:.2%})" if e.macro_acc > 0
                else f"  • {e.model_type}"
                for e in entries
            )
            reply = QMessageBox.question(
                self, "模型已存在",
                f"{ts} 已有训练记录:\n{entry_info}\n\n是否重新训练？（将覆盖现有模型）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            # Remove old predictor to force reload after re-training
            self.predictors.pop(ts, None)
        self._train_dlg = TrainDialog(self.cfg, self.repo, self, selected_stock=ts,
                                       model_registry=self.model_registry,
                                       fetcher=self.fetcher, gs=self.gs)
        self._train_dlg.training_done.connect(self._on_training_done)
        self._train_dlg.background_requested.connect(self._on_background_training)
        self._train_dlg.show()

    # ── Batch training queue (max 3 concurrent) ──

    def _batch_train_stocks(self, stocks: list[str]):
        """Queue selected stocks for training, max 3 at a time."""
        # Determine model to use
        if not self._train_model_name:
            from gui.dialogs import TrainDialog
            self._train_model_name = TrainDialog._last_model_name or "FT-iTransformer (时频协同)"

        # Ask user to confirm model and epochs
        from gui.dialogs import MODEL_REGISTRY
        model_names = list(MODEL_REGISTRY.keys())
        model_idx = model_names.index(self._train_model_name) if self._train_model_name in model_names else 0

        dialog = QMessageBox(self)
        dialog.setWindowTitle("批量训练确认")
        dialog.setText(
            f"即将训练 {len(stocks)} 只股票:\n"
            f"模型: {self._train_model_name}\n"
            f"模式: ⚡最高速 (单只训练, 3特征/5轮)\n\n"
            f"排队股票: {', '.join(stocks[:5])}"
            f"{'...' if len(stocks) > 5 else ''}"
        )
        dialog.addButton("开始", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("换模型", QMessageBox.ButtonRole.ActionRole)
        dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked is None or clicked.text() == "取消":
            return
        if clicked.text() == "换模型":
            # Simple model picker
            from PyQt6.QtWidgets import QInputDialog
            model_name, ok = QInputDialog.getItem(
                self, "选择模型", "模型:", model_names, model_idx, False
            )
            if not ok:
                return
            self._train_model_name = model_name

        # Deduplicate: don't re-queue stocks already training or queued
        active_codes = {t for _, t in self._bg_workers}
        active_codes |= set(self._train_queue)
        new_stocks = [s for s in stocks if s not in active_codes]
        if len(new_stocks) < len(stocks):
            skipped = len(stocks) - len(new_stocks)
            self.status_bar.showMessage(f"跳过 {skipped} 只已在队列中的股票")

        self._train_queue.extend(new_stocks)

        # Persist model choice for next time
        from gui.dialogs import TrainDialog
        TrainDialog._last_model_name = self._train_model_name

        self._process_train_queue()

    def _batch_predict_stocks(self, stocks: list[str]):
        """批量预测 — 优先 DeepSeek 远程，回退本地模型。"""
        predicted = 0
        for ts_code in stocks:
            try:
                # Try remote first
                if self.ai is not None:
                    pred = self._predict_remote(ts_code)
                    if pred and pred.get("direction_conf", 0) > 0:
                        rolling = [{
                            "step": i + 1,
                            "direction": pred["direction"],
                            "direction_conf": pred["direction_conf"] * (1 - i * 0.08),
                            "target_price": pred["target_price"] * (1 + 0.005 * i),
                            "price_lower": pred["price_lower"] * (1 + 0.005 * i),
                            "price_upper": pred["price_upper"] * (1 + 0.005 * i),
                        } for i in range(6)]
                        self._prediction_cache[ts_code] = (pred, rolling)
                        predicted += 1
                        self.status_bar.showMessage(f"云端预测: {predicted}/{len(stocks)} — {ts_code}")
                        QApplication.processEvents()
                        # 有机会时后台优化（不限置信度阈值）
                        self._try_optimize(ts_code, pred)
                        continue
                # Fallback to local
                if self._has_predictor(ts_code):
                    success = self._predict_with_rolling(ts_code)
                    if success:
                        self._prediction_cache.pop(ts_code, None)
                        predicted += 1
            except Exception:
                continue
        self._sync_trained_stocks()
        self._update_stock_colors()
        self.status_bar.showMessage(f"批量预测完成: {predicted}/{len(stocks)} 只")
        cur = self.stock_list.current_stock()
        if cur and cur in stocks:
            cached = self._prediction_cache.get(cur)
            if cached:
                self._load_chart(cur, self._auto_freq, show_prediction=True,
                                 cached_prediction=cached[0],
                                 cached_rolling_preds=cached[1])

    def _process_train_queue(self):
        """Start next worker from queue. Only 1 trains at a time (fastest mode)."""
        import os
        if self._train_queue and self._active_train_count < 1:
            ts_code = self._train_queue.pop(0)

            self._active_train_count += 1
            self._active_train_codes.add(ts_code)
            worker = TrainingWorker(
                None, self.repo, self.cfg, self.cfg.model.checkpoint_dir, scaler=None,
                ts_code=ts_code,
                selected_features=None,
                model_type=self._train_model_name,
                repo=self.repo,
                selected_stock=ts_code,
                model_name=self._train_model_name,
                fetcher=self.fetcher,
                gs=self.gs,
            )
            worker.finished.connect(
                lambda msg, tc=ts_code: self._on_queue_training_done(tc, msg)
            )
            worker.error.connect(
                lambda err, tc=ts_code: self._on_queue_training_error(tc, err)
            )
            self._bg_workers.append((worker, ts_code))
            worker.start()

            q_len = len(self._train_queue)
            model_short = self._train_model_name.split("(")[0].strip() if self._train_model_name else "模型"
            self._set_train_status(
                f"训练中: {model_short} — {ts_code} [⚡最高速]"
                f"{' (+' + str(q_len) + '排队)' if q_len > 0 else ''}",
                color="#4fc3f7"
            )
            self.status_bar.showMessage(
                f"训练中: {ts_code} [⚡最高速]"
                f"{' | 排队: ' + str(q_len) + ' 只' if q_len > 0 else ''}"
            )
    def _on_queue_training_done(self, ts_code: str, msg: str):
        """Queue worker finished — load model, update UI, start next."""
        self._active_train_count = max(0, self._active_train_count - 1)
        self._active_train_codes.discard(ts_code)
        self._remove_bg_worker(ts_code)
        self._sync_trained_stocks()
        self._load_model_info(ts_code)

        # Load the newly trained model
        ckpt = f"{self.cfg.model.checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt"
        import os
        if os.path.exists(ckpt):
            try:
                self.predictors[ts_code] = Predictor(ckpt, self.cfg)
            except Exception as e:
                self.status_bar.showMessage(f"批量训练: {ts_code} 模型加载失败: {e}")

        q_len = len(self._train_queue)
        self._set_train_status(
            f"✓ 训练完成: {ts_code}",
            color="#44dd44", auto_clear_s=5
        )
        self.status_bar.showMessage(
            f"训练完成: {ts_code} | {msg} | 剩余: {self._active_train_count} 进行中"
            f"{' | 排队: ' + str(q_len) + ' 只' if q_len > 0 else ''}"
        )
        # Auto-predict if this stock is currently selected
        if self.stock_list.current_stock() == ts_code:
            try:
                self._predict_with_rolling(ts_code)
                self._start_animated_prediction(ts_code)
            except Exception:
                pass

        self._process_train_queue()

    def _on_queue_training_error(self, ts_code: str, err: str):
        """Queue worker errored — log, decrement, continue."""
        self._active_train_count = max(0, self._active_train_count - 1)
        self._active_train_codes.discard(ts_code)
        self._remove_bg_worker(ts_code)
        self._set_train_status(
            f"✗ 训练失败: {ts_code}",
            color="#ff6666", auto_clear_s=5
        )
        self.status_bar.showMessage(f"训练失败 [{ts_code}]: {err}")
        self._process_train_queue()

    def _on_training_done(self, ts_code: str):
        """Called when training finishes — load model, rolling prediction, start animation."""
        self._sync_trained_stocks()
        self._load_model_info(ts_code)
        ckpt = f"{self.cfg.model.checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt"
        import os
        if not os.path.exists(ckpt):
            self.status_bar.showMessage("训练完成但未找到模型文件")
            self._process_train_queue()
            return
        try:
            self.predictors[ts_code] = Predictor(ckpt, self.cfg)
        except Exception as e:
            self.status_bar.showMessage(f"模型加载失败: {e}")
            self._process_train_queue()
            return
        self._set_train_status(
            f"✓ 训练完成: {ts_code}",
            color="#44dd44", auto_clear_s=5
        )
        # Run rolling predictions and display on chart + dashboard
        success = self._predict_with_rolling(ts_code)
        if success:
            self.status_bar.showMessage(
                f"训练完成 — {ts_code} | 预测已显示 | 点 [▶ 实时预测] 自动更新"
            )
            # Auto-start animated 3s prediction if this stock is selected
            if self.stock_list.current_stock() == ts_code:
                self._start_animated_prediction(ts_code)
        self._process_train_queue()
        # If not success, _predict_with_rolling already set the status message

    def _quick_predict(self):
        """预测当前股票 — 本地 PyTorch 模型。"""
        ts = self.stock_list.current_stock()
        if not ts:
            self.status_bar.showMessage("请先选择一只股票")
            return

        # Local PyTorch model prediction
        self.model_registry.refresh()
        entries = self.model_registry.get_models_for_stock(ts)
        if not entries:
            reply = QMessageBox.question(
                self, "无已训练模型",
                f"{ts} 尚无已训练模型。\n是否现在本地训练？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._open_train_dialog()
            return
        if len(entries) == 1:
            chosen = entries[0]
        else:
            dialog = QMessageBox(self)
            dialog.setWindowTitle(f"选择模型 — {ts}")
            info_parts = []
            for i, e in enumerate(entries):
                acc_str = f"{e.macro_acc:.2%}" if e.macro_acc > 0 else "N/A"
                info_parts.append(f"{i+1}. {e.model_type} (准确率 {acc_str})")
            dialog.setText(f"已训练 {len(entries)} 个模型，选择要使用的:\n\n" + "\n".join(info_parts))
            btn_texts = []
            for e in entries:
                short = e.model_type.split("(")[0].strip()[:12]
                btn_texts.append(short)
            for bt in btn_texts:
                dialog.addButton(bt, QMessageBox.ButtonRole.AcceptRole)
            dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            dialog.exec()
            clicked = dialog.clickedButton()
            if clicked is None or clicked.text() == "取消":
                return
            idx = btn_texts.index(clicked.text()) if clicked.text() in btn_texts else 0
            chosen = entries[min(idx, len(entries) - 1)]

        ckpt_path = chosen.file_path
        try:
            if ts in self.predictors:
                del self.predictors[ts]
            self.predictors[ts] = Predictor(ckpt_path, self.cfg)
        except Exception as e:
            self.status_bar.showMessage(f"模型加载失败: {e}")
            return

        self.status_bar.showMessage(f"正在用 {chosen.model_type} 预测 {ts} ...")
        self._pred_worker = QuickPredictWorker(
            self.predictors[ts], self.repo, ts, 20
        )
        self._pred_worker.result_ready.connect(
            lambda final_pred, rolling_preds: self._on_quick_predict_done(ts, chosen, final_pred, rolling_preds)
        )
        self._pred_worker.error.connect(lambda e: self.status_bar.showMessage(f"预测失败: {e}"))
        self._pred_worker.start()

    def _on_quick_predict_done(self, ts, chosen, final_pred, rolling_preds):
        """Called from QuickPredictWorker thread when prediction completes."""
        if final_pred is None:
            self.status_bar.showMessage("预测未能生成结果")
            return

        self._prediction_cache[ts] = (final_pred, rolling_preds)

        # Update chart (must be called from main thread)
        try:
            df = self.repo.get_minutes(ts)
            if not df.empty:
                df = df.sort_values("trade_time")
                self.chart.plot_kline(self._trim_for_display(df), ts, self._auto_freq,
                                      prediction=final_pred,
                                      rolling_preds=rolling_preds)
        except Exception:
            pass

        # Update prediction strip
        self.chart.set_prediction_strip(
            final_pred.get("direction", "flat"),
            final_pred.get("direction_conf", 0),
            final_pred.get("target_price", 0)
        )
        self._show_prediction(final_pred)
        self.dashboard.show_rolling_prediction(rolling_preds)

        self.status_bar.showMessage(
            f"预测完成 — {ts} | 模型: {chosen.model_type} | "
            f"方向: {final_pred.get('direction', '?')}"
        )
        if self.stock_list.current_stock() == ts:
            self._start_animated_prediction(ts)
        self.model_registry.refresh()
        self._update_stock_colors()

    def _on_background_training(self, worker, ts_code: str):
        """Training moved to background — track worker and connect signals."""
        worker.finished.connect(lambda msg, t=ts_code: self._on_bg_training_finished(msg, t))
        worker.error.connect(lambda err: self.status_bar.showMessage(f"后台训练失败 [{ts_code}]: {err}"))
        self._bg_workers.append((worker, ts_code))
        self._set_train_status(
            f"训练中: {ts_code} | 共 {len(self._bg_workers)} 个后台任务",
            color="#4fc3f7"
        )
        self.status_bar.showMessage(f"后台训练中: {ts_code} | 共 {len(self._bg_workers)} 个任务")

    def _remove_bg_worker(self, ts_code: str):
        """Remove a background worker from tracking."""
        self._bg_workers = [(w, t) for w, t in self._bg_workers if t != ts_code]

    def _on_bg_training_finished(self, msg: str, ts_code: str):
        """Background training completed — load model and start animation."""
        self._remove_bg_worker(ts_code)
        self._set_train_status(
            f"✓ 训练完成: {ts_code}",
            color="#44dd44", auto_clear_s=5
        )
        # Update count if other bg workers still running
        if self._bg_workers:
            codes = [t for _, t in self._bg_workers]
            self.status_bar.showMessage(f"后台训练完成 — {ts_code} | {msg} | 仍在训练: {', '.join(codes)}")
        else:
            self.status_bar.showMessage(f"后台训练完成 — {ts_code} | {msg}")
        self._sync_trained_stocks()
        self._load_model_info(ts_code)
        # Load model and start animation
        ckpt = f"{self.cfg.model.checkpoint_dir}/{sanitize_filename(ts_code)}_best_model.pt"
        import os
        if os.path.exists(ckpt):
            try:
                self.predictors[ts_code] = Predictor(ckpt, self.cfg)
            except Exception as e:
                self.status_bar.showMessage(f"后台模型加载失败: {e}")
                self._process_train_queue()  # continue queue on failure too
                return
            self._predict_with_rolling(ts_code)
            if self.stock_list.current_stock() == ts_code:
                self._start_animated_prediction(ts_code)
        self._process_train_queue()

    # ── Animated 3s prediction ──
    def _compute_max_steps(self, freq: str) -> int:
        """Always 6 steps = 6×20min = 2h prediction horizon."""
        return 6

    def _start_animated_prediction(self, ts_code: str):
        """Start 3-second interval animated prediction — one point per tick."""
        self._stop_animation()
        df = self.repo.get_minutes(ts_code)
        if df.empty:
            return
        df = df.sort_values("trade_time")
        self._anim_df_ext = df.copy()
        self._anim_indicator_cols = []
        self._anim_preds = []
        self._anim_step = 0
        self._anim_ts_code = ts_code
        self._anim_freq_min = 20  # 20 min per step
        self._anim_max_steps = self._compute_max_steps(self._auto_freq)
        self._anim_timer.start(200)
        self.status_bar.showMessage(
            f"{ts_code} | 2h预测动画 | 步长20min | 共{self._anim_max_steps}步"
        )

    def _stop_animation(self):
        """Stop animated prediction timer."""
        self._anim_timer.stop()
        self._anim_df_ext = None
        self._anim_preds = []
        self._anim_step = 0
        self._anim_ts_code = None
        self._anim_indicator_cols = []

    def _anim_tick(self):
        """One step of animated prediction: predict → append → update just the path line."""
        if (self._anim_df_ext is None or self._anim_ts_code is None
                or self._anim_step >= self._anim_max_steps):
            self._anim_timer.stop()
            self.status_bar.showMessage(
                f"{self._anim_ts_code} | 动画预测完成 | 共 {len(self._anim_preds)} 点"
            )
            return

        ts_code = self._anim_ts_code
        predictor = self._get_predictor(ts_code)
        if not predictor:
            self._stop_animation()
            return

        try:
            from data.features import compute_all_indicators
            from data.preprocessor import preprocess

            df_ext = self._anim_df_ext
            n = len(df_ext)

            # Only compute indicators on the last seq_len bars for efficiency
            seq_len = self.cfg.model.seq_len
            compute_start = max(0, n - seq_len - 1)

            # Drop old indicator columns and recompute only on the window
            if self._anim_indicator_cols:
                existing = [c for c in self._anim_indicator_cols if c in df_ext.columns]
                if existing:
                    df_ext = df_ext.drop(columns=existing)
                self._anim_indicator_cols = []

            # Compute indicators on the full df (needed for preprocess sliding window)
            ind = compute_all_indicators(df_ext)
            self._anim_indicator_cols = list(ind.columns)
            for col in self._anim_indicator_cols:
                df_ext[col] = ind[col].values

            # Only use last seq_len rows for prediction
            feat_arr, _ = preprocess(df_ext, fit_scaler=False)
            current_price = float(df_ext["close"].iloc[-1])
            pred = predictor.predict_one(feat_arr, current_price, ts_code, mc_samples=20)

            # Momentum decay + noise to prevent self-reinforcing straight line
            momentum_decay = max(0.2, 0.90 ** self._anim_step)
            noise = np.random.normal(0, 0.0002)
            base_delta = pred.get("price_delta", 0.0)
            adjusted_delta = base_delta * momentum_decay + noise
            adjusted_delta = float(adjusted_delta)

            target = current_price * (1 + adjusted_delta)
            pred["price_delta"] = round(float(adjusted_delta), 6)
            pred["target_price"] = round(target, 2)

            pred["step"] = self._anim_step + 1
            pred["ts_code"] = ts_code
            pred["created_at"] = pd.Timestamp.now()
            self._anim_preds.append(pred)

            # Build synthetic bar: blend prediction with current price
            avg_vol = float(df_ext["volume"].tail(20).mean()) if len(df_ext) >= 20 else float(df_ext["volume"].iloc[-1])
            blended_close = current_price * (1 + adjusted_delta * 0.35)
            move = abs(adjusted_delta * current_price)
            new_row_data = {
                "open": current_price,
                "close": blended_close,
                "high": max(blended_close, current_price) + move * np.random.uniform(0.05, 0.15),
                "low": min(blended_close, current_price) - move * np.random.uniform(0.05, 0.15),
                "volume": avg_vol * (1 + np.random.uniform(-0.1, 0.1)),
            }
            if "trade_time" in df_ext.columns:
                last_time = pd.to_datetime(df_ext["trade_time"].iloc[-1])
                new_row_data["trade_time"] = last_time + pd.Timedelta(seconds=3)
            for c in df_ext.columns:
                if c not in self._anim_indicator_cols and c not in new_row_data:
                    new_row_data[c] = df_ext[c].iloc[-1]
            new_row = pd.DataFrame([new_row_data])
            self._anim_df_ext = pd.concat([df_ext, new_row], ignore_index=True)

            # Fast update: only redraw the rolling prediction path, not full chart
            self.chart.plot_rolling_path(self._anim_preds)

            self._anim_step += 1
        except Exception as e:
            self.status_bar.showMessage(f"动画预测出错: {e}")
            self._stop_animation()

    # ── Auto-predict (rolling prediction) ──
    def _gather_trained_stocks(self) -> list[tuple[str, str, float]]:
        """Return list of (ts_code, model_type, accuracy) for stocks with local models."""
        trained = set(self.predictors.keys())
        import os
        if os.path.isdir(self.cfg.model.checkpoint_dir):
            for fname in os.listdir(self.cfg.model.checkpoint_dir):
                if fname.endswith("_best_model.pt"):
                    code = fname.replace("_best_model.pt", "")
                    if code not in trained:
                        p = self._get_predictor(code)
                        if p:
                            trained.add(code)
        # Enrich with model registry metadata
        result = []
        for code in sorted(trained):
            entry = self.model_registry.get_best_model(code)
            if entry:
                result.append((code, entry.model_type, entry.macro_acc))
            else:
                result.append((code, "未知", 0.0))
        return result

    def _show_auto_predict_dialog(self) -> list[str] | None:
        """Show a dialog for the user to select up to 5 stocks for real-time prediction.
        Only stocks with locally trained models are listed.
        Returns the selected stock codes, or None if cancelled."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                                      QCheckBox, QLabel, QScrollArea, QWidget, QFrame)

        trained = self._gather_trained_stocks()
        if not trained:
            QMessageBox.warning(self, "提示", "无已训练模型，请先训练至少一只股票再开启实时预测")
            return None

        dlg = QDialog(self)
        dlg.setWindowTitle("选择实时预测股票（最多5只）")
        dlg.resize(480, 420)
        dlg.setMinimumSize(400, 300)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel(
            f'<span style="color: #ffd700; font-size: 13px;">已训练模型: {len(trained)} 只</span>'
            f'<span style="color: #aaa; font-size: 12px;">&nbsp;&nbsp;|&nbsp;&nbsp;最多选择 5 只</span>'
            f'<span style="color: #888; font-size: 11px; margin-left: 8px;">（使用本地模型进行实时预测）</span>'
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Scrollable checkbox list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #333; background: #1e1e2e; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: #1e1e2e;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        scroll_layout.setSpacing(2)

        checkboxes: list[QCheckBox] = []
        self._auto_dlg_checkboxes = []  # temp storage for (checkbox, ts_code)

        for ts_code, model_type, acc in trained:
            row = QFrame()
            row.setStyleSheet("QFrame:hover { background: #2a2a3e; }")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(6, 2, 6, 2)

            cb = QCheckBox()
            cb.setStyleSheet("QCheckBox::indicator { width: 16px; height: 16px; }")
            cb.toggled.connect(lambda checked, cbs=checkboxes: self._limit_auto_selection(cbs))
            checkboxes.append(cb)

            acc_str = f"{acc:.1%}" if acc > 0 else "N/A"
            label = QLabel(f'{ts_code}&nbsp;&nbsp;|&nbsp;&nbsp;<b>{model_type}</b>&nbsp;&nbsp;|&nbsp;&nbsp;准确率: {acc_str}')
            label.setStyleSheet("color: #ccc; font-size: 12px;")

            row_layout.addWidget(cb)
            row_layout.addWidget(label)
            row_layout.addStretch()
            scroll_layout.addWidget(row)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Selection count
        count_label = QLabel('<span style="color: #888; font-size: 12px;">已选: 0 / 5</span>')
        layout.addWidget(count_label)

        # Update count when checkboxes change
        for cb in checkboxes:
            cb.toggled.connect(
                lambda: count_label.setText(
                    f'<span style="color: {"#ffd700" if sum(1 for c in checkboxes if c.isChecked()) == 5 else "#888"}; font-size: 12px;">已选: {sum(1 for c in checkboxes if c.isChecked())} / 5</span>'
                )
            )

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(
            "QPushButton { padding: 6px 24px; background: #444; color: #ddd; border: 1px solid #555; border-radius: 4px; }"
            "QPushButton:hover { background: #555; }"
        )
        cancel_btn.clicked.connect(dlg.reject)

        ok_btn = QPushButton("开始实时预测")
        ok_btn.setStyleSheet(
            "QPushButton { padding: 6px 24px; background: #d4380d; color: #fff; border: none; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #e65c3a; }"
            "QPushButton:disabled { background: #555; color: #888; }"
        )
        ok_btn.clicked.connect(dlg.accept)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        # Store refs for result extraction
        self._auto_dlg_checkboxes = list(zip(checkboxes, [t[0] for t in trained]))
        self._auto_dlg_ok = ok_btn

        # Auto-update OK button state
        def _update_ok():
            n_checked = sum(1 for c in checkboxes if c.isChecked())
            ok_btn.setEnabled(1 <= n_checked <= 5)
        for cb in checkboxes:
            cb.toggled.connect(_update_ok)
        ok_btn.setEnabled(False)

        if not dlg.exec():
            self._auto_dlg_checkboxes = []
            return None

        self._auto_dlg_checkboxes = []
        selected = [code for cb, code in zip(checkboxes, [t[0] for t in trained]) if cb.isChecked()]
        return selected if selected else None

    def _limit_auto_selection(self, checkboxes: list):
        """Prevent selecting more than 5 stocks."""
        checked = [cb for cb in checkboxes if cb.isChecked()]
        if len(checked) > 5:
            checked[-1].blockSignals(True)
            checked[-1].setChecked(False)
            checked[-1].blockSignals(False)

    def _toggle_auto_predict(self, checked: bool = None):
        """Start/stop the rolling prediction timer — user selects up to 5 stocks."""
        if checked is None:
            checked = not self._auto_active

        if checked:
            # Show selection dialog — user picks up to 5 stocks with local models
            selected = self._show_auto_predict_dialog()
            if not selected:
                self._auto_btn.setChecked(False)
                return
            self._auto_stocks = selected
            self._auto_active = True
            self._auto_btn.setChecked(True)
            self._auto_timer.start(60000)
            self._auto_btn.setText("⏸ 停止预测")
            self.status_bar.showMessage(
                f"实时预测已启动 | 已选 {len(self._auto_stocks)} 只股票 | 每分钟更新 | 本地模型"
            )
            # Run first tick immediately
            QTimer.singleShot(500, self._auto_predict_tick)
        else:
            self._auto_active = False
            self._auto_btn.setChecked(False)
            self._auto_timer.stop()
            self._auto_btn.setText("▶ 实时预测")
            self._auto_stocks = []
            self.status_bar.showMessage("实时预测已停止")

    def _auto_predict_tick(self):
        """Timer tick: local model prediction for user-selected stocks (max 5).

        - Only the currently selected stock gets full chart/dashboard update
        - All selected stocks get prediction cache updated for color display in list
        """
        if not self._auto_active or not self._auto_stocks:
            return

        current_ts = self.stock_list.current_stock()
        predicted_count = 0

        for ts_code in self._auto_stocks:
            try:
                # ── Refresh data for every stock first ──
                df_new = self._fetch_minutes(ts_code, freq=self._auto_freq, days=1)
                if df_new is not None and not df_new.empty:
                    self.repo.insert_minutes(df_new)

                # ── Non-selected stock: real rolling prediction ──
                if ts_code != current_ts:
                    predictor = self._get_predictor(ts_code)
                    if predictor:
                        df = self.repo.get_minutes(ts_code, freq=self._auto_freq)
                        if df is not None and len(df) >= self.cfg.model.seq_len:
                            df = df.sort_values("trade_time")
                            rolling = self._predict_rolling(df, ts_code, predictor, 6, 20)
                            if rolling:
                                final_pred = rolling[-1]
                                self._prediction_cache[ts_code] = (final_pred, rolling)
                                predicted_count += 1
                    continue

                # ── Current stock: full prediction pipeline ──
                self._auto_predict_current(ts_code)
                predicted_count += 1

            except Exception:
                pass

        self._update_stock_colors()

        now_str = datetime.now().strftime("%H:%M:%S")
        self.status_bar.showMessage(
            f"[{now_str}] 实时预测: {predicted_count}/{len(self._auto_stocks)} 只股票 | 本地模型"
        )

    def _auto_predict_current(self, ts: str):
        """Full local prediction pipeline for the currently selected stock."""
        from data.features import compute_all_indicators
        from data.preprocessor import preprocess

        predictor = self._get_predictor(ts)
        if not predictor:
            self.status_bar.showMessage(f"{ts} 无本地模型，请先训练")
            return

        # Load working data
        df = self.repo.get_minutes(ts, freq=self._auto_freq)
        if df.empty or len(df) < self.cfg.model.seq_len:
            self.status_bar.showMessage(f"{ts} 历史数据不足，等待更多数据...")
            return
        df = df.sort_values("trade_time")
        current_price = float(df["close"].iloc[-1])

        # Local rolling prediction
        rolling_preds = self._predict_rolling(df, ts, predictor, 6, 20)
        final_pred = rolling_preds[-1] if rolling_preds else None
        if not final_pred:
            return
        pred = final_pred

        # News adjustment
        if self._active_news_impacts:
            news_bias = compute_news_bias(self._active_news_impacts, datetime.now())
            if news_bias["active_count"] > 0 and abs(news_bias["bias"]) > 0.01:
                for i, rp in enumerate(rolling_preds):
                    step_weight = 1.0 - (i / max(len(rolling_preds), 1)) * 0.5
                    adj = apply_news_adjustment(
                        rp["target_price"], rp.get("price_delta", 0),
                        {"bias": news_bias["bias"] * step_weight,
                         "magnitude": news_bias["magnitude"] * step_weight,
                         "active_count": news_bias["active_count"]},
                        max_adj_pct=0.015
                    )
                    rp["target_price"] = adj["adjusted_target"]
                    rp["price_lower"] = round(adj["adjusted_target"] * 0.97, 2)
                    rp["price_upper"] = round(adj["adjusted_target"] * 1.03, 2)

        # Store
        final_pred["ts_code"] = ts
        final_pred["created_at"] = pd.Timestamp.now()
        self.repo.insert_predictions(pd.DataFrame([final_pred]))
        self._prediction_cache[ts] = (final_pred, rolling_preds)
        self._last_predicted_price = float(rolling_preds[0]["target_price"])

        # Update chart & dashboard (full df for prediction, trim for display)
        self.chart.plot_kline(self._trim_for_display(df), ts, self._auto_freq,
                              prediction=final_pred, rolling_preds=rolling_preds)
        self._show_prediction(final_pred)
        self.dashboard.show_rolling_prediction(rolling_preds)
        advice = generate_trade_advice(final_pred, df, ts_code=ts)
        self.dashboard.show_trade_advice(advice)

        # Behavior + quote
        self._refresh_realtime_quote(ts)
        result = analyze_trader_behavior(df)
        self.dashboard.show_behavior(result)

    def _predict_rolling(self, df, ts_code, predictor, steps: int,
                         freq_min: int = 20) -> list[dict]:
        """Delegate to shared predict_rolling for consistent results across all paths."""
        from gui.workers import predict_rolling
        return predict_rolling(df, ts_code, predictor, steps, freq_min)

    def _on_timeframe_changed(self, freq: str):
        self._auto_freq = freq
        ts = self.stock_list.current_stock()
        if ts:
            # Use cached prediction regardless of timeframe — same model works for all
            cached_entry = self._prediction_cache.get(ts)
            cached_pred, rolling_preds = cached_entry if cached_entry else (None, None)
            self._load_chart(ts, freq,
                             show_prediction=cached_pred is not None,
                             cached_prediction=cached_pred,
                             cached_rolling_preds=rolling_preds)
            self._load_behavior(ts)
        # Update animation max steps if running
        freq_name = {"1min": "分钟线", "daily": "日线"}.get(freq, freq)
        if self._anim_timer.isActive() and self._anim_ts_code == ts:
            self._anim_max_steps = self._compute_max_steps(freq)
            self.status_bar.showMessage(
                f"{ts} | 动画预测 | 最大节点更新为 {self._anim_max_steps} ({freq_name})"
            )
        # Restart timer on freq change (interval stays 60s)
        if self._auto_active:
            self._auto_timer.start(60000)
            self.status_bar.showMessage(f"实时预测频率切换至 {freq_name} | 每分钟更新")

    def _predict_single(self, ts_code: str):
        """Run one-shot prediction for a single stock — show on chart + dashboard."""
        predictor = self._get_predictor(ts_code)
        if not predictor:
            self.status_bar.showMessage(f"未找到 {ts_code} 的预测模型")
            return False
        try:
            df = self.repo.get_minutes(ts_code)
            if df.empty:
                self.status_bar.showMessage(f"{ts_code} 无分钟数据，请先刷新数据")
                return False
            df = df.sort_values("trade_time")
            from data.features import compute_all_indicators
            from data.preprocessor import preprocess
            ind = compute_all_indicators(df)
            for col in ind.columns:
                if col not in df.columns:
                    df[col] = ind[col].values
            feat_arr, _ = preprocess(df, fit_scaler=False)
            current_price = float(df["close"].iloc[-1])
            pred = predictor.predict_one(feat_arr, current_price, ts_code)
            pred["ts_code"] = ts_code
            pred["created_at"] = pd.Timestamp.now()
            self.repo.insert_predictions(pd.DataFrame([pred]))

            # Update chart with prediction overlay (full df for training, trim for display)
            self.chart.plot_kline(self._trim_for_display(df), ts_code, self._auto_freq, prediction=pred)

            # Update dashboard
            self._show_prediction(pred)
            advice = generate_trade_advice(pred, df, ts_code=ts_code)
            self.dashboard.show_trade_advice(advice)
            self._prediction_cache[ts_code] = (pred, [])
            self._update_stock_colors()
            try:
                df_b = self.repo.get_minutes(ts_code, freq=self._auto_freq)
                if not df_b.empty:
                    result = analyze_trader_behavior(df_b)
                    self.dashboard.show_behavior(result)
            except Exception:
                pass
            return True
        except Exception as e:
            self.status_bar.showMessage(f"预测失败: {e}")
            return False

    def _predict_with_rolling(self, ts_code: str) -> bool:
        """Run rolling iterative predictions for multiple future time periods.

        Important: do NOT pre-compute indicators on df before _predict_rolling.
        _predict_rolling internally calls compute_all_indicators and drops stale
        indicator columns at each iteration so synthetic bars get fresh values.
        """
        predictor = self._get_predictor(ts_code)
        if not predictor:
            self.status_bar.showMessage(f"未找到 {ts_code} 的预测模型")
            return False

        df = self.repo.get_minutes(ts_code)
        if df.empty:
            self.status_bar.showMessage(f"{ts_code} 无分钟数据，请先刷新数据")
            return False
        df = df.sort_values("trade_time")

        # 6 steps × 20min = 2h prediction horizon
        freq_min = 20
        rolling_steps = 6

        # ── Step 1: Run rolling predictions ──
        try:
            rolling_preds = self._predict_rolling(df, ts_code, predictor, rolling_steps, freq_min)
        except Exception as e:
            self.status_bar.showMessage(f"滚动预测计算失败: {e}")
            return False

        final_pred = rolling_preds[-1] if rolling_preds else None
        if not final_pred:
            self.status_bar.showMessage("滚动预测未能生成结果")
            return False

        final_pred["ts_code"] = ts_code
        final_pred["created_at"] = pd.Timestamp.now()

        # ── Step 2: Save to DB ──
        try:
            self.repo.insert_predictions(pd.DataFrame([final_pred]))
        except Exception as e:
            self.status_bar.showMessage(f"预测保存失败: {e}")
            return False

        # ── Step 3: Update chart (display trimmed, training used full df) ──
        try:
            display_days = 2 if self._auto_freq == "1min" else 30
            df_display = self._trim_trading_days(df, display_days)
            self.chart.plot_kline(df_display, ts_code, self._auto_freq,
                                  prediction=final_pred,
                                  rolling_preds=rolling_preds)
        except Exception as e:
            self.status_bar.showMessage(f"图表更新失败: {e}")
            return False

        # ── Step 4: Update dashboard ──
        try:
            self._show_prediction(final_pred)
            self.dashboard.show_rolling_prediction(rolling_preds)
            advice = generate_trade_advice(final_pred, df, ts_code=ts_code)
            self.dashboard.show_trade_advice(advice)
        except Exception as e:
            self.status_bar.showMessage(f"面板更新失败: {e}")
            return False

        # ── Step 5: Update behavior (best-effort, non-critical) ──
        try:
            freq = "5min"
            df_behavior = self.repo.get_minutes(ts_code, freq=freq)
            if df_behavior.empty:
                df_behavior = self.fetcher.fetch_recent_mins(ts_code, days=60, freq=freq)
            if not df_behavior.empty:
                result = analyze_trader_behavior(df_behavior)
                self.dashboard.show_behavior(result)
        except Exception:
            pass  # behavior is best-effort

        self._last_predicted_price = float(final_pred["target_price"])

        # Cache prediction + rolling path in memory so it persists across
        # stock and timeframe switches (same model works for all timeframes)
        self._prediction_cache[ts_code] = (final_pred, rolling_preds)
        self._update_stock_colors()
        return True

    # ── Auto-optimization ──

    def _try_optimize(self, ts_code: str, pred: dict):
        """后台优化：训练线程空闲时通过云端 AI 搜索更优参数，提高置信度."""
        # Debounce: 同一股票 1 小时内不重复优化
        now = datetime.now()
        if ts_code in self._optimize_cooldown:
            elapsed = (now - self._optimize_cooldown[ts_code]).total_seconds()
            if elapsed < 3600:
                return
        # 只在没有训练线程时触发，避免与训练争抢资源
        if self._active_train_count > 0:
            return
        if self.ai is None:
            return
        self._optimize_cooldown[ts_code] = now

        conf = pred.get("direction_conf", 0)
        self.status_bar.showMessage(
            f"后台优化: {ts_code} (当前置信 {conf:.0%}) — 搜索更优参数...")
        QApplication.processEvents()

        self._opt_worker = OptimizationWorker(
            self.ai, ts_code, pred, self.cfg.model,
            self.repo, self.model_registry,
        )
        self._opt_worker.result_ready.connect(self._on_optimization_ready)
        self._opt_worker.error.connect(lambda e: None)  # 静默失败，不显示文本
        self._opt_worker.start()

    def _on_optimization_ready(self, params: dict):
        """云端优化返回后直接用云 API 重新预测刷新，不触发本地训练."""
        ts_code = params.get("ts_code", "")
        reasoning = params.get("reasoning", "")
        expected = params.get("expected_improvement", 0)

        if "error" in params:
            # 优化失败静默，不显示文本
            return

        self.status_bar.showMessage(
            f"云端优化完成: {reasoning[:50]}... | "
            f"预期提升: {expected:.0%} | 正在刷新预测..."
        )
        QApplication.processEvents()

        if self.ai is None:
            return

        # 直接用云端 API 重新预测，不触发本地训练
        try:
            pred = self._predict_remote(ts_code)
            if pred and pred.get("direction_conf", 0) > 0:
                final_pred = pred
                final_pred["ts_code"] = ts_code
                final_pred["created_at"] = pd.Timestamp.now()
                self.repo.insert_predictions(pd.DataFrame([final_pred]))
                self._prediction_cache[ts_code] = (final_pred, [])
                self._update_stock_colors()

                new_conf = pred.get("direction_conf", 0)
                # 更新当前选中股票的图表
                if self.stock_list.current_stock() == ts_code:
                    self._show_prediction(final_pred)
                    self.dashboard.show_rolling_prediction([])
                    try:
                        df = self.repo.get_minutes(ts_code)
                        if not df.empty:
                            df = df.sort_values("trade_time")
                            self.chart.plot_kline(self._trim_for_display(df), ts_code, self._auto_freq,
                                                  prediction=final_pred, rolling_preds=[])
                    except Exception:
                        pass
                self.status_bar.showMessage(
                    f"✓ 云端优化完成: {ts_code} | 置信度 {new_conf:.0%} | {reasoning[:30]}")
            else:
                self.status_bar.showMessage(
                    f"云端优化完成但重预测无结果: {ts_code}")
        except Exception as e:
            pass  # 静默失败

        self._optimize_cooldown.pop(ts_code, None)

    def _clear_all_models(self):
        """Delete all trained models and refresh UI."""
        reply = QMessageBox.question(
            self, "确认清除",
            "确定要清除所有股票的已训练模型吗？\n\n此操作将删除所有模型文件和特征缓存，不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Clear predictor cache
        self.predictors.clear()
        self._prediction_cache.clear()

        count = self.model_registry.clear_all_models()
        self._sync_trained_stocks()
        self._update_stock_colors()
        self.dashboard.show_model_info(None)
        self.status_bar.showMessage(f"已清除 {count} 个模型文件")

    def _predict_all(self):
        stocks = self.repo.get_screened_stocks()
        if not stocks:
            QMessageBox.warning(self, "提示", "没有筛选的股票")
            return
        # Remote-first: batch through DeepSeek API
        if self.ai is not None:
            self._batch_predict_stocks(stocks)
            return
        # Fallback: local PyTorch worker
        self.status_bar.showMessage("预测中...")
        worker = PredictionWorker(self.repo, self.cfg.model.checkpoint_dir, self.cfg, stocks)
        worker.progress.connect(lambda n, t: self.status_bar.showMessage(f"预测中... {n}/{t}"))
        worker.finished.connect(lambda: (
            self.status_bar.showMessage("预测完成"),
            self._update_stock_colors()
        ))
        worker.start()

    def _analyze_current(self):
        if not self.ai:
            QMessageBox.warning(self, "提示", "请先在设置中填入DeepSeek API Key")
            return
        ts = self.stock_list.current_stock()
        if not ts:
            return
        preds = self.repo.get_latest_predictions()
        if preds.empty:
            return
        row = preds[preds["ts_code"] == ts]
        if row.empty:
            return
        pred = row.iloc[0].to_dict()
        # Get behavior data
        behavior = None
        df = self.repo.get_minutes(ts, freq="5min")
        if not df.empty:
            from data.behavior import analyze_trader_behavior
            behavior = analyze_trader_behavior(df)
        news_text = self.dashboard.get_news_context()
        self.worker = AIAnalysisWorker(self.ai, pred, behavior, news_text)
        self.worker.analysis_ready.connect(self.dashboard.show_analysis)
        self.worker.start()
