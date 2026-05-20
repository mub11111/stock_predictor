from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QLabel, QPushButton, QDialogButtonBox,
    QProgressBar, QPlainTextEdit, QHBoxLayout, QCheckBox, QComboBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from config import AppConfig
from storage.repository import Repository
from model.ft_transformer import FT_iTransformerWrapper
from model.freq_orchestrator import FreqOrchestratorWrapper
from model.node_transformer import NodeTransformerWrapper
from model.pinn_model import PINNWrapper
from model.ensemble import EnsembleWeightedWrapper, EnsembleVotingWrapper, EnsembleStackingWrapper, EnsemblePINNGuardWrapper, AllModelsWrapper
from model.dataset import StockDataset
from data.preprocessor import preprocess, build_targets
from data.features import compute_all_indicators
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Model registry: 1 所有模型 + 3 单模型 + 4 集成方法 = 8 选项
MODEL_REGISTRY = {
    "所有模型": AllModelsWrapper,
    "FT-iTransformer (时频协同)": FT_iTransformerWrapper,
    "Freq Orchestrator (频段协同)": FreqOrchestratorWrapper,
    "Node Transformer (图注意力·准确率偏低)": NodeTransformerWrapper,
    "PINN (物理约束)": PINNWrapper,
    "集成-加权平均": EnsembleWeightedWrapper,
    "集成-多数投票": EnsembleVotingWrapper,
    "集成-堆叠法 (Stacking)": EnsembleStackingWrapper,
    "集成-PINN Guard": EnsemblePINNGuardWrapper,
}


class SettingsDialog(QDialog):
    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        layout = QFormLayout(self)
        self.deepseek_key_input = QLineEdit(cfg.mcp.deepseek_api_key)
        self.deepseek_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow("DeepSeek API Key:", self.deepseek_key_input)
        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn.accepted.connect(self.accept)
        btn.rejected.connect(self.reject)
        layout.addRow(btn)


class TrainDialog(QDialog):
    training_done = pyqtSignal(str)  # emits ts_code when training completes
    background_requested = pyqtSignal(object, str)  # (TrainingWorker, ts_code) for bg training
    batch_started = pyqtSignal(list, str)  # (stocks, model_name) — batch training with queue

    _last_model_name: str = ""  # persist model selection across dialog instances

    def __init__(self, cfg: AppConfig, repo: Repository, parent=None, selected_stock: str | None = None,
                 selected_stocks: list[str] | None = None,
                 model_registry=None, fetcher=None, gs=None):
        super().__init__(parent)
        self.cfg = cfg
        self.repo = repo
        self.selected_stock = selected_stock
        self.selected_stocks = selected_stocks or ([selected_stock] if selected_stock else [])
        self._multi_mode = len(self.selected_stocks) > 1
        self.model_registry = model_registry
        self._fetcher = fetcher
        self._gs = gs
        self._worker: TrainingWorker | None = None
        self._attached_worker = False
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        if self._multi_mode:
            title = f"批量训练 - {len(self.selected_stocks)} 只股票"
        elif selected_stock:
            title = f"训练模型 - {selected_stock}"
        else:
            title = "训练模型 (全部)"
        self.setWindowTitle(title)
        self.resize(520, 500)
        layout = QVBoxLayout(self)

        if self._multi_mode:
            codes_short = [s.split(".")[0] for s in self.selected_stocks[:5]]
            preview = ", ".join(codes_short)
            if len(self.selected_stocks) > 5:
                preview += f" 等{len(self.selected_stocks)}只"
            self.stock_label = QLabel(f"批量训练: {preview}  |  最大并发: 3")
        else:
            self.stock_label = QLabel(f"股票: {selected_stock or '全部'}  |  数据: 5分钟线")
        self.stock_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.stock_label)

        # Existing models info
        self.existing_label = QLabel()
        self.existing_label.setVisible(False)
        self.existing_label.setStyleSheet(
            "background: #1a2a1a; border: 1px solid #338833; border-radius: 4px; "
            "padding: 6px; font-size: 12px; color: #aaddaa;"
        )
        self.existing_label.setWordWrap(True)
        layout.addWidget(self.existing_label)

        self.status = QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(180)
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型选择:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_REGISTRY.keys()))
        # Restore last selection
        if TrainDialog._last_model_name:
            idx = self.model_combo.findText(TrainDialog._last_model_name)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.model_combo.setToolTip("单模型 (3种):\n"
                                     "  FT-iTransformer: FFT频域+倒置Transformer (推荐)\n"
                                     "  Node Transformer: 图注意力+情绪注入 (准确率偏低)\n"
                                     "  PINN: 物理约束防过拟合\n"
                                     "集成方法 (4种):\n"
                                     "  加权平均: 动态调整2个基模型权重\n"
                                     "  多数投票: 2个模型一致才出信号\n"
                                     "  堆叠法: 元学习器组合基模型输出\n"
                                     "  PINN Guard: 物理约束守门集成预测\n"
                                     "全模型对比 (1种):\n"
                                     "  所有模型: 训练全部6个模型, 自动选准确率最高")
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        self.btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self._start_training)
        self.btn_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("后台运行")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._to_background)
        self.btn_layout.addWidget(self.cancel_btn)

        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setVisible(False)
        self.predict_btn.setStyleSheet("font-weight: bold; background: #2196F3; color: white; padding: 6px 16px;")
        self.predict_btn.clicked.connect(self._on_predict_clicked)
        self.btn_layout.addWidget(self.predict_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self._on_close)
        self.btn_layout.addWidget(close_btn)
        layout.addLayout(self.btn_layout)

        if self.selected_stock:
            self._refresh_existing_models()

    def _refresh_existing_models(self):
        """Show existing trained models for the selected stock."""
        if not self.model_registry or not self.selected_stock:
            return
        entries = self.model_registry.get_models_for_stock(self.selected_stock)
        if not entries:
            self.existing_label.setVisible(False)
            return
        lines = [f"已训练模型: {len(entries)} 个"]
        for e in entries:
            acc_str = f"{e.macro_acc:.2%}" if e.macro_acc > 0 else "N/A"
            best_str = f" (最佳子模型: {e.best_model_name})" if e.best_model_name else ""
            lines.append(f"  • {e.model_type} — 准确率 {acc_str}{best_str}")
        self.existing_label.setText("\n".join(lines))
        self.existing_label.setVisible(True)

    def _build_dataset(self, selected_features: list[str] | None = None):
        if self.selected_stock:
            stocks = [self.selected_stock]
        else:
            stocks = self.repo.get_screened_stocks()[:20]

        # Training data: last 30 days, 1min frequency (densest for 3s-step prediction)
        one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        sequences, targets_list, timestamps_list = [], [], []
        scaler = None
        for ts_code in stocks:
            # Try 1min first, fall back to 5min
            for freq in ["1min", "5min"]:
                df = self.repo.get_minutes(ts_code, freq=freq, start=one_month_ago)
                if df is not None and len(df) >= self.cfg.model.seq_len + 10:
                    break
            if df is None or len(df) < self.cfg.model.seq_len + 10:
                continue
            df = df.sort_values("trade_time")

            # Extract epoch timestamps for chronological splitting
            epoch_ts = pd.to_datetime(df["trade_time"]).apply(
                lambda t: t.timestamp()
            ).values.astype(float)
            timestamps_list.append(epoch_ts)

            raw_close = df["close"].values.astype(float)
            ind = compute_all_indicators(df)
            for col in ind.columns:
                if col not in df.columns:
                    df[col] = ind[col].values
            feat_arr, sc = preprocess(df, fit_scaler=True)
            # Apply feature mask if selected
            if selected_features is not None:
                from data.feature_selector import get_feature_mask
                mask = get_feature_mask(selected_features)
                if feat_arr.shape[1] == len(mask):
                    feat_arr = feat_arr[:, mask]
                    from sklearn.preprocessing import StandardScaler
                    from data.preprocessor import sanitize_scaler
                    sc2 = StandardScaler()
                    sc2.fit(feat_arr)
                    sanitize_scaler(sc2)
                    sc = sc2
            if scaler is None:
                scaler = sc
            direction, price_change = build_targets(raw_close, horizon=10)
            tgts = np.stack([direction, price_change], axis=1)
            sequences.append(feat_arr)
            targets_list.append(tgts)
        if not sequences:
            return None, None
        dense = self.selected_stock is not None
        ds = StockDataset(sequences, targets_list, self.cfg.model.seq_len, horizon=10,
                         dense=dense, timestamps=timestamps_list)
        return ds, scaler

    def _start_training(self):
        self.status.clear()
        model_name = self.model_combo.currentText()
        TrainDialog._last_model_name = model_name

        # ── Multi-stock batch mode: emit signal, close dialog ──
        if self._multi_mode:
            self.status.appendPlainText(f"批量训练: {len(self.selected_stocks)} 只股票已加入队列")
            self.status.appendPlainText(f"模型: {model_name}  |  最大并发: 3")
            self.status.appendPlainText("训练将在后台进行，可关闭此窗口")
            self.batch_started.emit(self.selected_stocks, model_name)
            self.accept()
            return

        self.status.appendPlainText("准备训练...")
        self.predict_btn.setVisible(False)

        from gui.workers import TrainingWorker
        self._worker = TrainingWorker(
            None, self.repo, self.cfg, self.cfg.model.checkpoint_dir, scaler=None,
            ts_code=self.selected_stock or "default",
            selected_features=None,
            model_type=model_name,
            repo=self.repo,
            selected_stock=self.selected_stock,
            model_name=model_name,
            fetcher=self._fetcher,
            gs=self._gs,
        )
        self._worker.phase_update.connect(lambda msg: self.status.appendPlainText(msg))
        self._worker.epoch_update.connect(self._on_epoch)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setRange(0, self.cfg.model.epochs)
        self.progress.setValue(0)
        self.status.appendPlainText("后台线程已启动，数据准备中...")
        self._worker.start()

    def _to_background(self):
        """Put training in background — hide dialog, worker continues running."""
        if self._attached_worker:
            self.close()
            return
        if self._worker and self._worker.isRunning():
            ts = self.selected_stock or "default"
            self.status.appendPlainText("训练已转入后台运行，可关闭此窗口")
            self.status.appendPlainText(f"后台任务: {ts} — 完成后自动通知")
            self.background_requested.emit(self._worker, ts)
            self._worker = None  # Detach ownership
        self.close()

    def _on_epoch(self, epoch: int, train_loss: float, val_loss: float, val_acc: float):
        self.progress.setValue(epoch + 1)
        self.status.appendPlainText(
            f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

    def _on_finished(self, msg: str):
        self.status.appendPlainText(msg)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(self.progress.maximum())
        if self.selected_stock:
            self.status.appendPlainText(
                "训练完成！点击下方 [开始预测] 按钮查看预测结果。\n"
                "之后可点击主界面工具栏 [▶ 实时预测] 开始自动更新。"
            )
            self.predict_btn.setVisible(True)

    def _on_predict_clicked(self):
        """User explicitly clicks predict — close dialog, main window handles prediction."""
        if self.selected_stock:
            self.training_done.emit(self.selected_stock)
        self.close()

    def _on_error(self, err: str):
        self.status.appendPlainText(f"错误: {err}")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def attach_existing_worker(self, worker, ts_code: str):
        """Reconnect dialog to a worker already running in background."""
        self._worker = worker
        self.selected_stock = ts_code
        self._attached_worker = True
        self.setWindowTitle(f"训练模型 - {ts_code} (后台运行中)")
        # Reconnect signals to this dialog
        worker.epoch_update.connect(self._on_epoch)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        # Update UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setText("停止训练并关闭")
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self._stop_and_close)
        self.progress.setRange(0, self.cfg.model.epochs)
        self.status.appendPlainText(f"已重新连接 — {ts_code} 后台训练进行中...")
        self.status.appendPlainText("点击 [停止训练并关闭] 可终止训练")

    def _stop_and_close(self):
        """Stop the attached worker and close dialog."""
        if self._worker and self._worker.isRunning():
            self.status.appendPlainText("正在停止训练...")
            self._worker.stop()
        self.close()

    def _on_close(self):
        """Close dialog. If training, move to background automatically."""
        if self._attached_worker:
            self.close()
            return
        if self._worker and self._worker.isRunning():
            ts = self.selected_stock or "default"
            self.background_requested.emit(self._worker, ts)
            self._worker = None
        self.close()
