"""Optimized chart canvas: caches historical plot items, incremental prediction updates."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabBar, QLabel, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import pandas as pd
import numpy as np



class TimeAxisState:
    """Shared state so price and volume axes show identical time labels."""
    __slots__ = ("times", "future_count", "day_boundaries", "day_labels", "time_labels")
    def __init__(self):
        self.times: list[tuple[int, int, bool]] = []        # (hour, minute, is_trading)
        self.future_count: int = 0
        self.day_boundaries: list[int] = []                  # bar indices where a new trading day starts
        self.day_labels: dict[int, str] = {}                 # bar_idx → "MM-DD" string
        self.time_labels: dict[int, str] = {}                # bar_idx → "HH:MM" string


class TimeAxisItem(pg.AxisItem):
    """Custom X axis that uses pyqtgraph's auto-spacing for overlap-free ticks.
    - Span > 1.5 days: date-only labels (MM-DD), day boundaries color-distinguished.
    - Span ≤ 1.5 days: HH:MM time labels only, no dates."""

    def __init__(self, orientation: str, state: TimeAxisState | None = None, show_labels: bool = True):
        super().__init__(orientation)
        if state is not None:
            self._state = state
        else:
            self._own = TimeAxisState()
            self._state = self._own
        self._span_days: float = 0.0  # total time span in days
        self._show_labels = show_labels
        self._freq: str = "1min"  # "1min" → HH:MM labels, "daily" → MM-DD labels

    @staticmethod
    def _is_trading(hour: int, minute: int) -> bool:
        t = hour * 60 + minute
        return (9 * 60 + 30 <= t <= 11 * 60 + 30) or (13 * 60 <= t <= 15 * 60)

    def set_timestamps(self, trade_times: pd.DatetimeIndex, future_count: int = 0, freq: str = "1min",
                        future_interval_min: int = 20):
        self._state.future_count = future_count
        self._freq = freq
        times: list[tuple[int, int, bool]] = []
        day_labels: dict[int, str] = {}
        time_labels: dict[int, str] = {}
        day_bounds: list[int] = []

        n = len(trade_times)

        # Compute total span in days
        if n >= 2:
            span_sec = (trade_times[-1] - trade_times[0]).total_seconds()
            future_span_sec = future_count * future_interval_min * 60 if future_count > 0 else 0
            self._span_days = span_sec / 86400.0 + future_span_sec / 86400.0
        else:
            self._span_days = 0

        prev_date = None
        seen_afternoon_today = False
        for i, t in enumerate(trade_times):
            # Skip NaT timestamps (gap breaks inserted by _insert_gap_breaks)
            if pd.isna(t):
                times.append((0, 0, False))
                time_labels[i] = ""
                continue
            h, m = t.hour, t.minute
            trading = self._is_trading(h, m)
            times.append((h, m, trading))

            d = t.date()
            is_new_day = (d != prev_date)
            if is_new_day:
                prev_date = d
                day_bounds.append(i)
                day_labels[i] = f"{d.month}-{d.day}"
                seen_afternoon_today = False  # reset for new day

            # 11:30 sits right next to 13:00 (no gap), skip to avoid label overlap
            if h == 11 and m == 30:
                time_labels[i] = ""
            elif h >= 13 and not seen_afternoon_today:
                seen_afternoon_today = True
                time_labels[i] = "13:00"
            else:
                time_labels[i] = f"{h}:{m:02d}"

        # Future bars — no labels (prediction zone is visually distinct via yellow path line)
        for fi in range(future_count):
            times.append((0, 0, False))

        self._state.times = times
        self._state.day_boundaries = day_bounds
        self._state.day_labels = day_labels
        self._state.time_labels = time_labels

        if not self._show_labels:
            self.ticks = None
        elif self._freq == "daily":
            # Smaller font for daily date labels
            f = QFont()
            f.setPointSize(8)
            self.setStyle(tickFont=f)
            # Date-only mode: show every 2nd trading day
            date_items = [(k, v) for k, v in day_labels.items()]
            date_items.sort(key=lambda x: x[0])
            # Stride 2: 30 trading days → ~15 labels
            filtered = [date_items[i] for i in range(0, len(date_items), 2)]
            date_ticks = [(float(k), v) for k, v in filtered]
            self.setTicks([date_ticks])
        else:
            # Reset to default font for minute labels
            f = QFont()
            f.setPointSize(10)
            self.setStyle(tickFont=f)
            # Time-only mode: show labels at trading-session boundaries + hour marks
            key_minutes = {0, 30}  # HH:00 and HH:30
            time_ticks = []
            seen_afternoon = False
            day_bounds = self._state.day_boundaries
            day_idx = 0
            for i, (h, m, trading) in enumerate(times[:n]):  # skip future bars
                # New trading day → reset afternoon flag
                if day_idx < len(day_bounds) and i == day_bounds[day_idx]:
                    day_idx += 1
                    seen_afternoon = False
                if not trading:
                    continue
                if h == 11 and m == 30:
                    continue  # skip 11:30 tick
                # Force "13:00" label at the first afternoon bar (e.g. 13:01 or 13:05)
                if h >= 13 and not seen_afternoon:
                    seen_afternoon = True
                    time_ticks.append((float(i), "13:00"))
                    continue
                if m in key_minutes:
                    time_ticks.append((float(i), f"{h}:{m:02d}"))
            if time_ticks:
                self.setTicks([time_ticks])
            else:
                self.ticks = None

    def tickSpacing(self, minVal, maxVal, size):
        """Return (major_spacing, minor_spacing) in bar-index units.
        Spacing is computed so tick labels never overlap (~40px minimum between ticks)."""
        # Guard against NaN/None/inf from corrupted data
        try:
            if size is None or (isinstance(size, float) and size != size) or size <= 0:
                return [(1, 0.0)]
            if minVal is None or maxVal is None:
                return [(1, 0.0)]
            minVal = float(minVal) if minVal == minVal else 0.0
            maxVal = float(maxVal) if maxVal == maxVal else 1.0
            if maxVal <= minVal:
                return [(1, 0.0)]
            visible_bars = maxVal - minVal
        except (TypeError, ValueError, OverflowError):
            return [(1, 0.0)]
        if visible_bars <= 0:
            return [(1, 0.0)]

        # Target: at least 40px between major ticks to prevent overlap
        target_major = max(2, size / 40)
        spacing = max(1, int(visible_bars / target_major))

        # Round spacing up to "nice" values
        nice = [1, 2, 5, 10, 15, 20, 30, 60, 120, 240, 480]
        for nv in nice:
            if spacing <= nv:
                spacing = nv
                break
        else:
            spacing = ((spacing + nice[-1] - 1) // nice[-1]) * nice[-1]

        # Minor ticks: half of major, or 0 if main spacing is already tight
        minor = spacing // 2 if spacing >= 4 else 0
        # Must return list of (spacing, offset) tuples — pyqtgraph iterates with len()
        if minor > 0:
            return [(spacing, 0.0), (minor, 0.0)]
        return [(spacing, 0.0)]

    def tickStrings(self, values, scale, spacing):
        """Convert bar-index tick values to displayed strings.
        - span > 1.5 days → date-only (MM-DD), no HH:MM
        - span ≤ 1.5 days → HH:MM only, no dates
        """
        if not self._show_labels:
            return ["" for _ in values]
        vals_int = [int(round(v)) for v in values]
        if self._freq == "daily":
            # Date-only mode: show MM-DD at day boundaries, empty for other bars
            day_labels = self._state.day_labels
            return [day_labels.get(v, "") for v in vals_int]
        else:
            # Time-only mode: show HH:MM for every bar with a label
            time_labels = self._state.time_labels
            return [time_labels.get(v, "") for v in vals_int]


class ChartCanvas(QWidget):
    timeframe_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._prediction: dict | None = None

        # Lookup data for hover
        self._x_vals: np.ndarray | None = None
        self._close_vals: np.ndarray | None = None
        self._timestamps_pd: pd.DatetimeIndex | None = None  # pandas naive timestamps, no tz conversion
        self._freq_min: int = 5
        self._freq_label: str = "1min"  # "1min" or "daily"
        self._time_labels: list[str] = []
        self._future_count: int = 0
        self._rolling_preds: list[dict] = []  # for hover in prediction zone
        self._step_mult: int = 1  # x-axis units per prediction step (20 // freq_min)

        # Cached plot items for incremental updates
        self._price_line: pg.PlotDataItem | None = None
        self._ma_items: list[pg.PlotDataItem] = []
        self._vol_bars: pg.BarGraphItem | None = None
        self._marker_line: pg.PlotDataItem | None = None
        self._day_lines: list = []  # day boundary separators (PlotDataItem + TextItem)
        self._pred_items: list = []  # all prediction overlay items
        self._legend_label: pg.TextItem | None = None  # 分时线 legend (top-right)
        # Set by _setup_ui()
        self._hover_label = None
        self._crosshair = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabBar()
        self.tabs.addTab("分钟线")
        self.tabs.addTab("日线")
        self.tabs.setCurrentIndex(0)
        self.tabs.currentChanged.connect(
            lambda i: self.timeframe_changed.emit(["1min", "daily"][i])
        )
        layout.addWidget(self.tabs)

        # Prediction strip
        self._pred_strip = QWidget()
        self._pred_strip.setStyleSheet("background-color: #16161e; border-radius: 6px;")
        strip_layout = QHBoxLayout(self._pred_strip)
        strip_layout.setContentsMargins(8, 2, 8, 2)
        strip_layout.setSpacing(8)

        self._strip_dir = QLabel("--")
        self._strip_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._strip_dir.setFixedSize(70, 36)
        self._strip_dir.setStyleSheet("font-size: 16px; font-weight: bold; border-radius: 6px;")
        strip_layout.addWidget(self._strip_dir)

        self._strip_conf = QProgressBar()
        self._strip_conf.setRange(0, 100)
        self._strip_conf.setTextVisible(True)
        self._strip_conf.setFixedHeight(24)
        self._strip_conf.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555; border-radius: 4px;
                background-color: #1a1a1a;
                font-size: 12px; font-weight: bold; text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 3px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff4444, stop:0.5 #ffaa00, stop:1 #44cc44);
            }
        """)
        strip_layout.addWidget(self._strip_conf, stretch=1)

        self._strip_price = QLabel("--")
        self._strip_price.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._strip_price.setFixedSize(100, 36)
        self._strip_price.setStyleSheet("font-size: 15px; font-weight: bold; color: #fff;"
                                         "background-color: #16213e; border-radius: 6px;")
        strip_layout.addWidget(self._strip_price)

        layout.addWidget(self._pred_strip)

        self.setMouseTracking(True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setMouseTracking(True)
        self.graphics.viewport().setMouseTracking(True)
        layout.addWidget(self.graphics)

        self._axis_state = TimeAxisState()  # shared across price + volume plots

        self.time_axis = TimeAxisItem(orientation="bottom", state=self._axis_state, show_labels=True)
        self.vol_axis = TimeAxisItem(orientation="bottom", state=self._axis_state, show_labels=True)
        self.price_plot = self.graphics.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis})
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setLabel("left", "价格", units="¥")

        self.vol_plot = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": self.vol_axis})
        self.vol_plot.showGrid(x=True, y=True, alpha=0.3)
        self.vol_plot.setLabel("left", "成交量", units="手")
        self.vol_plot.setLabel("bottom", "时间")
        self.vol_plot.setXLink(self.price_plot)

        # Hover tooltip - show time and price on mouse move
        self._hover_label = pg.TextItem("", anchor=(0, 1), color=(200, 200, 200),
                                        fill=pg.mkColor(20, 20, 28, 220))
        self._hover_label.setVisible(False)
        self.price_plot.addItem(self._hover_label)

        # Hover crosshair — vertical dashed line (InfiniteLine avoids auto-range interference)
        self._crosshair = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen("#888888", width=1, style=Qt.PenStyle.DashLine))
        self.price_plot.addItem(self._crosshair)
        self._crosshair.setVisible(False)

        # Connect mouse move to hover handler
        self.price_plot.scene().sigMouseMoved.connect(self._on_hover)

        # Prevent pan/zoom below bar 0 (negative time axis)
        self.price_plot.vb.setLimits(xMin=-0.5)
        self.vol_plot.vb.setLimits(xMin=-0.5, yMin=0)

        # Zoom only via mouse wheel — disable rectangle zoom
        self.price_plot.vb.setMouseMode(pg.ViewBox.PanMode)
        self.vol_plot.vb.setMouseMode(pg.ViewBox.PanMode)

        self.graphics.ci.layout.setRowStretchFactor(0, 4)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)

        # Intercept 'A' key to auto-range with Y at 80%-120% of data
        self.price_plot.vb.installEventFilter(self)
        self.vol_plot.vb.installEventFilter(self)

        
    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_A:
            self._auto_range_y()
            return True
        return super().eventFilter(obj, event)

    def _auto_range_y(self):
        """Set Y range to 80%-120% of data, then auto-range X to full data."""
        if self._close_vals is not None and len(self._close_vals) > 0:
            mn = float(np.nanmin(self._close_vals))
            mx = float(np.nanmax(self._close_vals))
            if mx > mn:
                self.price_plot.vb.autoRange()
                self.price_plot.setYRange(mn, mx, padding=0)
                return
        self.price_plot.vb.autoRange()

    # ── Public API ──────────────────────────────────────────────

    def plot_kline(self, df: pd.DataFrame, ts_code: str = "", freq: str = "1min",
                   prediction: dict | None = None,
                   rolling_preds: list[dict] | None = None):
        """Full plot: historical data + 2h prediction with 6×20min confidence bands."""
        df = df.copy()
        self._df = df.copy()
        self._prediction = prediction
        # Normalize timestamp column: daily data has trade_date, minute data has trade_time
        if "trade_time" not in df.columns and "trade_date" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_date"])
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"])
        if df.empty:
            return

        self._clear_cached_items()

        # Downsample for display if too many bars (>600 → ~300)
        n_raw = len(df)
        if n_raw > 600:
            stride = max(2, n_raw // 300)
            df = df.iloc[::stride].reset_index(drop=True)
            if "trade_time" in df.columns:
                df["trade_time"] = pd.to_datetime(df["trade_time"])

        freq_label = {"1min": "分钟线", "daily": "日线"}.get(freq, freq)
        self.price_plot.setTitle(f"{ts_code} [{freq_label}]")

        close0 = df["close"].values.astype(float)
        # Keep NaN for gap breaks; only clean Inf to prevent Qt render crash
        close0[np.isinf(close0)] = np.nan
        close = close0
        n = len(close)
        data_min = float(np.nanmin(close))
        data_max = float(np.nanmax(close))
        if np.isnan(data_min) or np.isnan(data_max):
            return
        price_range = data_max - data_min if data_max > data_min else max(abs(close[finite_mask][-1]) * 0.02, 0.01)

        # Prediction bars: 6 steps × 20min each. X-axis step_mult ensures they span
        # the same visual width as equivalent historical bars.
        freq_min_per_bar = {"1min": 1, "daily": 1440}.get(freq, 60)
        if freq == "daily":
            step_mult = 1
        else:
            step_mult = max(1, 20 // freq_min_per_bar)  # 1min→20, 5min→4
        self._step_mult = step_mult
        future_bars = 6 * step_mult if prediction else 0
        x = np.arange(n + future_bars)

        # Timestamps
        self._timestamps_pd = None
        if "trade_time" in df.columns:
            trade_times = pd.DatetimeIndex(df["trade_time"])
            self.time_axis.set_timestamps(trade_times, future_count=future_bars, freq=freq,
                                          future_interval_min=freq_min_per_bar)
            self.vol_axis.set_timestamps(trade_times, future_count=future_bars, freq=freq,
                                         future_interval_min=freq_min_per_bar)
            self._timestamps_pd = trade_times

        self._x_vals = x
        self._close_vals = close
        self._future_count = future_bars
        self._freq_min = freq_min_per_bar
        self._freq_label = freq
        self._time_labels = self._build_time_labels(df, future_bars, freq_min_per_bar)

        # ── Cache historical items ──
        is_minute = freq == "1min"

        # Price line — green for 分时线 and 日线
        # connect="finite" breaks the line at NaN (trading gaps), connect="all" bridges them
        price_pen = pg.mkPen("#00ff88", width=3) if is_minute else pg.mkPen("#00ff88", width=2)
        self._price_line = self.price_plot.plot(
            x[:n], close, pen=price_pen, name="收盘价", connect="finite"
        )

        self._ma_items = []

        # VWAP (均价线) for 分时线 — cumulative amount/volume reset per trading day
        if is_minute and "amount" in df.columns:
            vol_col_vwap = "volume" if "volume" in df.columns else "vol"
            if vol_col_vwap in df.columns:
                vol_arr = df[vol_col_vwap].values.astype(float)
                amt_arr = df["amount"].values.astype(float)
                vwap_y = np.full(n, np.nan)
                run_vol = 0.0; run_amt = 0.0; prev_date = None
                has_trade_time = "trade_time" in df.columns
                for i in range(n):
                    if np.isnan(vol_arr[i]) or np.isnan(amt_arr[i]):
                        continue  # gap break — VWAP stays NaN
                    d = pd.to_datetime(df["trade_time"].iloc[i]).date() if has_trade_time else None
                    if d != prev_date:
                        run_vol = 0.0; run_amt = 0.0; prev_date = d
                    run_vol += max(vol_arr[i], 0); run_amt += amt_arr[i]
                    vwap_y[i] = run_amt / max(run_vol, 1.0)
                self._ma_items.append(self.price_plot.plot(
                    x[:n], vwap_y, pen=pg.mkPen("#ff8800", width=1.5), name="均价", connect="finite"))

        # MA lines (日线 only) — clip below data_min so they don't drag the chart down
        if not is_minute:
            if n >= 5:
                ma5 = pd.Series(close).rolling(5).mean().values
                ma5[ma5 < data_min] = np.nan
                self._ma_items.append(self.price_plot.plot(
                    x[:n], ma5, pen=pg.mkPen("#ffffff", width=1, style=pg.QtCore.Qt.PenStyle.DotLine), connect="finite"))
            if n >= 10:
                ma10 = pd.Series(close).rolling(10).mean().values
                ma10[ma10 < data_min] = np.nan
                self._ma_items.append(self.price_plot.plot(
                    x[:n], ma10, pen=pg.mkPen("#ffdd00", width=1, style=pg.QtCore.Qt.PenStyle.DotLine), connect="finite"))
            if n >= 20:
                ma20 = pd.Series(close).rolling(20).mean().values
                ma20[ma20 < data_min] = np.nan
                self._ma_items.append(self.price_plot.plot(
                    x[:n], ma20, pen=pg.mkPen("#cc66ff", width=1.5, style=pg.QtCore.Qt.PenStyle.DotLine), connect="finite"))
            if n >= 60:
                ma60 = pd.Series(close).rolling(60).mean().values
                ma60[ma60 < data_min] = np.nan
                self._ma_items.append(self.price_plot.plot(
                    x[:n], ma60, pen=pg.mkPen("#00cc44", width=1, style=pg.QtCore.Qt.PenStyle.DotLine), connect="finite"))

        # ── Legend (分时线 / 日线: top-right) ──
        if is_minute:
            legend_html = (
                '<div style="background: rgba(20,20,28,0.85); padding: 4px 10px; '
                'border-radius: 4px; font-size: 12px; white-space: nowrap;">'
                '<span style="color: #00ff88; font-weight: bold;">━━ 分时价格</span>&nbsp;&nbsp;'
                '<span style="color: #ff8800; font-weight: bold;">━━ 均价</span>'
                '</div>'
            )
            self._legend_label = pg.TextItem(html=legend_html, anchor=(1, 1))
            self._legend_label.setPos(n - 1, data_max * 1.12)
            self.price_plot.addItem(self._legend_label)
        else:
            ma_parts = ['<span style="color: #00ff88; font-weight: bold;">━━ 收盘价</span>']
            if n >= 5:
                ma_parts.append('<span style="color: #ffffff;">···· 5日均线</span>')
            if n >= 10:
                ma_parts.append('<span style="color: #ffdd00;">···· 10日均线</span>')
            if n >= 20:
                ma_parts.append('<span style="color: #cc66ff;">···· 20日均线</span>')
            if n >= 60:
                ma_parts.append('<span style="color: #00cc44;">···· 60日均线</span>')
            if ma_parts:
                legend_html = (
                    '<div style="background: rgba(20,20,28,0.85); padding: 4px 10px; '
                    'border-radius: 4px; font-size: 12px; white-space: nowrap;">'
                    + '&nbsp;&nbsp;'.join(ma_parts) +
                    '</div>'
                )
                self._legend_label = pg.TextItem(html=legend_html, anchor=(1, 1))
                self._legend_label.setPos(n - 1, data_max * 1.12)
                self.price_plot.addItem(self._legend_label)

        # Current marker (disabled - yellow dashed line)
        # marker_top = max(close[-1], 0.01) * 1.10
        # marker_bottom = data_min - price_range * 0.05
        # self._marker_line = self.price_plot.plot(
        #     [n - 1, n - 1], [marker_bottom, marker_top],
        #     pen=pg.mkPen("#ffdd00", width=3, style=pg.QtCore.Qt.PenStyle.DashLine)
        # )

        # Volume bars — NaN from gap breaks become invisible (height=0)
        vol_col = "volume" if "volume" in df.columns else "vol"
        vol = np.nan_to_num(df[vol_col].values.astype(float), nan=0.0) if vol_col in df.columns else np.zeros(n)
        # For NaN close rows, set volume bar color to transparent
        is_nan_close = np.isnan(close)
        colors = ["#ff4444" if c >= df["open"].values.astype(float)[i] else "#00cc44"
                  for i, c in enumerate(close)]
        self._vol_bars = pg.BarGraphItem(x=x[:n], height=vol, width=0.8, brushes=colors)
        self.vol_plot.addItem(self._vol_bars)

        # ── Day boundary separators ──
        day_bounds = getattr(self._axis_state, 'day_boundaries', [])
        if day_bounds:
            data_high = data_max
            data_low = data_min
            day_colors = ["#665544", "#445566", "#556644", "#664455", "#446655", "#554466"]
            for idx, di in enumerate(day_bounds):
                color = day_colors[idx % len(day_colors)]
                line = self.price_plot.plot(
                    [di, di], [data_low, data_high],
                    pen=pg.mkPen(color, width=0.8, style=pg.QtCore.Qt.PenStyle.DashLine)
                )
                self._day_lines.append(line)
            # Date labels are shown on the volume axis (bottom) only —
            # removed from the price plot to keep the top chart clean

        self.price_plot.setYRange(data_min - price_range * 0.05, data_max * 1.15, padding=0)
        # Center current data point in the viewport
        half_window = max(15, min(60, n // 4))  # adaptive: 15~60 bars each side
        center = n - 1
        x_left = max(0, center - half_window)
        x_right = center + half_window + future_bars
        self.price_plot.setXRange(x_left, x_right, padding=0)
        self.price_plot.vb.setLimits(xMin=-0.5, xMax=n + future_bars + 10)

        self._rolling_preds = rolling_preds or []

        # Prediction overlay
        if prediction:
            self._draw_prediction(prediction, rolling_preds, close, n, future_bars)

    def plot_kline_fast(self, df: pd.DataFrame, ts_code: str = "", freq: str = "1min",
                         prediction: dict | None = None,
                         rolling_preds: list[dict] | None = None):
        """Fast update: only change prediction overlay, keep historical items."""
        df = df.copy()
        self._df = df.copy()
        self._prediction = prediction

        if df.empty or self._price_line is None:
            self.plot_kline(df, ts_code, freq, prediction, rolling_preds)
            return

        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"])

        close = df["close"].values.astype(float)
        n = len(close)
        freq_min_per_bar = {"1min": 1, "daily": 1440}.get(freq, 60)
        if freq == "daily":
            step_mult = 1
        else:
            step_mult = max(1, 20 // freq_min_per_bar)
        self._step_mult = step_mult
        future_bars = 6 * step_mult if prediction else 0
        x = np.arange(n + future_bars)

        # Quick update lookup data
        self._x_vals = x
        self._close_vals = close
        self._future_count = future_bars
        self._freq_min = freq_min_per_bar
        self._freq_label = freq

        self._rolling_preds = rolling_preds or []

        # Update timestamp data
        if "trade_time" in df.columns:
            trade_times = pd.DatetimeIndex(df["trade_time"])
            self.time_axis.set_timestamps(trade_times, future_count=future_bars, freq=freq,
                                          future_interval_min=freq_min_per_bar)
            self.vol_axis.set_timestamps(trade_times, future_count=future_bars, freq=freq,
                                         future_interval_min=freq_min_per_bar)
            self._timestamps_pd = trade_times

        # ── Clear only prediction items, keep history ──
        self._clear_prediction_items()

        # Update marker position
        if self._marker_line is not None:
            data_min, data_max = close.min(), close.max()
            price_range = data_max - data_min if data_max > data_min else close[-1] * 0.02
            marker_top = max(close[-1], 0.01) * 1.10
            marker_bottom = data_min - price_range * 0.05
            self._marker_line.setData([n - 1, n - 1], [marker_bottom, marker_top])

        if prediction:
            self._draw_prediction(prediction, rolling_preds, close, n, future_bars)

    def plot_rolling_path(self, rolling_preds: list[dict]):
        """Animation-only: update just the rolling prediction path without touching anything else."""
        if self._df is None or self._df.empty:
            return
        n = len(self._df)
        close = self._df["close"].values.astype(float)
        prediction = self._prediction
        if not prediction:
            return

        target = prediction.get("target_price", close[-1])

        # Remove old prediction path (last item in _pred_items is the path line)
        self._clear_prediction_items()

        direction = prediction.get("direction", "flat")

        # Redraw path + confidence band
        path_x = [n - 1]
        path_y = [close[-1]]
        upper_y = [close[-1]]
        lower_y = [close[-1]]
        conf_sum = 0.0; conf_count = 0
        step_mult = getattr(self, '_step_mult', 1)
        for rp in rolling_preds:
            step_x = n - 1 + rp.get("step", 1) * step_mult
            path_x.append(step_x)
            path_y.append(rp.get("target_price", target))
            upper_y.append(rp.get("price_upper", rp.get("target_price", target)))
            lower_y.append(rp.get("price_lower", rp.get("target_price", target)))
            cc = rp.get("direction_conf", 0.5)
            conf_sum += cc
            conf_count += 1
        avg_conf = conf_sum / max(conf_count, 1)

        # Center trajectory
        path_line = self.price_plot.plot(path_x, path_y, pen=pg.mkPen("#ffdd00", width=3))
        self._pred_items.append(path_line)

        # Confidence band
        upper_curve = self.price_plot.plot(path_x, upper_y, pen=pg.mkPen(None))
        lower_curve = self.price_plot.plot(path_x, lower_y, pen=pg.mkPen(None))
        band_alpha = max(30, min(140, int(avg_conf * 160)))
        band_color = (
            (255, 50, 50, band_alpha) if direction == "up" else
            (80, 200, 80, band_alpha) if direction == "down" else
            (160, 160, 160, band_alpha)
        )
        fill = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(band_color))
        self.price_plot.addItem(fill)
        self._pred_items.append(fill)
        self._pred_items.append(upper_curve)
        self._pred_items.append(lower_curve)

    def set_prediction_strip(self, direction: str, conf: float, target_price):
        """Update the prediction strip between tabs and chart."""
        # Guard against NaN/None/inf/non-numeric from corrupted predictions
        try:
            conf = float(conf)
            if conf != conf:  # NaN check
                conf = 0.0
        except (TypeError, ValueError):
            conf = 0.0
        try:
            target_price = float(target_price) if target_price != "--" else 0.0
            if target_price != target_price:  # NaN check
                target_price = 0.0
        except (TypeError, ValueError):
            target_price = 0.0

        if direction == "up":
            text_color = "#ff3333"; bg_color = "#2a1010"; border = "#cc0000"
            label_text = "▲ 看涨"
        elif direction == "down":
            text_color = "#33cc44"; bg_color = "#102a10"; border = "#00aa00"
            label_text = "▼ 看跌"
        else:
            text_color = "#888888"; bg_color = "#1a1a1a"; border = "#555555"
            label_text = "─ 横盘"

        self._strip_dir.setText(label_text)
        self._strip_dir.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {text_color};"
            f"background-color: {bg_color}; border: 2px solid {border}; border-radius: 6px;"
        )
        try:
            conf_val = int(conf * 100)
        except (ValueError, OverflowError):
            conf_val = 0
        self._strip_conf.setValue(conf_val)
        self._strip_conf.setFormat(f"{conf_val}%")
        self._strip_price.setText(f"¥{target_price:.2f}")
        self._strip_price.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: #ffffff;"
            f"background-color: {bg_color}; border: 1px solid {border}; border-radius: 6px;"
        )

    def clear_prediction_strip(self):
        """Reset prediction strip to empty state."""
        self._strip_dir.setText("")
        self._strip_dir.setStyleSheet(
            "font-size: 16px; font-weight: bold; border-radius: 6px;"
        )
        self._strip_conf.setValue(0)
        self._strip_conf.setFormat("")
        self._strip_price.setText("")
        self._strip_price.setStyleSheet(
            "font-size: 15px; font-weight: bold;"
        )

    def set_prediction(self, pred: dict):
        """Add prediction overlay to existing chart."""
        if self._df is None or self._df.empty:
            return
        self._prediction = pred

    def add_prediction_line(self, direction: str, target_price: float, current_price: float):
        """Legacy method — kept for compatibility."""
        if self._df is None or self._df.empty:
            return
        n = len(self._df)
        x_end = n + 12
        color = "#ff4444" if direction == "up" else ("#00cc44" if direction == "down" else "#888888")
        self.price_plot.plot(
            [n - 1, x_end],
            [current_price, target_price],
            pen=pg.mkPen(color, width=3, style=pg.QtCore.Qt.PenStyle.DashLine)
        )

    # ── Internal ────────────────────────────────────────────────

    def _build_time_labels(self, df, future_bars, future_interval_min=20):
        labels = []
        for i in range(len(df)):
            try:
                labels.append(pd.to_datetime(df["trade_time"].iloc[i]).strftime("%H:%M"))
            except Exception:
                labels.append(f"bar {i}")
        if future_bars > 0 and "trade_time" in df.columns and len(df) > 0:
            last_dt = pd.to_datetime(df["trade_time"].iloc[-1])
            for i in range(future_bars):
                future_dt = last_dt + pd.Timedelta(minutes=future_interval_min * (i + 1))
                labels.append(future_dt.strftime("%H:%M"))
        return labels

    @staticmethod
    def _insert_gap_breaks(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Insert NaN rows proportional to actual time gaps between bars.

        Lunch break (90 min) gets ~90 NaN rows for 1-min data so the chart
        shows a proportional gap. Overnight/weekend gaps are capped to avoid
        making the chart unusably sparse.
        """
        if df.empty or "trade_time" not in df.columns or len(df) < 2:
            return df
        times = pd.to_datetime(df["trade_time"])
        freq_min = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "60min": 60, "daily": 1440}.get(freq, 5)
        max_gap_rows = 300  # cap overnight/weekend gaps

        rows = []
        for i in range(len(df)):
            if i > 0:
                gap_min = (times.iloc[i] - times.iloc[i - 1]).total_seconds() / 60.0
                expected_gap = freq_min
                if gap_min > expected_gap * 1.5:
                    # Proportional NaN rows, capped
                    n_nan = min(int(gap_min / freq_min), max_gap_rows)
                    nan_row = {c: np.nan for c in df.columns}
                    for _ in range(n_nan):
                        rows.append(dict(nan_row))
            rows.append({c: df.iloc[i][c] for c in df.columns})
        return pd.DataFrame(rows)

    def _clear_cached_items(self):
        """Remove all cached plot items from plots."""
        if self._crosshair is not None:
            self._crosshair.setVisible(False)
        if self._hover_label is not None:
            self._hover_label.setVisible(False)
        if self._price_line is not None:
            self.price_plot.removeItem(self._price_line)
            self._price_line = None
        for item in self._ma_items:
            self.price_plot.removeItem(item)
        self._ma_items = []
        if self._marker_line is not None:
            self.price_plot.removeItem(self._marker_line)
            self._marker_line = None
        if self._vol_bars is not None:
            self.vol_plot.removeItem(self._vol_bars)
            self._vol_bars = None
        for line in self._day_lines:
            self.price_plot.removeItem(line)
        self._day_lines = []
        if self._legend_label is not None:
            self.price_plot.removeItem(self._legend_label)
            self._legend_label = None
        self._clear_prediction_items()

    def _on_hover(self, pos):
        """Show time and price on mouse hover — covers both historical and prediction bars."""
        try:
            if (self._crosshair is None or self._hover_label is None
                    or self._x_vals is None or self._close_vals is None
                    or len(self._close_vals) == 0):
                if self._crosshair is not None:
                    self._crosshair.setVisible(False)
                if self._hover_label is not None:
                    self._hover_label.setVisible(False)
                return
            vb = self.price_plot.vb
            if not vb.sceneBoundingRect().contains(pos):
                self._crosshair.setVisible(False)
                self._hover_label.setVisible(False)
                return

            mouse_pt = vb.mapSceneToView(pos)
            mx = mouse_pt.x()

            n_hist = len(self._close_vals)
            n_future = self._future_count
            x_min, x_max = self._x_vals[0], self._x_vals[-1]

            if mx < x_min - 0.5 or mx > x_max + 0.5:
                self._crosshair.setVisible(False)
                self._hover_label.setVisible(False)
                return

            # Snap to nearest bar index
            mx_clamped = max(x_min, min(x_max, mx))
            idx = int(round(mx_clamped))
            idx = max(0, min(len(self._x_vals) - 1, idx))

            # ── Prediction zone (future bars) ──
            if idx >= n_hist:
                if not self._rolling_preds:
                    self._crosshair.setVisible(False)
                    self._hover_label.setVisible(False)
                    return
                step_mult = getattr(self, '_step_mult', 1)
                future_step = (idx - n_hist) // step_mult
                if future_step < 0 or future_step >= len(self._rolling_preds):
                    self._crosshair.setVisible(False)
                    self._hover_label.setVisible(False)
                    return
                rp = self._rolling_preds[future_step]
                py = rp.get("target_price", float(self._close_vals[-1]))
                if np.isnan(py):
                    self._crosshair.setVisible(False)
                    self._hover_label.setVisible(False)
                    return

                # Snap x position to the actual step bar
                step_x = n_hist + future_step * step_mult

                # Prediction time label: +20m, +40m, +1h, etc.
                if self._freq_label == "daily":
                    time_str = f"+{future_step + 1}日"
                else:
                    total_min = (future_step + 1) * 20
                    if total_min < 60:
                        time_str = f"+{total_min}m"
                    else:
                        h = total_min // 60
                        m = total_min % 60
                        time_str = f"+{h}h" + (f"{m}m" if m > 0 else "")

                # Use yellow color for prediction hover
                label_html = (
                    f'<div style="background: rgba(20,20,28,0.90); padding: 3px 8px; '
                    f'border-radius: 3px; font-size: 12px; white-space: nowrap;">'
                    f'<span style="color: #ffdd00;">{time_str}</span>  '
                    f'<span style="color: #ffdd00; font-weight: bold;">¥{py:.2f}</span>'
                    f'</div>'
                )
                self._hover_label.setHtml(label_html)
                self._hover_label.setPos(float(step_x) + 0.5, py)
                self._hover_label.setVisible(True)

                # Crosshair vertical line at prediction step position
                self._crosshair.setPos(float(step_x))
                self._crosshair.setVisible(True)
                return

            # ── Historical zone ──
            idx_hist = max(0, min(n_hist - 1, idx))

            # Skip NaN (gap breaks) — walk outward to find nearest valid bar
            py = self._close_vals[idx_hist]
            if np.isnan(py):
                best_idx = None
                for offset in range(1, max(n_hist, 1)):
                    left = idx_hist - offset
                    right = idx_hist + offset
                    if left >= 0 and not np.isnan(self._close_vals[left]):
                        best_idx = left; break
                    if right < n_hist and not np.isnan(self._close_vals[right]):
                        best_idx = right; break
                if best_idx is None:
                    self._crosshair.setVisible(False)
                    self._hover_label.setVisible(False)
                    return
                idx_hist = best_idx
                py = self._close_vals[idx_hist]

            # Time format: HH:MM for minute charts, MM-DD for daily
            time_str = ""
            if self._timestamps_pd is not None and len(self._timestamps_pd) > idx_hist:
                t = self._timestamps_pd[idx_hist]
                try:
                    if self._freq_label == "daily":
                        time_str = f"{t.month}-{t.day}"
                    else:
                        time_str = t.strftime("%H:%M")
                except (ValueError, AttributeError):
                    time_str = ""
            elif self._time_labels and idx_hist < len(self._time_labels):
                time_str = self._time_labels[idx_hist]

            label_html = (
                f'<div style="background: rgba(20,20,28,0.90); padding: 3px 8px; '
                f'border-radius: 3px; font-size: 12px; white-space: nowrap;">'
                f'<span style="color: #aaa;">{time_str}</span>  '
                f'<span style="color: #00ff88; font-weight: bold;">¥{py:.2f}</span>'
                f'</div>'
            )
            self._hover_label.setHtml(label_html)
            self._hover_label.setPos(float(idx_hist) + 0.5, py)
            self._hover_label.setVisible(True)

            # Crosshair vertical line
            self._crosshair.setPos(float(idx_hist))
            self._crosshair.setVisible(True)
        except Exception:
            if self._crosshair is not None:
                self._crosshair.setVisible(False)
            if self._hover_label is not None:
                self._hover_label.setVisible(False)

    def _clear_prediction_items(self):
        """Remove only prediction overlay items, keep history intact."""
        for item in self._pred_items:
            self.price_plot.removeItem(item)
        self._pred_items = []

    def _draw_prediction(self, prediction, rolling_preds, close, n, future_bars):
        """Draw prediction path + arrowhead, centered-right in the prediction zone."""
        if n == 0 or len(close) == 0 or future_bars <= 0:
            return
        direction = prediction.get("direction", "flat")
        target = prediction.get("target_price", float(close[-1]) if len(close) > 0 else 0.0)

        price_range = close.max() - close.min()
        if price_range < close[-1] * 0.02:
            price_range = close[-1] * 0.02

        if direction == "up":
            color = "#e83030"; arrow_symbol = "▲"; label_text = "看涨"
        elif direction == "down":
            color = "#18b84a"; arrow_symbol = "▼"; label_text = "看跌"
        else:
            color = "#aaaaaa"; arrow_symbol = "—"; label_text = "横盘"

        pred_line_color = "#ffdd00"

        # ── Prediction path ──
        if rolling_preds and len(rolling_preds) >= 1:
            path_x = [n - 1]; path_y = [close[-1]]
            upper_y = [close[-1]]; lower_y = [close[-1]]
            conf_sum = 0.0; conf_count = 0
            step_mult = getattr(self, '_step_mult', 1)
            for rp in rolling_preds:
                step_x = n - 1 + rp.get("step", 1) * step_mult
                path_x.append(step_x)
                path_y.append(rp.get("target_price", target))
                upper_y.append(rp.get("price_upper", rp.get("target_price", target)))
                lower_y.append(rp.get("price_lower", rp.get("target_price", target)))
                cc = rp.get("direction_conf", 0.5)
                conf_sum += cc
                conf_count += 1
            avg_conf = conf_sum / max(conf_count, 1)

            # Center trajectory
            path_line = self.price_plot.plot(path_x, path_y, pen=pg.mkPen(pred_line_color, width=3))
            self._pred_items.append(path_line)

            # Confidence band via FillBetweenItem
            upper_curve = self.price_plot.plot(path_x, upper_y, pen=pg.mkPen(None))
            lower_curve = self.price_plot.plot(path_x, lower_y, pen=pg.mkPen(None))
            band_alpha = max(30, min(140, int(avg_conf * 160)))
            band_color = (
                (255, 50, 50, band_alpha) if direction == "up" else
                (80, 200, 80, band_alpha) if direction == "down" else
                (160, 160, 160, band_alpha)
            )
            fill = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(band_color))
            self.price_plot.addItem(fill)
            self._pred_items.append(fill)
            self._pred_items.append(upper_curve)
            self._pred_items.append(lower_curve)
        else:
            self.price_plot.plot(
                [n - 1, n + future_bars - 0.5], [close[-1], target],
                pen=pg.mkPen(pred_line_color, width=8)
            )
            arrow_line = self.price_plot.plot(
                [n - 1, n + future_bars - 0.5], [close[-1], target],
                pen=pg.mkPen(pred_line_color, width=3.5)
            )
            self._pred_items.append(arrow_line)
            start_dot = pg.ScatterPlotItem(
                [n - 1], [close[-1]], symbol="o", size=12,
                brush=pg.mkBrush("#ffffff"), pen=pg.mkPen(color, width=2.5)
            )
            self.price_plot.addItem(start_dot)
            self._pred_items.append(start_dot)

        # ── Arrowhead at end ──
        final_target = target
        if rolling_preds and len(rolling_preds) >= 1:
            final_target = rolling_preds[-1].get("target_price", target)
        arrow_marker = pg.TextItem(
            html=f'<span style="font-size: 22px; color: {color}; font-weight: bold;">{arrow_symbol}</span>',
            anchor=(0.5, 0.5)
        )
        arrow_marker.setPos(n + future_bars - 1, final_target)
        self.price_plot.addItem(arrow_marker)
        self._pred_items.append(arrow_marker)

        # ── Prediction label center-right in prediction zone ──
        label_x = n + future_bars * 0.55  # center-right of the 6-bar zone
        if direction == "up":
            label_y = min(close[-1], final_target) - price_range * 0.06
        elif direction == "down":
            label_y = max(close[-1], final_target) + price_range * 0.06
        else:
            label_y = close[-1]

        label_html = (
            f'<div style="background: rgba(20,20,28,0.90); '
            f'padding: 5px 10px; border: 1.5px solid {color}; border-radius: 6px; text-align: center;">'
            f'<span style="font-size: 13px; color: {color}; font-weight: bold;">{label_text}</span><br>'
            f'<span style="font-size: 15px; color: #ffffff; font-weight: bold;">¥{final_target:.2f}</span>'
            f'</div>'
        )
        badge = pg.TextItem(html=label_html, anchor=(0.5, 0.5))
        badge.setPos(label_x, label_y)
        self.price_plot.addItem(badge)
        self._pred_items.append(badge)
