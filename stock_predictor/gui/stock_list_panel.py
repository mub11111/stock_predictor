from __future__ import annotations
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMenu, QLabel, QPushButton
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QColor, QAction
from storage.repository import Repository


class StockListPanel(QWidget):
    stock_selected = pyqtSignal(str)
    batch_train_requested = pyqtSignal(list)  # list[str] — selected stock codes
    batch_predict_requested = pyqtSignal(list)  # list[str] — predict selected stocks
    cloud_search_requested = pyqtSignal(str)  # str — keyword to search on TDX cloud
    view_prediction_requested = pyqtSignal(str)  # str — ts_code to view prediction

    CLOUD_ROW_MARKER = "__cloud_search__"

    def __init__(self, repo: Repository):
        super().__init__()
        self.repo = repo
        self._predicted_stocks: set[str] = set()
        self._trained_stocks: set[str] = set()
        self._cached_df = None
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._do_filter)
        self._first_load = True
        self._cloud_keyword: str = ""  # current search keyword for cloud placeholder
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(3)

        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索代码或名称...")
        self.search.setStyleSheet("font-size: 12px; padding: 4px 8px;")
        self.search.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["全部", "筛选股票", "按行业"])
        self.filter_combo.setStyleSheet("font-size: 12px; padding: 2px;")
        self.filter_combo.currentTextChanged.connect(self._on_search_changed)
        layout.addWidget(self.filter_combo)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["代码", "名称", "最新价", "操作"])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultSectionSize(80)
        header.resizeSection(0, 70)
        header.resizeSection(1, 90)
        header.resizeSection(3, 50)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet("font-size: 12px; gridline-color: #333;")
        self.table.cellClicked.connect(self._on_click)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.table)

        # Load initial data immediately (not debounced)
        self._do_filter()

    def set_predicted_stocks(self, stocks: set[str]):
        """Update which stocks have predictions in memory (re-colors rows, adds buttons)."""
        self._predicted_stocks = stocks
        if self._cached_df is not None and not self._cached_df.empty:
            text = self.search.text().strip().upper()
            self._do_filter()
        else:
            self._apply_colors()

    def set_trained_stocks(self, stocks: set[str]):
        """Update which stocks have trained models on disk. Re-renders to sort trained-first."""
        self._trained_stocks = stocks
        # Re-run filter to re-sort (trained stocks on top)
        text = self.search.text().strip().upper()
        self._do_filter()

    def get_selected_stocks(self) -> list[str]:
        """Return raw codes of all selected rows."""
        codes = []
        for item in self.table.selectedItems():
            if item.column() == 0:
                codes.append(self._raw_code(item.text()))
        return list(dict.fromkeys(codes))  # dedup preserving order

    def _apply_colors(self):
        """Update row colors/markers for predicted and trained stocks."""
        for i in range(self.table.rowCount()):
            code_item = self.table.item(i, 0)
            if code_item is None:
                continue
            code = self._raw_code(code_item.text())
            name_item = self.table.item(i, 1)
            trained = code in self._trained_stocks
            predicted = code in self._predicted_stocks

            # Code: ★ prefix for trained, gold color
            display_code = f"★ {code}" if trained else code
            code_item.setText(display_code)
            if trained:
                code_item.setForeground(QColor("#ffd700"))
            elif predicted:
                code_item.setForeground(QColor("#ff6b35"))
            else:
                code_item.setForeground(QColor("#ffffff"))

            # Name: gold tint for trained
            if name_item:
                if trained:
                    name_item.setForeground(QColor("#ffd700"))
                elif predicted:
                    name_item.setForeground(QColor("#ffaa66"))
                else:
                    name_item.setForeground(QColor("#cccccc"))

    def refresh(self, repo: Repository):
        self.repo = repo
        self._cached_df = None  # invalidate cache
        self._first_load = True  # allow auto-select on refresh
        self._do_filter()  # immediate, not debounced

    def _on_search_changed(self):
        """Debounced handler — schedule a filter rather than running immediately."""
        self._schedule_filter()

    def _schedule_filter(self):
        self._debounce_timer.start()

    def _do_filter(self):
        """Perform the actual filtering — called after debounce delay."""
        text = self.search.text().strip().upper()

        if self._cached_df is None:
            self._cached_df = self.repo.get_all_stocks()
        df = self._cached_df

        if df is None or df.empty:
            self.table.setRowCount(0)
            self._cloud_keyword = text
            self._show_cloud_row(text)
            return

        if text:
            mask = df["ts_code"].str.contains(text) | df["name"].str.contains(text)
            filtered = df[mask]
        else:
            filtered = df

        # Sort: trained stocks (★) on top
        trained_mask = filtered["ts_code"].isin(self._trained_stocks)
        filtered = pd.concat([filtered[trained_mask], filtered[~trained_mask]], ignore_index=True)

        n = len(filtered)
        self._cloud_keyword = ""

        if n == 0 and text:
            # No local results → show cloud-search placeholder row
            self._show_cloud_row(text)
            return

        self._render_stock_rows(filtered)

        # Auto-select first row on initial load
        if self._first_load and self.table.rowCount() > 0:
            self._first_load = False
            self.table.selectRow(0)

    def _show_cloud_row(self, keyword: str):
        """Display a single placeholder row offering cloud search."""
        self._cloud_keyword = keyword
        self.table.setRowCount(1)
        self.table.blockSignals(True)
        code_item = QTableWidgetItem(keyword)
        code_item.setForeground(QColor("#888888"))
        name_item = QTableWidgetItem("🔍 点击从云端搜索...")
        name_item.setForeground(QColor("#5599ff"))
        price_item = QTableWidgetItem("")
        empty_item = QTableWidgetItem("")
        self.table.setItem(0, 0, code_item)
        self.table.setItem(0, 1, name_item)
        self.table.setItem(0, 2, price_item)
        self.table.setItem(0, 3, empty_item)
        self.table.blockSignals(False)

    def _render_stock_rows(self, filtered):
        """Populate table with stock rows from dataframe."""
        self.table.setRowCount(len(filtered))
        self.table.blockSignals(True)
        for i, (_, row) in enumerate(filtered.iterrows()):
            code = row["ts_code"]
            trained = code in self._trained_stocks
            predicted = code in self._predicted_stocks
            display_code = f"★ {code}" if trained else code
            code_item = QTableWidgetItem(display_code)
            name_item = QTableWidgetItem(row.get("name", ""))
            price_item = QTableWidgetItem("")
            btn_item = QTableWidgetItem("")

            if trained:
                code_item.setForeground(QColor("#ffd700"))
                name_item.setForeground(QColor("#ffd700"))
            elif predicted:
                code_item.setForeground(QColor("#ff6b35"))
                name_item.setForeground(QColor("#ffaa66"))

            self.table.setItem(i, 0, code_item)
            self.table.setItem(i, 1, name_item)
            self.table.setItem(i, 2, price_item)

            # "查看" button for stocks with predictions
            if predicted or trained:
                btn = QPushButton("查看")
                btn.setFixedSize(42, 22)
                btn.setStyleSheet(
                    "font-size: 11px; padding: 1px 4px; color: #ffd700; "
                    "background: #1a1a2e; border: 1px solid #555; border-radius: 3px;"
                )
                code_ref = code  # capture for closure
                btn.clicked.connect(lambda checked, c=code_ref: self.view_prediction_requested.emit(c))
                self.table.setCellWidget(i, 3, btn)
            else:
                # Remove any stale button widget from previous render
                self.table.removeCellWidget(i, 3)
                self.table.setItem(i, 3, btn_item)
        self.table.blockSignals(False)

    def select_stock(self, ts_code: str):
        """Select and scroll to a specific stock in the table."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                code = self._raw_code(item.text())
                if code == ts_code:
                    self.table.selectRow(row)
                    self.table.scrollToItem(item)
                    break

        # Only auto-select on first load, not on every filter keystroke
        if self._first_load and self.table.rowCount() > 0:
            self._first_load = False
            self.table.selectRow(0)

    @staticmethod
    def _raw_code(display: str) -> str:
        return display.removeprefix("★ ")

    def _on_click(self, row: int, _col: int):
        item = self.table.item(row, 0)
        if not item:
            return
        code_text = item.text()
        # Cloud-search placeholder row — trigger cloud lookup
        if self._cloud_keyword and code_text == self._cloud_keyword:
            self.cloud_search_requested.emit(self._cloud_keyword)
            return
        # Normal stock row
        self.stock_selected.emit(self._raw_code(code_text))

    def _on_context_menu(self, pos):
        selected = self.get_selected_stocks()
        if not selected:
            return
        menu = QMenu(self)
        train_action = QAction(f"批量训练 ({len(selected)} 只)", self)
        train_action.triggered.connect(lambda: self.batch_train_requested.emit(selected))
        menu.addAction(train_action)
        predict_action = QAction(f"预测选中 ({len(selected)} 只)", self)
        predict_action.triggered.connect(lambda: self.batch_predict_requested.emit(selected))
        menu.addAction(predict_action)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def current_stock(self) -> str | None:
        rows = self.table.selectedItems()
        if not rows:
            return None
        return self._raw_code(self.table.item(rows[0].row(), 0).text())
