import sys
import os
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer
import qdarkstyle
from config import AppConfig
from gui.main_window import MainWindow
from gui.workers import DataFetchWorker


class StockPredictorApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        # Pass empty list — PyCharm console injects debug args into sys.argv
        self.app = QApplication.instance() or QApplication([])
        self.app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6"))

    def run(self):
        try:
            self.window = MainWindow(self.cfg)
            self.window.show()
            ret = self.app.exec()
            if not self._is_interactive():
                sys.exit(ret)
        except Exception as e:
            # Show error in a dialog so it's visible in PyCharm console mode
            QMessageBox.critical(None, "启动错误",
                f"程序启动失败:\n\n{traceback.format_exc()}")
            raise

    @staticmethod
    def _is_interactive() -> bool:
        """Detect if running inside PyCharm/IPython console."""
        try:
            __IPYTHON__
            return True
        except NameError:
            pass
        if os.environ.get("PYCHARM_HOSTED") or os.environ.get("PYDEV_CONSOLE_ENCODING"):
            return True
        return hasattr(sys, "ps1") or sys.flags.interactive
