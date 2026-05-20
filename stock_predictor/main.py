"""A股智能预测系统 — 入口文件"""
import sys
import warnings
import logging

# ── Suppress logging noise from third-party libs ──
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logging.getLogger("stock_pred").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# ── Suppress known harmless warnings ──
# PyTorch: nested tensor disabled (Pre-LN with norm_first=True)
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")
# PyTorch: torch.load with weights_only=False is intentional for our checkpoints
warnings.filterwarnings("ignore", message=".*weights_only.*")
# PyTorch: checkpoint loading with missing/unexpected keys
warnings.filterwarnings("ignore", message=".*Missing key.*")
warnings.filterwarnings("ignore", message=".*Unexpected key.*")
# PyTorch: internal deprecation noise
warnings.filterwarnings("ignore", message=".*_pytree.*")
warnings.filterwarnings("ignore", message=".*non-full-backward.*")
# NumPy: edge cases in feature computation already guarded
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=".*divide by zero.*")
# NumPy: dtype rename deprecations (np.bool→bool, etc.)
warnings.filterwarnings("ignore", message=".*was renamed.*")
# pandas / sklearn / xgboost future compat
warnings.filterwarnings("ignore", category=FutureWarning)
# sklearn: feature names warning (we use raw numpy arrays)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*feature names.*")
# Deprecation: third-party libs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
# PyTorch: harmless serialization / storage notes
warnings.filterwarnings("ignore", message=".*TypedStorage.*")
# pytz / pandas timezone transition warnings
warnings.filterwarnings("ignore", message=".*pytz.*")

from config import load_config, AppConfig
from storage.database import init_database
from gui.app import StockPredictorApp


def main():
    cfg = load_config()
    init_database(cfg)
    app = StockPredictorApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
