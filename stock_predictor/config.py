from dataclasses import dataclass, field
from pathlib import Path
import os
import sys

def _app_dir() -> Path:
    """Directory containing the executable (frozen) or this script.
    All runtime data (key.txt, pretrained/, stock_data_v2.db, logs/)
    lives here so the app is fully self-contained."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


ROOT = _app_dir()


def _read_deepseek_key():
    """Read DeepSeek API key from key.txt or env var."""
    deepseek = os.getenv("DEEPSEEK_API_KEY", "")
    candidates = [_app_dir() / "key.txt", Path("D:/AI/key.txt")]
    for kf in candidates:
        if kf.exists():
            lines = [l.strip() for l in kf.read_text(errors="ignore").splitlines() if l.strip()]
            for line in lines:
                import re
                match = re.search(r'sk-[a-zA-Z0-9]+', line)
                if match and not deepseek:
                    deepseek = match.group(0)
    return deepseek


_deepseek_key = _read_deepseek_key()


@dataclass
class DataConfig:
    minute_retention_days: int = 90
    daily_retention_years: int = 3
    screened_count: int = 50
    db_path: str = field(default_factory=lambda: str(ROOT / "stock_data_v2.db"))
    # 金太阳 / TDX 实时行情服务器
    gs_enabled: bool = True
    gs_host: str = "127.0.0.1"
    gs_port: int = 7709
    gs_use_public: bool = True  # 本地不可用时自动使用公共 TDX 服务器


@dataclass
class ModelConfig:
    d_model: int = 128
    lstm_hidden: int = 128
    lstm_layers: int = 2
    transformer_layers: int = 3
    nhead: int = 8
    dropout: float = 0.15
    seq_len: int = 120
    patch_len: int = 5          # PatchTST: group bars into patches (240→48 tokens)
    feature_dim: int = 88
    batch_size: int = 64
    lr: float = 5e-4
    epochs: int = 100
    early_stop_patience: int = 20
    min_confidence: float = 0.75  # 预测置信度最低阈值，低于此值不采纳
    lambda_quant: float = 0.1       # PINN 量化边界惩罚权重
    use_pinn_loss: bool = True      # 是否启用 PINN 边界惩罚
    device: str = "cpu"
    checkpoint_dir: str = field(default_factory=lambda: str(ROOT / "pretrained"))


@dataclass
class GUIConfig:
    refresh_interval_sec: int = 300
    theme: str = "dark"


@dataclass
class MCPConfig:
    deepseek_api_key: str = _deepseek_key
    deepseek_model: str = "deepseek-chat"
    deepseek_max_tokens: int = 500


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)


def load_config() -> AppConfig:
    import torch
    cfg = AppConfig()
    if torch.cuda.is_available():
        cfg.model.device = "cuda"
        torch.backends.cudnn.benchmark = True
        print(f"[设备] GPU 加速已启用: {torch.cuda.get_device_name(0)}")
        print(f"[设备] 显存: {torch.cuda.get_device_properties(0).total_mem // 1024**2} MB")
    else:
        cfg.model.device = "cpu"
        # Limit CPU threads to avoid system freeze
        cpu_count = torch.get_num_threads()
        torch.set_num_threads(min(cpu_count, 8))
        print(f"[设备] 未检测到 GPU，使用 CPU (线程数: {min(cpu_count, 8)})")
    return cfg
