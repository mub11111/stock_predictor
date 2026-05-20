# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Stock Predictor (A股智能预测系统) — onedir mode.

Build:  pyinstaller --distpath D:/AI --workpath D:/AI/build_temp stock_predictor.spec
Output: D:/AI/StockPredictor/
        ├── StockPredictor.exe   (small launcher, ~5 MB)
        ├── _internal/           (all deps: PyTorch, PyQt6, etc.)
        ├── pretrained/          (copy after build)
        ├── stock_data_v2.db     (copy after build)
        ├── key.txt              (user provides API keys)
        └── logs/                (runtime logs)
"""
from pathlib import Path

PROJECT_ROOT = Path("D:/AI/stock_predictor")

# ── Hidden imports: project modules ─────────────────────
_project_pkgs = []
for pkg in ["data", "gui", "model", "storage", "mcp"]:
    pkg_dir = PROJECT_ROOT / pkg
    if pkg_dir.exists():
        for py_file in pkg_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                _project_pkgs.append(pkg)
            elif not py_file.name.startswith("_"):
                _project_pkgs.append(f"{pkg}.{py_file.stem}")

# ── Hidden imports: third-party modules ─────────────────
_third_party = [
    "torch", "torch.nn", "torch.optim", "torch.utils", "torch.utils.data",
    "torch.nn.functional", "torch.nn.init", "torch.serialization",
    "torch.backends", "torch.backends.mps",
    "PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets",
    "PyQt6.sip",
    "pyqtgraph", "pyqtgraph.graphicsItems",
    "qdarkstyle",
    "sklearn", "sklearn.preprocessing", "sklearn.metrics",
    "sklearn.utils", "sklearn.model_selection",
    "sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext",
    "openai",
    "efinance",
    "numpy", "numpy.core", "numpy.random",
    "pandas", "pandas.core",
]

hiddenimports = _project_pkgs + _third_party

# ── Excludes ────────────────────────────────────────────
excludes = [
    "cublas", "cudnn", "cufft", "cusolver", "cusparse", "cuda_runtime",
    "nvidia", "nvidia.*",
    "PyQt6.QtWebEngine", "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtMultimedia", "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtBluetooth", "PyQt6.QtNfc", "PyQt6.QtSensors",
    "PyQt6.QtSerialPort", "PyQt6.QtLocation", "PyQt6.QtHelp",
    "PyQt6.QtTextToSpeech", "PyQt6.QtXml", "PyQt6.QtSql",
    "PyQt6.QtTest", "PyQt6.QtDesigner", "PyQt6.QtDBus",
    "matplotlib",
    "PIL", "Pillow",
    "tkinter", "_tkinter",
    "IPython", "jupyter", "notebook",
    "pytest",
]

# ── Analysis ────────────────────────────────────────────
a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        ("D:/python/anaconda/envs/stock_pred/Lib/site-packages/akshare/file_fold", "akshare/file_fold"),
        ("D:/AI/stock_predictor/qt.conf", "."),
        ("D:/python/anaconda/envs/stock_pred/Lib/site-packages/efinance/data", "efinance/data"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ── Filter CUDA binaries & problematic conda DLLs ──────
_filter_patterns = [
    # CUDA (not needed for CPU inference)
    "cublas", "cudnn", "cufft", "cusolver", "cusparse",
    "cuda_", "nvrtc", "nvml", "nccl", "nvfuser", "cublaslt",
    # Conda-env DLLs that conflict with Qt6/System32
    "icuuc", "icudt",  # ICU Unicode & Data — must use System32 version for Qt6 compat
]
a.binaries = [
    (name, path, typ)
    for name, path, typ in a.binaries
    if not any(p in name.lower() for p in _filter_patterns)
]

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ── EXE: small launcher only (onedir) ───────────────────
# Do NOT pass a.binaries/a.zipfiles/a.datas — those go in COLLECT
exe = EXE(
    pyz,
    a.scripts,
    [],  # binaries → COLLECT
    [],  # zipfiles → COLLECT
    [],  # datas    → COLLECT
    [],
    name="StockPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir="D:/AI/tmp",
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# ── COLLECT: all deps alongside the launcher ────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="StockPredictorApp",
)
