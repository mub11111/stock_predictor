"""Model registry: lightweight JSON index of trained models — no torch.load at startup."""
import json
import re
import os
from dataclasses import dataclass, field
from pathlib import Path

# Windows-illegal filename characters: < > : " / \ | ? *
_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def sanitize_filename(name: str) -> str:
    """Replace characters that are illegal in Windows filenames with '_'."""
    return _ILLEGAL_FILENAME_CHARS.sub('_', name)


@dataclass
class ModelEntry:
    """Metadata for one trained model checkpoint."""
    ts_code: str
    model_type: str
    file_path: str
    val_acc: float = 0.0
    macro_acc: float = 0.0
    ic: float = 0.0
    rank_ic: float = 0.0
    selected_features: list[str] | None = None
    best_model_name: str = ""
    all_accuracies: dict = field(default_factory=dict)


# ── Index file helpers ──

def _index_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / "model_index.json"


def update_model_index(checkpoint_dir: str, ts_code: str, model_type: str,
                        val_acc: float = 0.0, macro_acc: float = 0.0,
                        ic: float = 0.0, rank_ic: float = 0.0,
                        selected_features: list[str] | None = None,
                        best_model_name: str = "",
                        all_accuracies: dict | None = None):
    """Called after training saves a checkpoint — updates the lightweight index."""
    idx_path = _index_path(checkpoint_dir)
    records: dict[str, dict] = {}
    if idx_path.exists():
        try:
            records = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            records = {}

    records[ts_code] = {
        "model_type": model_type,
        "val_acc": val_acc,
        "macro_acc": macro_acc,
        "ic": round(ic, 4),
        "rank_ic": round(rank_ic, 4),
        "selected_features": selected_features,
        "best_model_name": best_model_name,
        "all_accuracies": all_accuracies or {},
    }
    idx_path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                        encoding="utf-8")


class ModelRegistry:
    """Reads lightweight model_index.json — no pytorch dependency at startup."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._entries: dict[str, list[ModelEntry]] = {}
        self._all_entries: list[ModelEntry] = []
        self.refresh()

    def refresh(self):
        """Re-read model_index.json (sub-millisecond, no GPU/CPU loading)."""
        self._entries.clear()
        self._all_entries.clear()
        idx_path = _index_path(str(self.checkpoint_dir))
        if not idx_path.exists():
            return

        try:
            records = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            return

        for ts_code, meta in records.items():
            # Validate ts_code
            if not re.match(r'^\d{6}\.(SZ|SH)$', ts_code) and ts_code not in ("default", "test_diag"):
                continue
            file_path = str(self.checkpoint_dir / f"{sanitize_filename(ts_code)}_best_model.pt")
            if not os.path.exists(file_path):
                continue
            entry = ModelEntry(
                ts_code=ts_code,
                model_type=meta.get("model_type", "未知"),
                file_path=file_path,
                val_acc=round(meta.get("val_acc", 0.0), 4),
                macro_acc=round(meta.get("macro_acc", 0.0), 4),
                ic=round(meta.get("ic", 0.0), 4),
                rank_ic=round(meta.get("rank_ic", 0.0), 4),
                selected_features=meta.get("selected_features"),
                best_model_name=meta.get("best_model_name", ""),
                all_accuracies=meta.get("all_accuracies", {}),
            )
            self._all_entries.append(entry)
            self._entries.setdefault(ts_code, []).append(entry)

    def get_models_for_stock(self, ts_code: str) -> list[ModelEntry]:
        entries = self._entries.get(ts_code, [])
        return [e for e in entries if e.model_type != "未知"]

    def has_models(self, ts_code: str) -> bool:
        return any(e.model_type != "未知" for e in self._entries.get(ts_code, []))

    def get_best_model(self, ts_code: str) -> ModelEntry | None:
        entries = self.get_models_for_stock(ts_code)
        if not entries:
            return None
        return max(entries, key=lambda e: e.macro_acc)

    def list_trained_stocks(self) -> list[str]:
        return [ts for ts in self._entries if self.has_models(ts)]

    def delete_models_for_stock(self, ts_code: str) -> int:
        """Delete all checkpoints for a stock. Returns count of files removed."""
        count = 0
        for pattern in [f"{sanitize_filename(ts_code)}_best_model.pt", f"{sanitize_filename(ts_code)}_features.json"]:
            fpath = self.checkpoint_dir / pattern
            if fpath.exists():
                fpath.unlink()
                count += 1
        self._remove_index_entry(ts_code)
        self.refresh()
        return count

    def clear_all_models(self) -> int:
        """Delete all model checkpoints and features. Returns count of stocks cleared."""
        count = 0
        for ts_code in list(self._entries.keys()):
            for pattern in [f"{sanitize_filename(ts_code)}_best_model.pt", f"{sanitize_filename(ts_code)}_features.json"]:
                fpath = self.checkpoint_dir / pattern
                if fpath.exists():
                    fpath.unlink()
                    count += 1
        idx_path = _index_path(str(self.checkpoint_dir))
        if idx_path.exists():
            idx_path.unlink()
        self.refresh()
        return count

    def _remove_index_entry(self, ts_code: str):
        idx_path = _index_path(str(self.checkpoint_dir))
        if not idx_path.exists():
            return
        try:
            records = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if ts_code in records:
            del records[ts_code]
            idx_path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    def get_features_for_stock(self, ts_code: str) -> list[str] | None:
        fpath = self.checkpoint_dir / f"{sanitize_filename(ts_code)}_features.json"
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                return data.get("selected_features", None)
            except Exception:
                pass
        best = self.get_best_model(ts_code)
        if best and best.selected_features:
            return best.selected_features
        return None
