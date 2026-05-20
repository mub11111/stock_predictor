import logging
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from copy import deepcopy
from data.market_rules import get_price_limit_pct, clip_price_delta

logger = logging.getLogger("stock_pred")
from data.rule_engine import validate_prediction
from model.online_learning import PredictionResidualMonitor, LightweightFineTuner

# Model registry — same as gui/dialogs.py MODEL_REGISTRY, kept in sync
from model.lstm_transformer import HybridModel  # kept for backward compat with old checkpoints
from model.ft_transformer import FT_iTransformerWrapper
from model.freq_orchestrator import FreqOrchestratorWrapper
from model.node_transformer import NodeTransformerWrapper, NodeTransformerV1Wrapper
from model.pinn_model import PINNWrapper
from model.ensemble import (
    EnsembleWeightedWrapper, EnsembleVotingWrapper,
    EnsembleStackingWrapper, EnsemblePINNGuardWrapper, AllModelsWrapper,
    LegacyEnsembleWeightedWrapper, LegacyEnsembleVotingWrapper,
    LegacyEnsembleStackingWrapper, LegacyEnsemblePINNGuardWrapper,
    LegacyAllModelsWrapper,
)

MODEL_CLASSES = {
    "Hybrid LSTM+Transformer": HybridModel,  # legacy checkpoints
    "Node Transformer (图注意力)": NodeTransformerWrapper,
    "Node Transformer (图注意力·准确率偏低)": NodeTransformerWrapper,
    "FT-iTransformer (时频协同)": FT_iTransformerWrapper,
    "Freq Orchestrator (频段协同)": FreqOrchestratorWrapper,
    "PINN (物理约束)": PINNWrapper,
    "集成-加权平均": EnsembleWeightedWrapper,
    "集成-多数投票": EnsembleVotingWrapper,
    "集成-堆叠法 (Stacking)": EnsembleStackingWrapper,
    "集成-PINN Guard": EnsemblePINNGuardWrapper,
    "所有模型": AllModelsWrapper,
    "所有模型对比": AllModelsWrapper,  # legacy name
}

# Legacy class mapping for old v1 checkpoints
LEGACY_CLASSES = {
    NodeTransformerWrapper: NodeTransformerV1Wrapper,
    EnsembleWeightedWrapper: LegacyEnsembleWeightedWrapper,
    EnsembleVotingWrapper: LegacyEnsembleVotingWrapper,
    EnsembleStackingWrapper: LegacyEnsembleStackingWrapper,
    EnsemblePINNGuardWrapper: LegacyEnsemblePINNGuardWrapper,
    AllModelsWrapper: LegacyAllModelsWrapper,
}


def _is_legacy_checkpoint(state_dict: dict) -> bool:
    """Detect old v1 checkpoints by checking for v1-specific keys."""
    # Direct NodeTransformer v1: has ts_proj (v1) vs input_ln (v2)
    if "model.ts_proj.weight" in state_dict:
        return True
    # Ensemble v1: base_models.0 uses single dir_head (v1) vs dir_heads (v2)
    if "base_models.0.model.dir_head.0.weight" in state_dict:
        return True
    if "base_models.0.model.ts_proj.weight" in state_dict:
        return True
    return False


def _is_hybrid_checkpoint(state_dict: dict) -> bool:
    """Detect HybridModel checkpoints (must have channel_mixer + stem.proj)."""
    return ("channel_mixer.net.0.weight" in state_dict
            and "stem.proj.weight" in state_dict
            and "transformer.layers.0.self_attn.in_proj_weight" in state_dict)


def _detect_hybrid_dims(state_dict: dict) -> dict | None:
    """Extract architecture dimensions from a HybridModel checkpoint's state_dict.

    Old checkpoints were trained with different seq_len / feature_dim / patch_len
    than the current config defaults. Build the model to match the checkpoint exactly.
    """
    try:
        cm_w = state_dict.get("channel_mixer.net.0.weight")  # (hidden, input_dim)
        stem_w = state_dict.get("stem.proj.weight")           # (d_model, patch_len * input_dim)
        pos_enc = state_dict.get("pos_encoding")              # (1, effective_len, d_model)
        if cm_w is None or stem_w is None or pos_enc is None:
            return None
        input_dim = cm_w.shape[1]
        d_model = stem_w.shape[0]
        patch_len = stem_w.shape[1] // input_dim
        effective_len = pos_enc.shape[1]
        max_seq_len = effective_len * patch_len
        return {
            "input_dim": input_dim,
            "d_model": d_model,
            "patch_len": patch_len,
            "max_seq_len": max_seq_len,
        }
    except Exception:
        return None


def _get_legacy_class(model_type: str, current_cls):
    """Return the legacy model class for old checkpoints."""
    return LEGACY_CLASSES.get(current_cls, current_cls)


class Predictor:
    """Load trained model, run inference, support online correction.

    Implements CRC-inspired (2025) safe online correction with:
      1. Direction gate — weakens correction when sign mismatches
      2. Quantile clipping — bounds correction magnitude by historical percentiles
      3. Validation gating — keeps update only if loss decreases on recent window
      4. Snapshot rollback — reverts parameters when validation fails
    """

    def __init__(self, checkpoint_path: str, config):
        self.config = config
        self.device = torch.device(config.model.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load selected features (per-stock adaptive dim reduction)
        self.selected_features = ckpt.get("selected_features", None)
        feature_dim = config.model.feature_dim

        # Determine model class from checkpoint (supports multiple architectures)
        model_type = ckpt.get("model_type", "Node Transformer (图注意力)")
        model_cls = MODEL_CLASSES.get(model_type, MODEL_CLASSES["Node Transformer (图注意力)"])

        # Auto-detect legacy/old checkpoints and swap to compat classes
        state_dict = ckpt.get("model_state_dict", {})
        if _is_hybrid_checkpoint(state_dict):
            model_cls = HybridModel
            # Auto-detect dimensions from checkpoint to handle old checkpoints
            # trained with different seq_len / feature_dim / patch_len
            ckpt_dims = _detect_hybrid_dims(state_dict)
            if ckpt_dims:
                input_dim = ckpt_dims["input_dim"]
                max_seq_len = ckpt_dims["max_seq_len"]
                patch_len = ckpt_dims["patch_len"]
                d_model = ckpt_dims.get("d_model", config.model.d_model)
                # If old checkpoint has fewer features than current config,
                # and no selected_features was saved, use first input_dim columns
                if self.selected_features is None and input_dim < feature_dim:
                    from data.preprocessor import FEATURE_COLS
                    self.selected_features = FEATURE_COLS[:input_dim]
            else:
                input_dim = len(self.selected_features) if self.selected_features else feature_dim
                max_seq_len = config.model.seq_len
                patch_len = config.model.patch_len
                d_model = config.model.d_model
        elif _is_legacy_checkpoint(state_dict):
            model_cls = _get_legacy_class(model_type, model_cls)
            input_dim = len(self.selected_features) if self.selected_features else feature_dim
            max_seq_len = config.model.seq_len
            patch_len = config.model.patch_len
            d_model = config.model.d_model
        else:
            input_dim = len(self.selected_features) if self.selected_features else feature_dim
            max_seq_len = config.model.seq_len
            patch_len = config.model.patch_len
            d_model = config.model.d_model

        self.model = model_cls(
            input_dim=input_dim,
            d_model=d_model,
            lstm_hidden=config.model.lstm_hidden,
            lstm_layers=config.model.lstm_layers,
            transformer_layers=config.model.transformer_layers,
            nhead=config.model.nhead,
            dropout=config.model.dropout,
            max_seq_len=max_seq_len,
            patch_len=patch_len,
            feature_group_indices=ckpt.get("feature_group_indices")
        )
        # strict=False: old checkpoints may lack newer keys
        # (price_scale buffer, adaptive input_proj layer, quantile heads, etc.)
        # The model's __init__ initializes defaults for any missing parameters.
        missing_keys, unexpected_keys = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing_keys:
            # Filter out buffer keys (price_scale defaults are fine)
            real_missing = [k for k in missing_keys if not k.endswith("price_scale")]
            if real_missing:
                logger.error("[non-fatal] Missing keys in checkpoint %s (will use defaults): %s",
                             checkpoint_path, real_missing)
        if unexpected_keys:
            logger.error("[non-fatal] Unexpected keys in checkpoint %s (newer ckpt, ignored): %s",
                         checkpoint_path, unexpected_keys)
        # Store the effective seq_len for _preprocess_seq padding/truncation
        self._effective_seq_len = max_seq_len
        # Restore all-models best model selection
        if hasattr(self.model, 'best_model_idx') and "best_model_idx" in ckpt:
            self.model.best_model_idx = ckpt["best_model_idx"]
        if hasattr(self.model, 'best_model_name') and "best_model_name" in ckpt:
            self.model.best_model_name = ckpt["best_model_name"]
        if hasattr(self.model, 'all_accuracies') and "all_accuracies" in ckpt:
            self.model.all_accuracies = ckpt["all_accuracies"]
        self.model.to(self.device)
        self.model.eval()
        self.model_version = ckpt.get("model_version", "v1")
        self.scaler = ckpt.get("scaler", None)

        # Online correction optimizer (tiny lr for gradual adaptation)
        self._corr_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-6, weight_decay=0
        )
        self._corr_count = 0
        self._total_correction = 0.0
        self._revert_count = 0  # times correction was rejected by validation
        self._error_buffer: list[float] = []  # for quantile clipping (CRC layer 2)

        # 在线学习: 预测残差监控 + 轻量微调
        self.residual_monitor = PredictionResidualMonitor(
            window_size=getattr(config.model, 'online_window', 60),
            ic_threshold=getattr(config.model, 'online_ic_threshold', 0.05),
            breach_threshold=getattr(config.model, 'online_breach_threshold', 0.35),
        )
        self._recent_features: list[np.ndarray] = []  # 缓存最近特征用于微调

    def predict_one(self, seq: np.ndarray, current_price: float,
                    ts_code: str | None = None, mc_samples: int = 0,
                    baseline_price: float | None = None) -> dict:
        """Run inference on one sample.

        Args:
            seq: (T, F) feature array (unscaled raw values)
            current_price: last known close price (used as fallback baseline)
            ts_code: stock code for price limit clipping
            mc_samples: Monte Carlo samples (0 = single forward pass)
            baseline_price: optional baseline anchor for residual prediction.
                            If None, current_price is used.
                            Pass MA20 or other benchmark for residual mode.
        """
        x = self._preprocess_seq(seq)

        if mc_samples > 0:
            return self._predict_mc(x, current_price, ts_code, mc_samples, baseline_price)

        with torch.no_grad():
            dir_logits, price_quantiles = self.model(x)  # (1,2), (1,3)
        probs = torch.softmax(dir_logits, dim=1).cpu().numpy()[0]
        directions = ["down", "up"]
        best_idx = int(np.argmax(probs))
        # price_quantiles: (1, 3) → q10, q50, q90 as percentage deltas
        q10 = price_quantiles[0, 0].item()
        q50 = price_quantiles[0, 1].item()
        q90 = price_quantiles[0, 2].item()
        delta = q50 / 100.0  # percentage → ratio (median)

        delta = clip_price_delta(delta, ts_code or "000001.SZ")

        base = baseline_price if baseline_price is not None else current_price
        target = base * (1 + delta)
        # Use quantile bounds (± price_scale% range around q50)
        q_lower = base * (1 + q10 / 100.0)
        q_upper = base * (1 + q90 / 100.0)
        limit_pct = get_price_limit_pct(ts_code or "000001.SZ")
        result = {
            "direction": directions[best_idx],
            "direction_conf": float(probs[best_idx]),
            "target_price": round(target, 2),
            "price_delta": delta,
            "price_lower": round(max(base * (1 - limit_pct), min(q_lower, target * 0.97)), 2),
            "price_upper": round(min(base * (1 + limit_pct), max(q_upper, target * 1.03)), 2),
            "q10_pct": round(q10 / 100.0, 6),
            "q50_pct": round(q50 / 100.0, 6),
            "q90_pct": round(q90 / 100.0, 6),
            "limit_up": round(current_price * (1 + limit_pct), 2),
            "limit_down": round(current_price * (1 - limit_pct), 2),
            "horizon_minutes": 1,
            "model_version": self.model_version,
            "created_at": datetime.now(),
            "price_std": None,
            "direction_conf_std": None,
            "mc_samples": 0,
        }
        result["rule_check"] = validate_prediction(result, ts_code or "000001.SZ")
        return result

    def _predict_mc(self, x: torch.Tensor, current_price: float,
                    ts_code: str | None, mc_samples: int,
                    baseline_price: float | None = None) -> dict:
        """Monte Carlo dropout: run N stochastic forward passes, return mean ± std."""
        self.model.train()  # keep dropout active
        dir_probs_all = []
        q10_all, q50_all, q90_all = [], [], []
        with torch.no_grad():
            for _ in range(mc_samples):
                dir_logits, price_quantiles = self.model(x)  # (1,3)
                probs = torch.softmax(dir_logits, dim=1).cpu().numpy()[0]
                dir_probs_all.append(probs)
                q10_all.append(price_quantiles[0, 0].item())
                q50_all.append(price_quantiles[0, 1].item())
                q90_all.append(price_quantiles[0, 2].item())
        self.model.eval()

        dir_probs = np.array(dir_probs_all)  # (mc_samples, 2)
        dir_mean = dir_probs.mean(axis=0)
        dir_std = dir_probs.std(axis=0)

        # Aggregate quantile predictions (MC mean for each quantile)
        q10_mean = float(np.mean(q10_all)) / 100.0
        q50_mean = float(np.mean(q50_all)) / 100.0
        q90_mean = float(np.mean(q90_all)) / 100.0
        q50_std = float(np.std(q50_all)) / 100.0

        directions = ["down", "up"]
        best_idx = int(np.argmax(dir_mean))
        delta = clip_price_delta(q50_mean, ts_code or "000001.SZ")

        base = baseline_price if baseline_price is not None else current_price
        target = base * (1 + delta)
        limit_pct = get_price_limit_pct(ts_code or "000001.SZ")
        # Combine quantile bounds + MC uncertainty
        ci_mult = 2.0
        lower_q = base * (1 + q10_mean)
        upper_q = base * (1 + q90_mean)
        lower_mc = target * (1 - ci_mult * q50_std)
        upper_mc = target * (1 + ci_mult * q50_std)
        result = {
            "direction": directions[best_idx],
            "direction_conf": float(dir_mean[best_idx]),
            "direction_conf_std": float(dir_std[best_idx]),
            "target_price": round(target, 2),
            "price_delta": delta,
            "price_std": round(q50_std, 6),
            "price_lower": round(max(base * (1 - limit_pct), min(lower_q, lower_mc)), 2),
            "price_upper": round(min(base * (1 + limit_pct), max(upper_q, upper_mc)), 2),
            "q10_pct": round(q10_mean, 6),
            "q50_pct": round(q50_mean, 6),
            "q90_pct": round(q90_mean, 6),
            "limit_up": round(current_price * (1 + limit_pct), 2),
            "limit_down": round(current_price * (1 - limit_pct), 2),
            "horizon_minutes": 1,
            "model_version": self.model_version,
            "created_at": datetime.now(),
            "mc_samples": mc_samples,
        }
        result["rule_check"] = validate_prediction(result, ts_code or "000001.SZ")
        return result

    def _apply_feature_mask(self, seq: np.ndarray) -> np.ndarray:
        """Apply per-stock feature selection mask if available."""
        if self.selected_features is not None:
            from data.feature_selector import FEATURE_POOL, get_feature_mask
            mask = get_feature_mask(self.selected_features)
            if seq.shape[1] == len(FEATURE_POOL):
                seq = seq[:, mask]
        return seq

    def _preprocess_seq(self, seq: np.ndarray) -> torch.Tensor:
        """Pad, scale, and convert a sequence to a batch-1 tensor."""
        seq = self._apply_feature_mask(seq)
        seq_len = getattr(self, '_effective_seq_len', self.config.model.seq_len)
        if seq.shape[0] < seq_len:
            pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        seq = seq[-seq_len:]
        if self.scaler is not None:
            from data.preprocessor import sanitize_scaler
            sanitize_scaler(self.scaler)
            seq = self.scaler.transform(seq)
        return torch.from_numpy(seq).float().unsqueeze(0).to(self.device)

    def _compute_validation_loss(self, val_seq: np.ndarray, val_price_delta: float) -> float:
        """Compute MSE on a single validation sample (price head only)."""
        x = self._preprocess_seq(val_seq)
        with torch.no_grad():
            _, price_delta = self.model(x)
        target = torch.tensor([[val_price_delta * 100.0]], dtype=torch.float32, device=self.device)
        return nn.functional.mse_loss(price_delta, target).item()

    def online_correct(self, seq: np.ndarray, predicted_price: float,
                       actual_price: float) -> dict:
        """Legacy blind correction — delegates to safe version without validation."""
        return self.online_correct_safe(seq, predicted_price, actual_price,
                                        val_seq=None, val_delta=None)

    def online_correct_safe(self, seq: np.ndarray, predicted_price: float,
                            actual_price: float,
                            val_seq: np.ndarray | None = None,
                            val_delta: float | None = None) -> dict:
        """
        CRC-inspired safe online correction with validation gating.

        Safety layers:
          1. Direction gate — weaken correction when predicted & actual signs differ
          2. Quantile clipping — bound error by historical 95th percentile
          3. Validation gating — keep update only if val loss decreases
          4. Snapshot rollback — revert all params when validation fails

        Args:
            seq: feature array for the bar BEFORE the actual price was observed
            predicted_price: the model's previous prediction
            actual_price: the real price that just arrived
            val_seq: optional validation sample (features → predicted_target)
            val_delta: the actual price_delta for the validation sample

        Returns:
            Correction result dict with correction metadata
        """
        error_pct = (actual_price - predicted_price) / (predicted_price + 1e-10)

        # ── Guard: negligible error ──
        if abs(error_pct) < 0.0001:
            return {"corrected": False, "reason": "negligible_error",
                    "error_pct": error_pct, "weight": 0}

        # ── CRC Layer 1: Direction Gate ──
        # If the model predicted "up" but price went "down" (or vice versa),
        # the error may be noise rather than systematic bias. Reduce weight.
        weight = min(abs(error_pct) * 50, 1.0)
        pred_sign = 1 if predicted_price > 0 else 0
        actual_sign = 1 if actual_price > 0 else 0
        sign_match = (pred_sign == actual_sign)
        if not sign_match:
            weight *= 0.3  # weaken cross-direction corrections

        # ── CRC Layer 2: Quantile Clipping ──
        # Track error distribution; clip extreme outliers to 95th percentile
        self._error_buffer.append(abs(error_pct))
        if len(self._error_buffer) > 100:
            self._error_buffer = self._error_buffer[-100:]
        if len(self._error_buffer) >= 10:
            q95 = np.percentile(self._error_buffer, 95)
            if abs(error_pct) > q95 * 1.5:
                error_pct = np.sign(error_pct) * q95 * 1.5
                weight *= 0.7  # down-weight clipped corrections

        # ── Snapshot for rollback (CRC Layer 4) ──
        snapshot = {k: v.clone() for k, v in self.model.state_dict().items()}

        # ── Compute pre-correction validation loss ──
        pre_loss = None
        if val_seq is not None and val_delta is not None:
            pre_loss = self._compute_validation_loss(val_seq, val_delta)

        # ── Prepare input ──
        x = self._preprocess_seq(seq)
        actual_delta = (actual_price - predicted_price) / (predicted_price + 1e-10) * 100.0  # percentage
        target_delta = torch.tensor([[actual_delta]], dtype=torch.float32, device=self.device)

        # ── Apply correction gradient ──
        self.model.train()
        dir_logits, price_delta = self.model(x)
        price_loss = nn.functional.mse_loss(price_delta, target_delta)
        scaled_loss = price_loss * weight
        self._corr_optimizer.zero_grad()
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self._corr_optimizer.step()
        self.model.eval()

        # ── CRC Layer 3+4: Validation Gating & Rollback ──
        if pre_loss is not None:
            post_loss = self._compute_validation_loss(val_seq, val_delta)

            if post_loss > pre_loss:
                # Correction made things worse — revert to snapshot
                self.model.load_state_dict(snapshot)
                self._revert_count += 1
                return {
                    "corrected": False,
                    "reason": "validation_loss_increased",
                    "error_pct": error_pct,
                    "weight": weight,
                    "pre_loss": round(pre_loss, 6),
                    "post_loss": round(post_loss, 6),
                    "sign_match": sign_match,
                    "correction_count": self._corr_count,
                    "revert_count": self._revert_count,
                }

            improvement = pre_loss - post_loss
        else:
            improvement = None

        self._corr_count += 1
        self._total_correction += abs(error_pct)

        return {
            "corrected": True,
            "error_pct": error_pct,
            "weight": weight,
            "pre_loss": round(pre_loss, 6) if pre_loss is not None else None,
            "post_loss": round(post_loss, 6) if pre_loss is not None else None,
            "improvement": round(improvement, 6) if improvement is not None else None,
            "sign_match": sign_match,
            "correction_count": self._corr_count,
            "revert_count": self._revert_count,
            "avg_correction": self._total_correction / self._corr_count,
        }

    def record_prediction_result(self, q10_pct: float, q50_pct: float,
                                   q90_pct: float, actual_delta: float,
                                   base_price: float, actual_price: float,
                                   raw_seq: np.ndarray | None = None):
        """记录预测-实际对到残差监控器.

        在每次预测后, 当实际价格已知时调用.
        """
        self.residual_monitor.record(q10_pct, q50_pct, q90_pct,
                                     actual_delta, base_price, actual_price)
        if raw_seq is not None:
            self._recent_features.append(raw_seq)
            # 保持窗口大小
            max_buf = self.residual_monitor.window_size * 2
            if len(self._recent_features) > max_buf:
                self._recent_features = self._recent_features[-max_buf:]

    def check_and_finetune(self, branch: str = "last_layer") -> dict | None:
        """检查是否需要在线微调, 必要时执行.

        Args:
            branch: "last_layer" | "high_freq" | "all"

        Returns:
            微调结果 dict 或 None
        """
        if not self.residual_monitor.check_trigger():
            return None

        preds, actuals = self.residual_monitor.get_recent_deltas(30)
        n = min(len(self._recent_features), len(preds))
        if n < 5:
            return None

        # 预处理最近的特征
        features = np.stack([
            self._preprocess_seq(s).squeeze(0).cpu().numpy()
            for s in self._recent_features[-n:]
        ], axis=0)  # (n, S, C)

        tuner = LightweightFineTuner(self.model, self.device)
        result = tuner.finetune(features, preds[-n:], actuals[-n:], branch=branch)
        return result

    def get_monitor_summary(self) -> dict:
        """获取残差监控状态摘要."""
        return self.residual_monitor.summary()


def predict_batch(predictor: Predictor, stock_sequences: dict[str, tuple[np.ndarray, float]]) -> list[dict]:
    results = []
    for ts_code, (seq, price) in stock_sequences.items():
        pred = predictor.predict_one(seq, price)
        pred["ts_code"] = ts_code
        results.append(pred)
    return results
