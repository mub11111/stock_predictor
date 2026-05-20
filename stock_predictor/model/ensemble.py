"""Model Ensemble: 4 integration methods for combining multiple base models.

Methods (from simple to complex):
  1. Dynamic Weighted Averaging — adjust weights based on recent MSE
  2. Directional Consensus Voting — vote for direction signals
  3. Stacking (Meta-Learner) — train a meta-model on base predictions
  4. PINN-Guard Logic — use physics constraints to gate predictions

Base models (2): FT-iTransformer, PINN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


# ═══════════════════════════════════════════════════════════════
# Method 1: Dynamic Weighted Averaging
# ═══════════════════════════════════════════════════════════════

class DynamicWeightedEnsemble:
    """Weight predictions by recent performance — better models get higher weight.

    Weights updated by: w_i = (1/error_i) / sum(1/error_j)
    """

    def __init__(self, models: list, window: int = 20):
        self.models = models
        self.n_models = len(models)
        self.weights = [1.0 / self.n_models] * self.n_models
        self.window = window
        self.error_history = [[] for _ in range(self.n_models)]

    def update_weights(self, validation_data):
        """Recompute weights based on validation errors.

        validation_data: (x, y_dir, y_price) tensors
        """
        x, y_dir, y_price = validation_data
        errors = []

        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                dir_out, price_out = model(x)
                # Direction accuracy-based weighting (higher accuracy = lower error)
                dir_acc = (dir_out.argmax(1) == y_dir).float().mean().item()
                dir_error = 1.0 - dir_acc

                # Price MSE
                price_mse = ((price_out.squeeze(-1) - y_price) ** 2).mean().item()

                # Combined error
                combined_error = dir_error * 0.6 + min(price_mse, 10.0) / 10.0 * 0.4
                errors.append(combined_error)

            # Track history
            self.error_history[i].append(combined_error)
            if len(self.error_history[i]) > self.window:
                self.error_history[i] = self.error_history[i][-self.window:]

        # Use smoothed errors
        smoothed_errors = []
        for i in range(self.n_models):
            if self.error_history[i]:
                smoothed = np.mean(self.error_history[i])
            else:
                smoothed = errors[i]
            smoothed_errors.append(smoothed)

        # Convert to weights: higher weight for lower error
        inv_errors = [1.0 / (e + 1e-6) for e in smoothed_errors]
        total = sum(inv_errors)
        self.weights = [e / total for e in inv_errors]

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted average of all model predictions."""
        dir_preds = []
        price_preds = []

        for model, w in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                dir_out, price_out = model(x)
                dir_preds.append(F.softmax(dir_out, dim=-1) * w)
                price_preds.append(price_out * w)

        dir_ensemble = torch.stack(dir_preds).sum(dim=0)
        price_ensemble = torch.stack(price_preds).sum(dim=0)
        return dir_ensemble, price_ensemble


# ═══════════════════════════════════════════════════════════════
# Method 2: Directional Consensus Voting
# ═══════════════════════════════════════════════════════════════

class VotingEnsemble:
    """Vote on direction — only act when models reach consensus."""

    def __init__(self, models: list, consensus_threshold: int = 3):
        self.models = models
        self.n_models = len(models)
        self.consensus_threshold = consensus_threshold

    def predict_signal(self, x: torch.Tensor) -> dict:
        """Get directional consensus signal from all models.

        Returns:
          signal: "Strong Buy" | "Strong Sell" | "Buy" | "Sell" | "Wait / Neutral"
          confidence: 0-1 consensus strength
          direction: 1 (up), -1 (down), 0 (neutral)
        """
        signals = []
        confidences = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                dir_out, _ = model(x)
                probs = F.softmax(dir_out, dim=-1)
                pred_class = dir_out.argmax(1)
                confidence = probs.max(dim=-1).values
                signals.append(pred_class.cpu().numpy())
                confidences.append(confidence.cpu().numpy())

        signals = np.array(signals)  # [n_models, batch]
        confidences = np.array(confidences)

        B = signals.shape[1]
        results = []

        for b in range(B):
            sample_signals = signals[:, b]
            sample_confs = confidences[:, b]

            count_up = (sample_signals == 1).sum()
            count_down = (sample_signals == 0).sum()
            avg_conf = sample_confs.mean()

            if count_up >= self.consensus_threshold:
                signal = "Strong Buy"
                direction = 1
            elif count_down >= self.consensus_threshold:
                signal = "Strong Sell"
                direction = -1
            elif count_up > count_down:
                signal = "Buy"
                direction = 1
            elif count_down > count_up:
                signal = "Sell"
                direction = -1
            else:
                signal = "Wait / Neutral"
                direction = 0

            results.append({
                "signal": signal,
                "direction": direction,
                "confidence": float(avg_conf),
                "up_votes": int(count_up),
                "down_votes": int(count_down),
            })

        return results

    def predict_price(self, x: torch.Tensor) -> torch.Tensor:
        """Average price prediction (direction-agnostic)."""
        price_preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                _, price_out = model(x)
                price_preds.append(price_out)
        return torch.stack(price_preds).mean(dim=0)


# ═══════════════════════════════════════════════════════════════
# Method 3: Stacking Meta-Learner
# ═══════════════════════════════════════════════════════════════

class StackingMetaModel(nn.Module):
    """Meta-learner: learns to combine 2 base model outputs + market context.

    y_final = f_meta(y1, y2, market_volatility)
    """

    def __init__(self, n_base_models: int = 2, context_dim: int = 1,
                 price_dim: int = 3, hidden: int = 32):
        super().__init__()
        input_dim = n_base_models * 2 + context_dim  # dir_probs + price_preds per model + context

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),  # final direction logits
        )

        self.price_net = nn.Sequential(
            nn.Linear(n_base_models * price_dim + context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),  # final price prediction
        )

    def forward(self, base_outputs: list, market_context: torch.Tensor):
        """base_outputs: list of (dir_logits, price_pred) tuples from each model"""
        B = market_context.shape[0]
        device = market_context.device

        # Collect base predictions
        dir_features = []
        price_features = []

        for dir_logits, price_pred in base_outputs:
            # Direction probabilities
            dir_prob = F.softmax(dir_logits, dim=-1)  # [B, 2]
            dir_features.append(dir_prob)
            # Price predictions
            price_features.append(price_pred)

        # Concatenate
        dir_cat = torch.cat(dir_features, dim=-1)  # [B, n_models * 2]
        price_cat = torch.cat(price_features, dim=-1)  # [B, n_models]

        # Add market context
        dir_input = torch.cat([dir_cat, market_context], dim=-1)
        price_input = torch.cat([price_cat, market_context], dim=-1)

        # Meta predictions
        dir_meta = self.net(dir_input)
        price_meta = self.price_net(price_input)

        return dir_meta, price_meta


class StackingEnsemble:
    """Stacking: train a meta-learner on base model outputs.

    IMPORTANT: Train meta-learner on validation set predictions, NOT training set,
    to avoid severe overfitting.
    """

    def __init__(self, base_models: list, meta_model: StackingMetaModel | None = None,
                 context_dim: int = 1):
        self.base_models = base_models
        self.n_models = len(base_models)
        self.meta_model = meta_model or StackingMetaModel(
            n_base_models=len(base_models), context_dim=context_dim,
            price_dim=3
        )

    def get_base_predictions(self, x: torch.Tensor) -> list:
        """Get raw predictions from all base models."""
        outputs = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                dir_out, price_out = model(x)
                outputs.append((dir_out, price_out))
        return outputs

    def predict(self, x: torch.Tensor, market_context: torch.Tensor | None = None):
        """Meta-learner prediction combining all base models."""
        if market_context is None:
            market_context = torch.zeros(x.shape[0], 1, device=x.device)

        base_outputs = self.get_base_predictions(x)
        self.meta_model.eval()
        with torch.no_grad():
            dir_out, price_out = self.meta_model(base_outputs, market_context)
        return dir_out, price_out

    def train_meta(self, train_loader, val_loader, device, epochs: int = 30):
        """Train meta-learner using validation set predictions (prevents overfitting)."""
        self.meta_model.to(device)
        optimizer = torch.optim.AdamW(self.meta_model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Precompute base model predictions on validation set
        val_preds, val_targets = [], []
        for x_batch, t_batch in val_loader:
            x_batch = x_batch.to(device)
            y_dir = t_batch["direction"].to(device)
            y_price = t_batch["price_change"].to(device)

            with torch.no_grad():
                base_outputs = self.get_base_predictions(x_batch)

            # Market context: recent volatility (vol_expanding feature)
            market_ctx = x_batch[:, -1, 73:74]  # vol_expanding index
            if market_ctx.shape[-1] == 0:
                market_ctx = torch.zeros(x_batch.shape[0], 1, device=device)

            val_preds.append((base_outputs, market_ctx))
            val_targets.append((y_dir, y_price))

        # Train meta-learner
        for epoch in range(epochs):
            total_loss = 0
            for (base_outputs, market_ctx), (y_dir, y_price) in zip(val_preds, val_targets):
                dir_meta, price_meta = self.meta_model(base_outputs, market_ctx)
                loss_dir = F.cross_entropy(dir_meta, y_dir)
                loss_price = F.mse_loss(price_meta.squeeze(-1), y_price)
                loss = loss_dir + 0.3 * loss_price

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        return total_loss / len(val_preds)


# ═══════════════════════════════════════════════════════════════
# Method 4: PINN-Guard Logic
# ═══════════════════════════════════════════════════════════════

class PINNGuard:
    """Use PINN model as a physics-constraint guard for ensemble predictions.

    If ensemble prediction deviates too far from PINN's physics-consistent
    prediction, the PINN overrides with a weighted correction.
    """

    def __init__(self, pinn_model, deviation_threshold: float = 0.05,
                 guard_weight: float = 0.7):
        self.pinn_model = pinn_model
        self.deviation_threshold = deviation_threshold
        self.guard_weight = guard_weight

    def guarded_predict(self, x: torch.Tensor,
                        ensemble_pred_dir: torch.Tensor,
                        ensemble_pred_price: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply PINN guard to ensemble predictions.

        If |ensemble - pinn| / pinn > threshold:
          final = guard_weight * pinn + (1 - guard_weight) * ensemble
        else:
          final = ensemble
        """
        self.pinn_model.eval()
        with torch.no_grad():
            pinn_dir, pinn_price = self.pinn_model(x)

        # Compute deviation for price
        price_dev = torch.abs(ensemble_pred_price - pinn_price) / (torch.abs(pinn_price) + 1e-6)
        need_guard = (price_dev > self.deviation_threshold).float()

        # Apply guard for direction (use probability disagreement)
        ensemble_prob = F.softmax(ensemble_pred_dir, dim=-1)
        pinn_prob = F.softmax(pinn_dir, dim=-1)
        dir_kl = (ensemble_prob * (ensemble_prob / (pinn_prob + 1e-6)).log()).sum(dim=-1)
        dir_need_guard = (dir_kl > 0.5).float().unsqueeze(-1)

        # Blend: if guard active, PINN dominates
        guard_mask_price = need_guard.unsqueeze(-1)
        final_price = (
            guard_mask_price * (self.guard_weight * pinn_price + (1 - self.guard_weight) * ensemble_pred_price)
            + (1 - guard_mask_price) * ensemble_pred_price
        )

        guard_mask_dir = dir_need_guard
        final_dir_prob = (
            guard_mask_dir * (self.guard_weight * pinn_prob + (1 - self.guard_weight) * ensemble_prob)
            + (1 - guard_mask_dir) * ensemble_prob
        )

        return final_dir_prob, final_price


# ═══════════════════════════════════════════════════════════════
# Full Ensemble Pipeline
# ═══════════════════════════════════════════════════════════════

class FullEnsemble:
    """Complete ensemble combining all 4 methods in a pipeline.

    Strategy:
      1. Train 2 base models (FT-iTransformer, PINN)
      2. Use Dynamic Weighting as baseline (Method 1)
      3. Apply Voting for signal generation (Method 2)
      4. Train Meta-Learner stacking layer (Method 3)
      5. Add PINN Guard as safety layer (Method 4)
    """

    def __init__(self, models: list, pinn_model=None, device: str = "cpu"):
        self.models = models
        self.device = device

        self.weighted = DynamicWeightedEnsemble(models)
        self.voting = VotingEnsemble(models)
        self.stacking = StackingEnsemble(models)
        self.guard = PINNGuard(pinn_model) if pinn_model else None

    def predict(self, x: torch.Tensor, method: str = "weighted",
                market_context: torch.Tensor | None = None):
        """Predict using specified ensemble method.

        Methods: "weighted", "voting", "stacking", "all"
        """
        if method == "weighted":
            return self.weighted.predict(x)

        elif method == "voting":
            signals = self.voting.predict_signal(x)
            prices = self.voting.predict_price(x)
            return signals, prices

        elif method == "stacking":
            return self.stacking.predict(x, market_context)

        elif method == "all":
            # Weighted baseline
            dir_prob, price_pred = self.weighted.predict(x)

            # Apply stacking refinement if meta-model trained
            try:
                dir_prob_s, price_pred_s = self.stacking.predict(x, market_context)
                dir_prob = (dir_prob + dir_prob_s) / 2
                price_pred = (price_pred + price_pred_s) / 2
            except Exception:
                pass

            # Apply PINN guard if available
            if self.guard:
                dir_prob, price_pred = self.guard.guarded_predict(
                    x, torch.log(dir_prob + 1e-6), price_pred
                )
                # Convert back from log-prob
                dir_prob = F.softmax(dir_prob, dim=-1)

            return dir_prob, price_pred

        else:
            raise ValueError(f"Unknown method: {method}")


# ═══════════════════════════════════════════════════════════════
# Ensemble Wrappers — 2 base models (FT-iTransformer, PINN)
# + 4 ensemble methods (Weighted, Voting, Stacking, PINN-Guard)
# ═══════════════════════════════════════════════════════════════

BASE_MODEL_BUILDERS = {}
def _get_base_builders():
    if not BASE_MODEL_BUILDERS:
        from model.ft_transformer import FT_iTransformerWrapper
        from model.pinn_model import PINNWrapper
        BASE_MODEL_BUILDERS.update({
            "FT-iTransformer (时频协同)": FT_iTransformerWrapper,
            "PINN (物理约束)": PINNWrapper,
        })
    return BASE_MODEL_BUILDERS


def _build_base_models(kwargs: dict) -> nn.ModuleList:
    builders = _get_base_builders()
    return nn.ModuleList([
        builders["FT-iTransformer (时频协同)"](**kwargs),
        builders["PINN (物理约束)"](**kwargs),
    ])


class EnsembleWeightedWrapper(nn.Module):
    """集成-加权平均: train 2 base models, predict via dynamic weighted averaging."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        self.base_models = _build_base_models(kwargs)
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))
        self._weights = [0.5, 0.5]

    def forward(self, x):
        dir_preds, price_preds = [], []
        for m in self.base_models:
            d, p = m(x)
            dir_preds.append(F.softmax(d, dim=-1))
            price_preds.append(p)
        dir_out = sum(w * d for w, d in zip(self._weights, dir_preds))
        price_out = sum(w * p for w, p in zip(self._weights, price_preds))
        return dir_out, price_out

    def update_weights(self, errors):
        inv = [1.0 / (e + 1e-6) for e in errors]
        total = sum(inv)
        self._weights = [e / total for e in inv]


class EnsembleVotingWrapper(nn.Module):
    """集成-多数投票: train 2 base models, predict via consensus voting."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        self.base_models = _build_base_models(kwargs)
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        dirs, prices = [], []
        for m in self.base_models:
            d, p = m(x)
            dirs.append(d)
            prices.append(p)
        dir_mean = torch.stack(dirs).mean(dim=0)
        price_mean = torch.stack(prices).mean(dim=0)
        return dir_mean, price_mean

    def predict_signal(self, x):
        """Return voting result: Strong Buy / Buy / Neutral / Sell / Strong Sell."""
        dirs = []
        for m in self.base_models:
            m.eval()
            with torch.no_grad():
                d, _ = m(x)
                dirs.append(d.argmax(1).cpu().tolist())
        signals = []
        for b in range(x.shape[0]):
            votes = [d[b] for d in dirs]
            up = votes.count(1)
            down = votes.count(0)
            if up >= 2:
                signals.append("Strong Buy")
            elif down >= 2:
                signals.append("Strong Sell")
            elif up > down:
                signals.append("Buy")
            elif down > up:
                signals.append("Sell")
            else:
                signals.append("Neutral")
        return signals


class EnsembleStackingWrapper(nn.Module):
    """集成-堆叠法: train 2 base models + meta-learner stacking layer."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        self.base_models = _build_base_models(kwargs)
        self.meta = StackingMetaModel(n_base_models=2, context_dim=1, price_dim=3)
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))
        self._meta_trained = False

    def forward(self, x):
        base_outputs = []
        with torch.no_grad():
            for m in self.base_models:
                m.eval()
                base_outputs.append(m(x))
        ctx = x[:, -1, 73:74] if x.shape[-1] > 73 else torch.zeros(x.shape[0], 1, device=x.device)
        return self.meta(base_outputs, ctx)

    def train_meta(self, base_outputs_list, market_contexts, y_dirs, y_prices, epochs=30):
        device = next(self.meta.parameters()).device
        self.meta.to(device)
        opt = torch.optim.AdamW(self.meta.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            total_loss = 0
            for bo, ctx, yd, yp in zip(base_outputs_list, market_contexts, y_dirs, y_prices):
                bo_dev = [(d.to(device), p.to(device)) for d, p in bo]
                dir_out, price_out = self.meta(bo_dev, ctx.to(device))
                loss = F.cross_entropy(dir_out, yd.to(device)) \
                       + 0.3 * F.mse_loss(price_out.squeeze(-1), yp.to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
        self._meta_trained = True


class EnsemblePINNGuardWrapper(nn.Module):
    """集成-PINN Guard: train 2 base models, PINN as physics-constraint guard.

    If ensemble prediction deviates >5% from PINN's physics-consistent prediction,
    the PINN overrides with weighted correction:
        final = 0.7 * pinn + 0.3 * ensemble
    """

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        self.base_models = _build_base_models(kwargs)
        # PINN is the last base model (index 1), also serves as guard
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))
        self.deviation_threshold = 0.05
        self.guard_weight = 0.7

    def forward(self, x):
        dirs, prices = [], []
        for m in self.base_models:
            d, p = m(x)
            dirs.append(d)
            prices.append(p)

        # Ensemble: average of both base models (including PINN)
        ens_dir = torch.stack(dirs).mean(dim=0)
        ens_price = torch.stack(prices).mean(dim=0)

        # PINN is the last base model
        pinn_dir, pinn_price = dirs[1], prices[1]

        # Compute deviation
        price_dev = torch.abs(ens_price - pinn_price) / (torch.abs(pinn_price) + 1e-6)
        need_guard = (price_dev > self.deviation_threshold).float()

        # Guard: if deviation > 5%, PINN dominates
        final_price = need_guard * (self.guard_weight * pinn_price + (1 - self.guard_weight) * ens_price) \
                      + (1 - need_guard) * ens_price

        # Direction: use KL divergence between ensemble prob and PINN prob
        ens_prob = F.softmax(ens_dir, dim=-1)
        pinn_prob = F.softmax(pinn_dir, dim=-1)
        dir_kl = (ens_prob * (ens_prob / (pinn_prob + 1e-6)).log()).sum(dim=-1, keepdim=True)
        dir_guard = (dir_kl > 0.5).float()
        final_dir = dir_guard * (self.guard_weight * pinn_prob + (1 - self.guard_weight) * ens_prob) \
                    + (1 - dir_guard) * ens_prob

        return final_dir, final_price


# ═══════════════════════════════════════════════════════════════
# All-Models wrapper: train all 6, pick the best
# ═══════════════════════════════════════════════════════════════

ALL_MODEL_NAMES = [
    "FT-iTransformer (时频协同)",
    "PINN (物理约束)",
    "集成-加权平均",
    "集成-多数投票",
    "集成-堆叠法 (Stacking)",
    "集成-PINN Guard",
]


class AllModelsWrapper(nn.Module):
    """全部模型对比: train all 6 models, pick the highest accuracy one.

    base_models contains all 6 models. TrainingWorker trains each one,
    evaluates on validation set, and saves the best checkpoint.
    """

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)

        # Build all 6 models
        self.base_models = nn.ModuleList()
        # 2 single models
        builders = _get_base_builders()
        for name in ALL_MODEL_NAMES[:2]:
            self.base_models.append(builders[name](**kwargs))
        # 4 ensemble methods
        self.base_models.append(EnsembleWeightedWrapper(**kwargs))
        self.base_models.append(EnsembleVotingWrapper(**kwargs))
        self.base_models.append(EnsembleStackingWrapper(**kwargs))
        self.base_models.append(EnsemblePINNGuardWrapper(**kwargs))

        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))
        self.best_model_idx: int = 0
        self.best_model_name: str = ""
        self.all_accuracies: dict = {}

    def forward(self, x):
        # Use the best model (or first if not yet determined)
        return self.base_models[self.best_model_idx](x)


# ═══════════════════════════════════════════════════════════════
# Legacy wrappers for backward compatibility with v1 NodeTransformer checkpoints
# ═══════════════════════════════════════════════════════════════

def _build_base_models_v1(kwargs: dict) -> nn.ModuleList:
    """Same as _build_base_models (2 base models: FT-iTransformer + PINN)."""
    from model.ft_transformer import FT_iTransformerWrapper
    from model.pinn_model import PINNWrapper
    return nn.ModuleList([
        FT_iTransformerWrapper(**kwargs),
        PINNWrapper(**kwargs),
    ])


class LegacyEnsembleWeightedWrapper(nn.Module):
    """Legacy 集成-加权平均: uses V1 NodeTransformer for old checkpoint compat."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        self.base_models = _build_base_models_v1(kwargs)
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))
        self._weights = [0.5, 0.5]

    def forward(self, x):
        dir_preds, price_preds = [], []
        for m in self.base_models:
            d, p = m(x)
            dir_preds.append(F.softmax(d, dim=-1))
            price_preds.append(p)
        dir_out = sum(w * d for w, d in zip(self._weights, dir_preds))
        price_out = sum(w * p for w, p in zip(self._weights, price_preds))
        return dir_out, price_out

    def update_weights(self, errors):
        inv = [1.0 / (e + 1e-6) for e in errors]
        total = sum(inv)
        self._weights = [e / total for e in inv]


class LegacyEnsembleVotingWrapper(nn.Module):
    """Legacy 集成-多数投票: uses V1 NodeTransformer for old checkpoint compat."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        self.base_models = _build_base_models_v1(kwargs)
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        dirs, prices = [], []
        for m in self.base_models:
            d, p = m(x)
            dirs.append(d)
            prices.append(p)
        dir_mean = torch.stack(dirs).mean(dim=0)
        price_mean = torch.stack(prices).mean(dim=0)
        return dir_mean, price_mean


class LegacyEnsembleStackingWrapper(nn.Module):
    """Legacy 集成-堆叠法: uses V1 NodeTransformer for old checkpoint compat."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        self.base_models = _build_base_models_v1(kwargs)
        self.meta = StackingMetaModel(n_base_models=2, context_dim=1, price_dim=3)
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))
        self._meta_trained = False

    def forward(self, x):
        base_outputs = []
        with torch.no_grad():
            for m in self.base_models:
                m.eval()
                base_outputs.append(m(x))
        ctx = x[:, -1, 73:74] if x.shape[-1] > 73 else torch.zeros(x.shape[0], 1, device=x.device)
        return self.meta(base_outputs, ctx)


class LegacyEnsemblePINNGuardWrapper(nn.Module):
    """Legacy 集成-PINN Guard: uses V1 NodeTransformer for old checkpoint compat."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        self.base_models = _build_base_models_v1(kwargs)
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))
        self.deviation_threshold = 0.05
        self.guard_weight = 0.7

    def forward(self, x):
        dirs, prices = [], []
        for m in self.base_models:
            d, p = m(x)
            dirs.append(d)
            prices.append(p)
        dir_mean = torch.stack(dirs).mean(dim=0)
        price_mean = torch.stack(prices).mean(dim=0)
        pinn_dir, pinn_price = self.base_models[1](x)
        price_dev = (price_mean - pinn_price).abs().mean().item()
        if price_dev > self.deviation_threshold:
            price_mean = self.guard_weight * pinn_price + (1 - self.guard_weight) * price_mean
        return dir_mean, price_mean


class LegacyAllModelsWrapper(nn.Module):
    """Legacy 所有模型: uses V1 NodeTransformer for old checkpoint compat."""

    def __init__(self, input_dim: int = 85, d_model: int = 128,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 transformer_layers: int = 3, nhead: int = 4,
                 dropout: float = 0.15, max_seq_len: int = 120,
                 patch_len: int = 5):
        super().__init__()
        kwargs = dict(input_dim=input_dim, d_model=d_model, lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers, transformer_layers=transformer_layers,
                      nhead=nhead, dropout=dropout, max_seq_len=max_seq_len,
                      patch_len=patch_len)
        from model.ft_transformer import FT_iTransformerWrapper
        from model.pinn_model import PINNWrapper

        self.base_models = nn.ModuleList()
        # 2 single models (v1 compat)
        self.base_models.append(FT_iTransformerWrapper(**kwargs))
        self.base_models.append(PINNWrapper(**kwargs))
        # 4 ensemble methods (legacy versions)
        self.base_models.append(LegacyEnsembleWeightedWrapper(**kwargs))
        self.base_models.append(LegacyEnsembleVotingWrapper(**kwargs))
        self.base_models.append(LegacyEnsembleStackingWrapper(**kwargs))
        self.base_models.append(LegacyEnsemblePINNGuardWrapper(**kwargs))

        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_price = nn.Parameter(torch.zeros(1))
        self.best_model_idx: int = 0
        self.best_model_name: str = ""
        self.all_accuracies: dict = {}

    def forward(self, x):
        return self.base_models[self.best_model_idx](x)