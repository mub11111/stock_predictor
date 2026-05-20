"""Auto-optimization: uses DeepSeek API to suggest training improvements when
prediction confidence falls below the minimum threshold."""

from __future__ import annotations
import json

OPTIMIZER_SYSTEM_PROMPT = """你是一个量化交易系统的超参数优化专家，专精于时间序列深度学习模型调优。

当前系统架构：PatchEmbed + BiLSTM + Transformer + MHA Pool 混合模型，支持多种模型变体（FT-iTransformer、PINN、Node Transformer、集成方法）。

训练配置：AdamW优化器，OneCycleLR调度器，FocalLoss损失函数，EMA权重平滑，早停机制。

请在分析后返回严格JSON格式（不要markdown，只要JSON）：
{
  "hyperparams": {
    "lr": 学习率建议(float, 1e-5~1e-2),
    "d_model": 模型维度建议(int, 64~512),
    "dropout": dropout建议(float, 0.0~0.5),
    "seq_len": 序列长度建议(int, 60~480),
    "batch_size": batch大小建议(int, 8~128),
    "epochs": 训练轮数建议(int, 20~300),
    "early_stop_patience": 早停耐心建议(int, 5~50),
    "lstm_layers": LSTM层数建议(int, 1~4),
    "transformer_layers": Transformer层数建议(int, 1~6),
    "nhead": 注意力头数建议(int, 2~16, 必须能被d_model整除)
  },
  "data_suggestions": {
    "fetch_more_days": 建议额外获取的天数(int, 0~180),
    "use_frequency": "5min或60min或daily",
    "add_extra_features": true或false
  },
  "model_suggestion": "推荐切换的模型名称，如保持当前则填当前模型名",
  "reasoning": "优化建议的详细理由（中文，100字内）",
  "expected_improvement": 预期置信度提升(float, 0.0~0.3)
}

注意：
- 只建议真正有改善空间的参数，不要为改而改
- 金融时序数据对过拟合敏感，dropout建议不要过低
- 序列长度过长会增加噪声，过短会丢失趋势信息
- 如果数据量不足，优先建议扩大数据采集范围"""

# Safe ranges for parameter clamping
_SAFE_RANGES = {
    "lr": (1e-5, 1e-2),
    "d_model": (64, 512),
    "dropout": (0.0, 0.5),
    "seq_len": (60, 480),
    "batch_size": (8, 128),
    "epochs": (20, 300),
    "early_stop_patience": (5, 50),
    "lstm_layers": (1, 4),
    "transformer_layers": (1, 6),
    "nhead": (2, 16),
}


class AutoOptimizer:
    """Analyzes low-confidence predictions and suggests training improvements
    using the DeepSeek API as a hyperparameter optimization advisor."""

    def __init__(self, ai_client):
        self.ai = ai_client

    def analyze_and_suggest(
        self, ts_code: str, pred: dict, model_config,
        model_context: dict | None = None,
        df_info: dict | None = None,
    ) -> dict:
        """Send a structured prompt to DeepSeek requesting optimization suggestions.

        Returns a parsed dict with hyperparams, data_suggestions, model_suggestion,
        reasoning, and expected_improvement.
        """
        prompt = self._build_prompt(ts_code, pred, model_config, model_context, df_info)

        try:
            response = self.ai.client.chat.completions.create(
                model=self.ai.model,
                max_tokens=600,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
            result = json.loads(text)
        except Exception:
            return {"error": "API调用失败", "reasoning": "DeepSeek优化请求失败"}

        return self._clamp_and_validate(result, model_context)

    def _build_prompt(
        self, ts_code: str, pred: dict, model_config,
        model_context: dict | None, df_info: dict | None,
    ) -> str:
        """Build the optimization request prompt with full context."""
        conf = pred.get("direction_conf", 0)
        direction = pred.get("direction", "flat")
        target = pred.get("target_price", 0)

        lines = [
            f"股票: {ts_code}",
            f"当前预测置信度: {conf:.1%} (目标>={getattr(model_config, 'min_confidence', 0.75):.0%})",
            f"预测方向: {direction}",
            f"目标价: ¥{target:.2f}" if target else "",
            "",
            "当前模型配置:",
            f"  d_model={model_config.d_model}, lstm_hidden={model_config.lstm_hidden}",
            f"  lstm_layers={model_config.lstm_layers}, transformer_layers={model_config.transformer_layers}",
            f"  nhead={model_config.nhead}, dropout={model_config.dropout}",
            f"  seq_len={model_config.seq_len}, patch_len={model_config.patch_len}",
            f"  batch_size={model_config.batch_size}, lr={model_config.lr}",
            f"  epochs={model_config.epochs}, early_stop_patience={model_config.early_stop_patience}",
        ]

        if model_context:
            lines.extend([
                "",
                "当前训练模型信息:",
                f"  模型类型: {model_context.get('model_type', '未知')}",
                f"  准确率: {model_context.get('accuracy', 0):.1%}" if model_context.get("accuracy", 0) > 0 else "",
                f"  验证准确率: {model_context.get('val_accuracy', 0):.1%}" if model_context.get("val_accuracy", 0) > 0 else "",
            ])

        if df_info and df_info.get("samples", 0) > 0:
            lines.extend([
                "",
                "数据统计:",
                f"  样本数: {df_info.get('samples', 0)}",
                f"  时间范围: {df_info.get('date_range', '未知')}",
                f"  频率: {df_info.get('frequencies_available', '5min')}",
            ])

        lines.append("\n请分析当前配置并提供优化建议。重点关注：学习率、模型维度、序列长度、dropout、训练轮数。")
        return "\n".join(l for l in lines if l)

    def _clamp_and_validate(self, result: dict, model_context: dict | None) -> dict:
        """Clamp suggested parameters to safe ranges and validate structure."""
        hyper = result.get("hyperparams", {})

        clamped = {}
        for key, (lo, hi) in _SAFE_RANGES.items():
            if key in hyper:
                val = hyper[key]
                if isinstance(val, (int, float)):
                    clamped[key] = max(lo, min(hi, val))
                else:
                    continue  # skip non-numeric values

        # Ensure nhead divides d_model if both are present
        if "d_model" in clamped and "nhead" in clamped:
            d_model = int(clamped["d_model"])
            nhead = int(clamped["nhead"])
            if d_model % nhead != 0:
                # Find the nearest valid nhead
                for nh in range(nhead, 1, -1):
                    if d_model % nh == 0:
                        clamped["nhead"] = nh
                        break

        data_sugs = result.get("data_suggestions", {})
        if isinstance(data_sugs, dict):
            fetch_days = data_sugs.get("fetch_more_days", 0)
            if isinstance(fetch_days, (int, float)):
                data_sugs["fetch_more_days"] = max(0, min(180, int(fetch_days)))

        # Validate model_suggestion against actual registry
        raw_suggestion = result.get("model_suggestion", "")
        model_suggestion = _match_model_name(raw_suggestion)
        if not model_suggestion:
            model_suggestion = model_context.get("model_type", "") if model_context else ""

        return {
            "hyperparams": clamped,
            "data_suggestions": data_sugs,
            "model_suggestion": model_suggestion,
            "reasoning": result.get("reasoning", ""),
            "expected_improvement": float(result.get("expected_improvement", 0.05)),
        }


# 模型名常量 — 避免从 gui.dialogs 导入 PyQt6 依赖
_MODEL_NAMES = [
    "所有模型",
    "FT-iTransformer (时频协同)",
    "Node Transformer (图注意力·准确率偏低)",
    "PINN (物理约束)",
    "集成-加权平均",
    "集成-多数投票",
    "集成-堆叠法 (Stacking)",
    "集成-PINN Guard",
]


def _match_model_name(raw: str, registry: dict | None = None) -> str:
    """Fuzzy-match a model name against registry keys.
    Handles DeepSeek returning shorthand like 'PINN' instead of 'PINN (物理约束)'."""
    if not raw:
        return ""
    keys = list(registry.keys()) if registry else _MODEL_NAMES
    raw_lower = raw.strip().lower()
    for key in keys:
        if key.lower() == raw_lower:
            return key
    for key in keys:
        if raw_lower in key.lower():
            return key
    for key in keys:
        key_compact = key.lower().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        raw_compact = raw_lower.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if raw_compact in key_compact or key_compact in raw_compact:
            return key
    return ""
