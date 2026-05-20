"""DeepSeek API client for AI stock analysis and remote prediction."""
from __future__ import annotations
import json
import pandas as pd
from openai import OpenAI
from data.rule_engine import build_ai_system_prompt, build_rules_context
from datetime import datetime


def _empty_pred(ts_code: str) -> dict:
    return {
        "direction": "flat", "direction_conf": 0,
        "target_price": 0, "price_lower": 0, "price_upper": 0,
        "horizon_minutes": 60, "model_version": "empty",
        "created_at": datetime.now(), "rule_check": [],
    }


class AIAnalyzer:
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

    def predict_remote(self, ts_code: str, df_min: pd.DataFrame,
                       df_daily: pd.DataFrame | None = None,
                       current_price: float | None = None,
                       model_context: dict | None = None) -> dict:
        """使用 DeepSeek API 远程预测，附带本地模型完整参数和90天数据。

        本地模型的权重统计、架构参数、特征列表一并上传至云端。
        云端基于模型学到的知识 + 90天市场数据进行推理预测。
        """
        import math
        if df_min.empty:
            return _empty_pred(ts_code)

        close = df_min["close"].values.astype(float)
        volume = df_min["volume"].values.astype(float) if "volume" in df_min.columns else close * 0
        price = float(current_price or close[-1])

        # ── Multi-timeframe data block (90-day perspective) ──
        recent_n = min(60, len(close))
        recent_close = close[-recent_n:]
        recent_vol = volume[-recent_n:]
        trend = "上涨" if close[-1] > close[-recent_n] else "下跌"

        ma5 = float(pd.Series(close).rolling(5).mean().iloc[-1]) if len(close) >= 5 else price
        ma20 = float(pd.Series(close).rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
        ma60 = float(pd.Series(close).rolling(60).mean().iloc[-1]) if len(close) >= 60 else price
        high_n = float(max(recent_close))
        low_n = float(min(recent_close))
        vol_ratio = float(recent_vol[-1] / (recent_vol.mean() + 1e-8))

        # Volatility across timeframes
        vol_5 = float(pd.Series(close).pct_change().tail(5).std() * 100) if len(close) >= 5 else 0
        vol_20 = float(pd.Series(close).pct_change().tail(20).std() * 100) if len(close) >= 20 else 0

        # Multi-period trend
        if len(close) >= 20:
            chg_5 = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
            chg_20 = (close[-1] / close[-20] - 1) * 100
            short_trend = "上涨" if chg_5 > 0 else "下跌"
            mid_trend = "上涨" if chg_20 > 0 else "下跌"
        else:
            chg_5 = chg_20 = 0
            short_trend = mid_trend = "未知"

        data_block = f"""A股 {ts_code} 分钟线数据（90天范围）:
当前价: ¥{price:.2f}
MA5: ¥{ma5:.2f} | MA20: ¥{ma20:.2f} | MA60: ¥{ma60:.2f}
近{recent_n}根K线区间: ¥{low_n:.2f} - ¥{high_n:.2f}
量比: {vol_ratio:.2f} | 5日波动率: {vol_5:.2f}% | 20日波动率: {vol_20:.2f}%
短期趋势(5日): {short_trend} {chg_5:+.2f}% | 中期趋势(20日): {mid_trend} {chg_20:+.2f}%
价格 vs MA5: {'↑' if price > ma5 else '↓'} | vs MA20: {'↑' if price > ma20 else '↓'} | vs MA60: {'↑' if price > ma60 else '↓'}
近10根收盘: {[round(c, 2) for c in close[-10:].tolist()]}"""

        # Daily data block (up to 90 days)
        if df_daily is not None and not df_daily.empty:
            d_close = df_daily["close"].values.astype(float)
            d_high = df_daily["high"].values.astype(float) if "high" in df_daily.columns else d_close
            d_low = df_daily["low"].values.astype(float) if "low" in df_daily.columns else d_close
            nd = len(d_close)
            if nd >= 5:
                d_ma5 = float(pd.Series(d_close).rolling(5).mean().iloc[-1])
                d_ma20 = float(pd.Series(d_close).rolling(20).mean().iloc[-1]) if nd >= 20 else d_ma5
                d_ma60 = float(pd.Series(d_close).rolling(60).mean().iloc[-1]) if nd >= 60 else d_ma5
                d_90high = float(max(d_high[-min(90, nd):]))
                d_90low = float(min(d_low[-min(90, nd):]))
                d_chg_5 = (d_close[-1] / d_close[-5] - 1) * 100 if nd >= 5 else 0
                d_chg_20 = (d_close[-1] / d_close[-20] - 1) * 100 if nd >= 20 else 0
                d_chg_60 = (d_close[-1] / d_close[-60] - 1) * 100 if nd >= 60 else 0
                d_trend_20 = "上涨" if d_chg_20 > 0 else "下跌"
                data_block += f"""

日线数据（{nd}个交易日）:
当前价: ¥{d_close[-1]:.2f}
近90日区间: ¥{d_90low:.2f} - ¥{d_90high:.2f}
日线MA5: ¥{d_ma5:.2f} | MA20: ¥{d_ma20:.2f} | MA60: ¥{d_ma60:.2f}
5日涨跌: {d_chg_5:+.2f}% | 20日: {d_chg_20:+.2f}% | 60日: {d_chg_60:+.2f}%
中期日线趋势: {d_trend_20}"""

        # ── Model parameter block ──
        model_block = ""
        if model_context:
            acc_str = f"{model_context['accuracy']:.1%}" if model_context.get("accuracy", 0) > 0 else "N/A"
            model_block = f"""

本地训练模型元数据:
  模型类型: {model_context.get('model_type', '未知')}
  准确率: {acc_str} (验证: {model_context.get('val_accuracy', 0):.1%})
  最佳子模型: {model_context.get('best_model_name', '无')}"""

            # Full model parameter stats
            mp = model_context.get("model_params")
            if mp:
                arch = mp.get("architecture", {})
                model_block += f"""
  模型参数总数: {arch.get('total_params', 0):,}
  层数: {arch.get('num_layers', 0)}
  训练轮次: {arch.get('epoch', 0)}
  训练验证损失: {arch.get('val_loss', 0):.4f}

  模型权重统计（各层 mean±std [min..max]）:"""
                # Show top layers by param count (limit ~15 to stay within token budget)
                layers = sorted(mp.get("layer_params", []),
                               key=lambda x: x.get("params", 0), reverse=True)[:15]
                for l in layers:
                    n = l.get("params", 0)
                    if n >= 1000:
                        n_str = f"{n/1000:.1f}k"
                    else:
                        n_str = str(n)
                    qs = l.get("quantiles")
                    if qs:
                        model_block += f"\n    {l['name']}: [{l['shape']}] {n_str}参 "
                        model_block += f"μ={l['mean']:.4f} σ={l['std']:.4f} "
                        model_block += f"[{l['min']:.4f}..{l['max']:.4f}]"
                        model_block += f" q=[{qs[0]:.4f},{qs[1]:.4f},{qs[2]:.4f},{qs[3]:.4f},{qs[4]:.4f}]"
                    else:
                        model_block += f"\n    {l['name']}: [{l['shape']}] {n_str}参 "
                        model_block += f"μ={l['mean']:.4f} σ={l['std']:.4f} [{l['min']:.4f}..{l['max']:.4f}]"

                # Feature list
                feats = mp.get("feature_list")
                if feats:
                    nf = len(feats)
                    model_block += f"\n  输入特征({nf}个): {feats[:20]}"
                    if nf > 20:
                        model_block += f" ...等{nf}个"

            # Local model prediction
            local_pred = model_context.get("local_prediction")
            if local_pred:
                model_block += f"""

本地模型预测:
  方向: {local_pred.get('direction', 'flat')}
  置信度: {local_pred.get('confidence', 0):.1%}
  目标价: ¥{local_pred.get('target_price', 0):.2f}"""

        prompt = f"""{data_block}{model_block}

请基于以上A股90天分钟线+日线数据以及本地训练模型的完整参数知识，预测该股未来60分钟走势。
你需要运用本地模型学到的权重分布和特征知识，结合更丰富的90天数据进行综合推理。
返回严格JSON（不含markdown）：
{{"direction":"up或down或flat","direction_conf":0.0到1.0的置信度,"target_price":目标价数字,"price_lower":下限价,"price_upper":上限价,"reason":"理由30字内"}}"""

        system_msg = (
            "你是一个量化分析系统，已加载本地训练的PyTorch神经网络模型的完整参数（权重统计、架构、特征列表）。"
            "请基于该模型学到的参数知识，结合提供的90天市场数据（分钟线+日线），进行综合推理预测。"
            "只返回严格JSON，不输出任何其他内容。"
            "字段：direction(up/down/flat), direction_conf(0-1), target_price(元), price_lower(元), price_upper(元), reason(中文30字内)。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=400,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.strip()
            # Parse JSON — handle possible markdown wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
            result = json.loads(text)
        except Exception:
            return _empty_pred(ts_code)

        direction = str(result.get("direction", "flat"))
        if direction not in ("up", "down", "flat"):
            direction = "flat"
        conf = float(result.get("direction_conf", 0.5))
        target = float(result.get("target_price", price))

        return {
            "direction": direction,
            "direction_conf": min(max(conf, 0), 1),
            "target_price": round(target, 2),
            "price_lower": round(float(result.get("price_lower", price * 0.97)), 2),
            "price_upper": round(float(result.get("price_upper", price * 1.03)), 2),
            "horizon_minutes": 60,
            "model_version": f"deepseek-remote-{self.model}",
            "created_at": datetime.now(),
            "rule_check": [],
        }

    def analyze_prediction(self, pred: dict, behavior: dict | None = None,
                           news_text: str = "") -> str:
        ts_code = pred.get("ts_code", "000001.SZ")
        current_price = pred.get("target_price", 0)

        behavior_text = ""
        if behavior:
            inst_net = behavior.get('inst_net', 0)
            retail_net = behavior.get('retail_net', 0)
            behavior_text = f"""
持股人群分析:
  机构净流向: {inst_net:+.1%} ({'流入' if inst_net > 0 else '流出'})
  散户净流向: {retail_net:+.1%} ({'流入' if retail_net > 0 else '流出'})
  机构买入: {behavior.get('inst_buy_total', 0):.0f}手  机构卖出: {behavior.get('inst_sell_total', 0):.0f}手
  散户买入: {behavior.get('retail_buy_total', 0):.0f}手  散户卖出: {behavior.get('retail_sell_total', 0):.0f}手
  恐慌指数: {behavior.get('panic_index', 0):.0f}/100
  贪婪指数: {behavior.get('greed_index', 0):.0f}/100
  综合情绪: {behavior.get('sentiment', 0):.0f}/100
  CMF: {behavior.get('cmf', 0):+.3f}  |  MFI: {behavior.get('mfi', 50):.0f}
  主导类型: {behavior.get('dominant_type', '未知')}
"""

        news_block = ""
        if news_text:
            news_block = f"""
近期新闻面:
{news_text}
"""

        rules_ctx = build_rules_context(ts_code, current_price)
        system_prompt = build_ai_system_prompt(ts_code)

        # MC uncertainty info
        mc_info = ""
        if pred.get("mc_samples", 0) > 0:
            mc_info = f"""
MC不确定性估计 (N={pred['mc_samples']}):
  价格标准差: {pred.get('price_std', 'N/A')}
  置信度标准差: {pred.get('direction_conf_std', 'N/A')}
"""

        context = f"""分析这只A股预测结果，结合量化技术指标、实时新闻面、交易规则和深度学习模型输出：

股票代码: {ts_code}
预测方向: {pred.get('direction', '未知')}
置信度: {pred.get('direction_conf', 0):.1%}
目标价格: ¥{pred.get('target_price', '未知')}
预测区间: ¥{pred.get('price_lower', '未知')} - ¥{pred.get('price_upper', '未知')}
{mc_info}{behavior_text}{news_block}
{rules_ctx}

模型架构: PatchEmbed+BiLSTM+Transformer+MHA Pool混合模型，输入240分钟序列，66维特征(含MA/RSI/MACD/Boll/ATR/KDJ/CCI/OBV等)

请用中文提供:
1. 结合近期新闻面判断对该股的利多/利空影响
2. 基于量化指标的趋势判断和技术关键位（引用具体规则编号）
3. 综合行为分析和新闻面，判断市场情绪是否支持该预测
4. 具体风险提示（注意交易时段对下单的影响）和止损/止盈参考价位
要求简洁专业，不超过300字，规则引用标注编号。"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ]
        )
        return response.choices[0].message.content
