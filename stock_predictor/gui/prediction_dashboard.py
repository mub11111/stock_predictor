import webbrowser
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QGroupBox, QListWidget
)
from PyQt6.QtCore import Qt


class PredictionDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self._behavior_text_cache = ""
        self._news_cache: list[dict] = []
        # Trade section — cached HTML parts
        self._quote_html = ""
        self._pred_html = ""
        self._trade_html = ""
        self._ai_html = ""
        self._no_pred_hint = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        layout.setContentsMargins(3, 3, 3, 3)

        gb_base = """
            QGroupBox {{
                font-size: 12px; font-weight: bold;
                border: 1px solid #444; border-radius: 5px;
                margin-top: 8px; padding-top: 12px;
                background-color: #16161e;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 10px;
                padding: 0 4px; color: #999;
            }}
        """

        # ── Row 0: Model Info ──
        model_gb = QGroupBox("模型信息")
        model_gb.setStyleSheet(gb_base)
        model_layout = QVBoxLayout(model_gb)
        model_layout.setContentsMargins(6, 6, 6, 6)
        self.model_info_label = QLabel("未训练")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet(
            "font-size: 12px; color: #aaa; padding: 4px;"
        )
        model_layout.addWidget(self.model_info_label)
        layout.addWidget(model_gb, stretch=0)

        # ── Row 1: Behavior & Fingerprints ──
        analysis_gb = QGroupBox("持股人群分析")
        analysis_gb.setStyleSheet(gb_base)
        analysis_layout = QVBoxLayout(analysis_gb)
        analysis_layout.setContentsMargins(4, 6, 4, 4)
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlaceholderText("选择股票后自动分析...")
        self.analysis_text.setStyleSheet("font-size: 12px; background-color: #0d0d14; border: none;")
        analysis_layout.addWidget(self.analysis_text)
        layout.addWidget(analysis_gb, stretch=2)

        # ── Row 2: News ──
        news_gb = QGroupBox("实时新闻面")
        news_gb.setStyleSheet(gb_base)
        news_layout = QVBoxLayout(news_gb)
        news_layout.setContentsMargins(4, 6, 4, 4)
        self.news_list = QListWidget()
        self.news_list.setStyleSheet("font-size: 11px; background-color: #0d0d14; border: none;")
        self.news_list.itemDoubleClicked.connect(self._on_news_double_click)
        news_layout.addWidget(self.news_list)
        layout.addWidget(news_gb, stretch=2)

        # ── Row 3: Trading Advice (scrollable, all in one QTextEdit) ──
        trade_gb = QGroupBox("交易建议")
        trade_gb.setStyleSheet(gb_base)
        trade_layout = QVBoxLayout(trade_gb)
        trade_layout.setContentsMargins(4, 6, 4, 4)
        self.trade_text = QTextEdit()
        self.trade_text.setReadOnly(True)
        self.trade_text.setPlaceholderText("选择股票后加载行情与建议...")
        self.trade_text.setStyleSheet(
            "font-size: 12px; background-color: #0d0d14; border: none; color: #cccccc;"
        )
        trade_layout.addWidget(self.trade_text)
        layout.addWidget(trade_gb, stretch=3)

    def _rebuild_trade(self):
        """Combine cached HTML parts into the trade_text widget."""
        parts = []
        if self._no_pred_hint:
            parts.append(
                f'<div style="font-size: 12px; color: #888; padding: 8px 12px; '
                f'margin-bottom: 8px; background: #14141c; border-radius: 4px; '
                f'border-left: 2px solid #555;">{self._no_pred_hint}</div>'
            )
        if self._quote_html:
            parts.append(self._quote_html)
        if self._pred_html:
            parts.append(self._pred_html)
        if self._trade_html:
            parts.append(self._trade_html)
        if self._ai_html:
            parts.append(self._ai_html)
        self.trade_text.setHtml("".join(parts) if parts else "")

    def _rebuild_analysis(self):
        self.analysis_text.setHtml(self._behavior_text_cache)

    # ── Real-time quote (金太阳) ──
    def show_realtime_quote(self, quote: dict | None):
        if not quote:
            self._quote_html = ""
            self._rebuild_trade()
            return
        price = quote.get("price", 0)
        pre_close = quote.get("pre_close", 0)
        change = price - pre_close if pre_close else 0
        change_pct = (change / pre_close * 100) if pre_close else 0
        color = "#ff6666" if change > 0 else "#44dd44" if change < 0 else "#aaa"
        arrow = "▲" if change > 0 else "▼" if change < 0 else "—"
        vol_m = quote.get("volume", 0) / 10000
        bid1 = quote.get("bid1", 0)
        ask1 = quote.get("ask1", 0)

        parts = [
            '<div style="background: #14141c; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px;">',
            '<div style="display: flex; align-items: baseline; gap: 12px;">',
            f'<span style="font-size: 22px; font-weight: bold; color: {color};">¥{price:.2f}</span>',
            f'<span style="font-size: 15px; font-weight: bold; color: {color};">{arrow} {change:+.2f}</span>',
            f'<span style="font-size: 14px; color: {color};">{change_pct:+.2f}%</span>',
            '</div>',
        ]
        if vol_m > 0:
            parts.append(
                f'<div style="font-size: 12px; color: #888; margin-top: 6px;">'
                f'成交量: {vol_m:.0f}万手</div>'
            )
        if bid1 > 0 and ask1 > 0:
            parts.append(
                f'<div style="font-size: 11px; color: #666; margin-top: 4px;">'
                f'买一 {bid1:.2f} | 卖一 {ask1:.2f}</div>'
            )
        parts.append('</div>')
        self._quote_html = "".join(parts)
        self._rebuild_trade()

    # ── 2h Rolling prediction table ──
    def show_rolling_prediction(self, rolling_preds: list[dict] | None):
        """Display 6-step × 20min = 2h prediction timeline as a compact table."""
        if not rolling_preds:
            self._pred_html = ""
            self._rebuild_trade()
            return
        self._no_pred_hint = ""  # clear hint when prediction data arrives

        rows = []
        for rp in rolling_preds:
            step = rp.get("step", "?")
            target = rp.get("target_price", 0)
            direction = rp.get("direction", "flat")
            conf = rp.get("direction_conf", 0)
            low = rp.get("price_lower", 0)
            high = rp.get("price_upper", 0)

            if direction == "up":
                dir_color = "#ff6666"
                arrow = "▲"
            elif direction == "down":
                dir_color = "#44dd44"
                arrow = "▼"
            else:
                dir_color = "#888"
                arrow = "—"

            conf_color = "#ffd700" if conf >= 0.75 else "#aaa" if conf >= 0.6 else "#666"
            rows.append(
                f'<tr style="border-bottom: 1px solid #1a1a24;">'
                f'<td style="color: #888; padding: 4px 6px;">+{step * 20}min</td>'
                f'<td style="color: {dir_color}; font-weight: bold; padding: 4px 6px;">{arrow}</td>'
                f'<td style="color: #ddd; padding: 4px 6px;">¥{target:.2f}</td>'
                f'<td style="color: {conf_color}; font-size: 11px; padding: 4px 6px;">{conf:.0%}</td>'
                f'<td style="color: #666; font-size: 11px; padding: 4px 6px;">¥{low:.2f}</td>'
                f'<td style="color: #666; font-size: 11px; padding: 4px 6px;">¥{high:.2f}</td>'
                f'</tr>'
            )

        self._pred_html = (
            '<div style="background: #14141c; border-radius: 6px; padding: 10px 14px; '
            'margin: 10px 0;">'
            '<div style="font-size: 13px; font-weight: bold; color: #ffd700; '
            'margin-bottom: 6px;">未来2小时预测</div>'
            '<table style="font-size: 12px; border-collapse: collapse; width: 100%;">'
            '<tr style="color: #777; font-size: 11px; border-bottom: 1px solid #2a2a36;">'
            '<th align="left">时间</th><th></th><th align="left">目标价</th>'
            '<th align="left">置信</th><th align="left">下限</th><th align="left">上限</th>'
            '</tr>'
            + "".join(rows) +
            '</table></div>'
        )
        self._rebuild_trade()

    def clear_trade_advice(self):
        """Clear the trade advice section (no prediction available)."""
        self._trade_html = ""
        self._rebuild_trade()

    def clear_rolling_prediction(self):
        """Clear the rolling prediction table (no prediction available)."""
        self._pred_html = ""
        self._rebuild_trade()

    def show_no_prediction(self, message: str = ""):
        """Show a hint that the current stock has no prediction, without clearing
        existing prediction content from a previously viewed stock."""
        self._no_pred_hint = message
        self._rebuild_trade()

    # ── Trade advice ──
    def show_trade_advice(self, advice: dict):
        if not advice or not advice.get("action"):
            self.clear_trade_advice()
            return
        self._no_pred_hint = ""  # clear hint when trade advice arrives
        action = advice.get("action", "--")
        if action in ("观望", "数据不足", "就绪"):
            reason = advice.get("reason", "")
            risk = advice.get("risk_warning", "")
            plan_b = advice.get("plan_b", "")
            parts = [
                '<div style="margin: 8px 0;">',
                '<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">',
                f'<span style="font-size: 18px; font-weight: bold; color: #888; '
                f'background: #1e1e28; padding: 8px 16px; border-radius: 6px; '
                f'border: 1px solid #444;">{action}</span>',
                '</div>',
            ]
            if reason:
                parts.append(
                    f'<div style="font-size: 12px; color: #aaa; margin: 8px 0; '
                    f'padding: 8px 12px; background: #14141c; border-radius: 4px; '
                    f'border-left: 2px solid #555;">{reason}</div>'
                )
            if risk:
                parts.append(
                    f'<div style="font-size: 12px; color: #ffaa00; margin: 8px 0; '
                    f'padding: 6px 12px; background: #1a1a14; border-radius: 4px; '
                    f'border-left: 2px solid #ffaa00;">⚠ {risk}</div>'
                )
            if plan_b:
                parts.append(
                    f'<div style="font-size: 13px; color: #aaa; margin: 10px 0; '
                    f'padding: 8px 14px; background: #14141c; border-radius: 4px; '
                    f'line-height: 1.6;">'
                    f'Plan B: {plan_b}</div>'
                )
            parts.append('</div>')
            self._trade_html = "".join(parts)
            self._rebuild_trade()
            return

        direction = advice.get("direction", "")
        pct = advice.get("position_pct", 0)
        shares = advice.get("position_shares", 0)
        amount = advice.get("position_amount", 0)
        sl = advice.get("stop_loss", 0)
        tp = advice.get("take_profit", 0)
        rr = advice.get("risk_reward", 0)
        kelly = advice.get("kelly_raw", 0)
        risk = advice.get("risk_warning", "")
        plan_b = advice.get("plan_b", "")

        # Direction badge
        if direction == "up":
            badge_color = "#ff3333"
            badge_bg = "#2a1010"
            badge_border = "#cc0000"
            badge_text = "买入 ▲"
        else:
            badge_color = "#33cc44"
            badge_bg = "#102a10"
            badge_border = "#00aa00"
            badge_text = "卖出 ▼"

        # Position bar color
        pos_color = "#ff6666" if direction == "up" else "#44dd44"

        parts = [
            '<div style="margin: 8px 0;">',
            # ── Action badge ──
            '<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">',
            f'<span style="font-size: 18px; font-weight: bold; color: {badge_color}; '
            f'background: {badge_bg}; padding: 8px 16px; border-radius: 8px; '
            f'border: 2px solid {badge_border};">{badge_text}</span>',
            f'<span style="font-size: 14px; color: {pos_color}; font-weight: bold;">'
            f'{pct}% 仓位</span>',
            '</div>',

            # ── Position details card ──
            '<div style="background: #14141c; border-radius: 6px; padding: 10px 14px; '
            'margin-bottom: 10px;">',
            '<table style="font-size: 13px; border-collapse: collapse; width: 100%; table-layout: fixed;">',
            '<tr>',
            f'<td style="color: #888; text-align: left; padding: 5px 10px;">股数</td>'
            f'<td style="color: #ddd; font-weight: bold; text-align: center; padding: 5px 10px;">{shares:,}</td>',
            f'<td style="color: #888; text-align: center; padding: 5px 10px;">金额</td>'
            f'<td style="color: #ddd; font-weight: bold; text-align: right; padding: 5px 10px;">¥{amount:,.0f}</td>',
            '</tr><tr>',
            f'<td style="color: #888; text-align: left; padding: 5px 10px;">止损</td>'
            f'<td style="color: #ff6666; font-weight: bold; text-align: center; padding: 5px 10px;">¥{sl:.2f}</td>',
            f'<td style="color: #888; text-align: center; padding: 5px 10px;">止盈</td>'
            f'<td style="color: #44dd44; font-weight: bold; text-align: right; padding: 5px 10px;">¥{tp:.2f}</td>',
            '</tr><tr>',
            f'<td style="color: #888; text-align: left; padding: 5px 10px;">风险/收益</td>'
            f'<td style="color: #ffd700; font-weight: bold; text-align: center; padding: 5px 10px;">{rr}</td>',
            f'<td style="color: #888; text-align: center; padding: 5px 10px;">凯利比率</td>'
            f'<td style="color: {"#ff6666" if kelly > 0.05 else "#44dd44" if kelly < -0.05 else "#ffaa00"}; font-weight: bold; text-align: right; padding: 5px 10px;">{kelly:+.3f}</td>',
            '</tr>',
            '</table></div>',
        ]

        # Risk warning
        if risk:
            parts.append(
                f'<div style="font-size: 12px; color: #ffaa00; margin: 8px 0; '
                f'padding: 6px 12px; background: #1a1a14; border-radius: 4px; '
                f'border-left: 3px solid #ffaa00;">⚠ {risk}</div>'
            )
        if plan_b:
            parts.append(
                f'<div style="font-size: 13px; color: #aaa; margin: 10px 0; '
                f'padding: 8px 14px; background: #14141c; border-radius: 4px; '
                f'line-height: 1.6; border-left: 3px solid #555;">Plan B: {plan_b}</div>'
            )
        parts.append('</div>')

        self._trade_html = "".join(parts)
        self._rebuild_trade()

    # ── AI / CRC / News impact → all in _ai_html ──
    def show_analysis(self, text: str):
        self._ai_html = f'<div style="font-size: 11px; color: #bbb; margin-top: 6px; white-space: pre-wrap;">{text}</div>'
        self._rebuild_trade()

    def show_correction(self, correction: dict | None, history: list[dict]):
        if correction is None:
            return
        corrected = correction.get("corrected", False)
        reason = correction.get("reason", "")
        err = correction.get("error_pct", 0)
        w = correction.get("weight", 0)
        count = correction.get("correction_count", 0)
        avg = correction.get("avg_correction", 0)
        revert_count = correction.get("revert_count", 0)
        sign_match = correction.get("sign_match", True)

        lines = ['<div style="font-size: 10px; color: #888; margin-top: 4px;">']
        if corrected:
            lines.append(f"[CRC修正 #{count}] 误差: {err:+.4%} | 权重: {w:.2f} | 累计: {avg:.4%}<br>")
            pre_loss = correction.get("pre_loss")
            post_loss = correction.get("post_loss")
            improvement = correction.get("improvement")
            if pre_loss is not None and post_loss is not None:
                lines.append(f"损失: {pre_loss:.6f} → {post_loss:.6f} | 改善: {improvement:+.6f}<br>" if improvement else "")
            if not sign_match:
                lines.append("方向不匹配 — 权重×0.3<br>")
            if revert_count > 0:
                accept_rate = count / max(count + revert_count, 1)
                lines.append(f"接受率: {accept_rate:.0%} ({count}/{count + revert_count})<br>")
        else:
            if reason == "negligible_error":
                lines.append(f"[CRC跳过] 误差 {err:+.4%} 可忽略<br>")
            elif reason == "validation_loss_increased":
                lines.append(f"[CRC回滚 #{count}] 误差: {err:+.4%} | 权重: {w:.2f}<br>")
                pre_loss = correction.get("pre_loss", 0)
                post_loss = correction.get("post_loss", 0)
                lines.append(f"损失上升: {pre_loss:.6f} → {post_loss:.6f} | 回滚: {revert_count}次<br>")
            else:
                lines.append(f"[CRC跳过] {reason}<br>")
        lines.append('</div>')
        self._ai_html = "".join(lines)
        self._rebuild_trade()

    def show_news_impact(self, impacts: list[dict]):
        if not impacts:
            return
        lines = [
            '<div style="font-size: 12px; color: #aaa; margin-top: 6px; line-height: 1.6;">'
            '═══ 新闻影响力 ═══<br>'
        ]
        for imp in impacts[:5]:
            direction = "利好 ▲" if imp.get("direction") == "positive" else "利空 ▼"
            lines.append(
                f"• [{imp.get('level', '')}] {imp.get('keyword', '')}"
                f" | 强度:{imp.get('magnitude', 0):.0%}"
                f" | 持续:{imp.get('duration_min', 0) / 60:.1f}h"
                f" {direction}<br>"
            )
        lines.append('</div>')
        self._ai_html = "".join(lines)
        self._rebuild_trade()

    def show_news_bias(self, bias: dict):
        active_count = bias.get("active_count", 0)
        if active_count == 0:
            return
        net_bias = bias.get("bias", 0)
        direction = "偏多 ▲" if net_bias > 0 else "偏空 ▼" if net_bias < 0 else "中性"
        text = (
            f'<div style="font-size: 10px; color: #aaa; margin-top: 4px;">'
            f'[新闻影响] {direction} | 净偏差:{net_bias:+.3f} | 活跃:{active_count}条'
            f'</div>'
        )
        if "新闻影响" not in self._ai_html:
            self._ai_html = self._ai_html + text if self._ai_html else text
        self._rebuild_trade()

    def show_news_detail(self, news_item: dict):
        if not news_item:
            return
        text = (
            f'<div style="font-size: 11px; color: #ccc; margin-top: 4px;">'
            f'<b>{news_item.get("title", "")}</b><br>'
            f'<span style="color: #888;">来源: {news_item.get("source", "")}'
            f' | 时间: {news_item.get("time", "")}</span><br><br>'
            f'{news_item.get("content", "")[:800]}'
            f'</div>'
        )
        self._ai_html = text
        self._rebuild_trade()

    # ── Model info ──
    def show_model_info(self, entries: list | None):
        valid = [e for e in (entries or []) if e.model_type != "未知"]
        if not valid:
            self.model_info_label.setText(
                '<span style="color: #777;">未训练</span>  '
                '<span style="font-size: 10px;">点击 [训练] 按钮训练模型</span>'
            )
            return
        lines = []
        for e in valid:
            acc_str = f"{e.macro_acc:.1%}" if e.macro_acc > 0 else f"{e.val_acc:.1%}" if e.val_acc > 0 else "--"
            color = "#ffd700" if e.macro_acc > 0.5 else "#ffaa66" if e.macro_acc > 0.4 else "#ff6b35"
            lines.append(
                f'<span style="color: {color}; font-weight: bold;">● {e.model_type}</span>'
                f'<span style="color: #aaa;"> — 准确率 {acc_str}</span>'
            )
            if e.best_model_name:
                lines.append(
                    f'<span style="font-size: 10px; color: #ffd700; margin-left: 12px;">'
                    f'🏆 最佳: {e.best_model_name}</span>'
                )
            if e.all_accuracies:
                acc_parts = []
                for name, acc in e.all_accuracies.items():
                    acc_parts.append(f"{name.split('(')[0].strip()}: {acc:.1%}")
                lines.append(
                    f'<span style="font-size: 10px; color: #888; margin-left: 12px;">'
                    f'{",  ".join(acc_parts)}</span>'
                )
        self.model_info_label.setText("<br>".join(lines))

    # ── Behavior ──
    def show_behavior(self, behavior: dict):
        if not behavior or not behavior.get("data_sufficient", False):
            self._behavior_text_cache = (
                '<div style="color: #888; font-size: 12px; padding: 8px;">'
                '数据不足，请先刷新分钟数据（至少需要30根K线）</div>'
            )
            self._rebuild_analysis()
            return

        inst_net = behavior.get("inst_net", 0)
        retail_net = behavior.get("retail_net", 0)
        inst_buy = behavior.get("inst_buy_total", 0)
        inst_sell = behavior.get("inst_sell_total", 0)
        retail_buy = behavior.get("retail_buy_total", 0)
        retail_sell = behavior.get("retail_sell_total", 0)
        panic = behavior.get("panic_index", 0)
        greed = behavior.get("greed_index", 0)
        sentiment = behavior.get("sentiment", 0)
        herding = behavior.get("herding_index", 0)
        cmf = behavior.get("cmf", 0)
        mfi = behavior.get("mfi", 50)
        vwap_dev = behavior.get("vwap_dev", 0)
        obv_dev = behavior.get("obv_dev", 0)
        regime = behavior.get("regime", "未知")
        dominant = behavior.get("dominant_type", "未知")

        total_vol = inst_buy + inst_sell + retail_buy + retail_sell
        if total_vol > 0:
            inst_pct = (inst_buy + inst_sell) / total_vol * 100
            retail_pct = (retail_buy + retail_sell) / total_vol * 100
        else:
            inst_pct = retail_pct = 50

        def _fmt_vol(v):
            if v >= 1e8:
                return f"{v / 1e8:.2f}亿"
            elif v >= 1e4:
                return f"{v / 1e4:.0f}万"
            else:
                return f"{v:.0f}"

        def _sig(v, pos="#33cc44", neg="#ff3333", zero="#888"):
            if v > 0: return pos
            if v < 0: return neg
            return zero

        h = []

        # ── Title ──
        h.append(
            '<div style="font-size: 14px; font-weight: bold; color: #ffd700; '
            'margin-bottom: 12px; padding: 4px 2px 8px 2px; border-bottom: 1px solid #333;">'
            '📊 资金流向分析</div>'
        )

        # ── Multi-period net institutional flow ──
        weekly = behavior.get("weekly")
        monthly = behavior.get("monthly")

        def _period_row(data: dict | None, label: str) -> str:
            if data is None:
                return (f'<tr>'
                        f'<td style="color: #888; padding: 4px 8px;">{label}</td>'
                        f'<td style="color: #555; padding: 4px 8px;" colspan="3">数据不足</td>'
                        f'</tr>')
            inst_b = data.get(f"{label}主力买入量", 0) + data.get(f"{label}机构买入量", 0)
            inst_s = data.get(f"{label}主力卖出量", 0) + data.get(f"{label}机构卖出量", 0)
            net = inst_b - inst_s
            color = "#ff6666" if net > 0 else "#44dd44" if net < 0 else "#888"
            sign = "+" if net > 0 else ""
            return (f'<tr>'
                    f'<td style="color: #ddd; padding: 4px 8px;">{label}</td>'
                    f'<td style="color: #ff6666; padding: 4px 8px;">买 {_fmt_vol(inst_b)}</td>'
                    f'<td style="color: #44dd44; padding: 4px 8px;">卖 {_fmt_vol(inst_s)}</td>'
                    f'<td style="color: {color}; font-weight: bold; padding: 4px 8px;">净 {sign}{_fmt_vol(net)}</td>'
                    f'</tr>')

        daily_net = inst_buy - inst_sell
        daily_color = _sig(daily_net)
        daily_sign = "+" if daily_net > 0 else ""

        h.append(
            '<div style="font-size: 12px; color: #aaa; margin: 10px 0 4px 0;">多周期净流向</div>'
            '<table style="font-size: 12px; border-collapse: collapse; width: 100%; margin-bottom: 10px;">'
            f'<tr>'
            f'<td style="color: #ddd; padding: 4px 8px;">当日</td>'
            f'<td style="color: #ff6666; padding: 4px 8px;">买 {_fmt_vol(inst_buy)}</td>'
            f'<td style="color: #44dd44; padding: 4px 8px;">卖 {_fmt_vol(inst_sell)}</td>'
            f'<td style="color: {daily_color}; font-weight: bold; padding: 4px 8px;">净 {daily_sign}{_fmt_vol(daily_net)}</td>'
            f'</tr>'
            + _period_row(weekly, "本周")
            + _period_row(monthly, "30日")
            + '</table>'
        )

        # ── Section 1: 当日明细 ──
        h.append('<div style="font-size: 12px; color: #aaa; margin: 10px 0 4px 0;">当日明细</div>')

        # Daily flow blocks in a mini table
        flow_keys = ["主力", "机构", "大户", "小散户"]
        flow_rows = []
        for key in flow_keys:
            buy_key = f"当日{key}买入量"
            sell_key = f"当日{key}卖出量"
            bv = behavior.get(buy_key, 0)
            sv = behavior.get(sell_key, 0)
            net = bv - sv
            net_color = _sig(net)
            net_sign = "+" if net > 0 else ""
            flow_rows.append(
                f'<tr>'
                f'<td style="color: #ddd; width: 60px; padding: 3px 0;">{key}</td>'
                f'<td style="color: #ff6666; width: 70px; text-align: right; padding: 3px 0;">买 {_fmt_vol(bv)}</td>'
                f'<td style="color: #44dd44; width: 70px; text-align: right; padding: 3px 0;">卖 {_fmt_vol(sv)}</td>'
                f'<td style="color: {net_color}; width: 75px; text-align: right; font-weight: bold; padding: 3px 0;">'
                f'净 {net_sign}{_fmt_vol(net)}</td>'
                f'</tr>'
            )
        h.append(
            '<table style="font-size: 12px; border-collapse: collapse; width: 100%; margin-bottom: 10px;">'
            + "".join(flow_rows) +
            '</table>'
        )

        # ── Section 2: 买卖力量对比 ──
        h.append('<div style="font-size: 12px; color: #aaa; margin: 12px 0 4px 0;">买卖力量对比</div>')

        # Visual bar for inst vs retail
        h.append(
            '<div style="display: flex; align-items: center; margin: 6px 0; font-size: 12px;">'
            f'<span style="color: #ff6666; width: 50px;">机构</span>'
            f'<span style="background: #ff6666; height: 10px; width: {inst_pct}%; '
            f'display: inline-block; border-radius: 2px; min-width: 1%;"></span>'
            f'<span style="color: #ff6666; margin-left: 6px;">{inst_pct:.0f}%</span>'
            '</div>'
        )
        h.append(
            '<div style="display: flex; align-items: center; margin: 6px 0; font-size: 12px;">'
            f'<span style="color: #4fc3f7; width: 50px;">散户</span>'
            f'<span style="background: #4fc3f7; height: 10px; width: {retail_pct}%; '
            f'display: inline-block; border-radius: 2px; min-width: 1%;"></span>'
            f'<span style="color: #4fc3f7; margin-left: 6px;">{retail_pct:.0f}%</span>'
            '</div>'
        )

        net_color = _sig(inst_net)
        h.append(
            f'<div style="font-size: 12px; margin: 6px 0;">'
            f'<span style="color: #888;">机构净流向: </span>'
            f'<span style="color: {net_color}; font-weight: bold;">{inst_net:+.1%}</span>'
            f'<span style="color: #888; margin-left: 12px;">散户净流向: </span>'
            f'<span style="color: {_sig(retail_net)}; font-weight: bold;">{retail_net:+.1%}</span>'
            f'</div>'
        )

        # ── Section 3: 情绪指标 ──
        h.append('<div style="font-size: 12px; color: #aaa; margin: 14px 0 4px 0;">情绪指标</div>')

        # Panic / Greed gauge
        panic_color = "#33cc44" if panic < 30 else "#ffaa00" if panic < 60 else "#ff3333"
        greed_color = "#ff3333" if greed > 70 else "#ffaa00" if greed > 40 else "#33cc44"
        h.append(
            '<table style="font-size: 12px; border-collapse: collapse; width: 100%; margin-bottom: 4px;">'
            f'<tr>'
            f'<td style="color: #888; width: 55px; padding: 3px 0;">恐慌</td>'
            f'<td style="width: 80px; padding: 3px 0;"><span style="background: {panic_color}; height: 5px; '
            f'display: inline-block; border-radius: 2px; width: {panic}%; min-width: 2%;"></span></td>'
            f'<td style="color: {panic_color}; font-weight: bold; padding: 3px 0;">{panic:.0f}/100</td>'
            f'</tr>'
            f'<tr>'
            f'<td style="color: #888; padding: 3px 0;">贪婪</td>'
            f'<td style="padding: 3px 0;"><span style="background: {greed_color}; height: 5px; '
            f'display: inline-block; border-radius: 2px; width: {greed}%; min-width: 2%;"></span></td>'
            f'<td style="color: {greed_color}; font-weight: bold; padding: 3px 0;">{greed:.0f}/100</td>'
            f'</tr>'
            f'<tr>'
            f'<td style="color: #888; padding: 3px 0;">羊群</td>'
            f'<td colspan="2" style="color: {_sig(-herding + 0.3, "#33cc44", "#ffaa00")}; padding: 3px 0;">{herding:.2f} '
            f'{"(跟风强)" if herding > 0.6 else "(理性)" if herding < 0.3 else "(中性)"}</td>'
            f'</tr>'
            f'<tr>'
            f'<td style="color: #888; padding: 3px 0;">综合</td>'
            f'<td colspan="2" style="color: {_sig(sentiment)}; font-weight: bold; padding: 3px 0;">'
            f'{sentiment:+.0f} {"(偏多)" if sentiment > 20 else "(偏空)" if sentiment < -20 else "(中性)"}</td>'
            f'</tr>'
            '</table>'
        )

        # ── Section 4: 市场状态 ──
        h.append('<div style="font-size: 12px; color: #aaa; margin: 14px 0 4px 0;">市场状态</div>')
        regime_color = "#33cc44" if "牛" in regime or "趋势" in regime else \
                       "#ff3333" if "熊" in regime or "震荡下跌" in regime else "#ffaa00"
        h.append(
            f'<div style="font-size: 13px; margin: 4px 0;">'
            f'<span style="color: {regime_color}; font-weight: bold;">{regime}</span>'
            f'<span style="color: #888; margin-left: 10px;">主导: </span>'
            f'<span style="color: #ddd;">{dominant}</span>'
            f'</div>'
        )

        # ── Section 5: 技术面资金指标 ──
        h.append('<div style="font-size: 12px; color: #aaa; margin: 14px 0 4px 0;">技术面资金指标</div>')
        cmf_color = _sig(cmf - 0.05, "#ff6666", "#44dd44", "#ffaa00")
        mfi_color = "#ff3333" if mfi > 80 else "#33cc44" if mfi < 20 else "#ffaa00"
        vwap_color = _sig(vwap_dev - 0.002, "#ff6666", "#44dd44")

        h.append(
            '<table style="font-size: 12px; border-collapse: collapse; width: 100%;">'
            f'<tr>'
            f'<td style="color: #888; padding: 4px 8px;">CMF<small> (蔡金资金流)</small></td>'
            f'<td style="color: {cmf_color}; font-weight: bold; padding: 4px 8px;">{cmf:+.3f}</td>'
            f'<td style="color: #666; font-size: 11px; padding: 4px 8px;">'
            f'{"吸筹" if cmf > 0.05 else "派发" if cmf < -0.05 else "中性"}</td>'
            f'</tr>'
            f'<tr>'
            f'<td style="color: #888; padding: 4px 8px;">MFI<small> (资金流量指数)</small></td>'
            f'<td style="color: {mfi_color}; font-weight: bold; padding: 4px 8px;">{mfi:.0f}</td>'
            f'<td style="color: #666; font-size: 11px; padding: 4px 8px;">'
            f'{"超买" if mfi > 80 else "超卖" if mfi < 20 else "正常"}</td>'
            f'</tr>'
            f'<tr>'
            f'<td style="color: #888; padding: 4px 8px;">VWAP<small> (均价偏离)</small></td>'
            f'<td style="color: {vwap_color}; font-weight: bold; padding: 4px 8px;">{vwap_dev:+.2%}</td>'
            f'<td style="color: #666; font-size: 11px; padding: 4px 8px;">'
            f'{"高于均价" if vwap_dev > 0.002 else "低于均价" if vwap_dev < -0.002 else "均价附近"}</td>'
            f'</tr>'
            f'<tr>'
            f'<td style="color: #888; padding: 4px 8px;">OBV<small> (能量潮偏离)</small></td>'
            f'<td style="color: {_sig(obv_dev - 0.02, "#ff6666", "#44dd44", "#ffaa00")}; font-weight: bold; padding: 4px 8px;">{obv_dev:+.2%}</td>'
            f'<td style="color: #666; font-size: 11px; padding: 4px 8px;">'
            f'{"资金流入" if obv_dev > 0.05 else "资金流出" if obv_dev < -0.05 else "中性"}</td>'
            f'</tr>'
            '</table>'
        )

        self._behavior_text_cache = "".join(h)
        self._rebuild_analysis()

    # ── News ──
    def show_news(self, news_list: list[dict]):
        self._news_cache = news_list
        self.news_list.clear()
        if not news_list:
            self.news_list.addItem("暂无相关新闻")
            return
        for n in news_list[:10]:
            title = n.get("title", "")
            time_str = str(n.get("time", ""))[:10]
            self.news_list.addItem(f"[{time_str}] {title}")

    def _on_news_double_click(self, item):
        idx = self.news_list.row(item)
        if idx < len(self._news_cache):
            news = self._news_cache[idx]
            self.show_news_detail(news)
            url = news.get("url", "")
            if url and url != "nan":
                webbrowser.open(url)

    def show_prediction(self, pred: dict):
        pass

    def get_news_context(self) -> str:
        if not self._news_cache:
            return ""
        lines = []
        for n in self._news_cache[:5]:
            lines.append(f"- [{n.get('time', '')}] {n.get('title', '')}")
        return "\n".join(lines)
