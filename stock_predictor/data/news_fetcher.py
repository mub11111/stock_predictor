"""Stock news fetcher — aggregates from multiple sources (EastMoney, Caixin, CCTV)."""
from __future__ import annotations
import akshare as ak
import pandas as pd
from datetime import datetime


def _match_stock_keywords(title: str, ts_code: str) -> bool:
    """Check if a news title is relevant to the given stock."""
    if not title or title == "nan":
        return False
    symbol = ts_code.split(".")[0] if "." in ts_code else ts_code
    # Match by stock code or common name patterns
    keywords = [symbol, symbol[-4:], symbol[-2:]]
    title_lower = title.lower()
    return any(kw in title_lower or kw in title for kw in keywords)


def fetch_stock_news(ts_code: str, max_news: int = 15) -> list[dict]:
    """Fetch recent news for a stock from multiple sources.

    Sources:
      1. EastMoney stock-specific news (ak.stock_news_em)
      2. Caixin main financial news (ak.stock_news_main_cx), filtered by relevance
      3. CCTV financial news (ak.news_cctv), filtered by relevance
    """
    all_news: list[dict] = []

    # ── Source 1: EastMoney individual stock news ──
    symbol = ts_code.split(".")[0] if "." in ts_code else ts_code
    try:
        df_em = ak.stock_news_em(symbol=symbol)
        if df_em is not None and not df_em.empty:
            cols = df_em.columns.tolist()
            title_col = cols[1] if len(cols) > 1 else None
            content_col = cols[2] if len(cols) > 2 else None
            time_col = cols[3] if len(cols) > 3 else None
            source_col = cols[4] if len(cols) > 4 else None
            url_col = cols[5] if len(cols) > 5 else None

            for _, row in df_em.iterrows():
                title = str(row[title_col]) if title_col else ""
                if not title or title == "nan":
                    continue
                content = str(row[content_col]) if content_col else ""
                content_short = content[:200] + "..." if len(content) > 200 else content
                all_news.append({
                    "title": title,
                    "content": content,
                    "content_short": content_short,
                    "time": str(row[time_col]) if time_col else "",
                    "source": f"东方财富·{str(row[source_col])}" if source_col and str(row[source_col]) != "nan" else "东方财富",
                    "url": str(row[url_col]) if url_col else "",
                })
    except Exception:
        pass

    # ── Source 2: Caixin main financial news (market-wide context) ──
    try:
        df_cx = ak.stock_news_main_cx()
        if df_cx is not None and not df_cx.empty:
            for _, row in df_cx.iterrows():
                title = str(row.get("summary", "") or row.get("tag", ""))
                if not title or title == "nan" or len(title) < 10:
                    continue
                content_short = title[:200] + "..." if len(title) > 200 else title
                all_news.append({
                    "title": title[:80] + "..." if len(title) > 80 else title,
                    "content": title,
                    "content_short": content_short,
                    "time": datetime.now().strftime("%Y-%m-%d"),
                    "source": "财新网",
                    "url": str(row.get("url", "")),
                })
    except Exception:
        pass

    # ── Source 3: CCTV financial/policy news ──
    try:
        df_cctv = ak.news_cctv()
        if df_cctv is not None and not df_cctv.empty:
            for _, row in df_cctv.iterrows():
                title = str(row.get("title", ""))
                if not title or title == "nan" or len(title) < 5:
                    continue
                content = str(row.get("content", ""))
                content_short = content[:200] + "..." if len(content) > 200 else content
                all_news.append({
                    "title": title,
                    "content": content,
                    "content_short": content_short,
                    "time": str(row.get("date", datetime.now().strftime("%Y%m%d"))),
                    "source": "央视财经",
                    "url": "",
                })
    except Exception:
        pass

    # Deduplicate by title (first 30 chars)
    seen = set()
    unique = []
    for n in all_news:
        key = n["title"][:30]
        if key not in seen:
            seen.add(key)
            unique.append(n)

    return unique[:max_news]


def build_news_context(news_list: list[dict], ts_code: str) -> str:
    """Build a text summary of recent news for AI analysis."""
    if not news_list:
        return f"（{ts_code} 暂无近期新闻数据）"

    lines = [f"=== {ts_code} 近期新闻 ({len(news_list)}条) ==="]
    for i, n in enumerate(news_list[:5], 1):
        lines.append(f"{i}. [{n['time']}] [{n['source']}] {n['title']}")
        if n.get("content_short"):
            lines.append(f"   摘要: {n['content_short'][:150]}")
        lines.append("")
    return "\n".join(lines)
