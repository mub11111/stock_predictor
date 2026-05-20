"""Quick test: stock screener with akshare."""
import sys
sys.path.insert(0, "D:/AI/stock_predictor")
from data.stock_screener import screen_stocks

df = screen_stocks()
if not df.empty:
    print(f"Top {len(df)} stocks:")
    print(df[["ts_code", "name", "score", "trend"]].to_string(index=False))
else:
    print("No stocks passed screening.")
