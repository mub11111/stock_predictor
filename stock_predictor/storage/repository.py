from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from storage.database import get_session
from storage.models import Stock, DailyData, MinuteData, DailyIndicator, Prediction
from config import AppConfig


class Repository:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def _session(self) -> Session:
        return get_session(self.cfg)

    # ── Stocks ──
    def upsert_stocks(self, df: pd.DataFrame):
        s = self._session()
        try:
            with s.no_autoflush:
                for _, row in df.iterrows():
                    existing = s.query(Stock).filter(Stock.ts_code == row["ts_code"]).first()
                    if existing:
                        existing.name = row.get("name", existing.name)
                        existing.industry = row.get("industry", existing.industry)
                        existing.is_screened = row.get("is_screened", True)
                    else:
                        s.add(Stock(
                            ts_code=row["ts_code"], name=row.get("name"),
                            industry=row.get("industry"), list_date=row.get("list_date"),
                            is_screened=row.get("is_screened", False)
                        ))
            s.commit()
        finally:
            s.close()

    def sync_stocks_from_gs(self, stocks: list[tuple[str, str]]):
        """从 TDX 服务器同步股票列表 (ts_code, name)。"""
        if not stocks:
            return
        df = pd.DataFrame(stocks, columns=["ts_code", "name"])
        df["is_screened"] = True
        self.upsert_stocks(df)

    def get_screened_stocks(self) -> list[str]:
        s = self._session()
        try:
            rows = s.query(Stock.ts_code).filter(Stock.is_screened == True).all()
            return [r[0] for r in rows]
        finally:
            s.close()

    def get_all_stocks(self) -> pd.DataFrame:
        s = self._session()
        try:
            result = s.query(Stock).all()
            if not result:
                return pd.DataFrame()
            return pd.DataFrame([{
                "ts_code": r.ts_code, "name": r.name,
                "industry": r.industry, "is_screened": r.is_screened
            } for r in result])
        finally:
            s.close()

    # ── Daily Data ──
    def insert_daily(self, df: pd.DataFrame):
        s = self._session()
        try:
            for _, row in df.iterrows():
                s.merge(DailyData(**row.to_dict()))
            s.commit()
        finally:
            s.close()

    def get_daily(self, ts_code: str, start: str = None, end: str = None) -> pd.DataFrame:
        s = self._session()
        try:
            q = s.query(DailyData).filter(DailyData.ts_code == ts_code)
            if start:
                q = q.filter(DailyData.trade_date >= start)
            if end:
                q = q.filter(DailyData.trade_date <= end)
            rows = q.order_by(DailyData.trade_date).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([{c.name: getattr(r, c.name) for c in DailyData.__table__.columns} for r in rows])
        finally:
            s.close()

    # ── Minute Data ──
    def insert_minutes(self, df: pd.DataFrame):
        from data.market_rules import filter_trading_hours
        df = filter_trading_hours(df)
        if df.empty:
            return
        s = self._session()
        try:
            valid_cols = {c.name for c in MinuteData.__table__.columns}
            with s.no_autoflush:
                for _, row in df.iterrows():
                    data = {k: v for k, v in row.items() if k in valid_cols}
                    existing = s.query(MinuteData).filter(
                        MinuteData.ts_code == data["ts_code"],
                        MinuteData.trade_time == data["trade_time"],
                        MinuteData.freq == data.get("freq", "5min")
                    ).first()
                    if existing:
                        for k, v in data.items():
                            setattr(existing, k, v)
                    else:
                        s.add(MinuteData(**data))
            s.commit()
        finally:
            s.close()

    def get_minutes(self, ts_code: str, freq: str = None, start: str = None, end: str = None) -> pd.DataFrame:
        s = self._session()
        try:
            q = s.query(MinuteData).filter(MinuteData.ts_code == ts_code)
            if freq:
                q = q.filter(MinuteData.freq == freq)
            if start:
                q = q.filter(MinuteData.trade_time >= start)
            if end:
                q = q.filter(MinuteData.trade_time <= end)
            rows = q.order_by(MinuteData.trade_time).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([{c.name: getattr(r, c.name) for c in MinuteData.__table__.columns} for r in rows])
        finally:
            s.close()

    def purge_old_minutes(self, retention_days: int = 90):
        s = self._session()
        try:
            cutoff = datetime.now() - timedelta(days=retention_days)
            s.query(MinuteData).filter(MinuteData.trade_time < cutoff).delete()
            s.commit()
        finally:
            s.close()

    # ── Indicators ──
    def insert_indicators(self, df: pd.DataFrame):
        s = self._session()
        try:
            for _, row in df.iterrows():
                s.merge(DailyIndicator(**row.to_dict()))
            s.commit()
        finally:
            s.close()

    def get_indicators(self, ts_code: str, start: str = None, end: str = None) -> pd.DataFrame:
        s = self._session()
        try:
            q = s.query(DailyIndicator).filter(DailyIndicator.ts_code == ts_code)
            if start:
                q = q.filter(DailyIndicator.trade_date >= start)
            if end:
                q = q.filter(DailyIndicator.trade_date <= end)
            rows = q.order_by(DailyIndicator.trade_date).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([{c.name: getattr(r, c.name) for c in DailyIndicator.__table__.columns} for r in rows])
        finally:
            s.close()

    # ── Predictions ──
    def insert_predictions(self, df: pd.DataFrame):
        s = self._session()
        try:
            valid_cols = {c.name for c in Prediction.__table__.columns}
            with s.no_autoflush:
                for _, row in df.iterrows():
                    data = {k: v for k, v in row.items() if k in valid_cols}
                    s.merge(Prediction(**data))
            s.commit()
        finally:
            s.close()

    def get_latest_predictions(self) -> pd.DataFrame:
        s = self._session()
        try:
            q = text("""
                SELECT p.* FROM predictions p
                JOIN (SELECT ts_code, MAX(created_at) as max_t FROM predictions GROUP BY ts_code) latest
                ON p.ts_code = latest.ts_code AND p.created_at = latest.max_t
                ORDER BY p.direction_conf DESC
            """)
            result = s.execute(q).fetchall()
            if not result:
                return pd.DataFrame()
            return pd.DataFrame(result, columns=["id", "ts_code", "created_at", "direction",
                                                  "direction_conf", "target_price", "price_lower",
                                                  "price_upper", "horizon_minutes", "model_version"])
        finally:
            s.close()

    def delete_all_predictions(self):
        s = self._session()
        try:
            s.execute(text("DELETE FROM predictions"))
            s.commit()
        finally:
            s.close()

    def purge_old_predictions(self, retention_days: int = 30):
        s = self._session()
        try:
            cutoff = datetime.now() - timedelta(days=retention_days)
            s.query(Prediction).filter(Prediction.created_at < cutoff).delete()
            s.commit()
        finally:
            s.close()
