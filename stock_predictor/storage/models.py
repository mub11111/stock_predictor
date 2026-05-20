from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Boolean, Text, Index
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stocks"
    id = Column(Integer, primary_key=True)
    ts_code = Column(String(16), unique=True, nullable=False)
    name = Column(String(32))
    industry = Column(String(32))
    list_date = Column(Date)
    is_screened = Column(Boolean, default=False)
    idx = Index("ix_stocks_code", ts_code)


class DailyData(Base):
    __tablename__ = "daily_data"
    id = Column(Integer, primary_key=True)
    ts_code = Column(String(16), nullable=False)
    trade_date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    idx = Index("ix_daily_code_date", "ts_code", "trade_date")


class MinuteData(Base):
    __tablename__ = "minute_data"
    id = Column(Integer, primary_key=True)
    ts_code = Column(String(16), nullable=False)
    trade_time = Column(DateTime, nullable=False)
    freq = Column(String(8), default="5min")
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    amount = Column(Float)
    idx = Index("ix_minute_code_freq_time", "ts_code", "freq", "trade_time")


class DailyIndicator(Base):
    __tablename__ = "daily_indicators"
    id = Column(Integer, primary_key=True)
    ts_code = Column(String(16), nullable=False)
    trade_date = Column(Date, nullable=False)
    ma5 = Column(Float); ma10 = Column(Float); ma20 = Column(Float); ma60 = Column(Float)
    rsi6 = Column(Float); rsi14 = Column(Float)
    macd = Column(Float); macd_signal = Column(Float); macd_hist = Column(Float)
    bb_upper = Column(Float); bb_middle = Column(Float); bb_lower = Column(Float)
    atr14 = Column(Float)
    vol_ratio = Column(Float)
    idx = Index("ix_ind_code_date", "ts_code", "trade_date")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    ts_code = Column(String(16), nullable=False)
    created_at = Column(DateTime, nullable=False)
    direction = Column(String(8))
    direction_conf = Column(Float)
    target_price = Column(Float)
    price_lower = Column(Float)
    price_upper = Column(Float)
    horizon_minutes = Column(Integer)
    model_version = Column(String(32))
    idx = Index("ix_pred_code_time", "ts_code", "created_at")


class ModelRun(Base):
    __tablename__ = "model_runs"
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    model_version = Column(String(32))
    train_loss = Column(Float)
    val_loss = Column(Float)
    notes = Column(Text)
