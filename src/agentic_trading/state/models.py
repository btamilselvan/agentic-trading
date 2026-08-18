"""SQLAlchemy ORM models — the audit trail for buckets, LLM decisions, orders, and trades.

Kept as plain SQLAlchemy (no Supabase-specific column types/features) so the schema
works unmodified against any Postgres-compatible database.
"""

from __future__ import annotations

import enum
from datetime import UTC, date, datetime

from sqlalchemy import DateTime, ForeignKey, Numeric, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


# All timestamp columns are timezone-aware (TIMESTAMPTZ) and default to this.
_TZDateTime = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class OrderSide(enum.StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(enum.StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


class TradingModeEnum(enum.StrEnum):
    DRY_RUN = "DRY_RUN"
    LIVE = "LIVE"


class TradeStatus(enum.StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class DecisionType(enum.StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"


class Bucket(Base):
    """A single 5-minute microstructure snapshot for one ticker."""

    __tablename__ = "buckets"
    __table_args__ = (UniqueConstraint("ticker", "bucket_start", name="uq_bucket_ticker_start"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(index=True)
    bucket_start: Mapped[datetime] = mapped_column(_TZDateTime, index=True)
    bucket_end: Mapped[datetime] = mapped_column(_TZDateTime)

    open: Mapped[float] = mapped_column(Numeric(12, 4))
    high: Mapped[float] = mapped_column(Numeric(12, 4))
    low: Mapped[float] = mapped_column(Numeric(12, 4))
    close: Mapped[float] = mapped_column(Numeric(12, 4))
    volume: Mapped[int]

    est_buy_volume: Mapped[int]
    est_sell_volume: Mapped[int]

    bid_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    ask_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    bid_size: Mapped[int | None] = mapped_column(nullable=True)
    ask_size: Mapped[int | None] = mapped_column(nullable=True)
    spread: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    book_imbalance: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)

    candle_body: Mapped[float] = mapped_column(Numeric(12, 4))
    upper_wick: Mapped[float] = mapped_column(Numeric(12, 4))
    lower_wick: Mapped[float] = mapped_column(Numeric(12, 4))
    rvol: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    vwap: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)

    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_utcnow)

    decisions: Mapped[list[LlmDecision]] = relationship(back_populates="bucket")


class LlmDecision(Base):
    """One LLM evaluation for a ticker at a point in time, with the full audit trail."""

    __tablename__ = "llm_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(index=True)
    bucket_id: Mapped[int | None] = mapped_column(ForeignKey("buckets.id"), nullable=True)

    prompt: Mapped[str]
    raw_response: Mapped[str]

    decision: Mapped[DecisionType] = mapped_column(SAEnum(DecisionType))
    confidence_score: Mapped[float] = mapped_column(Numeric(4, 3))
    buy_limit_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    target_sell_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    max_holding_time_minutes: Mapped[int | None] = mapped_column(nullable=True)
    pattern_reasoning: Mapped[str | None] = mapped_column(nullable=True)

    acted_on: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_utcnow, index=True)

    bucket: Mapped[Bucket | None] = relationship(back_populates="decisions")


class Order(Base):
    """A single buy or sell limit order submitted through the broker execution client."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(index=True)
    side: Mapped[OrderSide] = mapped_column(SAEnum(OrderSide))
    limit_price: Mapped[float] = mapped_column(Numeric(12, 4))
    quantity: Mapped[float] = mapped_column(Numeric(14, 6))
    status: Mapped[OrderStatus] = mapped_column(SAEnum(OrderStatus), default=OrderStatus.PENDING)
    mode: Mapped[TradingModeEnum] = mapped_column(SAEnum(TradingModeEnum))

    broker_order_id: Mapped[str | None] = mapped_column(nullable=True)
    trade_id: Mapped[int | None] = mapped_column(ForeignKey("trades.id"), nullable=True)

    submitted_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_utcnow)
    filled_at: Mapped[datetime | None] = mapped_column(_TZDateTime, nullable=True)
    filled_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(_TZDateTime, nullable=True)

    trade: Mapped[Trade] = relationship(back_populates="orders", foreign_keys=[trade_id])


class Trade(Base):
    """A round-trip (buy -> sell) position lifecycle for one ticker."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(index=True)
    trade_date: Mapped[date] = mapped_column(index=True)
    status: Mapped[TradeStatus] = mapped_column(SAEnum(TradeStatus), default=TradeStatus.OPEN)

    # Carried over from the LLM decision that opened this trade, so order_manager
    # knows where to place the paired sell and when to force-exit without having to
    # re-query llm_decisions on every poll.
    llm_decision_id: Mapped[int | None] = mapped_column(
        ForeignKey("llm_decisions.id"), nullable=True
    )
    target_sell_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    max_holding_time_minutes: Mapped[int | None] = mapped_column(nullable=True)

    entry_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    quantity: Mapped[float | None] = mapped_column(Numeric(14, 6), nullable=True)
    pnl: Mapped[float | None] = mapped_column(Numeric(14, 4), nullable=True)

    opened_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(_TZDateTime, nullable=True)

    orders: Mapped[list[Order]] = relationship(
        back_populates="trade", foreign_keys=[Order.trade_id]
    )


class TickerDailyState(Base):
    """Per-ticker, per-day counters used to enforce the guardrails in spec section 4."""

    __tablename__ = "ticker_daily_state"
    __table_args__ = (UniqueConstraint("ticker", "trade_date", name="uq_ticker_daily_state"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(index=True)
    trade_date: Mapped[date] = mapped_column(index=True)

    completed_trades_count: Mapped[int] = mapped_column(default=0)
    open_positions_count: Mapped[int] = mapped_column(default=0)
    realized_pnl: Mapped[float] = mapped_column(Numeric(14, 4), default=0)

    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_utcnow, onupdate=_utcnow)
