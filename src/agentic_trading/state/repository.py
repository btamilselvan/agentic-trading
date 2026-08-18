"""Persistence functions used by the rest of the app.

Nothing outside this module issues raw SQL / touches ORM sessions for reads-then-writes
of these entities — market_data, llm, execution, and api all go through here. That keeps
query logic in one place and makes the guardrails/order_manager code easy to unit test
with a fake repository.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_trading.state.models import (
    Bucket,
    LlmDecision,
    Order,
    OrderSide,
    OrderStatus,
    TickerDailyState,
    Trade,
    TradeStatus,
)

# --- Buckets ----------------------------------------------------------------


async def save_bucket(session: AsyncSession, **fields) -> Bucket:
    bucket = Bucket(**fields)
    session.add(bucket)
    await session.flush()
    return bucket


async def get_buckets_for_ticker(
    session: AsyncSession, ticker: str, since: datetime
) -> list[Bucket]:
    """The full intraday bucket history for a ticker from `since` onward, oldest first.

    This is what gets fed to the LLM as "the complete array of collected 5-minute
    metric buckets" per spec section 3.2.
    """
    stmt = (
        select(Bucket)
        .where(Bucket.ticker == ticker, Bucket.bucket_start >= since)
        .order_by(Bucket.bucket_start.asc())
    )
    return list((await session.scalars(stmt)).all())


# --- LLM decisions ------------------------------------------------------------


async def save_llm_decision(session: AsyncSession, **fields) -> LlmDecision:
    decision = LlmDecision(**fields)
    session.add(decision)
    await session.flush()
    return decision


# --- Orders ---------------------------------------------------------------------


async def create_order(session: AsyncSession, **fields) -> Order:
    order = Order(**fields)
    session.add(order)
    await session.flush()
    return order


async def update_order_status(
    session: AsyncSession,
    order_id: int,
    status: OrderStatus,
    *,
    filled_at: datetime | None = None,
    filled_price: float | None = None,
    cancelled_at: datetime | None = None,
    broker_order_id: str | None = None,
) -> Order:
    order = await session.get(Order, order_id)
    if order is None:
        raise ValueError(f"Order {order_id} not found")
    order.status = status
    if filled_at is not None:
        order.filled_at = filled_at
    if filled_price is not None:
        order.filled_price = filled_price
    if cancelled_at is not None:
        order.cancelled_at = cancelled_at
    if broker_order_id is not None:
        order.broker_order_id = broker_order_id
    await session.flush()
    return order


async def get_open_orders(
    session: AsyncSession, *, ticker: str | None = None, side: OrderSide | None = None
) -> list[Order]:
    stmt = select(Order).where(Order.status == OrderStatus.PENDING)
    if ticker is not None:
        stmt = stmt.where(Order.ticker == ticker)
    if side is not None:
        stmt = stmt.where(Order.side == side)
    return list((await session.scalars(stmt)).all())


# --- Trades --------------------------------------------------------------------


async def open_trade(
    session: AsyncSession,
    *,
    ticker: str,
    trade_date: date,
    entry_price: float,
    quantity: float,
    llm_decision_id: int | None = None,
    target_sell_price: float | None = None,
    max_holding_time_minutes: int | None = None,
) -> Trade:
    trade = Trade(
        ticker=ticker,
        trade_date=trade_date,
        status=TradeStatus.OPEN,
        entry_price=entry_price,
        quantity=quantity,
        llm_decision_id=llm_decision_id,
        target_sell_price=target_sell_price,
        max_holding_time_minutes=max_holding_time_minutes,
    )
    session.add(trade)
    await session.flush()
    return trade


async def close_trade(
    session: AsyncSession, trade_id: int, *, exit_price: float, closed_at: datetime, pnl: float
) -> Trade:
    trade = await session.get(Trade, trade_id)
    if trade is None:
        raise ValueError(f"Trade {trade_id} not found")
    trade.status = TradeStatus.CLOSED
    trade.exit_price = exit_price
    trade.closed_at = closed_at
    trade.pnl = pnl
    await session.flush()
    return trade


async def get_open_trades(session: AsyncSession, *, ticker: str | None = None) -> list[Trade]:
    stmt = select(Trade).where(Trade.status == TradeStatus.OPEN)
    if ticker is not None:
        stmt = stmt.where(Trade.ticker == ticker)
    return list((await session.scalars(stmt)).all())


# --- Per-ticker daily state (guardrail bookkeeping) -----------------------------


async def get_or_create_daily_state(
    session: AsyncSession, ticker: str, trade_date: date
) -> TickerDailyState:
    stmt = select(TickerDailyState).where(
        TickerDailyState.ticker == ticker, TickerDailyState.trade_date == trade_date
    )
    state = (await session.scalars(stmt)).one_or_none()
    if state is None:
        state = TickerDailyState(ticker=ticker, trade_date=trade_date)
        session.add(state)
        await session.flush()
    return state


async def record_trade_opened(
    session: AsyncSession, ticker: str, trade_date: date
) -> TickerDailyState:
    state = await get_or_create_daily_state(session, ticker, trade_date)
    state.open_positions_count += 1
    await session.flush()
    return state


async def record_trade_closed(
    session: AsyncSession, ticker: str, trade_date: date, pnl: float
) -> TickerDailyState:
    state = await get_or_create_daily_state(session, ticker, trade_date)
    state.open_positions_count = max(0, state.open_positions_count - 1)
    state.completed_trades_count += 1
    state.realized_pnl = float(state.realized_pnl or 0) + pnl
    await session.flush()
    return state
