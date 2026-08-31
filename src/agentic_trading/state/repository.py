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
from sqlalchemy.orm import selectinload

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


async def save_bucket(session: AsyncSession, *, existing: Bucket | None = None, **fields) -> Bucket:
    """Insert a new bucket, or -- if `existing` (a previously-saved row for the same
    `(ticker, bucket_start)`, per the `uq_bucket_ticker_start` constraint) is passed --
    update it in place instead of inserting a duplicate.

    Schwab (and to a lesser extent Robinhood) keeps settling a same-day candle's
    volume for a short while after it first appears in a `pricehistory` response
    (late/consolidated-tape prints), so a bucket polled right as it opens can read
    thinner than the exact same 5-minute window reads a poll cycle later. Without
    this, `scheduler.py`'s "already polled this bucket" dedup check would silently
    discard that more-complete re-read forever; updating in place instead lets a
    later, fuller read correct an earlier provisional one -- see scheduler.py's
    `_poll_ticker` for the call site.
    """
    if existing is not None:
        for key, value in fields.items():
            setattr(existing, key, value)
        await session.flush()
        return existing
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
    stop_loss_price: float | None = None,
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
        stop_loss_price=stop_loss_price,
        max_holding_time_minutes=max_holding_time_minutes,
    )
    session.add(trade)
    await session.flush()
    return trade


async def update_trade_trailing_levels(
    session: AsyncSession, trade_id: int, *, target_sell_price: float, stop_loss_price: float
) -> Trade:
    """Phase 3: applies a one-way trailing-stop ratchet (see
    execution.invalidation.compute_trailing_stop) to an OPEN trade's resting
    levels. Callers are responsible for ensuring the new values are never less
    favorable than the current ones -- this just persists whatever it's given.
    """
    trade = await session.get(Trade, trade_id)
    if trade is None:
        raise ValueError(f"Trade {trade_id} not found")
    trade.target_sell_price = target_sell_price
    trade.stop_loss_price = stop_loss_price
    await session.flush()
    return trade


async def close_trade(
    session: AsyncSession,
    trade_id: int,
    *,
    exit_price: float,
    closed_at: datetime,
    pnl: float,
    exit_reason: str | None = None,
) -> Trade:
    trade = await session.get(Trade, trade_id)
    if trade is None:
        raise ValueError(f"Trade {trade_id} not found")
    trade.status = TradeStatus.CLOSED
    trade.exit_price = exit_price
    trade.closed_at = closed_at
    trade.pnl = pnl
    if exit_reason is not None:
        trade.exit_reason = exit_reason
    await session.flush()
    return trade


async def get_open_trades(session: AsyncSession, *, ticker: str | None = None) -> list[Trade]:
    stmt = select(Trade).where(Trade.status == TradeStatus.OPEN)
    if ticker is not None:
        stmt = stmt.where(Trade.ticker == ticker)
    return list((await session.scalars(stmt)).all())


async def get_open_trade_for_ticker(session: AsyncSession, ticker: str) -> Trade | None:
    """The single OPEN trade for `ticker`, if any, with its orders eager-loaded
    (selectinload -- see get_open_trades_missing_sell_order's docstring for why
    that's necessary with the async ORM) so callers (execution.order_manager.
    try_exit_position_early/apply_trailing_stop, scheduler._poll_ticker's
    IN_POSITION branch) can inspect trade.orders without a separate lazy-load.
    Assumes at most one OPEN trade per ticker (the default
    max_open_positions_per_ticker guardrail is 1) -- returns the first if that's
    ever violated by a looser guardrail config.
    """
    stmt = (
        select(Trade)
        .options(selectinload(Trade.orders))
        .where(Trade.ticker == ticker, Trade.status == TradeStatus.OPEN)
    )
    return (await session.scalars(stmt)).first()


async def get_open_trades_missing_sell_order(session: AsyncSession) -> list[Trade]:
    """OPEN trades whose buy leg has FILLED but which have no SELL order at all --
    covers the window where `_place_paired_sell` was attempted right after a fill
    was detected but failed (broker rejection, network blip, process restart
    between the two calls) and nothing else would ever look at this trade again,
    since it's no longer a PENDING buy order (get_open_orders won't return it) and
    isn't itself a pending SELL order either. Retried every sweep tick by
    order_manager.retry_missing_paired_sells until the paired sell finally lands.

    `selectinload(Trade.orders)` eager-loads each trade's orders in the same
    query -- necessary because the async ORM can't lazy-load a relationship
    on demand the way the sync ORM can (accessing an unloaded relationship
    attribute outside an awaited context raises MissingGreenlet).
    """
    stmt = (
        select(Trade)
        .options(selectinload(Trade.orders))
        .where(Trade.status == TradeStatus.OPEN)
    )
    trades = list((await session.scalars(stmt)).all())
    return [
        trade
        for trade in trades
        if any(o.side == OrderSide.BUY and o.status == OrderStatus.FILLED for o in trade.orders)
        and not any(o.side == OrderSide.SELL for o in trade.orders)
    ]


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


async def realized_pnl_today_all_tickers(
    session: AsyncSession, tickers: list[str], trade_date: date
) -> float:
    """Sum of realized PnL across every ticker for `trade_date` -- the daily
    drawdown circuit breaker guardrail (spec section 4) is scoped across the whole
    watchlist, not per-ticker, so callers checking that guardrail need this rather
    than a single ticker's TickerDailyState.realized_pnl.
    """
    total = 0.0
    for ticker in tickers:
        state = await get_or_create_daily_state(session, ticker, trade_date)
        total += float(state.realized_pnl or 0)
    return total
