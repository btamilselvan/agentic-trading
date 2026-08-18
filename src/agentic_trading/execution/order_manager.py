"""Buy -> sell order lifecycle: guardrail-gated entry, fill detection via
position-quantity polling, paired sell placement, order-timeout cancellation, and
EOD liquidation.

This is the one place that decides whether an LLM BUY decision actually becomes a
real order -- the LLM saying BUY with a high confidence_score is necessary but never
sufficient; every entry path here independently re-checks the guardrails (spec
section 4), regardless of what the LLM claimed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_trading.alerts.base import Notifier, NullNotifier
from agentic_trading.config import Settings
from agentic_trading.execution.broker_mcp_client import (
    BrokerExecutionClient,
    OrderReview,
    PlacedOrder,
)
from agentic_trading.execution.guardrails import evaluate_buy_guardrails, is_order_timed_out
from agentic_trading.llm.schema import TradeDecision
from agentic_trading.state import repository as repo
from agentic_trading.state.models import Order, OrderSide, OrderStatus, Trade, TradingModeEnum

logger = logging.getLogger(__name__)

_NULL_NOTIFIER = NullNotifier()


class DryRunBrokerClient:
    """BrokerExecutionClient for MODE=DRY_RUN -- simulates fills instantly at the
    requested limit price and tracks open quantity in-memory. Exercises the exact
    same order_manager state machine and DB writes as LIVE, just with zero network
    calls to the MCP and zero real money at risk.
    """

    def __init__(self) -> None:
        self._positions: dict[str, float] = {}
        self._next_order_id = 1

    async def get_open_position_quantity(self, ticker: str) -> float:
        return self._positions.get(ticker, 0.0)

    async def review_order(
        self, *, ticker: str, side: str, quantity: float, limit_price: float
    ) -> OrderReview:
        return OrderReview(warnings=[], estimated_price=limit_price)

    async def place_order(
        self, *, ticker: str, side: str, quantity: float, limit_price: float
    ) -> PlacedOrder:
        order_id = f"DRYRUN-{self._next_order_id}"
        self._next_order_id += 1
        delta = quantity if side == "buy" else -quantity
        self._positions[ticker] = self._positions.get(ticker, 0.0) + delta
        return PlacedOrder(broker_order_id=order_id, status="filled", fill_price=limit_price)

    async def cancel_order(self, broker_order_id: str) -> None:
        return None


def compute_order_quantity(limit_price: float, max_capital_per_trade_usd: float) -> float:
    """Position sizing: spend up to the per-trade capital cap, in whole shares."""
    if limit_price <= 0:
        return 0.0
    return float(math.floor(max_capital_per_trade_usd / limit_price))


@dataclass
class TradeEntryOutcome:
    opened: bool
    reason: str | None = None
    trade_id: int | None = None
    order_id: int | None = None


async def _mark_order_filled(
    session: AsyncSession, order: Order, fill_price: float, notifier: Notifier
) -> None:
    await repo.update_order_status(
        session,
        order.id,
        OrderStatus.FILLED,
        filled_at=datetime.now(UTC),
        filled_price=fill_price,
    )
    await notifier.notify(
        "Order filled",
        {
            "ticker": order.ticker,
            "side": order.side.value,
            "fill_price": fill_price,
            "quantity": float(order.quantity),
            "mode": order.mode.value,
        },
    )


async def try_enter_position(
    session: AsyncSession,
    broker: BrokerExecutionClient,
    *,
    ticker: str,
    decision: TradeDecision,
    llm_decision_id: int,
    settings: Settings,
    today: date,
    realized_pnl_today_all_tickers: float,
    notifier: Notifier | None = None,
) -> TradeEntryOutcome:
    """Attempt to open a position from a BUY decision. Callers should already have
    checked `decision.confidence_score >= settings.confidence_threshold` -- this
    function's job is everything after that: guardrails, sizing, and execution.
    """
    notifier = notifier or _NULL_NOTIFIER
    if decision.decision != "BUY":
        return TradeEntryOutcome(opened=False, reason="decision is HOLD")

    daily_state = await repo.get_or_create_daily_state(session, ticker, today)
    quantity = compute_order_quantity(decision.buy_limit_price, settings.max_capital_per_trade_usd)
    if quantity <= 0:
        return TradeEntryOutcome(opened=False, reason="computed order quantity is zero")

    order_notional = quantity * decision.buy_limit_price
    guardrail_result = evaluate_buy_guardrails(
        open_positions_count=daily_state.open_positions_count,
        max_open_positions_per_ticker=settings.max_open_positions_per_ticker,
        completed_trades_today=daily_state.completed_trades_count,
        daily_trade_cap_per_ticker=settings.daily_trade_cap_per_ticker,
        order_notional=order_notional,
        max_capital_per_trade_usd=settings.max_capital_per_trade_usd,
        realized_pnl_today_all_tickers=realized_pnl_today_all_tickers,
        max_daily_drawdown_usd=settings.max_daily_drawdown_usd,
    )
    if not guardrail_result.allowed:
        logger.info("Guardrail blocked BUY for %s: %s", ticker, guardrail_result.reason)
        return TradeEntryOutcome(opened=False, reason=guardrail_result.reason)

    review = await broker.review_order(
        ticker=ticker, side="buy", quantity=quantity, limit_price=decision.buy_limit_price
    )
    if review.warnings:
        logger.warning("Broker review warnings for %s BUY: %s", ticker, review.warnings)

    placed = await broker.place_order(
        ticker=ticker, side="buy", quantity=quantity, limit_price=decision.buy_limit_price
    )

    order = await repo.create_order(
        session,
        ticker=ticker,
        side=OrderSide.BUY,
        limit_price=decision.buy_limit_price,
        quantity=quantity,
        mode=TradingModeEnum(settings.mode.value),
        broker_order_id=placed.broker_order_id,
    )
    trade = await repo.open_trade(
        session,
        ticker=ticker,
        trade_date=today,
        entry_price=decision.buy_limit_price,
        quantity=quantity,
        llm_decision_id=llm_decision_id,
        target_sell_price=decision.target_sell_price,
        max_holding_time_minutes=decision.max_holding_time_minutes,
    )
    order.trade_id = trade.id
    await session.flush()
    await repo.record_trade_opened(session, ticker, today)

    if placed.status == "filled":
        fill_price = placed.fill_price or decision.buy_limit_price
        await _mark_order_filled(session, order, fill_price, notifier)
        await _place_paired_sell(session, broker, order, trade, notifier)

    return TradeEntryOutcome(opened=True, trade_id=trade.id, order_id=order.id)


async def _place_paired_sell(
    session: AsyncSession,
    broker: BrokerExecutionClient,
    buy_order: Order,
    trade: Trade,
    notifier: Notifier,
) -> None:
    """Spec 3.3: once a buy fills, immediately place the matching limit sell at the
    trade's target_sell_price."""
    if trade.target_sell_price is None:
        logger.error(
            "Cannot place paired sell for trade %s -- no target_sell_price recorded", trade.id
        )
        return

    placed = await broker.place_order(
        ticker=buy_order.ticker,
        side="sell",
        quantity=float(buy_order.quantity),
        limit_price=float(trade.target_sell_price),
    )
    sell_order = await repo.create_order(
        session,
        ticker=buy_order.ticker,
        side=OrderSide.SELL,
        limit_price=float(trade.target_sell_price),
        quantity=float(buy_order.quantity),
        mode=buy_order.mode,
        broker_order_id=placed.broker_order_id,
        trade_id=trade.id,
    )
    if placed.status == "filled":
        fill_price = placed.fill_price or float(trade.target_sell_price)
        await _close_trade_from_sell(session, trade, sell_order, fill_price, notifier)


async def _close_trade_from_sell(
    session: AsyncSession, trade: Trade, sell_order: Order, fill_price: float, notifier: Notifier
) -> None:
    await _mark_order_filled(session, sell_order, fill_price, notifier)
    pnl = (fill_price - float(trade.entry_price)) * float(trade.quantity)
    await repo.close_trade(
        session, trade.id, exit_price=fill_price, closed_at=datetime.now(UTC), pnl=pnl
    )
    await repo.record_trade_closed(session, trade.ticker, trade.trade_date, pnl=pnl)
    await notifier.notify(
        "Trade closed",
        {
            "ticker": trade.ticker,
            "entry_price": float(trade.entry_price),
            "exit_price": fill_price,
            "pnl": pnl,
        },
    )


async def poll_pending_buy_orders(
    session: AsyncSession,
    broker: BrokerExecutionClient,
    *,
    now: datetime,
    order_timeout_minutes: int,
    notifier: Notifier | None = None,
) -> None:
    """Runs every poll cycle: cancels unfilled BUY orders past the timeout guardrail
    (spec 4: Order Timeout), and detects fills on the rest (via a position-quantity
    increase -- see broker_mcp_client's module docstring for why), placing the
    paired sell immediately on fill.
    """
    notifier = notifier or _NULL_NOTIFIER
    pending_buys = await repo.get_open_orders(session, side=OrderSide.BUY)
    for order in pending_buys:
        if is_order_timed_out(order.submitted_at, now, order_timeout_minutes):
            if order.broker_order_id:
                await broker.cancel_order(order.broker_order_id)
            await repo.update_order_status(
                session, order.id, OrderStatus.TIMED_OUT, cancelled_at=now
            )
            logger.info("Cancelled timed-out BUY order %s for %s", order.id, order.ticker)
            continue

        position_qty = await broker.get_open_position_quantity(order.ticker)
        if position_qty >= float(order.quantity):
            await _mark_order_filled(session, order, float(order.limit_price), notifier)
            trade = await session.get(Trade, order.trade_id)
            if trade is not None:
                await _place_paired_sell(session, broker, order, trade, notifier)


async def poll_pending_sell_orders(
    session: AsyncSession, broker: BrokerExecutionClient, notifier: Notifier | None = None
) -> None:
    """Detects fills on pending SELL orders via a position-quantity decrease, closes
    the trade, and records PnL. Not subject to the order-timeout guardrail -- that's
    a BUY-only safety valve against unfilled entries; an unfilled sell is instead
    swept up by liquidate_all_open_positions at end of day.
    """
    notifier = notifier or _NULL_NOTIFIER
    pending_sells = await repo.get_open_orders(session, side=OrderSide.SELL)
    for order in pending_sells:
        position_qty = await broker.get_open_position_quantity(order.ticker)
        if position_qty <= 0:
            trade = await session.get(Trade, order.trade_id)
            if trade is not None:
                await _close_trade_from_sell(
                    session, trade, order, float(order.limit_price), notifier
                )


async def liquidate_all_open_positions(
    session: AsyncSession,
    broker: BrokerExecutionClient,
    *,
    now: datetime,
    liquidation_prices: dict[str, float],
    mode: TradingModeEnum,
    notifier: Notifier | None = None,
) -> None:
    """Spec 4 / 3.3: Day-End Liquidation Rule. Cancels every still-pending order and
    force-sells every open position at `liquidation_prices[ticker]` (a marketable
    price -- e.g. current bid -- the caller/scheduler is responsible for fetching so
    this stays pure of market-data I/O).
    """
    notifier = notifier or _NULL_NOTIFIER
    for order in await repo.get_open_orders(session):
        if order.broker_order_id:
            await broker.cancel_order(order.broker_order_id)
        await repo.update_order_status(session, order.id, OrderStatus.CANCELLED, cancelled_at=now)

    for trade in await repo.get_open_trades(session):
        position_qty = await broker.get_open_position_quantity(trade.ticker)
        if position_qty <= 0:
            continue
        price = liquidation_prices.get(trade.ticker)
        if price is None:
            logger.error(
                "No liquidation price available for %s -- position left open at EOD sweep",
                trade.ticker,
            )
            continue
        placed = await broker.place_order(
            ticker=trade.ticker, side="sell", quantity=position_qty, limit_price=price
        )
        sell_order = await repo.create_order(
            session,
            ticker=trade.ticker,
            side=OrderSide.SELL,
            limit_price=price,
            quantity=position_qty,
            mode=mode,
            broker_order_id=placed.broker_order_id,
            trade_id=trade.id,
        )
        fill_price = placed.fill_price or price
        await _close_trade_from_sell(session, trade, sell_order, fill_price, notifier)
