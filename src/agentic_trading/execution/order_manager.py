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
from agentic_trading.state.models import (
    Order,
    OrderSide,
    OrderStatus,
    Trade,
    TradeStatus,
    TradingModeEnum,
)

logger = logging.getLogger(__name__)

_NULL_NOTIFIER = NullNotifier()


@dataclass
class _PendingSell:
    broker_order_id: str
    quantity: float
    limit_price: float


class DryRunBrokerClient:
    """BrokerExecutionClient for MODE=DRY_RUN/PAPER_TRADING -- tracks open quantity
    in-memory, exercising the exact same order_manager state machine and DB writes
    as LIVE, just with zero network calls to the MCP and zero real money at risk.

    BUY orders are a deliberate simplification and still fill instantly at the
    requested limit price -- entries are assumed marketable. SELL orders are NOT:
    they behave like a real resting limit order at the exchange (matching
    McpBrokerClient/LIVE -- see broker_mcp_client.py's module docstring: "real
    orders fill asynchronously, unlike DryRunBrokerClient's instant simulated
    fill"), staying PENDING until `mark_price` reports the market has traded at or
    above the limit. That covers the target sell and a trailed-up target
    (order_manager._place_paired_sell / apply_trailing_stop) resting untouched
    until price actually gets there. A SELL only fills immediately when it's
    already marketable at the time it's placed -- true for the early-exit/EOD
    liquidation paths, which call `_mark_price` with their own exit price right
    before placing (see try_exit_position_early / liquidate_all_open_positions).
    """

    def __init__(self) -> None:
        self._positions: dict[str, float] = {}
        self._next_order_id = 1
        self._last_price: dict[str, float] = {}
        self._pending_sells: dict[str, list[_PendingSell]] = {}

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
        if side == "buy":
            self._positions[ticker] = self._positions.get(ticker, 0.0) + quantity
            return PlacedOrder(broker_order_id=order_id, status="filled", fill_price=limit_price)

        current_price = self._last_price.get(ticker)
        if current_price is not None and current_price >= limit_price:
            self._positions[ticker] = self._positions.get(ticker, 0.0) - quantity
            return PlacedOrder(broker_order_id=order_id, status="filled", fill_price=limit_price)

        self._pending_sells.setdefault(ticker, []).append(
            _PendingSell(broker_order_id=order_id, quantity=quantity, limit_price=limit_price)
        )
        return PlacedOrder(broker_order_id=order_id, status="pending", fill_price=None)

    async def cancel_order(self, broker_order_id: str) -> None:
        for pending in self._pending_sells.values():
            pending[:] = [o for o in pending if o.broker_order_id != broker_order_id]

    def mark_price(self, ticker: str, price: float) -> None:
        """Feed the latest known market price for `ticker` -- the simulator has no
        real market connection of its own, so this is what lets a resting SELL
        order "cross" and fill, the same way a real limit order at the exchange
        would as the market trades through it. Called by
        scheduler.run_order_management_sweep once per tick for every ticker with an
        open trade, using a real quote -- see that function's docstring.
        """
        self._last_price[ticker] = price
        pending = self._pending_sells.get(ticker)
        if not pending:
            return
        still_pending = []
        for order in pending:
            if price >= order.limit_price:
                self._positions[ticker] = self._positions.get(ticker, 0.0) - order.quantity
            else:
                still_pending.append(order)
        self._pending_sells[ticker] = still_pending


def _mark_price(broker: BrokerExecutionClient, ticker: str, price: float) -> None:
    """Best-effort DryRunBrokerClient.mark_price call, made just before placing a
    marketable exit order (early exit / EOD liquidation) so it fills immediately
    the same way a real marketable order would -- without depending on whether the
    periodic order-management sweep has already primed a price for this ticker
    this tick. A no-op for any broker without `mark_price` (e.g. the real
    McpBrokerClient -- the exchange handles this itself, nothing to simulate).
    """
    mark_price = getattr(broker, "mark_price", None)
    if mark_price is not None:
        mark_price(ticker, price)


def compute_order_quantity(limit_price: float, max_capital_per_trade_usd: float) -> float:
    """Position sizing: spend up to the per-trade capital cap, in whole shares."""
    if limit_price <= 0:
        return 0.0
    return float(math.floor(max_capital_per_trade_usd / limit_price))


@dataclass
class TradeEntryOutcome:
    opened: bool
    reason: str | None = None
    trade_id: int | None = None  # our trades.id (internal PK, also Order.trade_id's FK target)
    order_id: int | None = None  # our orders.id (internal PK) for the BUY leg
    broker_order_id: str | None = None  # the real broker-side order id (e.g. MCP order UUID)


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

    logger.info(
        "All guardrails cleared for %s BUY order, quantity %s limit_price %s",
        ticker,
        quantity,
        decision.buy_limit_price,
    )

    review = await broker.review_order(
        ticker=ticker, side="buy", quantity=quantity, limit_price=decision.buy_limit_price
    )

    logger.info(
        "Placing BUY order for %s (qty=%s @ $%s)", ticker, quantity, decision.buy_limit_price
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
        stop_loss_price=decision.stop_loss_price,
        max_holding_time_minutes=decision.max_holding_time_minutes,
    )
    order.trade_id = trade.id
    await session.flush()
    await repo.record_trade_opened(session, ticker, today)

    if placed.status == "filled":
        fill_price = placed.fill_price or decision.buy_limit_price
        await _mark_order_filled(session, order, fill_price, notifier)
        await _place_paired_sell(session, broker, order, trade, notifier)

    return TradeEntryOutcome(
        opened=True, trade_id=trade.id, order_id=order.id, broker_order_id=order.broker_order_id
    )


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
        await _close_trade_from_sell(
            session, trade, sell_order, fill_price, notifier, exit_reason="TARGET_HIT"
        )


async def _close_trade_from_sell(
    session: AsyncSession,
    trade: Trade,
    sell_order: Order,
    fill_price: float,
    notifier: Notifier,
    *,
    exit_reason: str | None = None,
) -> None:
    await _mark_order_filled(session, sell_order, fill_price, notifier)
    pnl = (fill_price - float(trade.entry_price)) * float(trade.quantity)
    await repo.close_trade(
        session,
        trade.id,
        exit_price=fill_price,
        closed_at=datetime.now(UTC),
        pnl=pnl,
        exit_reason=exit_reason,
    )
    await repo.record_trade_closed(session, trade.ticker, trade.trade_date, pnl=pnl)
    await notifier.notify(
        "Trade closed",
        {
            "ticker": trade.ticker,
            "entry_price": float(trade.entry_price),
            "exit_price": fill_price,
            "pnl": pnl,
            "exit_reason": exit_reason,
        },
    )


def _pending_sell_order(trade: Trade) -> Order | None:
    """The trade's resting SELL order, if one is still PENDING -- requires
    trade.orders to already be eager-loaded (see repo.get_open_trade_for_ticker)."""
    return next(
        (o for o in trade.orders if o.side == OrderSide.SELL and o.status == OrderStatus.PENDING),
        None,
    )


def _trade_mode(trade: Trade) -> TradingModeEnum:
    """Mode (DRY_RUN/LIVE) is recorded per-Order, not per-Trade, but every order
    for one trade always carries the same mode -- so any existing order (there's
    always at least the BUY leg once a trade exists) tells us which, without
    needing to re-derive it from ambient settings."""
    if not trade.orders:
        raise ValueError(f"Trade {trade.id} has no orders to infer its trading mode from")
    return trade.orders[0].mode


async def try_exit_position_early(
    session: AsyncSession,
    broker: BrokerExecutionClient,
    *,
    trade: Trade,
    exit_price: float,
    exit_reason: str,
    notifier: Notifier | None = None,
) -> bool:
    """Phase 3 (requirements.md section 8): force-closes an OPEN position ahead of
    its resting target sell -- called by scheduler.py when
    execution.invalidation.evaluate_exit_guardrails forces an exit (STOP_LOSS/
    MOMENTUM_BREAK), or the LLM itself signals SELL with thesis_continuity_flag
    false for an IN_POSITION ticker (an LLM-judged catalyst, or its own read of the
    setup). `exit_price` is a marketable price (e.g. current bid) the caller is
    responsible for fetching, same pattern as liquidate_all_open_positions'
    `liquidation_prices`.

    `trade` must come from repo.get_open_trade_for_ticker (trade.orders needs to
    already be eager-loaded). Cancels the resting target-sell order first, if one
    is still pending, so it can't fill out from under this new exit order.

    Returns True once an exit order has been placed (whether or not it filled
    instantly -- an unfilled exit is still picked up by the ordinary
    poll_pending_sell_orders sweep next tick, same as any other pending sell).
    Returns False without doing anything if the trade is no longer OPEN, or the
    broker reports no position left to sell (both are races with something else
    having already closed it out this same cycle).
    """
    notifier = notifier or _NULL_NOTIFIER
    if trade.status != TradeStatus.OPEN:
        logger.info(
            "try_exit_position_early: trade %s (%s) is no longer OPEN -- nothing to do",
            trade.id,
            trade.ticker,
        )
        return False

    position_qty = await broker.get_open_position_quantity(trade.ticker)
    if position_qty <= 0:
        logger.info(
            "try_exit_position_early: %s has no open broker position -- nothing to exit",
            trade.ticker,
        )
        return False

    pending_sell = _pending_sell_order(trade)
    if pending_sell is not None:
        if pending_sell.broker_order_id:
            await broker.cancel_order(pending_sell.broker_order_id)
        await repo.update_order_status(
            session, pending_sell.id, OrderStatus.CANCELLED, cancelled_at=datetime.now(UTC)
        )

    logger.info(
        "Forcing early exit for %s (trade %s): %s @ ~$%s",
        trade.ticker,
        trade.id,
        exit_reason,
        exit_price,
    )
    _mark_price(broker, trade.ticker, exit_price)
    placed = await broker.place_order(
        ticker=trade.ticker, side="sell", quantity=position_qty, limit_price=exit_price
    )
    sell_order = await repo.create_order(
        session,
        ticker=trade.ticker,
        side=OrderSide.SELL,
        limit_price=exit_price,
        quantity=position_qty,
        mode=_trade_mode(trade),
        broker_order_id=placed.broker_order_id,
        trade_id=trade.id,
    )
    if placed.status == "filled":
        fill_price = placed.fill_price or exit_price
        await _close_trade_from_sell(
            session, trade, sell_order, fill_price, notifier, exit_reason=exit_reason
        )
    return True


async def apply_trailing_stop(
    session: AsyncSession,
    broker: BrokerExecutionClient,
    *,
    trade: Trade,
    new_target: float,
    new_stop: float,
    notifier: Notifier | None = None,
) -> None:
    """Phase 3 one-way trailing-stop ratchet (requirements.md section 8). Callers
    (scheduler.py, gated by settings.trailing_stop_enabled) are responsible for
    ensuring new_target/new_stop are never less favorable than the trade's current
    values -- see execution.invalidation.compute_trailing_stop -- this function
    just applies whatever it's given.

    Only a raised target touches the broker: stop_loss is purely a Python-side
    check each cycle (execution.invalidation.evaluate_exit_guardrails), not a
    resting order, so trailing it up is a pure data update. A raised target means
    cancelling the old resting sell and placing a new one at the higher price --
    `trade` must come from repo.get_open_trade_for_ticker (orders eager-loaded).
    """
    notifier = notifier or _NULL_NOTIFIER
    current_target = float(trade.target_sell_price) if trade.target_sell_price is not None else None
    target_raised = current_target is None or new_target > current_target

    if target_raised:
        pending_sell = _pending_sell_order(trade)
        if pending_sell is not None:
            if pending_sell.broker_order_id:
                await broker.cancel_order(pending_sell.broker_order_id)
            await repo.update_order_status(
                session, pending_sell.id, OrderStatus.CANCELLED, cancelled_at=datetime.now(UTC)
            )
            placed = await broker.place_order(
                ticker=trade.ticker,
                side="sell",
                quantity=float(pending_sell.quantity),
                limit_price=new_target,
            )
            sell_order = await repo.create_order(
                session,
                ticker=trade.ticker,
                side=OrderSide.SELL,
                limit_price=new_target,
                quantity=float(pending_sell.quantity),
                mode=pending_sell.mode,
                broker_order_id=placed.broker_order_id,
                trade_id=trade.id,
            )
            if placed.status == "filled":
                fill_price = placed.fill_price or new_target
                await _close_trade_from_sell(
                    session, trade, sell_order, fill_price, notifier, exit_reason="TARGET_HIT"
                )
            logger.info(
                "Trailed target up for %s (trade %s): %s -> %s",
                trade.ticker,
                trade.id,
                current_target,
                new_target,
            )
        # else: no resting sell to replace (e.g. still mid-retry via
        # retry_missing_paired_sells) -- update_trade_trailing_levels below still
        # records the new target, so whenever that sell does get placed it uses
        # the trailed value.

    if trade.status == TradeStatus.OPEN:
        await repo.update_trade_trailing_levels(
            session, trade.id, target_sell_price=new_target, stop_loss_price=new_stop
        )


async def poll_pending_buy_orders(
    session: AsyncSession,
    broker: BrokerExecutionClient,
    *,
    now: datetime,
    order_timeout_minutes: int,
    notifier: Notifier | None = None,
) -> None:
    """Runs every poll cycle: detects fills (via a position-quantity increase -- see
    broker_mcp_client's module docstring for why), placing the paired sell
    immediately on fill, and cancels unfilled BUY orders past the timeout guardrail
    (spec 4: Order Timeout).

    Fill is checked BEFORE timeout, not after: an order that fills right around its
    timeout deadline must never reach the cancel call below -- cancelling an
    already-filled order gets rejected by the broker (raises, per
    broker_mcp_client.unwrap_tool_result's is_error check) and, more importantly,
    would leave a real filled position marked TIMED_OUT/CANCELLED in our own DB.

    Each order is isolated in its own try/except: one order's broker error (a
    rejected cancel, a network blip) must not abort the rest of the sweep, and
    since this whole function runs inside one shared session_scope() transaction
    (see run_order_management_sweep), an uncaught exception here would roll back
    every other ticker's already-processed updates from this same tick too, not
    just the failing one.
    """
    notifier = notifier or _NULL_NOTIFIER
    pending_buys = await repo.get_open_orders(session, side=OrderSide.BUY)
    for order in pending_buys:
        try:
            position_qty = await broker.get_open_position_quantity(order.ticker)
            if position_qty >= float(order.quantity):
                await _mark_order_filled(session, order, float(order.limit_price), notifier)
                trade = await session.get(Trade, order.trade_id)
                if trade is not None:
                    await _place_paired_sell(session, broker, order, trade, notifier)
                continue

            if is_order_timed_out(order.submitted_at, now, order_timeout_minutes):
                if order.broker_order_id:
                    await broker.cancel_order(order.broker_order_id)
                await repo.update_order_status(
                    session, order.id, OrderStatus.TIMED_OUT, cancelled_at=now
                )
                logger.info("Cancelled timed-out BUY order %s for %s", order.id, order.ticker)
        except Exception:
            logger.exception(
                "Failed processing pending BUY order %s (%s) -- will retry next sweep",
                order.id,
                order.ticker,
            )


async def poll_pending_sell_orders(
    session: AsyncSession, broker: BrokerExecutionClient, notifier: Notifier | None = None
) -> None:
    """Detects fills on pending SELL orders via a position-quantity decrease, closes
    the trade, and records PnL. Not subject to the order-timeout guardrail -- that's
    a BUY-only safety valve against unfilled entries; an unfilled sell is instead
    swept up by liquidate_all_open_positions at end of day.

    Each order isolated in its own try/except -- see poll_pending_buy_orders'
    docstring for why.
    """
    notifier = notifier or _NULL_NOTIFIER
    pending_sells = await repo.get_open_orders(session, side=OrderSide.SELL)
    for order in pending_sells:
        try:
            position_qty = await broker.get_open_position_quantity(order.ticker)
            if position_qty <= 0:
                trade = await session.get(Trade, order.trade_id)
                if trade is not None:
                    # exit_reason intentionally left unset here: this generic
                    # fill-detection path can't tell whether `order` was the
                    # ordinary paired target sell, a trailed-up target sell, or
                    # an early-exit sell from try_exit_position_early -- all look
                    # identical once resting at the broker. Only the callers that
                    # already know for certain (an instant-fill right after
                    # placing) pass an explicit exit_reason.
                    await _close_trade_from_sell(
                        session, trade, order, float(order.limit_price), notifier
                    )
        except Exception:
            logger.exception(
                "Failed processing pending SELL order %s (%s) -- will retry next sweep",
                order.id,
                order.ticker,
            )


async def retry_missing_paired_sells(
    session: AsyncSession, broker: BrokerExecutionClient, notifier: Notifier | None = None
) -> None:
    """Runs every sweep tick, after poll_pending_buy_orders: covers the case where a
    buy fill was detected but placing its paired sell then failed (broker
    rejection, network blip, process restart between the two calls) -- without
    this, such a trade drops out of every other query (its buy order is no longer
    PENDING, and it has no SELL order to be found by poll_pending_sell_orders
    either) and would sit OPEN with a real, unmanaged position until
    liquidate_all_open_positions force-closes it at end of day. This closes that
    gap sooner, at the trade's actual target_sell_price rather than an EOD dump.
    """
    notifier = notifier or _NULL_NOTIFIER
    for trade in await repo.get_open_trades_missing_sell_order(session):
        buy_order = next(
            (o for o in trade.orders if o.side == OrderSide.BUY and o.status == OrderStatus.FILLED),
            None,
        )
        if buy_order is None:
            continue  # shouldn't happen given the query's own filter; defensive only
        try:
            await _place_paired_sell(session, broker, buy_order, trade, notifier)
        except Exception:
            logger.exception(
                "Retry: still failed to place paired sell for trade %s (%s) -- "
                "will retry again next sweep",
                trade.id,
                trade.ticker,
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

    Each order/trade is isolated in its own try/except -- this is the last line of
    defense against overnight exposure (spec 4), so one ticker's broker error must
    never stop the rest of the watchlist from being liquidated (see
    poll_pending_buy_orders' docstring for the same reasoning).
    """
    notifier = notifier or _NULL_NOTIFIER
    for order in await repo.get_open_orders(session):
        try:
            if order.broker_order_id:
                await broker.cancel_order(order.broker_order_id)
            await repo.update_order_status(
                session, order.id, OrderStatus.CANCELLED, cancelled_at=now
            )
        except Exception:
            logger.exception(
                "EOD: failed to cancel order %s (%s) -- continuing with the rest "
                "of the liquidation sweep",
                order.id,
                order.ticker,
            )

    for trade in await repo.get_open_trades(session):
        try:
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
            _mark_price(broker, trade.ticker, price)
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
            await _close_trade_from_sell(
                session, trade, sell_order, fill_price, notifier, exit_reason="EOD"
            )
        except Exception:
            logger.exception(
                "EOD: failed to liquidate trade %s (%s) -- position may remain open "
                "overnight, needs manual attention",
                trade.id,
                trade.ticker,
            )
