from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from agentic_trading.config import Settings, TradingMode
from agentic_trading.execution import order_manager as om
from agentic_trading.execution.broker_mcp_client import OrderReview, PlacedOrder
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

pytestmark = pytest.mark.asyncio

TODAY = date(2026, 8, 17)


def _settings(**overrides) -> Settings:
    kwargs = dict(
        mode=TradingMode.DRY_RUN,
        max_open_positions_per_ticker=1,
        daily_trade_cap_per_ticker=3,
        max_capital_per_trade_usd=1000.0,
        max_daily_drawdown_usd=1000.0,
        order_timeout_minutes=15,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _buy_decision(**overrides) -> TradeDecision:
    kwargs = dict(
        decision="BUY",
        confidence_score=0.9,
        buy_limit_price=100.0,
        target_sell_price=102.0,
        stop_loss_price=98.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
        thesis_continuity_flag=True,
    )
    kwargs.update(overrides)
    return TradeDecision(**kwargs)


class FakeLaggyBroker:
    """A broker whose orders start PENDING and only "fill" once test code explicitly
    flips the simulated position -- used to exercise poll_pending_buy_orders instead
    of the instant-fill DryRunBrokerClient.
    """

    def __init__(self):
        self.positions: dict[str, float] = {}
        self.cancelled: list[str] = []
        self._next_id = 1

    async def get_open_position_quantity(self, ticker: str) -> float:
        return self.positions.get(ticker, 0.0)

    async def review_order(self, *, ticker, side, quantity, limit_price) -> OrderReview:
        return OrderReview(warnings=[], estimated_price=limit_price)

    async def place_order(self, *, ticker, side, quantity, limit_price) -> PlacedOrder:
        order_id = f"FAKE-{self._next_id}"
        self._next_id += 1
        return PlacedOrder(broker_order_id=order_id, status="pending", fill_price=None)

    async def cancel_order(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)


async def _seed_decision(session, ticker="AAPL") -> int:
    decision = await repo.save_llm_decision(
        session,
        ticker=ticker,
        bucket_id=None,
        prompt="p",
        raw_response="{}",
        decision="BUY",
        confidence_score=0.9,
        buy_limit_price=100.0,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
    )
    return decision.id


async def test_try_enter_position_opens_trade_and_fills_instantly_in_dry_run(db_session):
    broker = om.DryRunBrokerClient()
    decision_id = await _seed_decision(db_session)

    outcome = await om.try_enter_position(
        db_session,
        broker,
        ticker="AAPL",
        decision=_buy_decision(),
        llm_decision_id=decision_id,
        settings=_settings(),
        today=TODAY,
        realized_pnl_today_all_tickers=0.0,
    )

    assert outcome.opened
    trade = await db_session.get(Trade, outcome.trade_id)
    # DryRunBrokerClient fills the buy instantly, which places+fills the paired sell
    # instantly too -- so the trade should already be closed with realized PnL.
    assert trade.status == TradeStatus.CLOSED
    assert trade.exit_price == 102.0
    assert float(trade.pnl) == pytest.approx((102.0 - 100.0) * float(trade.quantity))

    daily_state = await repo.get_or_create_daily_state(db_session, "AAPL", TODAY)
    assert daily_state.completed_trades_count == 1
    assert daily_state.open_positions_count == 0


async def test_try_enter_position_blocked_by_open_position_guardrail(db_session):
    broker = om.DryRunBrokerClient()
    decision_id = await _seed_decision(db_session)
    await repo.record_trade_opened(db_session, "AAPL", TODAY)  # simulate an already-open position

    outcome = await om.try_enter_position(
        db_session,
        broker,
        ticker="AAPL",
        decision=_buy_decision(),
        llm_decision_id=decision_id,
        settings=_settings(),
        today=TODAY,
        realized_pnl_today_all_tickers=0.0,
    )

    assert not outcome.opened
    assert "position" in outcome.reason


async def test_try_enter_position_blocked_by_circuit_breaker(db_session):
    broker = om.DryRunBrokerClient()
    decision_id = await _seed_decision(db_session)

    outcome = await om.try_enter_position(
        db_session,
        broker,
        ticker="AAPL",
        decision=_buy_decision(),
        llm_decision_id=decision_id,
        settings=_settings(max_daily_drawdown_usd=500.0),
        today=TODAY,
        realized_pnl_today_all_tickers=-500.0,
        )

    assert not outcome.opened
    assert "circuit breaker" in outcome.reason


async def test_poll_pending_buy_orders_cancels_timed_out_order(db_session):
    broker = FakeLaggyBroker()
    order = await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.BUY,
        limit_price=100.0,
        quantity=5,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-1",
    )
    order.submitted_at = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    await db_session.flush()

    now = datetime(2026, 8, 17, 9, 46, tzinfo=UTC)  # 16 minutes later
    await om.poll_pending_buy_orders(db_session, broker, now=now, order_timeout_minutes=15)

    refreshed = await db_session.get(Order, order.id)
    assert refreshed.status == OrderStatus.TIMED_OUT
    assert "FAKE-1" in broker.cancelled


async def test_poll_pending_buy_orders_detects_fill_and_places_paired_sell(db_session):
    broker = FakeLaggyBroker()
    decision_id = await _seed_decision(db_session)
    trade = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=100.0,
        quantity=5,
        llm_decision_id=decision_id,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
    )
    order = await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.BUY,
        limit_price=100.0,
        quantity=5,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-1",
        trade_id=trade.id,
    )
    order.submitted_at = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    await db_session.flush()

    # Simulate the buy fill by making the position appear on the broker.
    broker.positions["AAPL"] = 5

    now = datetime(2026, 8, 17, 9, 32, tzinfo=UTC)
    await om.poll_pending_buy_orders(db_session, broker, now=now, order_timeout_minutes=15)

    refreshed_buy = await db_session.get(Order, order.id)
    assert refreshed_buy.status == OrderStatus.FILLED

    sell_orders = await repo.get_open_orders(db_session, ticker="AAPL", side=OrderSide.SELL)
    assert len(sell_orders) == 1
    assert float(sell_orders[0].limit_price) == 102.0


async def test_poll_pending_buy_orders_fills_instead_of_cancelling_on_timeout_race(db_session):
    # The order is both past its timeout AND has actually filled at the broker --
    # fill must win. Cancelling an already-filled order is wrong (and, against the
    # real MCP, raises -- see broker_mcp_client.unwrap_tool_result's is_error check).
    broker = FakeLaggyBroker()
    decision_id = await _seed_decision(db_session)
    trade = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=100.0,
        quantity=5,
        llm_decision_id=decision_id,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
    )
    order = await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.BUY,
        limit_price=100.0,
        quantity=5,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-1",
        trade_id=trade.id,
    )
    order.submitted_at = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    await db_session.flush()

    broker.positions["AAPL"] = 5  # filled...
    now = datetime(2026, 8, 17, 9, 46, tzinfo=UTC)  # ...16 minutes later, i.e. also timed out

    await om.poll_pending_buy_orders(db_session, broker, now=now, order_timeout_minutes=15)

    refreshed = await db_session.get(Order, order.id)
    assert refreshed.status == OrderStatus.FILLED
    assert broker.cancelled == []
    sell_orders = await repo.get_open_orders(db_session, ticker="AAPL", side=OrderSide.SELL)
    assert len(sell_orders) == 1


class FlakyBroker(FakeLaggyBroker):
    """FakeLaggyBroker that raises on get_open_position_quantity for one specific
    ticker, to test that a sweep isolates one order's failure from the rest."""

    def __init__(self, fail_ticker: str):
        super().__init__()
        self.fail_ticker = fail_ticker

    async def get_open_position_quantity(self, ticker: str) -> float:
        if ticker == self.fail_ticker:
            raise RuntimeError(f"simulated broker failure for {ticker}")
        return await super().get_open_position_quantity(ticker)


async def test_poll_pending_buy_orders_isolates_one_orders_failure_from_the_rest(db_session):
    broker = FlakyBroker(fail_ticker="AAPL")
    decision_id = await _seed_decision(db_session, ticker="MSFT")
    aapl_order = await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.BUY,
        limit_price=100.0,
        quantity=5,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-AAPL",
    )
    msft_trade = await repo.open_trade(
        db_session,
        ticker="MSFT",
        trade_date=TODAY,
        entry_price=50.0,
        quantity=2,
        llm_decision_id=decision_id,
        target_sell_price=51.0,
        max_holding_time_minutes=30,
    )
    msft_order = await repo.create_order(
        db_session,
        ticker="MSFT",
        side=OrderSide.BUY,
        limit_price=50.0,
        quantity=2,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-MSFT",
        trade_id=msft_trade.id,
    )
    for order in (aapl_order, msft_order):
        order.submitted_at = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    await db_session.flush()

    broker.positions["MSFT"] = 2  # MSFT filled; AAPL's check raises instead

    now = datetime(2026, 8, 17, 9, 32, tzinfo=UTC)
    # must not raise, despite AAPL's broker call throwing
    await om.poll_pending_buy_orders(db_session, broker, now=now, order_timeout_minutes=15)

    refreshed_aapl = await db_session.get(Order, aapl_order.id)
    assert refreshed_aapl.status == OrderStatus.PENDING  # untouched, will retry next sweep
    refreshed_msft = await db_session.get(Order, msft_order.id)
    assert refreshed_msft.status == OrderStatus.FILLED
    sell_orders = await repo.get_open_orders(db_session, ticker="MSFT", side=OrderSide.SELL)
    assert len(sell_orders) == 1


async def test_retry_missing_paired_sells_places_the_sell_for_a_filled_buy_missing_one(
    db_session,
):
    # Simulates _place_paired_sell having failed right after the fill was detected:
    # buy order is FILLED, trade is OPEN, but no SELL order was ever recorded.
    broker = FakeLaggyBroker()
    decision_id = await _seed_decision(db_session)
    trade = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=100.0,
        quantity=5,
        llm_decision_id=decision_id,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
    )
    order = await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.BUY,
        limit_price=100.0,
        quantity=5,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-1",
        trade_id=trade.id,
    )
    await repo.update_order_status(
        db_session, order.id, OrderStatus.FILLED, filled_at=datetime.now(UTC), filled_price=100.0
    )

    await om.retry_missing_paired_sells(db_session, broker)

    sell_orders = await repo.get_open_orders(db_session, ticker="AAPL", side=OrderSide.SELL)
    assert len(sell_orders) == 1
    assert float(sell_orders[0].limit_price) == 102.0
    refreshed_trade = await db_session.get(Trade, trade.id)
    assert refreshed_trade.status == TradeStatus.OPEN  # sell placed but not yet filled


async def test_retry_missing_paired_sells_is_a_noop_once_a_sell_order_exists(db_session):
    broker = FakeLaggyBroker()
    decision_id = await _seed_decision(db_session)
    trade = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=100.0,
        quantity=5,
        llm_decision_id=decision_id,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
    )
    buy_order = await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.BUY,
        limit_price=100.0,
        quantity=5,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-1",
        trade_id=trade.id,
    )
    await repo.update_order_status(
        db_session,
        buy_order.id,
        OrderStatus.FILLED,
        filled_at=datetime.now(UTC),
        filled_price=100.0,
    )
    await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.SELL,
        limit_price=102.0,
        quantity=5,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-2",
        trade_id=trade.id,
    )

    await om.retry_missing_paired_sells(db_session, broker)

    sell_orders = await repo.get_open_orders(db_session, ticker="AAPL", side=OrderSide.SELL)
    assert len(sell_orders) == 1  # no duplicate sell placed


async def test_poll_pending_sell_orders_closes_trade_on_exit(db_session):
    broker = FakeLaggyBroker()
    decision_id = await _seed_decision(db_session)
    trade = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=100.0,
        quantity=5,
        llm_decision_id=decision_id,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
    )
    sell_order = await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.SELL,
        limit_price=102.0,
        quantity=5,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="FAKE-2",
        trade_id=trade.id,
    )
    await db_session.flush()

    broker.positions["AAPL"] = 0  # sell has fully exited the position

    await om.poll_pending_sell_orders(db_session, broker)

    refreshed_sell = await db_session.get(Order, sell_order.id)
    assert refreshed_sell.status == OrderStatus.FILLED
    refreshed_trade = await db_session.get(Trade, trade.id)
    assert refreshed_trade.status == TradeStatus.CLOSED
    assert float(refreshed_trade.pnl) == pytest.approx((102.0 - 100.0) * 5)


async def test_liquidate_all_open_positions_force_sells_and_cancels_pending(db_session):
    broker = om.DryRunBrokerClient()
    broker._positions["AAPL"] = 5  # noqa: SLF001 -- test seeding an open position directly
    decision_id = await _seed_decision(db_session)
    trade = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=100.0,
        quantity=5,
        llm_decision_id=decision_id,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
    )
    stale_order = await repo.create_order(
        db_session,
        ticker="MSFT",
        side=OrderSide.BUY,
        limit_price=50.0,
        quantity=2,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="STALE-1",
    )
    await db_session.flush()

    now = datetime(2026, 8, 17, 15, 45, tzinfo=UTC)
    await om.liquidate_all_open_positions(
        db_session,
        broker,
        now=now,
        liquidation_prices={"AAPL": 99.5},
        mode=TradingModeEnum.DRY_RUN,
    )

    refreshed_trade = await db_session.get(Trade, trade.id)
    assert refreshed_trade.status == TradeStatus.CLOSED
    assert float(refreshed_trade.exit_price) == 99.5

    refreshed_stale = await db_session.get(Order, stale_order.id)
    assert refreshed_stale.status == OrderStatus.CANCELLED


class PartiallyFailingEodBroker:
    """Behaves like DryRunBrokerClient (instant fills) but raises for one specific
    ticker's get_open_position_quantity call -- tests that liquidate_all_open_positions
    isolates one ticker's broker failure from the rest of the EOD sweep (spec 4's
    overnight-exposure guardrail must not be undermined by an unrelated ticker's error).
    """

    def __init__(self, fail_ticker: str):
        self.fail_ticker = fail_ticker
        self.positions: dict[str, float] = {}
        self._next_id = 1

    async def get_open_position_quantity(self, ticker: str) -> float:
        if ticker == self.fail_ticker:
            raise RuntimeError(f"simulated broker failure for {ticker}")
        return self.positions.get(ticker, 0.0)

    async def review_order(self, **kwargs) -> OrderReview:
        return OrderReview(warnings=[], estimated_price=kwargs.get("limit_price"))

    async def place_order(self, *, ticker, side, quantity, limit_price) -> PlacedOrder:
        order_id = f"EOD-{self._next_id}"
        self._next_id += 1
        return PlacedOrder(broker_order_id=order_id, status="filled", fill_price=limit_price)

    async def cancel_order(self, broker_order_id: str) -> None:
        return None


async def test_liquidate_all_open_positions_isolates_one_tickers_failure_from_the_rest(
    db_session,
):
    broker = PartiallyFailingEodBroker(fail_ticker="AAPL")
    broker.positions["MSFT"] = 2
    decision_id = await _seed_decision(db_session, ticker="AAPL")
    aapl_trade = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=100.0,
        quantity=5,
        llm_decision_id=decision_id,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
    )
    msft_decision_id = await _seed_decision(db_session, ticker="MSFT")
    msft_trade = await repo.open_trade(
        db_session,
        ticker="MSFT",
        trade_date=TODAY,
        entry_price=50.0,
        quantity=2,
        llm_decision_id=msft_decision_id,
        target_sell_price=51.0,
        max_holding_time_minutes=30,
    )

    now = datetime(2026, 8, 17, 15, 45, tzinfo=UTC)
    await om.liquidate_all_open_positions(  # must not raise
        db_session,
        broker,
        now=now,
        liquidation_prices={"AAPL": 99.5, "MSFT": 50.5},
        mode=TradingModeEnum.DRY_RUN,
    )

    refreshed_aapl = await db_session.get(Trade, aapl_trade.id)
    assert refreshed_aapl.status == TradeStatus.OPEN  # untouched -- broker call failed
    refreshed_msft = await db_session.get(Trade, msft_trade.id)
    assert refreshed_msft.status == TradeStatus.CLOSED
    assert float(refreshed_msft.exit_price) == 50.5


async def _open_position_with_pending_sell(
    db_session, *, entry_price=100.0, target_sell_price=102.0, stop_loss_price=98.0, quantity=5
):
    """Phase 3 test fixture: an OPEN trade with a FILLED buy leg and a resting
    PENDING sell at target_sell_price -- the state try_exit_position_early/
    apply_trailing_stop expect to find via repo.get_open_trade_for_ticker.
    """
    decision_id = await _seed_decision(db_session)
    trade = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=entry_price,
        quantity=quantity,
        llm_decision_id=decision_id,
        target_sell_price=target_sell_price,
        stop_loss_price=stop_loss_price,
        max_holding_time_minutes=30,
    )
    buy_order = await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.BUY,
        limit_price=entry_price,
        quantity=quantity,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="BUY-1",
        trade_id=trade.id,
    )
    await repo.update_order_status(db_session, buy_order.id, OrderStatus.FILLED)
    await repo.create_order(
        db_session,
        ticker="AAPL",
        side=OrderSide.SELL,
        limit_price=target_sell_price,
        quantity=quantity,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="SELL-1",
        trade_id=trade.id,
    )
    await db_session.flush()
    return await repo.get_open_trade_for_ticker(db_session, "AAPL")


async def test_try_exit_position_early_cancels_resting_sell_and_closes_at_exit_price(db_session):
    broker = om.DryRunBrokerClient()
    broker._positions["AAPL"] = 5  # noqa: SLF001 -- test seeding an open position directly
    trade = await _open_position_with_pending_sell(db_session)

    exited = await om.try_exit_position_early(
        db_session, broker, trade=trade, exit_price=97.0, exit_reason="STOP_LOSS"
    )

    assert exited is True
    assert trade.status == TradeStatus.CLOSED
    assert float(trade.exit_price) == 97.0
    assert trade.exit_reason == "STOP_LOSS"
    assert float(trade.pnl) == pytest.approx((97.0 - 100.0) * 5)

    orders = await repo.get_open_orders(db_session, ticker="AAPL")
    assert orders == []  # the old resting sell was cancelled, the new one filled
    # trade.orders was eager-loaded before the new exit order was created in this
    # same session, so it won't reflect it -- query fresh instead.
    all_sell_orders = (
        await db_session.scalars(
            select(Order).where(Order.ticker == "AAPL", Order.side == OrderSide.SELL)
        )
    ).all()
    assert len(all_sell_orders) == 2
    cancelled = next(o for o in all_sell_orders if o.broker_order_id == "SELL-1")
    assert cancelled.status == OrderStatus.CANCELLED


async def test_try_exit_position_early_does_nothing_when_trade_already_closed(db_session):
    broker = om.DryRunBrokerClient()
    trade = await _open_position_with_pending_sell(db_session)
    # Same session -> repo.close_trade's session.get(Trade, ...) returns this exact
    # object (identity map), so `trade.status` is already CLOSED after this call.
    await repo.close_trade(
        db_session, trade.id, exit_price=102.0, closed_at=datetime.now(UTC), pnl=10.0
    )
    assert trade.status == TradeStatus.CLOSED

    exited = await om.try_exit_position_early(
        db_session, broker, trade=trade, exit_price=97.0, exit_reason="STOP_LOSS"
    )

    assert exited is False


async def test_try_exit_position_early_does_nothing_when_broker_has_no_position(db_session):
    broker = om.DryRunBrokerClient()  # no position seeded -- get_open_position_quantity == 0
    trade = await _open_position_with_pending_sell(db_session)

    exited = await om.try_exit_position_early(
        db_session, broker, trade=trade, exit_price=97.0, exit_reason="STOP_LOSS"
    )

    assert exited is False
    assert trade.status == TradeStatus.OPEN  # untouched


async def test_apply_trailing_stop_replaces_resting_sell_when_target_raised(db_session):
    broker = om.DryRunBrokerClient()
    broker._positions["AAPL"] = 5  # noqa: SLF001 -- test seeding an open position directly
    trade = await _open_position_with_pending_sell(db_session)

    await om.apply_trailing_stop(db_session, broker, trade=trade, new_target=105.0, new_stop=99.0)

    orders = await repo.get_open_orders(db_session, ticker="AAPL")
    assert orders == []  # DryRunBrokerClient fills the replacement instantly
    assert trade.status == TradeStatus.CLOSED  # filled at the new, higher target
    assert float(trade.exit_price) == 105.0
    assert trade.exit_reason == "TARGET_HIT"


async def test_apply_trailing_stop_leaves_resting_order_untouched_when_target_not_raised(
    db_session,
):
    broker = FakeLaggyBroker()  # orders stay PENDING -- proves no replacement was attempted
    trade = await _open_position_with_pending_sell(db_session)

    # Only the stop is trailed up this cycle; the target proposal equals the
    # current target, so nothing about the resting sell order should change.
    await om.apply_trailing_stop(db_session, broker, trade=trade, new_target=102.0, new_stop=99.0)

    assert broker.cancelled == []
    refreshed_trade = await db_session.get(Trade, trade.id)
    assert float(refreshed_trade.stop_loss_price) == 99.0
    assert float(refreshed_trade.target_sell_price) == 102.0


async def test_apply_trailing_stop_updates_levels_even_without_a_pending_sell(db_session):
    # No resting sell order exists yet (e.g. still mid-retry via
    # retry_missing_paired_sells) -- must not crash, and still records the
    # trailed values so the eventual paired sell uses them.
    broker = om.DryRunBrokerClient()
    decision_id = await _seed_decision(db_session)
    trade_row = await repo.open_trade(
        db_session,
        ticker="AAPL",
        trade_date=TODAY,
        entry_price=100.0,
        quantity=5,
        llm_decision_id=decision_id,
        target_sell_price=102.0,
        stop_loss_price=98.0,
        max_holding_time_minutes=30,
    )
    await db_session.flush()
    trade = await repo.get_open_trade_for_ticker(db_session, "AAPL")
    assert trade.id == trade_row.id

    await om.apply_trailing_stop(db_session, broker, trade=trade, new_target=105.0, new_stop=99.0)

    refreshed_trade = await db_session.get(Trade, trade.id)
    assert float(refreshed_trade.target_sell_price) == 105.0
    assert float(refreshed_trade.stop_loss_price) == 99.0
