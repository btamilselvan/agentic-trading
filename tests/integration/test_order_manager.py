from datetime import UTC, date, datetime

import pytest

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
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
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
