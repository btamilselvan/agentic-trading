from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from agentic_trading import scheduler
from agentic_trading.config import Settings, TradingMode
from agentic_trading.execution.broker_mcp_client import OrderReview, PlacedOrder
from agentic_trading.execution.order_manager import DryRunBrokerClient
from agentic_trading.llm.schema import TickerState, TradeDecision
from agentic_trading.market_data.robinhood_client import HistoricalBar, Quote
from agentic_trading.state import repository as repo
from agentic_trading.state.db import session_scope
from agentic_trading.state.models import (
    LlmDecision,
    OrderSide,
    TradingModeEnum,
)

TODAY = date(2026, 8, 17)
BUCKET_START = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
# Settings() defaults to timezone="America/New_York" -- run_poll_cycle now enforces
# the exact MARKET_OPEN_TIME-EVALUATION_WINDOW_END_TIME window against "now", so
# tests need a "now" that actually falls inside it rather than relying on whatever
# the real wall clock happens to be when the suite runs.
MID_WINDOW = datetime(2026, 8, 17, 10, 0, tzinfo=ZoneInfo("America/New_York"))


def _settings(**overrides) -> Settings:
    kwargs = dict(
        mode=TradingMode.DRY_RUN,
        watchlist=["AAPL"],
        confidence_threshold=0.7,
        max_open_positions_per_ticker=1,
        daily_trade_cap_per_ticker=3,
        max_capital_per_trade_usd=1000.0,
        max_daily_drawdown_usd=1000.0,
        order_timeout_minutes=15,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


class FakeLLMClient:
    def __init__(self, decision: TradeDecision):
        self.decision = decision
        self.calls = 0

    async def decide(self, ticker, bucket_history, ticker_state: TickerState):
        self.calls += 1
        return self.decision, "prompt-text", "{}"


class ExplodingLLMClient:
    async def decide(self, ticker, bucket_history, ticker_state):
        raise AssertionError("LLM should not have been called for an ineligible ticker")


def _patch_market_data(monkeypatch, bar: HistoricalBar, quote: Quote | None = None):
    monkeypatch.setattr(
        scheduler.rh, "get_5min_historicals", lambda ticker, span="day", bounds="regular": [bar]
    )
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: quote)
    scheduler._rvol_lookback_cache.clear()


async def _bucket_count(ticker: str) -> int:
    async with session_scope() as session:
        since = datetime(2020, 1, 1, tzinfo=UTC)
        rows = await repo.get_buckets_for_ticker(session, ticker, since=since)
        return len(rows)


async def _decision_count(ticker: str) -> int:
    async with session_scope() as session:
        result = await session.execute(select(LlmDecision).where(LlmDecision.ticker == ticker))
        return len(result.scalars().all())


def test_is_within_poll_window_enforces_exact_minutes_not_just_the_hour():
    settings = _settings()  # market_open_time="09:30", evaluation_window_end_time="11:30"

    assert not scheduler._is_within_poll_window(time(9, 0), settings)  # before 09:30
    assert scheduler._is_within_poll_window(time(9, 30), settings)  # exactly open
    assert scheduler._is_within_poll_window(time(10, 45), settings)
    assert scheduler._is_within_poll_window(time(11, 30), settings)  # exactly close
    assert not scheduler._is_within_poll_window(time(11, 45), settings)  # after 11:30


async def test_run_poll_cycle_skips_entirely_outside_the_poll_window(db_session, monkeypatch):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    _patch_market_data(monkeypatch, bar)
    outside_window = datetime(2026, 8, 17, 20, 0, tzinfo=ZoneInfo("America/New_York"))

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(),
        llm_client=ExplodingLLMClient(),
        settings=_settings(),
        now=outside_window,
    )

    assert await _bucket_count("AAPL") == 0  # never even fetched market data


async def test_poll_cycle_persists_bucket_and_skips_reprocessing_same_bucket(
    db_session, monkeypatch
):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    _patch_market_data(monkeypatch, bar)
    llm = FakeLLMClient(TradeDecision(decision="HOLD", confidence_score=0.2))
    broker = DryRunBrokerClient()
    settings = _settings()

    await scheduler.run_poll_cycle(broker=broker, llm_client=llm, settings=settings, now=MID_WINDOW)
    await scheduler.run_poll_cycle(broker=broker, llm_client=llm, settings=settings, now=MID_WINDOW)

    assert await _bucket_count("AAPL") == 1
    assert llm.calls == 1  # second call was a no-op (bucket already recorded)


async def test_poll_cycle_opens_and_closes_trade_on_high_confidence_buy(db_session, monkeypatch):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    quote = Quote("AAPL", 100.4, 100.6, 500, 400, 100.5, None)
    _patch_market_data(monkeypatch, bar, quote)
    decision = TradeDecision(
        decision="BUY",
        confidence_score=0.9,
        buy_limit_price=100.5,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
    )
    llm = FakeLLMClient(decision)
    broker = DryRunBrokerClient()

    await scheduler.run_poll_cycle(
        broker=broker, llm_client=llm, settings=_settings(), now=MID_WINDOW
    )

    async with session_scope() as session:
        result = await session.execute(select(LlmDecision).where(LlmDecision.ticker == "AAPL"))
        saved = result.scalars().one()
        assert saved.acted_on is True

        daily_state = await repo.get_or_create_daily_state(session, "AAPL", TODAY)
        assert daily_state.completed_trades_count == 1
        assert daily_state.open_positions_count == 0


class FakeExplodingBroker:
    """Any real call is a bug in OBSERVE mode -- it must never touch the broker."""

    async def get_open_position_quantity(self, ticker: str) -> float:
        raise AssertionError("OBSERVE mode must not query the broker")

    async def review_order(self, **kwargs):
        raise AssertionError("OBSERVE mode must not review an order")

    async def place_order(self, **kwargs):
        raise AssertionError("OBSERVE mode must not place an order")

    async def cancel_order(self, broker_order_id: str) -> None:
        raise AssertionError("OBSERVE mode must not cancel an order")


class FakeNotifier:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def notify(self, title, fields):
        self.calls.append((title, fields))


async def test_poll_cycle_reports_but_never_acts_in_observe_mode(db_session, monkeypatch):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    quote = Quote("AAPL", 100.4, 100.6, 500, 400, 100.5, None)
    _patch_market_data(monkeypatch, bar, quote)
    decision = TradeDecision(
        decision="BUY",
        confidence_score=0.9,
        buy_limit_price=100.5,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
    )
    llm = FakeLLMClient(decision)
    notifier = FakeNotifier()

    await scheduler.run_poll_cycle(
        broker=FakeExplodingBroker(),
        llm_client=llm,
        settings=_settings(mode=TradingMode.OBSERVE),
        notifier=notifier,
        now=MID_WINDOW,
    )

    async with session_scope() as session:
        result = await session.execute(select(LlmDecision).where(LlmDecision.ticker == "AAPL"))
        saved = result.scalars().one()
        assert saved.decision.value == "BUY"
        assert saved.acted_on is False  # reported, never acted on

        daily_state = await repo.get_or_create_daily_state(session, "AAPL", TODAY)
        assert daily_state.completed_trades_count == 0
        assert daily_state.open_positions_count == 0

    assert len(notifier.calls) == 1
    title, fields = notifier.calls[0]
    assert title == "Buying opportunity (observation only)"
    assert fields["ticker"] == "AAPL"


async def test_poll_cycle_skips_llm_when_ticker_already_has_open_position(db_session, monkeypatch):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    _patch_market_data(monkeypatch, bar)
    async with session_scope() as session:
        await repo.record_trade_opened(session, "AAPL", TODAY)

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(),
        llm_client=ExplodingLLMClient(),
        settings=_settings(),
        now=MID_WINDOW,
    )

    assert await _bucket_count("AAPL") == 1  # bucket is still recorded
    assert await _decision_count("AAPL") == 0  # but the LLM was never consulted


class FakePendingBroker:
    def __init__(self):
        self.cancelled: list[str] = []

    async def get_open_position_quantity(self, ticker: str) -> float:
        return 0.0

    async def review_order(self, **kwargs) -> OrderReview:
        return OrderReview(warnings=[], estimated_price=None)

    async def place_order(self, **kwargs) -> PlacedOrder:
        raise AssertionError("not expected in this test")

    async def cancel_order(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)


async def test_order_management_sweep_cancels_timed_out_orders(db_session):
    async with session_scope() as session:
        order = await repo.create_order(
            session,
            ticker="AAPL",
            side=OrderSide.BUY,
            limit_price=100.0,
            quantity=5,
            mode=TradingModeEnum.DRY_RUN,
            broker_order_id="STALE-1",
        )
        # Submitted well beyond the timeout window, relative to real "now" -- avoids
        # having to freeze datetime.now() just for this one guardrail check.
        order.submitted_at = datetime.now(UTC) - timedelta(minutes=30)

    broker = FakePendingBroker()
    settings = _settings(order_timeout_minutes=15)

    await scheduler.run_order_management_sweep(broker=broker, settings=settings)

    assert "STALE-1" in broker.cancelled


async def test_eod_liquidation_closes_open_trade_at_quoted_bid(db_session, monkeypatch):
    monkeypatch.setattr(
        scheduler.rh, "get_quote", lambda ticker: Quote(ticker, 99.5, 99.7, 100, 100, 99.6, None)
    )
    async with session_scope() as session:
        decision = await repo.save_llm_decision(
            session,
            ticker="AAPL",
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
        await repo.open_trade(
            session,
            ticker="AAPL",
            trade_date=TODAY,
            entry_price=100.0,
            quantity=5,
            llm_decision_id=decision.id,
            target_sell_price=102.0,
            max_holding_time_minutes=30,
        )

    broker = DryRunBrokerClient()
    broker._positions["AAPL"] = 5  # noqa: SLF001 -- seeding an open position for the test

    await scheduler.run_eod_liquidation(broker=broker, settings=_settings())

    async with session_scope() as session:
        trades = await repo.get_open_trades(session, ticker="AAPL")
        assert trades == []  # trade was closed by the liquidation sweep
