from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
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
    Order,
    OrderSide,
    OrderStatus,
    TradingModeEnum,
)
from agentic_trading.state.ticker_state_store import InMemoryTickerStateStore

TODAY = date(2026, 8, 17)
BUCKET_START = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
# Settings() defaults to timezone="America/New_York" -- run_poll_cycle now enforces
# the exact MARKET_OPEN_TIME-EVALUATION_WINDOW_END_TIME window against "now", so
# tests need a "now" that actually falls inside it rather than relying on whatever
# the real wall clock happens to be when the suite runs.
MID_WINDOW = datetime(2026, 8, 17, 10, 0, tzinfo=ZoneInfo("America/New_York"))
# Shared across the many tests below that don't care about the decision content
# itself, just that a HOLD came back -- thesis_continuity_flag is required now
# (requirements.md section 8) but irrelevant to what these tests assert.
_HOLD_DECISION = TradeDecision(decision="HOLD", confidence_score=0.2, thesis_continuity_flag=True)


@pytest.fixture(autouse=True)
def _no_schwab(monkeypatch):
    """scheduler.py now calls market_data.market_data_client (Schwab-primary,
    Robinhood-fallback) instead of robinhood_client directly (Phase 4). Every test
    below still monkeypatches scheduler.rh.get_quote/get_5min_historicals -- that
    keeps working because market_data_client falls back to the same robinhood_client
    module object, but only if Schwab itself reports unavailable first. Force that
    deterministically here rather than relying on SCHWAB_CLIENT_ID/SECRET happening
    to be unset in whatever environment the suite runs in.
    """
    monkeypatch.setattr(scheduler.mdc.schwab_client, "get_quote", lambda ticker: None)
    monkeypatch.setattr(
        scheduler.mdc.schwab_client,
        "get_5min_historicals",
        lambda ticker, start_datetime, end_datetime: [],
    )


@pytest.fixture(autouse=True)
def _fake_ticker_state_store(monkeypatch):
    """run_poll_cycle defaults its ticker_state_store via
    scheduler.get_ticker_state_store() when the caller doesn't pass one -- every
    test below relies on that default (none pass ticker_state_store explicitly),
    so patch it to an isolated in-memory fake. Without this, these tests would
    silently talk to (and leave test data in) whatever real Redis happens to be
    reachable at REDIS_URL.
    """
    store = InMemoryTickerStateStore()
    monkeypatch.setattr(scheduler, "get_ticker_state_store", lambda: store)
    return store


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


class CapturingLLMClient:
    """Records the TickerState it was called with, per ticker, so tests can assert
    on what scheduler.py threaded through (e.g. market_context fields) without
    having to parse the rendered prompt text.
    """

    def __init__(self, decision: TradeDecision):
        self.decision = decision
        self.ticker_states: dict[str, TickerState] = {}

    async def decide(self, ticker, bucket_history, ticker_state: TickerState):
        self.ticker_states[ticker] = ticker_state
        return self.decision, "prompt-text", "{}"


class ExplodingLLMClient:
    async def decide(self, ticker, bucket_history, ticker_state):
        raise AssertionError("LLM should not have been called for an ineligible ticker")


def _patch_catalyst_context(monkeypatch):
    """Neutral (no news, no float) stub for the catalyst-context fetch every
    _poll_ticker call now makes -- without this, tests that don't care about gap 8
    at all would fall through to the real robin_stocks calls.
    """
    monkeypatch.setattr(scheduler.rh, "get_latest_news", lambda ticker: None)
    monkeypatch.setattr(scheduler.rh, "get_float_shares", lambda ticker: None)
    scheduler._float_shares_cache.clear()


def _patch_market_data(monkeypatch, bar: HistoricalBar, quote: Quote | None = None):
    monkeypatch.setattr(
        scheduler.rh, "get_5min_historicals", lambda ticker, span="day", bounds="regular": [bar]
    )
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: quote)
    scheduler._rvol_lookback_cache.clear()
    _patch_catalyst_context(monkeypatch)


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


async def test_run_poll_cycle_bypass_window_runs_outside_the_poll_window(db_session, monkeypatch):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    _patch_market_data(monkeypatch, bar)
    outside_window = datetime(2026, 8, 17, 20, 0, tzinfo=ZoneInfo("America/New_York"))
    llm = FakeLLMClient(_HOLD_DECISION)

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(),
        llm_client=llm,
        settings=_settings(),
        now=outside_window,
        bypass_window=True,
    )

    assert await _bucket_count("AAPL") == 1  # ran despite being outside the window


async def test_poll_cycle_persists_bucket_and_skips_reprocessing_same_bucket(
    db_session, monkeypatch
):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    _patch_market_data(monkeypatch, bar)
    llm = FakeLLMClient(_HOLD_DECISION)
    broker = DryRunBrokerClient()
    settings = _settings()

    await scheduler.run_poll_cycle(broker=broker, llm_client=llm, settings=settings, now=MID_WINDOW)
    await scheduler.run_poll_cycle(broker=broker, llm_client=llm, settings=settings, now=MID_WINDOW)

    assert await _bucket_count("AAPL") == 1
    assert llm.calls == 1  # second call was a no-op (bucket already recorded)


async def test_poll_cycle_refreshes_volume_of_already_recorded_bucket(db_session, monkeypatch):
    """Schwab (and to a lesser extent Robinhood) keeps settling a same-day candle's
    volume for a short while after it first appears -- if a later poll re-fetches
    the exact same bucket_start with a fuller/more-settled volume reading, that
    should correct the already-saved row in place rather than being silently
    discarded by the "already polled this bucket" dedup check.
    """
    bars = [HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)]

    def fake_get_5min_historicals(ticker, span="day", bounds="regular"):
        return bars

    monkeypatch.setattr(scheduler.rh, "get_5min_historicals", fake_get_5min_historicals)
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: None)
    scheduler._rvol_lookback_cache.clear()
    _patch_catalyst_context(monkeypatch)

    llm = FakeLLMClient(_HOLD_DECISION)
    broker = DryRunBrokerClient()
    settings = _settings()

    await scheduler.run_poll_cycle(broker=broker, llm_client=llm, settings=settings, now=MID_WINDOW)

    # Same bucket_start, but Schwab now reports the fully-settled volume for that
    # exact same 5-minute window.
    bars = [HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 254_321)]
    await scheduler.run_poll_cycle(broker=broker, llm_client=llm, settings=settings, now=MID_WINDOW)

    assert await _bucket_count("AAPL") == 1  # still one row, not a duplicate
    assert llm.calls == 1  # no re-processing of an already-decided bucket
    async with session_scope() as session:
        rows = await repo.get_buckets_for_ticker(
            session, "AAPL", since=datetime(2020, 1, 1, tzinfo=UTC)
        )
    assert rows[0].volume == 254_321  # refreshed in place, not left stale at 1000


async def test_run_poll_cycle_fetches_market_context_once_and_threads_it_in(
    db_session, monkeypatch
):
    spy_bar = HistoricalBar("SPY", BUCKET_START, 400, 406, 398, 404, 5_000)
    call_counts: dict[tuple[str, str], int] = {}

    def fake_get_5min_historicals(ticker, span="day", bounds="regular"):
        key = (ticker, span)
        call_counts[key] = call_counts.get(key, 0) + 1
        if ticker == "SPY":
            return [spy_bar]
        return [HistoricalBar(ticker, BUCKET_START, 100, 101, 99, 100.5, 1_000)]

    monkeypatch.setattr(scheduler.rh, "get_5min_historicals", fake_get_5min_historicals)
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: None)
    scheduler._rvol_lookback_cache.clear()
    _patch_catalyst_context(monkeypatch)

    llm = CapturingLLMClient(_HOLD_DECISION)
    settings = _settings(watchlist=["AAPL", "TSLA"], market_benchmark_ticker="SPY")

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(), llm_client=llm, settings=settings, now=MID_WINDOW
    )

    # fetched once for the whole cycle (day bars + week lookback), not once per
    # watchlist ticker -- with 2 tickers, a bug that re-fetched per ticker would
    # show up here as 4 calls instead of 2.
    assert call_counts[("SPY", "day")] == 1
    assert call_counts[("SPY", "week")] == 1
    aapl_state = llm.ticker_states["AAPL"]
    tsla_state = llm.ticker_states["TSLA"]
    assert aapl_state.market_benchmark_ticker == "SPY"
    # both tickers see the identical market snapshot for this cycle
    assert aapl_state.market_range_pct == tsla_state.market_range_pct == (406 - 398) / 400 * 100


async def test_market_context_fetch_failure_degrades_gracefully(db_session, monkeypatch):
    aapl_bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1_000)

    def fake_get_5min_historicals(ticker, span="day", bounds="regular"):
        if ticker == "SPY":
            raise RuntimeError("benchmark fetch failed")
        return [aapl_bar]

    monkeypatch.setattr(scheduler.rh, "get_5min_historicals", fake_get_5min_historicals)
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: None)
    scheduler._rvol_lookback_cache.clear()
    _patch_catalyst_context(monkeypatch)

    llm = CapturingLLMClient(_HOLD_DECISION)
    settings = _settings(watchlist=["AAPL"], market_benchmark_ticker="SPY")

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(), llm_client=llm, settings=settings, now=MID_WINDOW
    )

    assert await _bucket_count("AAPL") == 1  # per-ticker polling still succeeded
    assert llm.ticker_states["AAPL"].market_benchmark_ticker is None  # degraded, not crashed


async def test_market_benchmark_ticker_empty_disables_market_context(db_session, monkeypatch):
    aapl_bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1_000)
    call_counts: dict[str, int] = {}

    def fake_get_5min_historicals(ticker, span="day", bounds="regular"):
        call_counts[ticker] = call_counts.get(ticker, 0) + 1
        return [aapl_bar]

    monkeypatch.setattr(scheduler.rh, "get_5min_historicals", fake_get_5min_historicals)
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: None)
    scheduler._rvol_lookback_cache.clear()
    _patch_catalyst_context(monkeypatch)

    llm = CapturingLLMClient(_HOLD_DECISION)
    settings = _settings(watchlist=["AAPL"], market_benchmark_ticker="")

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(), llm_client=llm, settings=settings, now=MID_WINDOW
    )

    assert set(call_counts) == {"AAPL"}  # never fetched a benchmark at all
    assert llm.ticker_states["AAPL"].market_benchmark_ticker is None


async def test_poll_cycle_threads_catalyst_context_into_ticker_state(db_session, monkeypatch):
    from agentic_trading.market_data.robinhood_client import NewsItem

    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1_000)
    _patch_market_data(monkeypatch, bar)
    news = NewsItem(
        title="Company announces buyback",
        summary="Summary text",
        published_at=BUCKET_START,
        source="Reuters",
    )
    monkeypatch.setattr(scheduler.rh, "get_latest_news", lambda ticker: news)
    monkeypatch.setattr(scheduler.rh, "get_float_shares", lambda ticker: 15_000_000)

    llm = CapturingLLMClient(_HOLD_DECISION)

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(), llm_client=llm, settings=_settings(), now=MID_WINDOW
    )

    state = llm.ticker_states["AAPL"]
    assert state.news_headline == "Company announces buyback"
    assert state.news_summary == "Summary text"
    assert state.news_published_at == BUCKET_START
    assert state.float_shares == 15_000_000


async def test_catalyst_context_fetch_failure_degrades_gracefully(db_session, monkeypatch):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1_000)
    _patch_market_data(monkeypatch, bar)

    def raise_error(ticker):
        raise RuntimeError("news feed unavailable")

    monkeypatch.setattr(scheduler.rh, "get_latest_news", raise_error)
    monkeypatch.setattr(scheduler.rh, "get_float_shares", raise_error)

    llm = CapturingLLMClient(_HOLD_DECISION)

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(), llm_client=llm, settings=_settings(), now=MID_WINDOW
    )

    assert await _bucket_count("AAPL") == 1  # per-ticker polling still succeeded
    state = llm.ticker_states["AAPL"]
    assert state.news_headline is None
    assert state.float_shares is None  # degraded, not crashed


async def test_float_shares_is_fetched_once_per_ticker_across_poll_cycles(db_session, monkeypatch):
    # Static intraday, so re-fetching every 5-minute poll is wasted work -- see
    # scheduler._float_shares_cache.
    float_call_counts: dict[str, int] = {}
    day_call_counts: dict[str, int] = {}

    def fake_get_5min_historicals(ticker, span="day", bounds="regular"):
        if span != "day":
            return []
        day_call_counts[ticker] = day_call_counts.get(ticker, 0) + 1
        # A distinct bucket_start per call so the second poll cycle isn't skipped
        # as "already recorded".
        start = BUCKET_START + timedelta(minutes=5 * (day_call_counts[ticker] - 1))
        return [HistoricalBar(ticker, start, 100, 101, 99, 100.5, 1_000)]

    def fake_get_float_shares(ticker):
        float_call_counts[ticker] = float_call_counts.get(ticker, 0) + 1
        return 15_000_000

    monkeypatch.setattr(scheduler.rh, "get_5min_historicals", fake_get_5min_historicals)
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: None)
    monkeypatch.setattr(scheduler.rh, "get_latest_news", lambda ticker: None)
    monkeypatch.setattr(scheduler.rh, "get_float_shares", fake_get_float_shares)
    scheduler._rvol_lookback_cache.clear()
    scheduler._float_shares_cache.clear()

    llm = CapturingLLMClient(_HOLD_DECISION)
    settings = _settings(market_benchmark_ticker="")  # no SPY fetch to account for

    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(), llm_client=llm, settings=settings, now=MID_WINDOW
    )
    await scheduler.run_poll_cycle(
        broker=DryRunBrokerClient(), llm_client=llm, settings=settings, now=MID_WINDOW
    )

    assert day_call_counts["AAPL"] == 2  # sanity-check both cycles actually ran
    assert float_call_counts["AAPL"] == 1


async def test_poll_cycle_opens_and_closes_trade_on_high_confidence_buy(db_session, monkeypatch):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    quote = Quote("AAPL", 100.4, 100.6, 500, 400, 100.5, None)
    _patch_market_data(monkeypatch, bar, quote)
    decision = TradeDecision(
        decision="BUY",
        confidence_score=0.9,
        buy_limit_price=100.5,
        target_sell_price=102.0,
        stop_loss_price=99.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
        thesis_continuity_flag=True,
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


class FakeNotifier:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def notify(self, title, fields):
        self.calls.append((title, fields))


async def test_poll_cycle_paper_trades_exactly_like_dry_run(db_session, monkeypatch):
    """PAPER_TRADING runs the identical simulated order lifecycle DRY_RUN does --
    real data/LLM in, simulated buy+sell fill out, real DB updates -- the only
    difference is Order.mode is tagged PAPER_TRADING, not DRY_RUN, so paper-
    trading performance can be queried apart from ad hoc dev DRY_RUN runs.
    """
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    quote = Quote("AAPL", 100.4, 100.6, 500, 400, 100.5, None)
    _patch_market_data(monkeypatch, bar, quote)
    decision = TradeDecision(
        decision="BUY",
        confidence_score=0.9,
        buy_limit_price=100.5,
        target_sell_price=102.0,
        stop_loss_price=99.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
        thesis_continuity_flag=True,
    )
    llm = FakeLLMClient(decision)
    notifier = FakeNotifier()
    broker = DryRunBrokerClient()

    await scheduler.run_poll_cycle(
        broker=broker,
        llm_client=llm,
        settings=_settings(mode=TradingMode.PAPER_TRADING),
        notifier=notifier,
        now=MID_WINDOW,
    )

    async with session_scope() as session:
        result = await session.execute(select(LlmDecision).where(LlmDecision.ticker == "AAPL"))
        saved = result.scalars().one()
        assert saved.decision.value == "BUY"
        assert saved.acted_on is True  # simulated, but genuinely acted on

        daily_state = await repo.get_or_create_daily_state(session, "AAPL", TODAY)
        assert daily_state.completed_trades_count == 1
        assert daily_state.open_positions_count == 0

        orders = (
            await session.execute(select(Order).where(Order.ticker == "AAPL"))
        ).scalars().all()
        assert len(orders) == 2  # buy + paired sell
        assert all(o.mode == TradingModeEnum.PAPER_TRADING for o in orders)

    titles = [title for title, _ in notifier.calls]
    assert "BUY signal" in titles
    assert "Order filled" in titles
    assert "Trade closed" in titles
    buy_signal_fields = next(fields for title, fields in notifier.calls if title == "BUY signal")
    assert buy_signal_fields["mode"] == "PAPER_TRADING"


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


# --- Phase 3: IN_POSITION re-evaluation branch (requirements.md section 8) -----


class PendingFillBroker:
    """Like DryRunBrokerClient but every order stays PENDING rather than filling
    instantly -- used where a test needs to observe an OPEN trade's updated
    resting-order state without an instant fill immediately closing it out.
    """

    def __init__(self, position_qty: float = 0.0):
        self._position_qty = position_qty
        self.cancelled: list[str] = []
        self._next_id = 1

    async def get_open_position_quantity(self, ticker: str) -> float:
        return self._position_qty

    async def review_order(self, **kwargs) -> OrderReview:
        return OrderReview(warnings=[], estimated_price=kwargs.get("limit_price"))

    async def place_order(self, **kwargs) -> PlacedOrder:
        order_id = f"PENDING-{self._next_id}"
        self._next_id += 1
        return PlacedOrder(broker_order_id=order_id, status="pending", fill_price=None)

    async def cancel_order(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)


async def _seed_open_trade(
    session,
    *,
    ticker="AAPL",
    entry_price=100.0,
    target_sell_price=105.0,
    stop_loss_price=97.0,
    quantity=5,
) -> None:
    """An OPEN trade with a FILLED buy leg and a resting PENDING sell -- the
    state _manage_open_position (via repo.get_open_trade_for_ticker) expects to
    find for a ticker already IN_POSITION.
    """
    decision = await repo.save_llm_decision(
        session,
        ticker=ticker,
        bucket_id=None,
        prompt="p",
        raw_response="{}",
        decision="BUY",
        confidence_score=0.9,
        buy_limit_price=entry_price,
        target_sell_price=target_sell_price,
        stop_loss_price=stop_loss_price,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
        thesis_continuity_flag=True,
    )
    trade = await repo.open_trade(
        session,
        ticker=ticker,
        trade_date=TODAY,
        entry_price=entry_price,
        quantity=quantity,
        llm_decision_id=decision.id,
        target_sell_price=target_sell_price,
        stop_loss_price=stop_loss_price,
        max_holding_time_minutes=30,
    )
    buy_order = await repo.create_order(
        session,
        ticker=ticker,
        side=OrderSide.BUY,
        limit_price=entry_price,
        quantity=quantity,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="BUY-1",
        trade_id=trade.id,
    )
    await repo.update_order_status(session, buy_order.id, OrderStatus.FILLED)
    await repo.create_order(
        session,
        ticker=ticker,
        side=OrderSide.SELL,
        limit_price=target_sell_price,
        quantity=quantity,
        mode=TradingModeEnum.DRY_RUN,
        broker_order_id="SELL-1",
        trade_id=trade.id,
    )
    await repo.record_trade_opened(session, ticker, TODAY)


def _notified(notifier: FakeNotifier, title: str) -> dict:
    return next(fields for t, fields in notifier.calls if t == title)


async def test_poll_ticker_forces_stop_loss_exit_without_an_llm_call(
    db_session, monkeypatch, _fake_ticker_state_store
):
    bar = HistoricalBar("AAPL", BUCKET_START, 96, 96.5, 94, 95.0, 1000)  # closes at 95
    quote = Quote("AAPL", 94.9, 95.1, 500, 500, 95.0, None)
    _patch_market_data(monkeypatch, bar, quote)
    async with session_scope() as session:
        await _seed_open_trade(session, stop_loss_price=97.0)  # 95 <= 97 -> breached

    broker = DryRunBrokerClient()
    broker._positions["AAPL"] = 5  # noqa: SLF001 -- seeding an open position for the test
    llm = FakeLLMClient(_HOLD_DECISION)  # must not be called at all
    notifier = FakeNotifier()

    await scheduler.run_poll_cycle(
        broker=broker, llm_client=llm, settings=_settings(), notifier=notifier, now=MID_WINDOW
    )

    assert llm.calls == 0  # deterministic exit -- no LLM round-trip needed
    async with session_scope() as session:
        assert await repo.get_open_trades(session, ticker="AAPL") == []
    assert _notified(notifier, "Trade closed")["exit_reason"] == "STOP_LOSS"
    assert await _fake_ticker_state_store.get("AAPL", TODAY) is None  # cleared on close


async def test_poll_ticker_exits_on_llm_sell_decision(
    db_session, monkeypatch, _fake_ticker_state_store
):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    quote = Quote("AAPL", 100.4, 100.6, 500, 500, 100.5, None)
    _patch_market_data(monkeypatch, bar, quote)
    async with session_scope() as session:
        await _seed_open_trade(session, stop_loss_price=90.0)  # nowhere near breached

    broker = DryRunBrokerClient()
    broker._positions["AAPL"] = 5  # noqa: SLF001 -- seeding an open position for the test
    sell_decision = TradeDecision(
        decision="SELL",
        confidence_score=0.8,
        thesis_continuity_flag=False,
        pattern_reasoning="momentum stalled, exiting early",
    )
    llm = FakeLLMClient(sell_decision)
    notifier = FakeNotifier()

    await scheduler.run_poll_cycle(
        broker=broker, llm_client=llm, settings=_settings(), notifier=notifier, now=MID_WINDOW
    )

    assert llm.calls == 1
    async with session_scope() as session:
        assert await repo.get_open_trades(session, ticker="AAPL") == []
    assert _notified(notifier, "Trade closed")["exit_reason"] == "LLM_THESIS_BREAK"
    assert await _fake_ticker_state_store.get("AAPL", TODAY) is None


async def test_poll_ticker_holds_and_records_continuity_when_no_invalidation(
    db_session, monkeypatch, _fake_ticker_state_store
):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    quote = Quote("AAPL", 100.4, 100.6, 500, 500, 100.5, None)
    _patch_market_data(monkeypatch, bar, quote)
    async with session_scope() as session:
        await _seed_open_trade(session, stop_loss_price=90.0)

    broker = PendingFillBroker(position_qty=5)
    hold_decision = TradeDecision(
        decision="HOLD",
        confidence_score=0.7,
        thesis_continuity_flag=True,
        pattern_reasoning="still intact",
    )
    llm = FakeLLMClient(hold_decision)

    await scheduler.run_poll_cycle(
        broker=broker, llm_client=llm, settings=_settings(), now=MID_WINDOW
    )

    assert llm.calls == 1
    async with session_scope() as session:
        trades = await repo.get_open_trades(session, ticker="AAPL")
        assert len(trades) == 1  # still open -- nothing invalidated
    state = await _fake_ticker_state_store.get("AAPL", TODAY)
    assert state.status == "IN_POSITION"
    assert len(state.decision_history) == 1
    assert state.decision_history[0].decision == "HOLD"


async def test_poll_ticker_trails_target_and_stop_upward_when_enabled(
    db_session, monkeypatch, _fake_ticker_state_store
):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 105, 99, 104.0, 1000)
    quote = Quote("AAPL", 103.9, 104.1, 500, 500, 104.0, None)
    _patch_market_data(monkeypatch, bar, quote)
    async with session_scope() as session:
        await _seed_open_trade(session, target_sell_price=102.0, stop_loss_price=98.0)

    broker = PendingFillBroker(position_qty=5)
    hold_decision = TradeDecision(
        decision="HOLD",
        confidence_score=0.75,
        thesis_continuity_flag=True,
        target_sell_price=106.0,  # better than current 102.0
        stop_loss_price=99.0,  # better than current 98.0
        pattern_reasoning="still trending, ratchet up",
    )
    llm = FakeLLMClient(hold_decision)

    await scheduler.run_poll_cycle(
        broker=broker,
        llm_client=llm,
        settings=_settings(trailing_stop_enabled=True),
        now=MID_WINDOW,
    )

    async with session_scope() as session:
        trades = await repo.get_open_trades(session, ticker="AAPL")
        assert len(trades) == 1
        trade = trades[0]
        assert float(trade.target_sell_price) == 106.0
        assert float(trade.stop_loss_price) == 99.0
    # The old resting sell (target 102.0) was cancelled and replaced.
    assert "SELL-1" in broker.cancelled


async def test_poll_ticker_does_not_trail_when_disabled(
    db_session, monkeypatch, _fake_ticker_state_store
):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 105, 99, 104.0, 1000)
    quote = Quote("AAPL", 103.9, 104.1, 500, 500, 104.0, None)
    _patch_market_data(monkeypatch, bar, quote)
    async with session_scope() as session:
        await _seed_open_trade(session, target_sell_price=102.0, stop_loss_price=98.0)

    broker = PendingFillBroker(position_qty=5)
    hold_decision = TradeDecision(
        decision="HOLD",
        confidence_score=0.75,
        thesis_continuity_flag=True,
        target_sell_price=106.0,
        stop_loss_price=99.0,
        pattern_reasoning="still trending",
    )
    llm = FakeLLMClient(hold_decision)

    await scheduler.run_poll_cycle(
        broker=broker,
        llm_client=llm,
        settings=_settings(trailing_stop_enabled=False),
        now=MID_WINDOW,
    )

    assert broker.cancelled == []  # no order replacement attempted
    async with session_scope() as session:
        trades = await repo.get_open_trades(session, ticker="AAPL")
        assert float(trades[0].target_sell_price) == 102.0  # unchanged
        assert float(trades[0].stop_loss_price) == 98.0  # unchanged


async def test_poll_ticker_initializes_redis_in_position_state_on_buy_entry(
    db_session, monkeypatch, _fake_ticker_state_store
):
    bar = HistoricalBar("AAPL", BUCKET_START, 100, 101, 99, 100.5, 1000)
    quote = Quote("AAPL", 100.4, 100.6, 500, 400, 100.5, None)
    _patch_market_data(monkeypatch, bar, quote)
    decision = TradeDecision(
        decision="BUY",
        confidence_score=0.9,
        buy_limit_price=100.5,
        target_sell_price=102.0,
        stop_loss_price=99.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
        thesis_continuity_flag=True,
    )
    llm = FakeLLMClient(decision)
    # A pending-fill broker so the trade opens but doesn't instantly round-trip
    # closed again within this same cycle (DryRunBrokerClient would fill both
    # legs instantly, making status IN_POSITION observably wrong by the time this
    # test checks it).
    broker = PendingFillBroker()

    await scheduler.run_poll_cycle(
        broker=broker, llm_client=llm, settings=_settings(), now=MID_WINDOW
    )

    state = await _fake_ticker_state_store.get("AAPL", TODAY)
    assert state is not None
    assert state.status == "IN_POSITION"
    assert state.active_thesis == "breakout"
    assert state.initial_entry_price == 100.5
    assert state.target_price == 102.0
    assert state.stop_loss == 99.0
