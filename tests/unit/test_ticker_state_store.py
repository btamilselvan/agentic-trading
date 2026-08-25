from datetime import date, datetime

from agentic_trading.state.ticker_state_store import (
    DecisionLogEntry,
    InMemoryTickerStateStore,
    TickerEvaluationState,
)

TODAY = date(2026, 8, 24)


async def test_fresh_state_defaults_to_flat_with_no_history():
    state = TickerEvaluationState.fresh("AAPL", TODAY)
    assert state.status == "FLAT"
    assert state.active_thesis is None
    assert state.decision_history == []


async def test_store_returns_none_for_unknown_ticker():
    store = InMemoryTickerStateStore()
    assert await store.get("AAPL", TODAY) is None


async def test_store_round_trips_state():
    store = InMemoryTickerStateStore()
    state = TickerEvaluationState(
        ticker="AAPL",
        trade_date=TODAY,
        status="IN_POSITION",
        active_thesis="morning breakout continuation",
        initial_entry_price=100.0,
        target_price=102.0,
        stop_loss=98.5,
    )

    await store.save(state)
    loaded = await store.get("AAPL", TODAY)

    assert loaded is not None
    assert loaded.status == "IN_POSITION"
    assert loaded.initial_entry_price == 100.0
    assert loaded.target_price == 102.0
    assert loaded.stop_loss == 98.5
    assert loaded.updated_at is not None  # stamped by save()


async def test_store_keys_are_isolated_by_ticker_and_date():
    store = InMemoryTickerStateStore()
    await store.save(TickerEvaluationState(ticker="AAPL", trade_date=TODAY, status="BUY"))
    await store.save(TickerEvaluationState(ticker="TSLA", trade_date=TODAY, status="HOLD"))
    await store.save(
        TickerEvaluationState(ticker="AAPL", trade_date=date(2026, 8, 25), status="FLAT")
    )

    aapl_today = await store.get("AAPL", TODAY)
    tsla_today = await store.get("TSLA", TODAY)
    aapl_tomorrow = await store.get("AAPL", date(2026, 8, 25))

    assert aapl_today.status == "BUY"
    assert tsla_today.status == "HOLD"
    assert aapl_tomorrow.status == "FLAT"


async def test_clear_removes_state():
    store = InMemoryTickerStateStore()
    await store.save(TickerEvaluationState(ticker="AAPL", trade_date=TODAY, status="IN_POSITION"))

    await store.clear("AAPL", TODAY)

    assert await store.get("AAPL", TODAY) is None


async def test_with_decision_appended_trims_to_max_history():
    state = TickerEvaluationState.fresh("AAPL", TODAY)
    for i in range(7):
        state = state.with_decision_appended(
            DecisionLogEntry(
                bucket_start=datetime(2026, 8, 24, 9, 30 + i),
                decision="HOLD",
                confidence_score=0.5,
                thesis_continuity_flag=True,
                pattern_reasoning=f"bar {i}",
            ),
            max_history=5,
        )

    assert len(state.decision_history) == 5
    # Oldest entries fell off the front -- the most recent 5 of 0..6 survive.
    assert [e.pattern_reasoning for e in state.decision_history] == [
        "bar 2",
        "bar 3",
        "bar 4",
        "bar 5",
        "bar 6",
    ]


async def test_with_decision_appended_does_not_mutate_original():
    state = TickerEvaluationState.fresh("AAPL", TODAY)
    entry = DecisionLogEntry(
        bucket_start=datetime(2026, 8, 24, 9, 30),
        decision="BUY",
        confidence_score=0.9,
        thesis_continuity_flag=True,
    )

    new_state = state.with_decision_appended(entry, max_history=5)

    assert state.decision_history == []
    assert len(new_state.decision_history) == 1
