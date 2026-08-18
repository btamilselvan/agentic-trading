import json
from datetime import UTC, datetime

from agentic_trading.llm.prompt import build_prompt
from agentic_trading.llm.schema import TickerState
from agentic_trading.market_data.bucket_builder import build_bucket
from agentic_trading.market_data.robinhood_client import HistoricalBar


def test_prompt_contains_full_bucket_history_and_ticker_state():
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    t1 = datetime(2026, 8, 17, 9, 35, tzinfo=UTC)
    bars = [
        HistoricalBar("AAPL", t0, 100, 101, 99, 100.5, 1000),
        HistoricalBar("AAPL", t1, 100.5, 103, 100, 102.5, 1500),
    ]
    buckets = [build_bucket(bar, quote=None, lookback_bars=[]) for bar in bars]
    state = TickerState(completed_trades_today=1, open_positions=0, realized_pnl_today=12.5)

    prompt = build_prompt("AAPL", buckets, state)

    assert "AAPL" in prompt
    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert payload["ticker"] == "AAPL"
    assert len(payload["buckets"]) == 2
    assert payload["buckets"][0]["close"] == 100.5
    assert payload["ticker_state_today"] == {
        "completed_trades": 1,
        "open_positions": 0,
        "realized_pnl": 12.5,
    }


def test_prompt_includes_decision_contract_fields():
    prompt = build_prompt("AAPL", [], TickerState(0, 0, 0.0))
    for field in (
        "decision",
        "confidence_score",
        "buy_limit_price",
        "target_sell_price",
        "max_holding_time_minutes",
        "pattern_reasoning",
    ):
        assert field in prompt
