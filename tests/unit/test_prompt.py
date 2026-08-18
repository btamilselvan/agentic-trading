import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

from agentic_trading.llm.prompt import build_prompt
from agentic_trading.llm.schema import TickerState
from agentic_trading.market_data.bucket_builder import build_bucket
from agentic_trading.market_data.robinhood_client import HistoricalBar, Quote


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


def test_prompt_includes_book_depth_fields():
    # Regression test: bid_size/ask_size/book_imbalance (spec 3.1's "order book depth
    # imbalance") were computed and persisted but never reached the LLM payload
    # before _bucket_to_dict included them -- guard against that gap recurring.
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    quote = Quote(
        symbol="AAPL", bid_price=99.9, ask_price=100.1, bid_size=750, ask_size=250,
        last_trade_price=100.0, updated_at=None,
    )
    bucket = build_bucket(
        HistoricalBar("AAPL", t0, 100, 101, 99, 100.5, 1000), quote=quote, lookback_bars=[]
    )
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [bucket], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    bucket_payload = payload["buckets"][0]
    assert bucket_payload["bid_size"] == 750
    assert bucket_payload["ask_size"] == 250
    assert bucket_payload["book_imbalance"] == 0.5
    assert "book_imbalance" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions


def test_prompt_serializes_when_lookback_buckets_carry_decimal_fields():
    # Regression test: lookback buckets read back from the DB come back as Decimal
    # (state/models.py's Bucket columns are SQLAlchemy Numeric, despite the
    # `Mapped[float]` type hint), while the freshly-built current bucket holds plain
    # floats. json.dumps chokes on a raw Decimal, so a real poll cycle -- which mixes
    # both in one bucket_history list -- failed with "Object of type Decimal is not
    # JSON serializable" before _bucket_to_dict normalized everything to float.
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    t1 = datetime(2026, 8, 17, 9, 35, tzinfo=UTC)
    db_bucket = build_bucket(
        HistoricalBar("AAPL", t0, 100, 101, 99, 100.5, 1000), quote=None, lookback_bars=[]
    )
    db_bucket = replace(
        db_bucket,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        bid_price=Decimal("100.4"),
        ask_price=Decimal("100.6"),
        spread=Decimal("0.2"),
        candle_body=Decimal("0.5"),
        upper_wick=Decimal("0.5"),
        lower_wick=Decimal("1.0"),
        rvol=Decimal("1.2311"),
    )
    fresh_bucket = build_bucket(
        HistoricalBar("AAPL", t1, 100.5, 103, 100, 102.5, 1500), quote=None, lookback_bars=[]
    )
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [db_bucket, fresh_bucket], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert payload["buckets"][0]["close"] == 100.5
    assert isinstance(payload["buckets"][0]["close"], float)


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
