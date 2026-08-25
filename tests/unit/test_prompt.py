import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from agentic_trading.llm.prompt import build_prompt
from agentic_trading.llm.schema import TickerState
from agentic_trading.market_data.bucket_builder import build_bucket
from agentic_trading.market_data.robinhood_client import HistoricalBar, Quote
from agentic_trading.state.ticker_state_store import DecisionLogEntry


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
        "prior_close": None,
        "today_open": 100.0,
        "gap_pct": None,
    }
    assert "market_context" not in payload  # no benchmark configured on this state


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


def test_prompt_includes_gap_context_from_prior_close():
    # Regression test for gap 2: with only today's own bars in view, the LLM can't
    # tell if today opened with a gap -- prior_close/today_open/gap_pct fix that.
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bucket = build_bucket(
        HistoricalBar("AAPL", t0, 103, 104, 102, 103.5, 1000), quote=None, lookback_bars=[]
    )
    state = TickerState(
        completed_trades_today=0, open_positions=0, realized_pnl_today=0.0, prior_close=100.0
    )

    prompt = build_prompt("AAPL", [bucket], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    ticker_state_payload = payload["ticker_state_today"]
    assert ticker_state_payload["prior_close"] == 100.0
    assert ticker_state_payload["today_open"] == 103.0
    assert round(ticker_state_payload["gap_pct"], 2) == 3.0  # (103-100)/100 * 100
    assert "gap_pct" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions


def test_prompt_gap_pct_is_none_without_prior_close_or_buckets():
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    ticker_state_payload = payload["ticker_state_today"]
    assert ticker_state_payload["prior_close"] is None
    assert ticker_state_payload["today_open"] is None
    assert ticker_state_payload["gap_pct"] is None


def test_prompt_includes_vwap_and_deviation():
    # Regression test for gap 3: VWAP wasn't computed at all before -- the standard
    # intraday momentum reference line most strategies anchor to was invisible.
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    t1 = t0.replace(minute=35)
    bar0 = HistoricalBar("AAPL", t0, 100, 102, 98, 100, 1_000)  # typical=100, value=100_000
    bar1 = HistoricalBar("AAPL", t1, 100, 104, 100, 102, 3_000)  # typical=102, value=306_000
    bucket0 = build_bucket(bar0, quote=None, lookback_bars=[], today_bars=[bar0])
    bucket1 = build_bucket(bar1, quote=None, lookback_bars=[], today_bars=[bar0, bar1])
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [bucket0, bucket1], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    second_bucket = payload["buckets"][1]
    assert second_bucket["vwap"] == 101.5  # (100_000 + 306_000) / 4_000
    assert round(second_bucket["vwap_deviation_pct"], 4) == round((102 - 101.5) / 101.5 * 100, 4)
    assert "vwap_deviation_pct" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions


def test_prompt_includes_market_context_when_present():
    # Regression test for gap 4: a ticker-specific setup was evaluated with no
    # sense of whether the broad market was moving with or against it.
    state = TickerState(
        completed_trades_today=0,
        open_positions=0,
        realized_pnl_today=0.0,
        market_benchmark_ticker="SPY",
        market_change_pct=1.5,
        market_vwap_deviation_pct=0.3,
        market_range_pct=0.8,
    )

    prompt = build_prompt("AAPL", [], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert payload["market_context"] == {
        "benchmark_ticker": "SPY",
        "change_pct": 1.5,
        "vwap_deviation_pct": 0.3,
        "range_pct": 0.8,
    }
    assert "market_context" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions


def test_prompt_omits_market_context_when_no_benchmark_configured():
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert "market_context" not in payload


def test_prompt_first_bucket_has_no_sequential_signals():
    # Regression test for gap 5: nothing to compare the first bucket of the day
    # against -- close_change_pct/volume_change_pct/vwap_cross must all be null
    # rather than crash or compare against garbage.
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bucket = build_bucket(
        HistoricalBar("AAPL", t0, 100, 101, 99, 100.5, 1_000), quote=None, lookback_bars=[]
    )
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [bucket], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    first = payload["buckets"][0]
    assert first["close_change_pct"] is None
    assert first["volume_change_pct"] is None
    assert first["vwap_cross"] is None


def test_prompt_computes_close_and_volume_change_between_consecutive_buckets():
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    t1 = t0.replace(minute=35)
    bar0 = HistoricalBar("AAPL", t0, 100, 101, 99, 100.0, 1_000)
    bar1 = HistoricalBar("AAPL", t1, 100, 104, 100, 103.0, 1_500)
    bucket0 = build_bucket(bar0, quote=None, lookback_bars=[], today_bars=[bar0])
    bucket1 = build_bucket(bar1, quote=None, lookback_bars=[], today_bars=[bar0, bar1])
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [bucket0, bucket1], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    second = payload["buckets"][1]
    assert round(second["close_change_pct"], 4) == round((103.0 - 100.0) / 100.0 * 100, 4)
    assert round(second["volume_change_pct"], 4) == round((1_500 - 1_000) / 1_000 * 100, 4)
    assert "close_change_pct" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions
    assert "vwap_cross" in prompt.split("Input:\n", 1)[0]


def test_prompt_flags_vwap_cross_on_reclaim():
    # Bucket 0 closes below its own point-in-time VWAP; bucket 1 closes above its
    # (cumulative, now larger) VWAP -- a reclaim, so vwap_cross == "up".
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    t1 = t0.replace(minute=35)
    bar0 = HistoricalBar("AAPL", t0, 100, 101, 95, 96.0, 1_000)  # typical ~97.33, closes below
    bar1 = HistoricalBar("AAPL", t1, 96, 110, 96, 109.0, 1_000)  # closes well above VWAP now
    bucket0 = build_bucket(bar0, quote=None, lookback_bars=[], today_bars=[bar0])
    bucket1 = build_bucket(bar1, quote=None, lookback_bars=[], today_bars=[bar0, bar1])
    assert bucket0.close < bucket0.vwap  # sanity-check the fixture
    assert bucket1.close > bucket1.vwap
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [bucket0, bucket1], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert payload["buckets"][0]["vwap_cross"] is None  # no prior bucket
    assert payload["buckets"][1]["vwap_cross"] == "up"


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
        book_imbalance=Decimal("0.5"),
        vwap=Decimal("100.25"),
        rsi=Decimal("55.25"),
    )
    fresh_bucket = build_bucket(
        HistoricalBar("AAPL", t1, 100.5, 103, 100, 102.5, 1500), quote=None, lookback_bars=[]
    )
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [db_bucket, fresh_bucket], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert payload["buckets"][0]["close"] == 100.5
    assert isinstance(payload["buckets"][0]["close"], float)


def test_prompt_includes_session_time_context():
    # Regression test for gap 6: minutes_since_open/session_phase weren't computed
    # at all before -- the LLM had no sense of where in the day a bucket fell.
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    t1 = t0.replace(minute=35)
    t2 = t0.replace(hour=11, minute=45)  # 135 minutes after session start
    bars = [
        HistoricalBar("AAPL", t0, 100, 101, 99, 100, 1_000),
        HistoricalBar("AAPL", t1, 100, 101, 99, 100, 1_000),
        HistoricalBar("AAPL", t2, 100, 101, 99, 100, 1_000),
    ]
    buckets = [build_bucket(bar, quote=None, lookback_bars=[]) for bar in bars]
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", buckets, state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    first, second, third = payload["buckets"]
    assert first["minutes_since_open"] == 0
    assert first["session_phase"] == "OPENING_VOLATILITY"
    assert second["minutes_since_open"] == 5
    assert second["session_phase"] == "OPENING_VOLATILITY"
    assert third["minutes_since_open"] == 135
    assert third["session_phase"] == "MIDDAY_CHOP"
    assert "session_phase" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions


def test_prompt_includes_rsi_and_centerline_cross():
    # Regression test for gap 7: RSI wasn't computed at all before -- the standard
    # overbought/oversold oscillator most intraday strategies also watch.
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bars = [
        HistoricalBar("AAPL", t0 + timedelta(minutes=5 * i), c, c, c, c, 100)
        for i, c in enumerate([100, 102, 101])
    ]
    buckets = [
        build_bucket(bars[i], quote=None, lookback_bars=[], today_bars=bars[: i + 1], rsi_period=2)
        for i in range(len(bars))
    ]
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", buckets, state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    first, second, third = payload["buckets"]
    assert first["rsi"] is None  # only 1 close -- not enough for period=2
    assert second["rsi"] is None  # only 2 closes -- still not enough (needs 3)
    assert round(third["rsi"], 4) == round(100 - 100 / 3, 4)
    # second bucket's rsi is None, so there's no prior rsi for the third bucket to
    # compare against -- no crossing can be recorded regardless of third's own rsi.
    assert third["rsi_centerline_cross"] is None
    assert "rsi_centerline_cross" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions


def test_prompt_includes_catalyst_context_when_present():
    # Regression test for gap 8: news/float were fetched but never reached the LLM.
    state = TickerState(
        completed_trades_today=0,
        open_positions=0,
        realized_pnl_today=0.0,
        news_headline="Company announces buyback",
        news_summary="Summary text",
        news_published_at=datetime(2026, 8, 19, 9, 0, tzinfo=UTC),
        float_shares=15_000_000,
    )

    prompt = build_prompt("AAPL", [], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert payload["catalyst_context"] == {
        "news_headline": "Company announces buyback",
        "news_summary": "Summary text",
        "news_published_at": "2026-08-19T09:00:00+00:00",
        "float_shares": 15_000_000,
    }
    assert "catalyst_context" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions
    assert "short interest" in prompt.split("Input:\n", 1)[0].lower()


def test_prompt_omits_catalyst_context_when_nothing_fetched():
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert "catalyst_context" not in payload


def test_prompt_includes_catalyst_context_with_float_only():
    # No news this cycle, but float_shares alone is still worth surfacing.
    state = TickerState(
        completed_trades_today=0,
        open_positions=0,
        realized_pnl_today=0.0,
        float_shares=8_000_000,
    )

    prompt = build_prompt("AAPL", [], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert payload["catalyst_context"]["float_shares"] == 8_000_000
    assert payload["catalyst_context"]["news_headline"] is None


def test_prompt_includes_decision_contract_fields():
    prompt = build_prompt("AAPL", [], TickerState(0, 0, 0.0))
    for field in (
        "decision",
        "confidence_score",
        "buy_limit_price",
        "target_sell_price",
        "stop_loss_price",
        "max_holding_time_minutes",
        "pattern_reasoning",
        "thesis_continuity_flag",
    ):
        assert field in prompt


def test_prompt_includes_position_context_defaults_when_flat():
    state = TickerState(completed_trades_today=0, open_positions=0, realized_pnl_today=0.0)

    prompt = build_prompt("AAPL", [], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    assert payload["position_context"] == {
        "status": "FLAT",
        "active_thesis": None,
        "initial_entry_price": None,
        "current_target_price": None,
        "current_stop_loss": None,
        "recent_decisions": [],
    }
    assert "position_context" in prompt.split("Input:\n", 1)[0]  # mentioned in instructions
    assert "hysteresis" in prompt.split("Input:\n", 1)[0].lower()


def test_prompt_includes_position_context_with_active_thesis_and_history():
    state = TickerState(
        completed_trades_today=0,
        open_positions=1,
        realized_pnl_today=0.0,
        status="IN_POSITION",
        active_thesis="morning breakout continuation",
        initial_entry_price=100.0,
        current_target_price=103.0,
        current_stop_loss=98.5,
        decision_history=[
            DecisionLogEntry(
                bucket_start=datetime(2026, 8, 24, 9, 30, tzinfo=UTC),
                decision="BUY",
                confidence_score=0.85,
                thesis_continuity_flag=True,
                pattern_reasoning="breakout",
            ),
            DecisionLogEntry(
                bucket_start=datetime(2026, 8, 24, 9, 35, tzinfo=UTC),
                decision="HOLD",
                confidence_score=0.8,
                thesis_continuity_flag=True,
                pattern_reasoning="still holding above VWAP",
            ),
        ],
    )

    prompt = build_prompt("AAPL", [], state)

    payload = json.loads(prompt.split("Input:\n", 1)[1])
    position_context = payload["position_context"]
    assert position_context["status"] == "IN_POSITION"
    assert position_context["active_thesis"] == "morning breakout continuation"
    assert position_context["current_target_price"] == 103.0
    assert position_context["current_stop_loss"] == 98.5
    assert len(position_context["recent_decisions"]) == 2
    assert position_context["recent_decisions"][0]["decision"] == "BUY"
    assert position_context["recent_decisions"][1]["pattern_reasoning"] == (
        "still holding above VWAP"
    )
