from datetime import UTC, datetime, timedelta

from agentic_trading.market_data.bucket_builder import (
    build_bucket,
    build_market_context,
    compute_rsi,
    compute_rvol,
    compute_vwap,
    detect_vwap_cross,
    find_prior_close,
    minutes_since_open,
    pct_change,
    rsi_centerline_cross,
    session_phase,
)
from agentic_trading.market_data.robinhood_client import HistoricalBar, Quote


def _bar(begins_at, open_, high, low, close, volume, symbol="AAPL"):
    return HistoricalBar(
        symbol=symbol,
        begins_at=begins_at,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def test_bullish_bar_estimates_more_buy_than_sell_volume():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 105, 99, 104.5, 10_000)
    bucket = build_bucket(bar, quote=None, lookback_bars=[])
    assert bucket.est_buy_volume > bucket.est_sell_volume
    assert bucket.est_buy_volume + bucket.est_sell_volume == bar.volume


def test_bearish_bar_estimates_more_sell_than_buy_volume():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 95, 95.5, 10_000)
    bucket = build_bucket(bar, quote=None, lookback_bars=[])
    assert bucket.est_sell_volume > bucket.est_buy_volume
    assert bucket.est_buy_volume + bucket.est_sell_volume == bar.volume


def test_flat_range_bar_splits_volume_evenly():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 100, 100, 100, 10_000)
    bucket = build_bucket(bar, quote=None, lookback_bars=[])
    assert bucket.est_buy_volume == bucket.est_sell_volume == 5_000


def test_candle_stats_computed_from_ohlc():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 106, 98, 103, 1_000)
    bucket = build_bucket(bar, quote=None, lookback_bars=[])
    assert bucket.candle_body == 3  # |close - open|
    assert bucket.upper_wick == 3  # high - max(open, close)
    assert bucket.lower_wick == 2  # min(open, close) - low


def test_bucket_end_is_five_minutes_after_start():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    bucket = build_bucket(bar, quote=None, lookback_bars=[])
    assert bucket.bucket_end - bucket.bucket_start == timedelta(minutes=5)


def test_quote_populates_bid_ask_and_spread():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    quote = Quote(
        symbol="AAPL", bid_price=99.9, ask_price=100.1, bid_size=500, ask_size=300,
        last_trade_price=100.0, updated_at=None,
    )
    bucket = build_bucket(bar, quote=quote, lookback_bars=[])
    assert bucket.bid_price == 99.9
    assert bucket.ask_price == 100.1
    assert bucket.bid_size == 500
    assert bucket.ask_size == 300
    assert round(bucket.spread, 2) == 0.2


def test_no_quote_leaves_bid_ask_spread_none():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    bucket = build_bucket(bar, quote=None, lookback_bars=[])
    assert bucket.bid_price is None
    assert bucket.spread is None
    assert bucket.book_imbalance is None


def test_book_imbalance_positive_when_bid_size_dominates():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    quote = Quote(
        symbol="AAPL", bid_price=99.9, ask_price=100.1, bid_size=750, ask_size=250,
        last_trade_price=100.0, updated_at=None,
    )
    bucket = build_bucket(bar, quote=quote, lookback_bars=[])
    assert bucket.book_imbalance == 0.5  # (750-250)/1000


def test_book_imbalance_negative_when_ask_size_dominates():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    quote = Quote(
        symbol="AAPL", bid_price=99.9, ask_price=100.1, bid_size=100, ask_size=300,
        last_trade_price=100.0, updated_at=None,
    )
    bucket = build_bucket(bar, quote=quote, lookback_bars=[])
    assert round(bucket.book_imbalance, 4) == -0.5  # (100-300)/400


def test_book_imbalance_zero_when_sizes_are_balanced():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    quote = Quote(
        symbol="AAPL", bid_price=99.9, ask_price=100.1, bid_size=400, ask_size=400,
        last_trade_price=100.0, updated_at=None,
    )
    bucket = build_bucket(bar, quote=quote, lookback_bars=[])
    assert bucket.book_imbalance == 0.0


def test_book_imbalance_none_when_sizes_are_both_zero():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    quote = Quote(
        symbol="AAPL", bid_price=99.9, ask_price=100.1, bid_size=0, ask_size=0,
        last_trade_price=100.0, updated_at=None,
    )
    bucket = build_bucket(bar, quote=quote, lookback_bars=[])
    assert bucket.book_imbalance is None


def test_rvol_none_without_lookback_history():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    assert compute_rvol(bar, lookback_bars=[]) is None


def test_rvol_averages_same_time_of_day_across_prior_days():
    today = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bar = _bar(today, 100, 101, 99, 100, 3_000)
    lookback = [
        _bar(today - timedelta(days=1), 0, 0, 0, 0, 1_000),
        _bar(today - timedelta(days=2), 0, 0, 0, 0, 2_000),
        # different time-of-day slot -- must be excluded from the baseline
        _bar(today - timedelta(days=1, minutes=-5), 0, 0, 0, 0, 999_999),
    ]
    # baseline = mean(1000, 2000) = 1500; rvol = 3000 / 1500 = 2.0
    assert compute_rvol(bar, lookback) == 2.0


def test_rvol_none_when_baseline_is_zero():
    today = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bar = _bar(today, 100, 101, 99, 100, 3_000)
    lookback = [_bar(today - timedelta(days=1), 0, 0, 0, 0, 0)]
    assert compute_rvol(bar, lookback) is None


def test_find_prior_close_none_without_lookback_history():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 1_000)
    assert find_prior_close(bar, lookback_bars=[]) is None


def test_find_prior_close_ignores_same_day_bars():
    today = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bar = _bar(today, 100, 101, 99, 100, 1_000)
    # only earlier bars from today itself -- not a "prior" day
    lookback = [_bar(today - timedelta(minutes=25), 0, 0, 0, 42, 1)]
    assert find_prior_close(bar, lookback) is None


def test_find_prior_close_picks_the_last_bar_of_the_most_recent_prior_day():
    today = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bar = _bar(today, 100, 101, 99, 100, 1_000)
    lookback = [
        # two days ago -- should be ignored in favor of yesterday
        _bar(today - timedelta(days=2, hours=6), 0, 0, 0, 88, 1),
        # yesterday, earlier in the session -- not the last bar of that day
        _bar(today - timedelta(days=1, hours=6), 0, 0, 0, 95, 1),
        # yesterday's actual last bar (closest to market close)
        _bar(today - timedelta(days=1, hours=1), 0, 0, 0, 97.5, 1),
    ]
    assert find_prior_close(bar, lookback) == 97.5


def test_compute_vwap_none_without_any_volume():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 101, 99, 100, 0)
    assert compute_vwap([bar]) is None


def test_compute_vwap_single_bar_is_its_own_typical_price():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 106, 98, 103, 1_000)
    # typical price = (high + low + close) / 3
    assert compute_vwap([bar]) == (106 + 98 + 103) / 3


def test_compute_vwap_accumulates_volume_weighted_across_bars():
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bars = [
        _bar(t0, 100, 102, 98, 100, 1_000),  # typical = 100, value = 100_000
        # typical = 102, value = 306_000
        _bar(t0 + timedelta(minutes=5), 100, 104, 100, 102, 3_000),
    ]
    # vwap = (100_000 + 306_000) / (1_000 + 3_000) = 101.5
    assert compute_vwap(bars) == 101.5


def test_build_bucket_defaults_vwap_to_single_bar_when_today_bars_omitted():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 106, 98, 103, 1_000)
    bucket = build_bucket(bar, quote=None, lookback_bars=[])
    assert bucket.vwap == (106 + 98 + 103) / 3


def test_build_bucket_uses_today_bars_for_vwap_when_provided():
    t0 = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)
    bar0 = _bar(t0, 100, 102, 98, 100, 1_000)
    bar1 = _bar(t1, 100, 104, 100, 102, 3_000)
    bucket = build_bucket(bar1, quote=None, lookback_bars=[], today_bars=[bar0, bar1])
    assert bucket.vwap == 101.5  # same accumulation as test_compute_vwap_accumulates_...


def test_pct_change_basic():
    assert pct_change(103.0, 100.0) == 3.0
    assert pct_change(97.0, 100.0) == -3.0


def test_pct_change_none_when_either_side_missing_or_reference_zero():
    assert pct_change(None, 100.0) is None
    assert pct_change(100.0, None) is None
    assert pct_change(100.0, 0.0) is None


def test_build_market_context_none_fields_without_any_bars():
    ctx = build_market_context("SPY", bars_today=[], lookback_bars=[])
    assert ctx.ticker == "SPY"
    assert ctx.change_pct is None
    assert ctx.vwap_deviation_pct is None
    assert ctx.range_pct is None


def test_build_market_context_computes_change_vwap_deviation_and_range():
    today = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    # prior day's last bar closes at 400 -- the benchmark's "prior close"
    lookback = [_bar(today - timedelta(days=1, hours=1), 0, 0, 0, 400.0, 1, symbol="SPY")]
    bars_today = [
        _bar(today, 404, 406, 402, 404, 1_000, symbol="SPY"),
        _bar(today + timedelta(minutes=5), 404, 410, 404, 408, 1_000, symbol="SPY"),
    ]

    ctx = build_market_context("SPY", bars_today, lookback)

    assert ctx.ticker == "SPY"
    # latest close vs prior close
    assert round(ctx.change_pct, 4) == round((408 - 400) / 400 * 100, 4)
    vwap = compute_vwap(bars_today)
    assert round(ctx.vwap_deviation_pct, 4) == round((408 - vwap) / vwap * 100, 4)
    assert round(ctx.range_pct, 4) == round((410 - 402) / 404 * 100, 4)  # day high/low vs day open


def test_detect_vwap_cross_up_on_reclaim_from_below():
    assert detect_vwap_cross(prev_close=99.0, prev_vwap=100.0, close=101.0, vwap=100.5) == "up"


def test_detect_vwap_cross_down_on_breakdown_from_above():
    assert detect_vwap_cross(prev_close=101.0, prev_vwap=100.0, close=99.0, vwap=100.5) == "down"


def test_detect_vwap_cross_none_when_staying_on_the_same_side():
    # stayed above both times
    assert detect_vwap_cross(prev_close=101.0, prev_vwap=100.0, close=102.0, vwap=100.5) is None
    # stayed below both times
    assert detect_vwap_cross(prev_close=99.0, prev_vwap=100.0, close=98.0, vwap=100.5) is None


def test_detect_vwap_cross_none_at_exact_vwap_boundary():
    # close == vwap is treated as "not above" on both sides -- no crossing recorded
    assert detect_vwap_cross(prev_close=100.0, prev_vwap=100.0, close=100.5, vwap=100.5) is None


def test_detect_vwap_cross_none_when_any_input_missing():
    assert detect_vwap_cross(None, 100.0, 101.0, 100.5) is None
    assert detect_vwap_cross(99.0, None, 101.0, 100.5) is None
    assert detect_vwap_cross(99.0, 100.0, None, 100.5) is None
    assert detect_vwap_cross(99.0, 100.0, 101.0, None) is None


def test_minutes_since_open_zero_at_session_start():
    session_start = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    assert minutes_since_open(session_start, session_start) == 0


def test_minutes_since_open_computes_elapsed_minutes():
    session_start = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bucket_start = session_start + timedelta(minutes=45)
    assert minutes_since_open(bucket_start, session_start) == 45


def test_session_phase_opening_volatility_before_30_minutes():
    assert session_phase(0) == "OPENING_VOLATILITY"
    assert session_phase(29) == "OPENING_VOLATILITY"


def test_session_phase_morning_trend_between_30_and_120_minutes():
    assert session_phase(30) == "MORNING_TREND"
    assert session_phase(119) == "MORNING_TREND"


def test_session_phase_midday_chop_after_120_minutes():
    assert session_phase(120) == "MIDDAY_CHOP"
    assert session_phase(240) == "MIDDAY_CHOP"


def _closes_bars(closes, start=datetime(2026, 8, 17, 9, 30, tzinfo=UTC)):
    return [
        _bar(start + timedelta(minutes=5 * i), c, c, c, c, 100) for i, c in enumerate(closes)
    ]


def test_compute_rsi_none_without_enough_bars():
    # period=2 needs 2 deltas -> 3 closes; only 2 given
    assert compute_rsi(_closes_bars([100, 102]), period=2) is None


def test_compute_rsi_seeds_from_simple_average_of_first_period_deltas():
    # closes [100, 102, 101] -> deltas [+2, -1] -> avg_gain=1, avg_loss=0.5
    # rs=2 -> rsi = 100 - 100/3 = 66.6667
    rsi = compute_rsi(_closes_bars([100, 102, 101]), period=2)
    assert round(rsi, 4) == round(100 - 100 / 3, 4)


def test_compute_rsi_100_when_every_bar_in_window_gained():
    rsi = compute_rsi(_closes_bars([100, 101, 102]), period=2)
    assert rsi == 100.0


def test_compute_rsi_0_when_every_bar_in_window_lost():
    rsi = compute_rsi(_closes_bars([102, 101, 100]), period=2)
    assert rsi == 0.0


def test_compute_rsi_applies_wilder_smoothing_beyond_the_seed_window():
    # closes [100, 102, 101, 103] -> deltas [+2, -1, +2]
    # seed (first 2 deltas): avg_gain=1, avg_loss=0.5
    # smoothed with 3rd delta (+2, loss=0): avg_gain=(1*1+2)/2=1.5, avg_loss=(0.5*1+0)/2=0.25
    # rs=6 -> rsi = 100 - 100/7
    rsi = compute_rsi(_closes_bars([100, 102, 101, 103]), period=2)
    assert round(rsi, 4) == round(100 - 100 / 7, 4)


def test_build_bucket_rsi_none_when_today_bars_too_short():
    bar = _bar(datetime(2026, 8, 17, 9, 30, tzinfo=UTC), 100, 106, 98, 103, 1_000)
    bucket = build_bucket(bar, quote=None, lookback_bars=[], today_bars=[bar], rsi_period=14)
    assert bucket.rsi is None


def test_build_bucket_uses_today_bars_and_rsi_period_for_rsi():
    bars = _closes_bars([100, 102, 101])
    bucket = build_bucket(bars[-1], quote=None, lookback_bars=[], today_bars=bars, rsi_period=2)
    assert round(bucket.rsi, 4) == round(100 - 100 / 3, 4)


def test_rsi_centerline_cross_up_when_crossing_above_50():
    assert rsi_centerline_cross(prev_rsi=45.0, rsi=55.0) == "up"


def test_rsi_centerline_cross_down_when_crossing_below_50():
    assert rsi_centerline_cross(prev_rsi=55.0, rsi=45.0) == "down"


def test_rsi_centerline_cross_none_when_staying_on_the_same_side():
    assert rsi_centerline_cross(prev_rsi=55.0, rsi=60.0) is None
    assert rsi_centerline_cross(prev_rsi=45.0, rsi=40.0) is None


def test_rsi_centerline_cross_none_when_either_rsi_missing():
    assert rsi_centerline_cross(None, 55.0) is None
    assert rsi_centerline_cross(45.0, None) is None
