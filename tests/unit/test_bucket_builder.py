from datetime import UTC, datetime, timedelta

from agentic_trading.market_data.bucket_builder import (
    build_bucket,
    build_market_context,
    compute_rvol,
    compute_vwap,
    find_prior_close,
    pct_change,
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
