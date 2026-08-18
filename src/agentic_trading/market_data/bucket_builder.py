"""Turns raw robin_stocks market data into a MetricBucket ready to persist.

Neither robin_stocks nor the Robinhood MCP expose a trade-by-trade tape, so "buy
volume" / "sell volume" (spec section 3.1) is an ESTIMATE, not a read of classified
trades: a Chaikin-style money-flow-multiplier proxy based on where the bar closed
within its high/low range. Closing near the high implies buying pressure dominated
the bar; closing near the low implies selling pressure did. This is a standard,
well-understood approximation for OHLCV-only data.

Similarly "order book depth imbalance" is approximated from the top-of-book bid/ask
*size* only (no L2 depth feed is available from either data source) -- see
`_book_imbalance`, a normalized [-1, 1] reading of which side has more resting size.

VWAP is likewise a bar-based approximation (typical price = (H+L+C)/3 per bar,
volume-weighted across the session) rather than computed from a real trade tape --
see `compute_vwap`. This is the standard approach when only OHLCV bars are
available and is what most retail charting tools compute anyway.

`build_market_context` gives the LLM broad-market context (a ticker in isolation
looks the same during a market-wide rally as during a market-wide selloff, but the
setup's reliability is very different) -- built from a benchmark ticker (SPY by
default, see config.market_benchmark_ticker) using the same bar-based machinery as
everything else here. No VIX: Robinhood/robin_stocks don't reliably expose index
quotes, so `range_pct` (today's benchmark high-low range as a % of its open) stands
in as a realized-volatility proxy instead.

`detect_vwap_cross` labels the specific bucket-to-bucket transition where price
crosses its own session VWAP (a "VWAP reclaim" or "VWAP breakdown") -- llm/prompt.py
uses it, plus per-bucket close_change_pct/volume_change_pct, so a small local model
doesn't have to infer momentum/volume trend or a crossing event by eyeballing a raw
series of buckets itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from agentic_trading.market_data.robinhood_client import HistoricalBar, Quote


@runtime_checkable
class BucketLike(Protocol):
    """Structural type satisfied by both MetricBucket (below) and the Bucket ORM
    model (state/models.py) -- the LLM prompt builder only needs attribute access, so
    buckets read back from the DB can be fed to it directly, with no conversion step.
    """

    ticker: str
    bucket_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    est_buy_volume: int
    est_sell_volume: int
    bid_price: float | None
    ask_price: float | None
    bid_size: int | None
    ask_size: int | None
    spread: float | None
    book_imbalance: float | None
    candle_body: float
    upper_wick: float
    lower_wick: float
    rvol: float | None
    vwap: float | None


@dataclass(frozen=True)
class MetricBucket:
    ticker: str
    bucket_start: datetime
    bucket_end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    est_buy_volume: int
    est_sell_volume: int
    bid_price: float | None
    ask_price: float | None
    bid_size: int | None
    ask_size: int | None
    spread: float | None
    book_imbalance: float | None
    candle_body: float
    upper_wick: float
    lower_wick: float
    rvol: float | None
    vwap: float | None


def pct_change(value: float | None, reference: float | None) -> float | None:
    """`value` vs. `reference` as a percentage -- e.g. gap_pct (today's open vs.
    prior close), vwap_deviation_pct (close vs. vwap), and market_context's
    change_pct/vwap_deviation_pct all reduce to this. Centralized here (rather than
    duplicated per call site) so a small local LLM is never asked to do this
    arithmetic itself -- see gap 5 (computed indicators).
    """
    if value is None or reference is None or reference == 0:
        return None
    return (value - reference) / reference * 100


def _estimate_buy_sell_volume(bar: HistoricalBar) -> tuple[int, int]:
    rng = bar.high - bar.low
    if rng <= 0 or bar.volume <= 0:
        half = bar.volume // 2
        return half, bar.volume - half
    buy_ratio = ((bar.close - bar.low) - (bar.high - bar.close)) / rng  # in [-1, 1]
    buy_fraction = (1 + buy_ratio) / 2  # in [0, 1]
    est_buy = round(bar.volume * buy_fraction)
    est_sell = bar.volume - est_buy
    return est_buy, est_sell


def _candle_stats(bar: HistoricalBar) -> tuple[float, float, float]:
    body = abs(bar.close - bar.open)
    upper_wick = bar.high - max(bar.open, bar.close)
    lower_wick = min(bar.open, bar.close) - bar.low
    return body, upper_wick, lower_wick


def _book_imbalance(bid_size: int | None, ask_size: int | None) -> float | None:
    """Top-of-book depth imbalance (spec 3.1's "order book depth imbalance"), the
    piece of spec 3.1 that was computed (bid_size/ask_size, see quote fetch) and
    persisted but never actually reached the LLM prompt -- see llm/prompt.py.

    Normalized to [-1, 1]: positive means more resting size on the bid (buying
    pressure at the top of book), negative means more on the ask (selling
    pressure). None if either side's size is unavailable or both are zero.
    """
    if bid_size is None or ask_size is None:
        return None
    total = bid_size + ask_size
    if total <= 0:
        return None
    return (bid_size - ask_size) / total


def compute_rvol(bar: HistoricalBar, lookback_bars: list[HistoricalBar]) -> float | None:
    """`bar` volume relative to the historical average volume in the same
    time-of-day 5-minute slot, using `lookback_bars` (typically a week of 5-min
    history, excluding today). None if there isn't enough history to compare against.
    """
    same_slot_volumes = [
        b.volume
        for b in lookback_bars
        if b.begins_at.time() == bar.begins_at.time() and b.begins_at.date() != bar.begins_at.date()
    ]
    if not same_slot_volumes:
        return None
    baseline = sum(same_slot_volumes) / len(same_slot_volumes)
    if baseline <= 0:
        return None
    return bar.volume / baseline


def find_prior_close(bar: HistoricalBar, lookback_bars: list[HistoricalBar]) -> float | None:
    """The most recent prior trading day's closing price, from `lookback_bars` (the
    same week of 5-min history already fetched for the RVOL baseline -- no extra API
    call needed). Lets the LLM detect a gap-up/gap-down open, one of the four named
    intraday setups (spec 3.2, "morning breakout") that's otherwise invisible when
    only today's own bars are in view. None if there's no prior-day history yet.
    """
    prior_day_bars = [b for b in lookback_bars if b.begins_at.date() < bar.begins_at.date()]
    if not prior_day_bars:
        return None
    return max(prior_day_bars, key=lambda b: b.begins_at).close


def compute_vwap(bars_today: list[HistoricalBar]) -> float | None:
    """Session VWAP as of the most recent bar in `bars_today` (today's 5-min bars
    from market open through now, inclusive): cumulative(typical_price * volume) /
    cumulative(volume), where typical_price = (high + low + close) / 3 per bar. The
    standard intraday momentum reference line -- price above VWAP with volume
    confirms buying pressure, below confirms selling pressure -- and one most
    intraday strategies anchor to, but which this engine didn't compute at all
    before. None if there's no volume yet.
    """
    total_volume = sum(b.volume for b in bars_today)
    if total_volume <= 0:
        return None
    total_value = sum(((b.high + b.low + b.close) / 3) * b.volume for b in bars_today)
    return total_value / total_volume


def build_bucket(
    bar: HistoricalBar,
    quote: Quote | None,
    lookback_bars: list[HistoricalBar],
    today_bars: list[HistoricalBar] | None = None,
) -> MetricBucket:
    """`today_bars` should be all of today's 5-min bars from market open through
    `bar`, inclusive (used for the session VWAP) -- defaults to just `[bar]` itself
    if omitted, which degrades gracefully to a single-bar VWAP rather than raising,
    since most callers (tests, anything not wiring up the live poll loop) don't care
    about VWAP accumulation.
    """
    est_buy, est_sell = _estimate_buy_sell_volume(bar)
    body, upper_wick, lower_wick = _candle_stats(bar)
    spread = None
    if quote and quote.bid_price is not None and quote.ask_price is not None:
        spread = quote.ask_price - quote.bid_price
    bid_size = quote.bid_size if quote else None
    ask_size = quote.ask_size if quote else None
    return MetricBucket(
        ticker=bar.symbol,
        bucket_start=bar.begins_at,
        bucket_end=bar.begins_at + timedelta(minutes=5),
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        est_buy_volume=est_buy,
        est_sell_volume=est_sell,
        bid_price=quote.bid_price if quote else None,
        ask_price=quote.ask_price if quote else None,
        bid_size=bid_size,
        ask_size=ask_size,
        spread=spread,
        book_imbalance=_book_imbalance(bid_size, ask_size),
        candle_body=body,
        upper_wick=upper_wick,
        lower_wick=lower_wick,
        rvol=compute_rvol(bar, lookback_bars),
        vwap=compute_vwap(today_bars if today_bars is not None else [bar]),
    )


@dataclass(frozen=True)
class MarketContext:
    """Broad-market snapshot (spec 3.2 doesn't name this explicitly, but every
    named intraday setup -- breakout, absorption, mean reversion, continuation --
    reads very differently depending on whether the whole market is moving with the
    ticker or against it). Ephemeral: built fresh each poll cycle from the
    benchmark's own bars, never persisted per-ticker-bucket.
    """

    ticker: str
    change_pct: float | None  # benchmark's move today, prior close -> latest close
    vwap_deviation_pct: float | None  # benchmark's latest close vs. its own session VWAP
    range_pct: float | None  # today's high-low range as a % of open -- volatility proxy


def build_market_context(
    ticker: str, bars_today: list[HistoricalBar], lookback_bars: list[HistoricalBar]
) -> MarketContext:
    if not bars_today:
        return MarketContext(
            ticker=ticker, change_pct=None, vwap_deviation_pct=None, range_pct=None
        )

    latest = bars_today[-1]
    prior_close = find_prior_close(latest, lookback_bars)
    vwap = compute_vwap(bars_today)
    day_open = bars_today[0].open
    day_high = max(b.high for b in bars_today)
    day_low = min(b.low for b in bars_today)
    return MarketContext(
        ticker=ticker,
        change_pct=pct_change(latest.close, prior_close),
        vwap_deviation_pct=pct_change(latest.close, vwap),
        range_pct=(day_high - day_low) / day_open * 100 if day_open else None,
    )


def detect_vwap_cross(
    prev_close: float | None,
    prev_vwap: float | None,
    close: float | None,
    vwap: float | None,
) -> str | None:
    """"up" if the previous bucket closed at/below its own session VWAP and this
    bucket closed above (a VWAP reclaim); "down" for the reverse (a VWAP breakdown);
    None if there's no crossing, no prior bucket to compare against, or either
    side's VWAP/close is unavailable.
    """
    if None in (prev_close, prev_vwap, close, vwap):
        return None
    was_above = prev_close > prev_vwap
    is_above = close > vwap
    if was_above == is_above:
        return None
    return "up" if is_above else "down"
