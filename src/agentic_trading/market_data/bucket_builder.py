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


def build_bucket(
    bar: HistoricalBar, quote: Quote | None, lookback_bars: list[HistoricalBar]
) -> MetricBucket:
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
    )
