"""Builds the LLM prompt from the full intraday bucket history + ticker trade state.

Per spec section 3.2, the LLM sees the COMPLETE array of the day's 5-minute buckets
for the ticker (not just the latest one) so it can evaluate for same-day intraday
setups (breakout, volume absorption, mean reversion, momentum continuation).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from decimal import Decimal

from agentic_trading.llm.schema import TickerState
from agentic_trading.market_data.bucket_builder import BucketLike, detect_vwap_cross, pct_change

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTIONS = """\
You are an intraday momentum trading analyst. You are given a time-ordered series of \
5-minute market microstructure buckets for one ticker, plus that ticker's trading \
state for today. Evaluate STRICTLY for same-day intraday setups (e.g. morning \
breakout, volume absorption, quick mean reversion, momentum continuation). Do not \
consider multi-day or swing setups -- any position must be closeable within the same \
session. Each bucket's book_imbalance is the top-of-book depth skew in [-1, 1]: \
positive means more resting size on the bid (buying pressure), negative means more \
on the ask (selling pressure). ticker_state_today.gap_pct is today's open versus the \
prior session's close (positive = gapped up, negative = gapped down, null = no prior \
close available yet) -- a large gap is a precondition for a genuine "morning \
breakout" setup, as opposed to ordinary intraday drift. Each bucket's \
vwap_deviation_pct is that bucket's close versus the session VWAP so far: \
sustained positive readings with rising volume support momentum continuation or a \
breakout holding; a move back toward/through zero suggests the move is fading or \
being rejected. If present, market_context describes the broad market today via a \
benchmark index ETF (change_pct and vwap_deviation_pct mean the same thing as \
above, but for the whole market; range_pct is today's high-low range as a % of \
open, a volatility proxy). Weigh the ticker's own setup against this: a breakout \
aligned with a trending market (same direction, comparable magnitude) is more \
reliable than the same-looking breakout while the broad market is flat or moving \
the other way -- treat market_context as a modifier on confidence, not a veto. \
Each bucket's close_change_pct and volume_change_pct are versus the immediately \
preceding bucket (null on the first bucket, since there's nothing prior to compare \
against): a string of positive close_change_pct with rising volume_change_pct is \
the clearest signature of momentum continuation, while volume fading as price \
keeps climbing is a warning sign of exhaustion. vwap_cross is "up" on the bucket \
where price reclaims its session VWAP from below, "down" where it breaks below \
from above, and null otherwise -- a reclaim aligned with volume is a stronger \
breakout signal than price merely drifting above VWAP without ever having crossed \
it intraday.

Respond with a single JSON object matching this contract:
- decision: "BUY" or "HOLD"
- confidence_score: 0.0-1.0, how strongly the setup supports entering now
- buy_limit_price: required if decision is BUY -- a limit entry price near the \
current ask, sized for immediate fill
- target_sell_price: required if decision is BUY -- the intraday profit-target limit \
sell price, greater than buy_limit_price
- max_holding_time_minutes: required if decision is BUY -- max minutes to hold \
before a mandatory exit
- pattern_reasoning: a concise explanation of the order-flow setup you detected

If no valid intraday setup is present, or ticker_state_today shows this ticker \
already has an open position or has hit its trade cap for the day, respond HOLD.
"""


def _num(value: float | Decimal | None) -> float | None:
    """Normalize a numeric bucket field to float.

    Bucket rows read back from the DB come back as Decimal (state/models.py's
    columns are SQLAlchemy Numeric, despite the `Mapped[float]` type hint), while
    a freshly-built MetricBucket for the current bucket holds plain floats. Mixing
    the two in one payload made json.dumps below raise "Object of type Decimal is
    not JSON serializable" the moment lookback history was involved.
    """
    return None if value is None else float(value)


def _bucket_to_dict(bucket: BucketLike, previous: BucketLike | None) -> dict:
    close = _num(bucket.close)
    vwap = _num(bucket.vwap)
    prev_close = _num(previous.close) if previous else None
    prev_vwap = _num(previous.vwap) if previous else None
    return {
        "bucket_start": bucket.bucket_start.isoformat(),
        "open": _num(bucket.open),
        "high": _num(bucket.high),
        "low": _num(bucket.low),
        "close": close,
        "volume": bucket.volume,
        "est_buy_volume": bucket.est_buy_volume,
        "est_sell_volume": bucket.est_sell_volume,
        "bid_price": _num(bucket.bid_price),
        "ask_price": _num(bucket.ask_price),
        "bid_size": bucket.bid_size,
        "ask_size": bucket.ask_size,
        "spread": _num(bucket.spread),
        "book_imbalance": _num(bucket.book_imbalance),
        "candle_body": _num(bucket.candle_body),
        "upper_wick": _num(bucket.upper_wick),
        "lower_wick": _num(bucket.lower_wick),
        "rvol": _num(bucket.rvol),
        "vwap": vwap,
        "vwap_deviation_pct": pct_change(close, vwap),
        # Sequential signals -- vs. the immediately preceding bucket, not raw levels
        # -- so a small local model isn't left to infer momentum/volume trend or a
        # VWAP crossing itself by diffing raw numbers across the bucket array.
        "close_change_pct": pct_change(close, prev_close),
        "volume_change_pct": pct_change(bucket.volume, previous.volume if previous else None),
        "vwap_cross": detect_vwap_cross(prev_close, prev_vwap, close, vwap),
    }


def build_prompt(
    ticker: str, bucket_history: Sequence[BucketLike], ticker_state: TickerState
) -> str:
    today_open = _num(bucket_history[0].open) if bucket_history else None
    prior_close = ticker_state.prior_close
    buckets_payload = []
    previous: BucketLike | None = None
    for bucket in bucket_history:
        buckets_payload.append(_bucket_to_dict(bucket, previous))
        previous = bucket
    payload = {
        "ticker": ticker,
        "buckets": buckets_payload,
        "ticker_state_today": {
            "completed_trades": ticker_state.completed_trades_today,
            "open_positions": ticker_state.open_positions,
            "realized_pnl": ticker_state.realized_pnl_today,
            "prior_close": prior_close,
            "today_open": today_open,
            "gap_pct": pct_change(today_open, prior_close),
        },
    }
    if ticker_state.market_benchmark_ticker:
        # Omitted entirely (not sent as an all-null section) when the benchmark
        # fetch failed outright this cycle -- scheduler.py leaves
        # market_benchmark_ticker unset in that case. Individual fields inside can
        # still be null even when present (e.g. no prior-day history yet for the
        # benchmark), same as ticker_state_today's own prior_close/gap_pct.
        payload["market_context"] = {
            "benchmark_ticker": ticker_state.market_benchmark_ticker,
            "change_pct": ticker_state.market_change_pct,
            "vwap_deviation_pct": ticker_state.market_vwap_deviation_pct,
            "range_pct": ticker_state.market_range_pct,
        }
    logger.debug("payload %s", payload)
    return f"{_SYSTEM_INSTRUCTIONS}\nInput:\n{json.dumps(payload, indent=2)}"
