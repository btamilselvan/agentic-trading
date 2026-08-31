"""Builds the LLM prompt from the full intraday bucket history + ticker trade state.

Per spec section 3.2, the LLM sees the COMPLETE array of the day's 5-minute buckets
for the ticker (not just the latest one) so it can evaluate for same-day intraday
setups (breakout, volume absorption, mean reversion, momentum continuation).

Per spec section 8, it also sees position_context -- the continuity state
(status/active_thesis/stop/target/recent decisions) scheduler.py loaded from Redis
(state.ticker_state_store) before this cycle -- so an active BUY/IN_POSITION/HOLD
call isn't re-derived from scratch (and flip-flopped on noise) every 5 minutes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime

from agentic_trading.llm.schema import TickerState
from agentic_trading.market_data.bucket_builder import (
    BucketLike,
    detect_vwap_cross,
    minutes_since_open,
    pct_change,
    rsi_centerline_cross,
    session_phase,
)
from agentic_trading.market_data.bucket_builder import to_float as _num

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTIONS = """\
You are an intraday momentum trading analyst. You are given a time-ordered series of \
5-minute market microstructure buckets for one ticker, plus that ticker's trading \
state for today. Evaluate STRICTLY for same-day intraday setups (e.g. morning \
breakout, volume absorption, quick mean reversion, momentum continuation). Do not \
consider multi-day or swing setups -- any position must be closeable within the same \
session. Each bucket's book_imbalance is the top-of-book depth skew in [-1, 1]: \
positive means more resting size on the bid (buying pressure), negative means more \
on the ask (selling pressure). Each bucket's buy_pressure_pct estimates buying vs. \
selling pressure within that bar, in [-100, 100] (positive = more estimated buy \
volume, negative = more estimated sell volume) -- derived from where the bar closed \
within its high/low range times volume, NOT a real trade-by-trade classification, so \
treat it as a soft proxy rather than a precise read. Only the most recent bucket \
(not earlier ones) also carries raw bid_price/ask_price/bid_size/ask_size -- the \
live top-of-book quote, useful when pricing buy_limit_price/target_sell_price/ \
stop_loss_price; earlier buckets omit these since spread/book_imbalance already \
summarize the relationship between them. ticker_state_today.gap_pct is today's open \
versus the prior session's close (positive = gapped up, negative = gapped down, \
null = no prior close available yet) -- a large gap is a precondition for a genuine \
"morning breakout" setup, as opposed to ordinary intraday drift. Each bucket's vwap \
is the session-cumulative volume-weighted average price through that bucket -- the \
standard intraday momentum reference line, and a useful anchor price when setting \
target_sell_price/stop_loss_price levels above/below it. Each bucket's \
vwap_deviation_pct is that bucket's close versus its vwap: sustained positive \
readings with rising volume support momentum continuation or a breakout holding; a \
move back toward/through zero suggests the move is fading or being rejected. If \
present, market_context describes the broad market today via a \
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
it intraday. Each bucket's session_phase places it within \
the trading day: "OPENING_VOLATILITY" (first 30 minutes) breakouts are common but \
noisier and more prone to failing, "MORNING_TREND" (30-120 minutes) is where trend \
continuation setups are most reliable, and "MIDDAY_CHOP" (120+ minutes) favors \
mean-reversion/absorption setups over fresh breakouts -- weigh confidence \
accordingly rather than treating every phase's setups as equally reliable. Each \
bucket's rsi is a 0-100 momentum oscillator (null until enough bars have \
accumulated intraday); above 70 is conventionally overbought (a breakout here is \
more likely to be extended/exhausted), below 30 oversold (a bounce here is more \
likely to be a mean-reversion setup than continuation), and rsi_centerline_cross \
is "up"/"down" on the specific bucket where rsi crosses its 50 centerline (a \
sharper momentum-shift signal than the raw level alone) or null otherwise. Also \
watch for divergence across the bucket series: price making a new high while rsi \
fails to make a new high (or the reverse) warns the move may be losing momentum \
even though price is still advancing. If present, catalyst_context surfaces \
same-day qualitative signals: news_headline/news_summary/news_published_at (a \
recent story that could plausibly explain an otherwise-unexplained volume/price \
move) and float_shares (shares available for trading -- below ~20M is \
conventionally a "low float", which tends to amplify breakout moves but can also \
reverse just as sharply, so treat it as a reason for a tighter target/faster exit \
rather than higher confidence on its own). Short interest is not available from \
the current data sources and is never included -- do not assume its absence means \
low short interest.

position_context carries this ticker's continuity state from prior cycles today -- \
status is one of FLAT (no thesis, never entered), HOLD (a thesis is being tracked \
but no position yet), BUY (a position was just opened this cycle or is pending \
fill), or IN_POSITION (already holding shares from an earlier cycle today). \
active_thesis is the rationale carried forward from whichever cycle first \
established the current status; initial_entry_price/current_target_price/ \
current_stop_loss are only meaningful once IN_POSITION. recent_decisions is the \
last few cycles' decisions for this ticker (bucket_start, decision, \
confidence_score, thesis_continuity_flag, pattern_reasoning), oldest first.

HYSTERESIS -- this is the most important rule in this prompt. Do not abandon an \
active BUY, IN_POSITION, or HOLD status because of a single bar of noise. If \
status is already HOLD or IN_POSITION with an active_thesis, your default is to \
maintain that same call and set thesis_continuity_flag=true UNLESS one of these \
specific invalidation criteria is met on the latest bucket:
  1. Price has crossed the current_stop_loss boundary.
  2. Primary momentum alignment has broken -- e.g. rsi_centerline_cross is "down" \
against the position's direction, or vwap_cross is "down" (price lost the session \
VWAP it was holding above).
  3. A major, high-impact NEGATIVE catalyst headline just appeared in \
catalyst_context that specifically undermines the active_thesis.
Only when one of these is true should you set thesis_continuity_flag=false and, if \
status is IN_POSITION, respond SELL. Ordinary intraday chop -- a red bar inside an \
otherwise intact uptrend, a brief dip that doesn't touch current_stop_loss -- is \
NOT sufficient grounds to flip.

SELL is only ever appropriate when status is IN_POSITION (there is a real position \
to exit); never respond SELL for a FLAT/HOLD ticker. BUY is only appropriate when \
status is FLAT or HOLD (not already holding); if status is IN_POSITION, your only \
choices are HOLD (continue holding) or SELL (exit now) -- a fresh BUY would double \
up the position, which is never correct here.

TRAILING TARGETS -- once IN_POSITION, target_sell_price/stop_loss_price may only \
move in the position's favor (for a long: stop_loss up, target up), never the \
other way. If you propose new target_sell_price/stop_loss_price values while \
IN_POSITION and momentum still supports the thesis, they must be at or above the \
current current_target_price/current_stop_loss respectively -- never suggest \
lowering either on noise. If you have no better level than what's already active, \
just repeat the current values back.

Respond with a single JSON object matching this contract:
- decision: "BUY", "HOLD", or "SELL" (see SELL/BUY eligibility above)
- confidence_score: 0.0-1.0, how strongly the setup (or the continuing thesis) \
supports this decision
- buy_limit_price: required if decision is BUY -- a limit entry price near the \
current ask, sized for immediate fill
- target_sell_price: required if decision is BUY -- the intraday profit-target limit \
sell price, greater than buy_limit_price
- stop_loss_price: required if decision is BUY -- the protective exit price, below \
buy_limit_price; this becomes current_stop_loss for every later cycle's hysteresis \
check while the position is open
- max_holding_time_minutes: required if decision is BUY -- max minutes to hold \
before a mandatory exit
- pattern_reasoning: a concise explanation of the order-flow setup (or, for a \
continuing HOLD/IN_POSITION, why the thesis still holds or why it just broke)
- thesis_continuity_flag: true if the active thesis (if any) from position_context \
still holds; false only when an invalidation criterion above was met this cycle

If no valid intraday setup is present, or ticker_state_today shows this ticker \
already has an open position or has hit its trade cap for the day, respond HOLD.
"""


def _buy_pressure_pct(est_buy_volume: int, est_sell_volume: int, volume: int) -> float | None:
    """(est_buy_volume - est_sell_volume) / volume, rescaled to [-100, 100] -- same
    normalized-ratio shape as book_imbalance, so the LLM reads a single signed signal
    instead of having to divide two raw six-figure volume counts itself (unreliable
    arithmetic for a small local model). None if volume is zero (nothing to compute
    a ratio from), matching pct_change's zero-reference guard elsewhere in this
    module.
    """
    if not volume:
        return None
    return (est_buy_volume - est_sell_volume) / volume * 100


def _bucket_to_dict(
    bucket: BucketLike, previous: BucketLike | None, session_start: datetime, is_latest: bool
) -> dict:
    close = _num(bucket.close)
    vwap = _num(bucket.vwap)
    prev_close = _num(previous.close) if previous else None
    prev_vwap = _num(previous.vwap) if previous else None
    rsi = _num(bucket.rsi)
    prev_rsi = _num(previous.rsi) if previous else None
    minutes_open = minutes_since_open(bucket.bucket_start, session_start)
    payload = {
        "bucket_start": bucket.bucket_start.isoformat(),
        "open": _num(bucket.open),
        "high": _num(bucket.high),
        "low": _num(bucket.low),
        "close": close,
        "volume": bucket.volume,
        "buy_pressure_pct": _buy_pressure_pct(
            bucket.est_buy_volume, bucket.est_sell_volume, bucket.volume
        ),
    }
    if is_latest:
        # Raw top-of-book quote -- only needed on the most recent bucket, to price
        # buy_limit_price/target_sell_price/stop_loss_price near the live bid/ask.
        # Every bucket (including earlier ones) already carries the normalized
        # spread/book_imbalance derived from these, so carrying the raw quote across
        # the whole history would just be redundant tokens.
        payload["bid_price"] = _num(bucket.bid_price)
        payload["ask_price"] = _num(bucket.ask_price)
        payload["bid_size"] = bucket.bid_size
        payload["ask_size"] = bucket.ask_size
    payload.update(
        {
            "spread": _num(bucket.spread),
            "book_imbalance": _num(bucket.book_imbalance),
            "candle_body": _num(bucket.candle_body),
            "upper_wick": _num(bucket.upper_wick),
            "lower_wick": _num(bucket.lower_wick),
            "rvol": _num(bucket.rvol),
            "vwap": vwap,
            "vwap_deviation_pct": pct_change(close, vwap),
            # Sequential signals -- vs. the immediately preceding bucket, not raw
            # levels -- so a small local model isn't left to infer momentum/volume
            # trend or a VWAP crossing itself by diffing raw numbers across the
            # bucket array.
            "close_change_pct": pct_change(close, prev_close),
            "volume_change_pct": pct_change(
                bucket.volume, previous.volume if previous else None
            ),
            "vwap_cross": detect_vwap_cross(prev_close, prev_vwap, close, vwap),
            # Session time context (requirements.md section 6) -- where this bucket
            # falls within the trading day, so the LLM weighs a setup differently at
            # 09:35 than at 11:15. See bucket_builder.session_phase for the
            # boundaries. minutes_since_open itself isn't sent -- session_phase is
            # the same information already bucketed into the categorical label the
            # LLM is told how to use, so sending both is redundant.
            "session_phase": session_phase(minutes_open),
            # RSI (requirements.md section 6) -- Wilder-smoothed, plain Python (see
            # bucket_builder.compute_rsi); null until enough bars have accumulated.
            "rsi": rsi,
            "rsi_centerline_cross": rsi_centerline_cross(prev_rsi, rsi),
        }
    )
    return payload


def build_prompt(
    ticker: str, bucket_history: Sequence[BucketLike], ticker_state: TickerState
) -> str:
    today_open = _num(bucket_history[0].open) if bucket_history else None
    prior_close = ticker_state.prior_close
    buckets_payload = []
    previous: BucketLike | None = None
    if bucket_history:
        session_start = bucket_history[0].bucket_start
        last_index = len(bucket_history) - 1
        for index, bucket in enumerate(bucket_history):
            buckets_payload.append(
                _bucket_to_dict(bucket, previous, session_start, index == last_index)
            )
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
    # Continuity context (requirements.md section 8) -- always present, unlike
    # market_context/catalyst_context below, since status/decision_history are
    # meaningful (if empty/FLAT) for every ticker on every cycle, not just when an
    # optional side-fetch happened to succeed this cycle.
    payload["position_context"] = {
        "status": ticker_state.status,
        "active_thesis": ticker_state.active_thesis,
        "initial_entry_price": ticker_state.initial_entry_price,
        "current_target_price": ticker_state.current_target_price,
        "current_stop_loss": ticker_state.current_stop_loss,
        "recent_decisions": [
            {
                "bucket_start": entry.bucket_start.isoformat(),
                "decision": entry.decision,
                "confidence_score": entry.confidence_score,
                "thesis_continuity_flag": entry.thesis_continuity_flag,
                "pattern_reasoning": entry.pattern_reasoning,
            }
            for entry in ticker_state.decision_history
        ],
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
    if ticker_state.news_headline is not None or ticker_state.float_shares is not None:
        # Omitted entirely (not sent as an all-null section) when neither fetch
        # turned up anything this cycle -- same treatment as market_context.
        # Short interest is deliberately absent -- see llm/prompt.py's docstring
        # note in _SYSTEM_INSTRUCTIONS and robinhood_client.py's module docstring.
        payload["catalyst_context"] = {
            "news_headline": ticker_state.news_headline,
            "news_summary": ticker_state.news_summary,
            "news_published_at": (
                ticker_state.news_published_at.isoformat()
                if ticker_state.news_published_at
                else None
            ),
            "float_shares": ticker_state.float_shares,
        }
    logger.debug("payload %s", payload)
    return f"{_SYSTEM_INSTRUCTIONS}\nInput:\n{json.dumps(payload, indent=2)}"
