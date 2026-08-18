"""Builds the LLM prompt from the full intraday bucket history + ticker trade state.

Per spec section 3.2, the LLM sees the COMPLETE array of the day's 5-minute buckets
for the ticker (not just the latest one) so it can evaluate for same-day intraday
setups (breakout, volume absorption, mean reversion, momentum continuation).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from agentic_trading.llm.schema import TickerState
from agentic_trading.market_data.bucket_builder import BucketLike

_SYSTEM_INSTRUCTIONS = """\
You are an intraday momentum trading analyst. You are given a time-ordered series of \
5-minute market microstructure buckets for one ticker, plus that ticker's trading \
state for today. Evaluate STRICTLY for same-day intraday setups (e.g. morning \
breakout, volume absorption, quick mean reversion, momentum continuation). Do not \
consider multi-day or swing setups -- any position must be closeable within the same \
session.

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


def _bucket_to_dict(bucket: BucketLike) -> dict:
    return {
        "bucket_start": bucket.bucket_start.isoformat(),
        "open": bucket.open,
        "high": bucket.high,
        "low": bucket.low,
        "close": bucket.close,
        "volume": bucket.volume,
        "est_buy_volume": bucket.est_buy_volume,
        "est_sell_volume": bucket.est_sell_volume,
        "bid_price": bucket.bid_price,
        "ask_price": bucket.ask_price,
        "spread": bucket.spread,
        "candle_body": bucket.candle_body,
        "upper_wick": bucket.upper_wick,
        "lower_wick": bucket.lower_wick,
        "rvol": bucket.rvol,
    }


def build_prompt(
    ticker: str, bucket_history: Sequence[BucketLike], ticker_state: TickerState
) -> str:
    payload = {
        "ticker": ticker,
        "buckets": [_bucket_to_dict(b) for b in bucket_history],
        "ticker_state_today": {
            "completed_trades": ticker_state.completed_trades_today,
            "open_positions": ticker_state.open_positions,
            "realized_pnl": ticker_state.realized_pnl_today,
        },
    }
    return f"{_SYSTEM_INSTRUCTIONS}\nInput:\n{json.dumps(payload, indent=2)}"
