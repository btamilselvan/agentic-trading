"""Schwab-primary / Robinhood-fallback market data orchestrator (Phase 4).

requirements.md Phase 4: Schwab's Market Data Production API (`schwab_client.py`)
is the primary source for quotes and 5-minute historicals; `robinhood_client.py`
(robin_stocks) is the fallback whenever Schwab fails, so this is the one module
`scheduler.py`/`api/routes.py` should call for market data instead of reaching into
either provider module directly. Both provider modules already fail closed --
they log and return `None`/`[]` rather than raise -- so "did Schwab work" here is
just "was the result non-empty", no exception handling needed at this layer.

News (`get_latest_news`) and float shares (`get_float_shares`) are NOT wrapped
here -- they stay Robinhood-only (see robinhood_client.py); Schwab's Market Data
API has no confirmed equivalent, and requirements.md Phase 4 only calls out
quotes/5-min historicals (OHLCV), not catalyst metadata.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from agentic_trading.config import get_settings
from agentic_trading.market_data import robinhood_client, schwab_client
from agentic_trading.market_data.models import HistoricalBar, Quote

logger = logging.getLogger(__name__)

# Calendar days back for the "week" span -- generously covers the "last 5 trading
# days" the RVOL baseline wants even across a weekend/holiday, mirroring what
# robin_stocks' span="week" already returns.
_WEEK_SPAN_LOOKBACK_DAYS = 8


def market_open_today(now: datetime) -> datetime:
    """Today's market-open instant (settings.market_open_time, in settings.timezone),
    expressed in UTC. Public (not `_`-prefixed) so callers like api/routes.py's
    Schwab connectivity check can reuse it directly rather than re-deriving "today's
    market open" themselves -- naively doing `now.replace(hour=..., minute=...)` on
    an already-UTC `now` gives midnight UTC (evening of the previous day in US
    Eastern), not market open; the local-time round-trip below is required.
    """
    settings = get_settings()
    tz = ZoneInfo(settings.timezone)
    hour, _, minute = settings.market_open_time.partition(":")
    local_now = now.astimezone(tz)
    return local_now.replace(
        hour=int(hour), minute=int(minute), second=0, microsecond=0
    ).astimezone(UTC)


def get_quote(symbol: str) -> Quote | None:
    """Current top-of-book quote. Tries Schwab first, falls back to Robinhood."""
    quote = schwab_client.get_quote(symbol)
    if quote is not None:
        logger.debug("get_quote(%s) served by Schwab", symbol)
        return quote
    logger.info("get_quote(%s): Schwab unavailable, falling back to Robinhood", symbol)
    return robinhood_client.get_quote(symbol)


def get_5min_historicals(symbol: str, span: str = "day") -> list[HistoricalBar]:
    """5-minute OHLCV bars. `span="day"` returns today's bars so far; `span="week"`
    returns roughly the last 5 trading days (used for the RVOL baseline). Tries
    Schwab first (translating span into an explicit date range, since Schwab has no
    "day"/"week" shorthand), falls back to Robinhood on an empty result.
    """
    now = datetime.now(UTC)
    if span == "day":
        start = market_open_today(now)
    elif span == "week":
        start = now - timedelta(days=_WEEK_SPAN_LOOKBACK_DAYS)
    else:
        raise ValueError(f"Unsupported span: {span!r}")

    bars = schwab_client.get_5min_historicals(symbol, start_datetime=start, end_datetime=now)
    if bars:
        logger.debug("get_5min_historicals(%s, span=%s) served by Schwab", symbol, span)
        return bars
    logger.info(
        "get_5min_historicals(%s, span=%s): Schwab unavailable, falling back to Robinhood",
        symbol,
        span,
    )
    return robinhood_client.get_5min_historicals(symbol, span=span)
