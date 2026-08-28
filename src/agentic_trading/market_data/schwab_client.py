"""schwab-py wrapper for market data polling (Phase 4).

Primary source for quotes and 5-minute historicals -- `market_data_client.py` tries
this module first and falls back to `robinhood_client.py` on any failure. This is
the ONLY module that imports schwab-py, matching the project's convention of
isolating each external brokerage integration behind one narrow module (see
`robinhood_client.py`'s own docstring and `execution/broker_mcp_client.py`).

Unlike robin_stocks, Schwab's Market Data API genuinely needs an authenticated,
refreshable OAuth session for every call. The running app never performs the
interactive browser consent itself -- that's a one-time (well, periodic: Schwab
refresh tokens go stale after ~7 days unused, see schwab-py's `easy_client`
`max_token_age`) operator step via `scripts/bootstrap_schwab_oauth.py`. This module
only ever reads the cached token file (`client_from_token_file`) and lets schwab-py
silently refresh it in the background; if that token is missing, expired past
recovery, or the account has no market-data entitlement, every call here fails
closed (logged, returns `None`/`[]`) so the fallback layer can take over -- it never
raises out to callers, same contract as `robinhood_client.py`.

Response field names (`bidPrice`, `askPrice`, `lastPrice`, `quoteTime`, `candles`,
etc.) follow Schwab's published Market Data Production API schema. Parsing is
deliberately defensive (broad try/except around the whole mapping step) since this
hasn't been validated against a live account in this environment -- an
unrecognized/changed response shape degrades to "Schwab unavailable this call"
rather than an unhandled exception.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from agentic_trading.config import get_settings
from agentic_trading.market_data.models import HistoricalBar, Quote

logger = logging.getLogger(__name__)

# Mirrors robinhood_client.py's login-retry cooldown: a broken/expired token or a
# down Schwab API shouldn't be retried on every single call (once per poll cycle,
# again per ticker) -- that's a retry hammer against an already-failing auth flow.
_CLIENT_INIT_RETRY_COOLDOWN = timedelta(seconds=90)
_client: object | None = None
_client_init_retry_after: datetime | None = None


def _get_client() -> object | None:
    """Lazily construct (and cache) the schwab-py client from the on-disk token.

    Returns None -- logged, not raised -- if credentials/token aren't configured
    or the client can't be constructed, so callers degrade to "Schwab unavailable"
    rather than crashing the poll cycle.
    """
    global _client, _client_init_retry_after
    if _client is not None:
        return _client
    now = datetime.now(UTC)
    if _client_init_retry_after is not None and now < _client_init_retry_after:
        return None
    settings = get_settings()
    if not settings.schwab_client_id or not settings.schwab_client_secret:
        logger.debug("Schwab client_id/client_secret not configured; skipping Schwab.")
        _client_init_retry_after = now + _CLIENT_INIT_RETRY_COOLDOWN
        return None
    try:
        from schwab.auth import client_from_token_file

        _client = client_from_token_file(
            token_path=settings.schwab_token_path,
            api_key=settings.schwab_client_id,
            app_secret=settings.schwab_client_secret,
        )
    except Exception:
        logger.warning(
            "Failed to load Schwab client from token file at %s -- has "
            "`uv run scripts/bootstrap_schwab_oauth.py` been run (or re-run, if the "
            "token is stale)? Falling back to Robinhood for market data until the "
            "next retry window.",
            settings.schwab_token_path,
            exc_info=True,
        )
        _client_init_retry_after = now + _CLIENT_INIT_RETRY_COOLDOWN
        return None
    _client_init_retry_after = None
    return _client


def _to_float(value: object) -> float | None:
    return float(value) if value not in (None, "") else None


def _to_int(value: object) -> int | None:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else None


def get_quote(symbol: str) -> Quote | None:
    """Top-of-book quote (bid/ask price + size), same shape as
    `robinhood_client.get_quote` regardless of which provider actually served it.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.get_quote(symbol)
        if response.status_code != 200:
            logger.warning(
                "Schwab get_quote(%s) returned HTTP %s: %s",
                symbol,
                response.status_code,
                response.text[:500],
            )
            return None
        payload = response.json()
        row = payload.get(symbol) or next(iter(payload.values()), None)
        if not row:
            return None
        quote = row.get("quote") or {}
        quote_time_ms = quote.get("quoteTime") or quote.get("tradeTime")
        updated_at = (
            datetime.fromtimestamp(quote_time_ms / 1000, tz=UTC) if quote_time_ms else None
        )
        return Quote(
            symbol=symbol,
            bid_price=_to_float(quote.get("bidPrice")),
            ask_price=_to_float(quote.get("askPrice")),
            bid_size=_to_int(quote.get("bidSize")),
            ask_size=_to_int(quote.get("askSize")),
            last_trade_price=_to_float(quote.get("lastPrice")),
            updated_at=updated_at,
        )
    except Exception:
        logger.warning("Schwab get_quote(%s) failed", symbol, exc_info=True)
        return None


def get_5min_historicals(
    symbol: str, start_datetime: datetime, end_datetime: datetime
) -> list[HistoricalBar]:
    """5-minute OHLCV bars over an explicit date range (Schwab has no `span="day"` /
    `span="week"` shorthand like robin_stocks -- `market_data_client.py` computes
    the equivalent range and passes it in).
    """
    client = _get_client()
    if client is None:
        return []
    try:
        response = client.get_price_history_every_five_minutes(
            symbol,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            need_extended_hours_data=False,
        )
        if response.status_code != 200:
            logger.warning(
                "Schwab get_price_history(%s) returned HTTP %s: %s",
                symbol,
                response.status_code,
                response.text[:500],
            )
            return []
        payload = response.json()
        candles = payload.get("candles") or []
        bars: list[HistoricalBar] = []
        for candle in candles:
            bars.append(
                HistoricalBar(
                    symbol=symbol,
                    begins_at=datetime.fromtimestamp(candle["datetime"] / 1000, tz=UTC),
                    open=float(candle["open"]),
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    close=float(candle["close"]),
                    volume=int(candle["volume"]),
                )
            )
        return bars
    except Exception:
        logger.warning("Schwab get_price_history(%s) failed", symbol, exc_info=True)
        return []
