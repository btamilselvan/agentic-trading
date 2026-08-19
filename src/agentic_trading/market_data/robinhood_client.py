"""robin_stocks wrapper for market data polling only.

This is the ONLY module that imports robin_stocks. Order execution never goes through
here — that's the Robinhood MCP client's job (see execution/broker_mcp_client.py).
Keeping the two brokerage integrations behind separate, narrow modules is what lets
market-data and order-execution be swapped independently later.

`get_latest_news`/`get_float_shares` back requirements.md section 6's "Qualitative
Catalyst & Metadata" (Phase 2): the most recent news story and float size, both from
robin_stocks (`get_news`/`get_fundamentals`). Short interest %, the third field that
section asks for, is NOT available from robin_stocks or the Robinhood API at all --
no field, no endpoint -- so it's simply not computed anywhere in this project (same
treatment as the missing VIX feed in bucket_builder.build_market_context).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import robin_stocks.robinhood as rh

from agentic_trading.config import get_settings

_logged_in = False

logger = logging.getLogger(__name__)

def ensure_login() -> None:
    """Log in once per process, reusing a cached session pickle if present.

    On first run (no cached session), robin_stocks will prompt for an MFA code on
    stdin — this must be run interactively at least once before the scheduler relies
    on it unattended.
    """
    global _logged_in
    if _logged_in:
        return
    settings = get_settings()
    token_path = Path(settings.robinhood_token_path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    rh.login(
        username=settings.robinhood_username,
        password=settings.robinhood_password,
        store_session=True,
        pickle_path=str(token_path.parent),
        pickle_name=token_path.stem,
    )
    _logged_in = True


@dataclass(frozen=True)
class HistoricalBar:
    symbol: str
    begins_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid_price: float | None
    ask_price: float | None
    bid_size: int | None
    ask_size: int | None
    last_trade_price: float | None
    updated_at: datetime | None


@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: str | None
    published_at: datetime | None
    source: str | None


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _to_float(value: object) -> float | None:
    return float(value) if value not in (None, "") else None


def _to_int(value: object) -> int | None:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else None


def get_5min_historicals(
    symbol: str, span: str = "day", bounds: str = "regular"
) -> list[HistoricalBar]:
    """5-minute OHLCV bars.

    `span="day"` returns today's bars so far (used for the live poll loop);
    `span="week"` returns the last 5 trading days (used by bucket_builder to compute
    an RVOL baseline for the same time-of-day slot).
    """
    ensure_login()
    raw = rh.stocks.get_stock_historicals(symbol, interval="5minute", span=span, bounds=bounds)
    bars: list[HistoricalBar] = []
    for row in raw or []:
        if not row:
            continue
        bars.append(
            HistoricalBar(
                symbol=symbol,
                begins_at=_parse_timestamp(row["begins_at"]),
                open=float(row["open_price"]),
                high=float(row["high_price"]),
                low=float(row["low_price"]),
                close=float(row["close_price"]),
                volume=int(float(row["volume"])),
            )
        )
    return bars


def get_quote(symbol: str) -> Quote | None:
    """Top-of-book quote (bid/ask price + size). Used for spread and the depth-imbalance
    proxy — this is top-of-book only, not a full order-book depth feed."""
    ensure_login()
    raw = rh.stocks.get_quotes(symbol)
    if not raw or not raw[0]:
        return None
    row = raw[0]
    updated_at = _parse_timestamp(row["updated_at"]) if row.get("updated_at") else None
    return Quote(
        symbol=symbol,
        bid_price=_to_float(row.get("bid_price")),
        ask_price=_to_float(row.get("ask_price")),
        bid_size=_to_int(row.get("bid_size")),
        ask_size=_to_int(row.get("ask_size")),
        last_trade_price=_to_float(row.get("last_trade_price")),
        updated_at=updated_at,
    )


def get_latest_news(symbol: str) -> NewsItem | None:
    """Most recent news story for `symbol`, or None if there's no news at all, or
    every row is missing a title (robin_stocks occasionally returns placeholder/
    empty rows). get_news doesn't document a guaranteed order, so this explicitly
    picks the max by published_at rather than assuming raw[0] is the latest; if no
    row has a usable timestamp, it falls back to whatever order the feed returned.
    """
    ensure_login()
    raw = rh.stocks.get_news(symbol)
    if not raw:
        return None
    items = [
        NewsItem(
            title=row["title"],
            summary=row.get("summary") or None,
            published_at=(
                _parse_timestamp(row["published_at"]) if row.get("published_at") else None
            ),
            source=row.get("source") or None,
        )
        for row in raw
        if row and row.get("title")
    ]
    if not items:
        return None
    dated = [item for item in items if item.published_at is not None]
    return max(dated, key=lambda item: item.published_at) if dated else items[0]


def get_float_shares(symbol: str) -> int | None:
    """Float size (requirements.md section 6's "<20M shares indication") from
    get_fundamentals. None if fundamentals are unavailable or the float field is
    missing/blank for this symbol.
    """
    ensure_login()
    raw = rh.stocks.get_fundamentals(symbol)
    if not raw or not raw[0]:
        return None
    return _to_int(raw[0].get("float"))
