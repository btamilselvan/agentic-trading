"""robin_stocks wrapper for market data polling only.

This is the ONLY module that imports robin_stocks. Order execution never goes through
here — that's the Robinhood MCP client's job (see execution/broker_mcp_client.py).
Keeping the two brokerage integrations behind separate, narrow modules is what lets
market-data and order-execution be swapped independently later.
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
