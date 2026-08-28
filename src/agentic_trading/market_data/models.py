"""Shared market-data dataclasses, provider-agnostic.

Both `robinhood_client.py` and `schwab_client.py` (Phase 4) map their respective
provider's raw responses into these same shapes, so `market_data_client.py`'s
fallback orchestration and downstream consumers (`bucket_builder.py`,
`scheduler.py`) don't need to know or care which provider actually served a
given call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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
