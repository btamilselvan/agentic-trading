"""Tests for the three read-only connectivity-check endpoints in api/routes.py:
GET /market-data/{ticker} (robin_stocks), GET /market-data/schwab/{ticker} (Schwab,
Phase 4), and GET /broker/positions/{ticker} (Robinhood MCP). All external calls are
faked -- these tests are about the HTTP plumbing (status codes, response shape, error
mapping), not the real robin_stocks/Schwab/MCP wire protocols, which are exercised
manually (see README's "Going live"/"Schwab market data" sections,
scripts/bootstrap_mcp_oauth.py, and scripts/bootstrap_schwab_oauth.py).
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI

from agentic_trading.api import routes
from agentic_trading.execution.broker_mcp_client import McpBrokerClient
from agentic_trading.market_data.robinhood_client import HistoricalBar, Quote

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(routes.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_market_data_returns_quote_and_latest_bar(client, monkeypatch):
    quote = Quote(
        symbol="AAPL", bid_price=100.4, ask_price=100.6, bid_size=500, ask_size=400,
        last_trade_price=100.5, updated_at=None,
    )
    bar = HistoricalBar(
        symbol="AAPL", begins_at=datetime(2026, 8, 17, 9, 30, tzinfo=UTC),
        open=100, high=101, low=99, close=100.5, volume=1000,
    )
    monkeypatch.setattr(routes.rh_market, "get_quote", lambda ticker: quote)
    monkeypatch.setattr(
        routes.rh_market, "get_5min_historicals", lambda ticker, span="day": [bar]
    )

    response = await client.get("/market-data/aapl")

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"  # normalized to uppercase
    assert body["quote"]["bid_price"] == 100.4
    assert body["latest_bar"]["close"] == 100.5
    assert body["bars_today"] == 1


async def test_market_data_handles_no_quote_or_bars(client, monkeypatch):
    monkeypatch.setattr(routes.rh_market, "get_quote", lambda ticker: None)
    monkeypatch.setattr(routes.rh_market, "get_5min_historicals", lambda ticker, span="day": [])

    response = await client.get("/market-data/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["quote"] is None
    assert body["latest_bar"] is None
    assert body["bars_today"] == 0


async def test_market_data_maps_robin_stocks_failure_to_502(client, monkeypatch):
    def _boom(ticker):
        raise RuntimeError("robin_stocks session expired")

    monkeypatch.setattr(routes.rh_market, "get_quote", _boom)

    response = await client.get("/market-data/AAPL")

    assert response.status_code == 502
    assert "robin_stocks session expired" in response.text


async def test_market_data_schwab_returns_quote_and_latest_bar(client, monkeypatch):
    quote = Quote(
        symbol="AAPL", bid_price=100.4, ask_price=100.6, bid_size=500, ask_size=400,
        last_trade_price=100.5, updated_at=None,
    )
    bar = HistoricalBar(
        symbol="AAPL", begins_at=datetime(2026, 8, 17, 9, 30, tzinfo=UTC),
        open=100, high=101, low=99, close=100.5, volume=1000,
    )
    monkeypatch.setattr(routes.schwab_client, "get_quote", lambda ticker: quote)
    monkeypatch.setattr(
        routes.schwab_client,
        "get_5min_historicals",
        lambda ticker, start_datetime, end_datetime: [bar],
    )

    response = await client.get("/market-data/schwab/aapl")

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"  # normalized to uppercase
    assert body["quote"]["bid_price"] == 100.4
    assert body["latest_bar"]["close"] == 100.5
    assert body["bars_today"] == 1


async def test_market_data_schwab_requests_todays_market_open_not_utc_midnight(
    client, monkeypatch
):
    """Regression test: this endpoint used to pass now.replace(hour=0, minute=0, ...)
    straight through on an already-UTC `now`, which lands on UTC midnight -- 8pm the
    *previous* day in US Eastern, not today's market open. It must instead request
    from settings.market_open_time (09:30) converted from settings.timezone
    (America/New_York) into UTC, via market_data_client.market_open_today."""
    monkeypatch.setattr(routes.schwab_client, "get_quote", lambda ticker: None)
    seen = {}

    def _capture(ticker, start_datetime, end_datetime):
        seen["start_datetime"] = start_datetime
        seen["end_datetime"] = end_datetime
        return []

    monkeypatch.setattr(routes.schwab_client, "get_5min_historicals", _capture)

    await client.get("/market-data/schwab/AAPL")

    expected_start = routes.mdc.market_open_today(seen["end_datetime"])
    assert seen["start_datetime"] == expected_start
    # The literal bug being regression-tested: UTC midnight is NOT market open.
    assert seen["start_datetime"] != seen["end_datetime"].replace(
        hour=0, minute=0, second=0, microsecond=0
    )


async def test_market_data_schwab_never_502s_on_failure(client, monkeypatch):
    """Unlike /market-data/{ticker}, schwab_client fails closed (returns None/[]
    rather than raising) -- a down/unauthorized Schwab surfaces as null fields here,
    not an HTTP error, since that's exactly the signal market_data_client's fallback
    logic already relies on."""
    monkeypatch.setattr(routes.schwab_client, "get_quote", lambda ticker: None)
    monkeypatch.setattr(
        routes.schwab_client,
        "get_5min_historicals",
        lambda ticker, start_datetime, end_datetime: [],
    )

    response = await client.get("/market-data/schwab/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["quote"] is None
    assert body["latest_bar"] is None
    assert body["bars_today"] == 0


async def test_broker_position_returns_quantity(client, monkeypatch):
    async def fake_get_open_position_quantity(self, ticker: str) -> float:
        assert ticker == "AAPL"
        return 12.0

    monkeypatch.setattr(
        McpBrokerClient, "get_open_position_quantity", fake_get_open_position_quantity
    )

    response = await client.get("/broker/positions/aapl")

    assert response.status_code == 200
    assert response.json() == {"ticker": "AAPL", "open_position_quantity": 12.0}


async def test_broker_position_maps_mcp_failure_to_502(client, monkeypatch):
    async def fake_get_open_position_quantity(self, ticker: str) -> float:
        raise RuntimeError("No valid MCP OAuth token on disk")

    monkeypatch.setattr(
        McpBrokerClient, "get_open_position_quantity", fake_get_open_position_quantity
    )

    response = await client.get("/broker/positions/AAPL")

    assert response.status_code == 502
    assert "No valid MCP OAuth token on disk" in response.text
