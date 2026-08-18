"""Tests for the two read-only connectivity-check endpoints in api/routes.py:
GET /market-data/{ticker} (robin_stocks) and GET /broker/positions/{ticker}
(Robinhood MCP). Both external calls are faked -- these tests are about the HTTP
plumbing (status codes, response shape, error mapping), not the real robin_stocks/MCP
wire protocol, which is exercised manually (see README's "Going live" section and
scripts/bootstrap_mcp_oauth.py).
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
