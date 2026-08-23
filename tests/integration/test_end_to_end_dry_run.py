"""As close to a real end-to-end DRY_RUN validation as this environment allows.

No live Robinhood credentials or Ollama server are available here, so:
  - market data (robin_stocks) is stubbed at the function level -- it needs a real
    Robinhood login this sandbox doesn't have.
  - the LLM and the webhook alert are NOT mocked at the function level -- they run
    against real local HTTP servers standing in for Ollama and a Slack-style
    webhook receiver, so OllamaClient and WebhookNotifier's actual HTTP code paths
    get exercised over the wire, not just asserted against a mocked transport.

This is the manual DRY_RUN validation called for in the implementation plan's
Verification section, automated so it re-runs on every test invocation instead of
being a one-off manual step.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from sqlalchemy import select

from agentic_trading import scheduler
from agentic_trading.alerts.webhook_notifier import WebhookNotifier
from agentic_trading.config import Settings, TradingMode
from agentic_trading.execution.order_manager import DryRunBrokerClient
from agentic_trading.llm.ollama_client import OllamaClient
from agentic_trading.market_data.robinhood_client import HistoricalBar, Quote
from agentic_trading.state import repository as repo
from agentic_trading.state.db import session_scope
from agentic_trading.state.models import LlmDecision

pytestmark = pytest.mark.asyncio

OLLAMA_PORT = 8799
WEBHOOK_PORT = 8798
TICKER = "AAPL"


def _run_uvicorn(app: FastAPI, port: int) -> None:
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


@pytest.fixture(scope="module")
def fake_ollama_server():
    """A real HTTP server implementing just enough of Ollama's /api/chat to let
    the real OllamaClient run its full request/parse/validate path against it."""
    app = FastAPI()

    @app.post("/api/chat")
    async def chat(request: Request) -> dict:
        body = await request.json()
        prompt = body["messages"][0]["content"]
        # Structured-output constraint is honored implicitly here: we return a
        # payload that always satisfies TradeDecision's schema.
        decision = {
            "decision": "BUY",
            "confidence_score": 0.91,
            "buy_limit_price": 100.5,
            "target_sell_price": 102.0,
            "max_holding_time_minutes": 30,
            "pattern_reasoning": f"breakout detected for prompt of length {len(prompt)}",
        }
        return {"message": {"role": "assistant", "content": json.dumps(decision)}}

    thread = threading.Thread(target=_run_uvicorn, args=(app, OLLAMA_PORT), daemon=True)
    thread.start()
    _wait_for_server(f"http://127.0.0.1:{OLLAMA_PORT}/docs")
    yield f"http://127.0.0.1:{OLLAMA_PORT}"


@pytest.fixture(scope="module")
def fake_webhook_server():
    """A real HTTP server capturing every alert WebhookNotifier posts, so we can
    assert the real alerts pipeline actually fired over the wire."""
    app = FastAPI()
    received: list[dict] = []

    @app.post("/webhook")
    async def webhook(request: Request) -> dict:
        received.append(await request.json())
        return {"ok": True}

    thread = threading.Thread(target=_run_uvicorn, args=(app, WEBHOOK_PORT), daemon=True)
    thread.start()
    _wait_for_server(f"http://127.0.0.1:{WEBHOOK_PORT}/docs")
    yield received


def _wait_for_server(url: str, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=0.5)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"server at {url} did not come up in time")


async def test_full_dry_run_cycle_over_real_http_llm_and_webhook(
    db_session, monkeypatch, fake_ollama_server, fake_webhook_server
):
    # Market data faked at the function level (needs real Robinhood credentials);
    # everything downstream of this is the real code path.
    bucket_start = datetime(2026, 8, 17, 9, 30, tzinfo=UTC)
    bar = HistoricalBar(TICKER, bucket_start, 100, 101, 99, 100.5, 10_000)
    quote = Quote(TICKER, 100.4, 100.6, 500, 400, 100.5, None)
    monkeypatch.setattr(
        scheduler.rh, "get_5min_historicals", lambda ticker, span="day", bounds="regular": [bar]
    )
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: quote)
    scheduler._rvol_lookback_cache.clear()

    settings = Settings(
        mode=TradingMode.DRY_RUN,
        watchlist=[TICKER],
        confidence_threshold=0.7,
        llm_provider="ollama",
        llm_model="gemma3",
        ollama_host=fake_ollama_server,
        webhook_url=f"http://127.0.0.1:{WEBHOOK_PORT}/webhook",
        max_open_positions_per_ticker=1,
        daily_trade_cap_per_ticker=3,
        max_capital_per_trade_usd=1000.0,
        max_daily_drawdown_usd=1000.0,
        order_timeout_minutes=15,
    )

    llm_client = OllamaClient(host=settings.ollama_host, model=settings.llm_model)
    broker = DryRunBrokerClient()
    notifier = WebhookNotifier(settings.webhook_url)

    mid_window = datetime(2026, 8, 17, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    await scheduler.run_poll_cycle(
        broker=broker, llm_client=llm_client, settings=settings, notifier=notifier, now=mid_window
    )

    # 1. The real Ollama HTTP call produced a real, schema-valid BUY decision that
    #    got persisted.
    async with session_scope() as session:
        result = await session.execute(select(LlmDecision).where(LlmDecision.ticker == TICKER))
        saved = result.scalars().one()
        assert saved.decision.value == "BUY"
        assert saved.acted_on is True
        assert "breakout" in saved.pattern_reasoning

    # 2. DryRunBrokerClient filled the buy instantly, which placed+filled the paired
    #    sell instantly too -- the trade should be closed with realized PnL, and the
    #    ticker's daily state reflects a completed round trip.
    async with session_scope() as session:
        daily_state = await repo.get_or_create_daily_state(session, TICKER, date(2026, 8, 17))
        assert daily_state.completed_trades_count == 1
        assert daily_state.open_positions_count == 0
        trades = await repo.get_open_trades(session, ticker=TICKER)
        assert trades == []

    # 3. Real webhook POSTs actually reached the fake receiver over HTTP.
    texts = "\n---\n".join(item["text"] for item in fake_webhook_server)
    assert "BUY signal" in texts
    assert "Order filled" in texts
    assert "Trade closed" in texts

    # 4. Order-management sweep and EOD liquidation both run cleanly against this
    #    already-closed state with zero pending orders/open trades left to act on.
    await scheduler.run_order_management_sweep(broker=broker, settings=settings, notifier=notifier)
    monkeypatch.setattr(scheduler.rh, "get_quote", lambda ticker: quote)
    await scheduler.run_eod_liquidation(broker=broker, settings=settings, notifier=notifier)

    async with session_scope() as session:
        assert (await repo.get_open_trades(session, ticker=TICKER)) == []
