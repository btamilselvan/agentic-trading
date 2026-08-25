from datetime import date

import httpx
import pytest

from agentic_trading import config
from agentic_trading.config import TradingMode
from agentic_trading.main import create_app, lifespan
from agentic_trading.state import repository as repo
from agentic_trading.state.db import session_scope

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client(db_session):
    app = create_app()
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app = app  # exposed so tests can inspect/override app.state (e.g. broker)
            yield c


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_status_reports_mode_and_scheduled_jobs(client):
    resp = await client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    # Deliberately not pinned to one mode -- this reads real ambient Settings (via
    # .env), so it should stay valid regardless of what MODE a developer/deployment
    # has configured locally.
    assert body["mode"] in {m.value for m in TradingMode}
    assert body["halted"] is False
    job_ids = {job["id"] for job in body["jobs"]}
    assert job_ids == {"poll_cycle", "order_management_sweep", "eod_liquidation"}


async def test_decisions_and_trades_endpoints_reflect_db_state(client):
    async with session_scope() as session:
        decision = await repo.save_llm_decision(
            session,
            ticker="AAPL",
            bucket_id=None,
            prompt="p",
            raw_response="{}",
            decision="BUY",
            confidence_score=0.85,
            buy_limit_price=100.0,
            target_sell_price=102.0,
            max_holding_time_minutes=30,
            pattern_reasoning="breakout",
        )
        await repo.open_trade(
            session,
            ticker="AAPL",
            trade_date=date(2026, 8, 17),
            entry_price=100.0,
            quantity=5,
            llm_decision_id=decision.id,
            target_sell_price=102.0,
            max_holding_time_minutes=30,
        )

    decisions_resp = await client.get("/decisions")
    assert decisions_resp.status_code == 200
    decisions = decisions_resp.json()
    assert len(decisions) == 1
    assert decisions[0]["ticker"] == "AAPL"
    assert decisions[0]["decision"] == "BUY"

    trades_resp = await client.get("/trades")
    assert trades_resp.status_code == 200
    trades = trades_resp.json()
    assert len(trades) == 1
    assert trades[0]["status"] == "OPEN"
    assert trades[0]["entry_price"] == 100.0


async def test_manual_order_entry_opens_a_trade_in_dry_run(monkeypatch, client):
    from agentic_trading.execution.order_manager import DryRunBrokerClient

    # Force a known-safe simulator regardless of the ambient MODE this process
    # happened to boot with -- this test must never risk a real broker call.
    client.app.state.broker = DryRunBrokerClient()
    config.get_settings.cache_clear()
    monkeypatch.setenv("MODE", "DRY_RUN")
    # Isolate from whatever MAX_CAPITAL_PER_TRADE_USD the ambient .env happens to
    # have (e.g. a deliberately tiny cap for real-money live testing) -- this test
    # is about the entry/paired-sell/close chain, not guardrail sizing, so give it
    # headroom regardless of local config.
    monkeypatch.setenv("MAX_CAPITAL_PER_TRADE_USD", "10000")
    try:
        resp = await client.post(
            "/orders/manual-entry",
            json={
                "ticker": "aapl",
                "buy_limit_price": 100.0,
                "target_sell_price": 102.0,
                "stop_loss_price": 98.0,
                "max_holding_time_minutes": 15,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "AAPL"
        assert body["mode"] == "DRY_RUN"
        assert body["opened"] is True
        assert body["trade_id"] is not None
        assert body["order_id"] is not None
        # trade_id/order_id are our own internal DB primary keys (different tables,
        # not the same value) -- broker_order_id is the actual broker-side id
        # (DryRunBrokerClient's simulated "DRYRUN-N" format here).
        assert body["broker_order_id"] is not None
        assert body["broker_order_id"] != body["trade_id"]
        assert body["broker_order_id"].startswith("DRYRUN-")

        trades_resp = await client.get("/trades")
        trades = trades_resp.json()
        # DryRunBrokerClient fills both legs instantly, so by the time we check,
        # try_enter_position has already placed the paired sell and closed the
        # trade -- this exercises that whole chain, not just the initial entry.
        assert any(
            t["ticker"] == "AAPL"
            and t["status"] == "CLOSED"
            and t["entry_price"] == 100.0
            and t["exit_price"] == 102.0
            for t in trades
        )
    finally:
        config.get_settings.cache_clear()


async def test_manual_order_entry_in_live_mode_requires_confirm(monkeypatch, client):
    from agentic_trading.execution.order_manager import DryRunBrokerClient

    # Guardrail-under-test is the confirm gate itself, checked before any broker
    # call -- still force a safe simulator so a regression here can never reach a
    # real broker.
    client.app.state.broker = DryRunBrokerClient()
    config.get_settings.cache_clear()
    monkeypatch.setenv("MODE", "LIVE")
    try:
        resp = await client.post(
            "/orders/manual-entry",
            json={
                "ticker": "AAPL",
                "buy_limit_price": 100.0,
                "target_sell_price": 102.0,
                "stop_loss_price": 98.0,
            },
        )
        assert resp.status_code == 400
        assert "confirm=true" in resp.json()["detail"]
    finally:
        config.get_settings.cache_clear()


async def test_manual_order_entry_surfaces_broker_rejection_as_502(monkeypatch, client):
    # Simulates the real failure mode this endpoint hit on 2026-08-21: the MCP
    # rejects the order (e.g. market closed) and review_order raises -- this must
    # come back as a clear 502 with the real reason, not a bare 500, and must not
    # leave a dangling llm_decision row behind (session_scope rolls the whole
    # transaction back on any exception).
    class _RejectingBroker:
        async def get_open_position_quantity(self, ticker: str) -> float:
            return 0.0

        async def review_order(self, **kwargs):
            raise RuntimeError("MCP tool call failed: Market is closed for regular trading hours")

        async def place_order(self, **kwargs):
            raise AssertionError("should not be reached")

        async def cancel_order(self, broker_order_id: str) -> None:
            return None

    client.app.state.broker = _RejectingBroker()
    config.get_settings.cache_clear()
    monkeypatch.setenv("MODE", "DRY_RUN")
    try:
        decisions_before = (await client.get("/decisions")).json()

        resp = await client.post(
            "/orders/manual-entry",
            json={
                "ticker": "CLOV",
                "buy_limit_price": 4.11,
                "target_sell_price": 4.20,
                "stop_loss_price": 4.00,
            },
        )
        assert resp.status_code == 502
        assert "Market is closed for regular trading hours" in resp.json()["detail"]

        decisions_after = (await client.get("/decisions")).json()
        assert decisions_after == decisions_before
    finally:
        config.get_settings.cache_clear()


async def test_manual_order_entry_refuses_while_halted(monkeypatch, client):
    config.get_settings.cache_clear()
    monkeypatch.setenv("MODE", "DRY_RUN")
    try:
        await client.post("/kill-switch")
        resp = await client.post(
            "/orders/manual-entry",
            json={
                "ticker": "AAPL",
                "buy_limit_price": 100.0,
                "target_sell_price": 102.0,
                "stop_loss_price": 98.0,
            },
        )
        assert resp.status_code == 409
    finally:
        config.get_settings.cache_clear()


async def test_kill_switch_halts_and_resume_restarts(client):
    kill_resp = await client.post("/kill-switch")
    assert kill_resp.status_code == 200
    assert kill_resp.json() == {"status": "halted"}

    status_resp = await client.get("/status")
    assert status_resp.json()["halted"] is True

    resume_resp = await client.post("/resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json() == {"status": "resumed"}

    status_resp2 = await client.get("/status")
    assert status_resp2.json()["halted"] is False
