from datetime import date

import httpx
import pytest

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
