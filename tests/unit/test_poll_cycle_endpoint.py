"""Tests for POST /poll-cycle (api/routes.py): the on-demand trigger for
run_poll_cycle. run_poll_cycle itself is exercised elsewhere (test_scheduler.py);
these are about the HTTP plumbing -- that it reuses app.state's broker/llm_client
rather than fresh ones, that ?force= maps to bypass_window, and that it refuses to
run at all while halted.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from agentic_trading.api import routes

pytestmark = pytest.mark.asyncio


class _SentinelBroker:
    pass


class _SentinelLLMClient:
    pass


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(routes.router)
    app.state.halted = False
    app.state.broker = _SentinelBroker()
    app.state.llm_client = _SentinelLLMClient()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, app


async def test_poll_cycle_trigger_reuses_app_state_broker_and_llm_client(client, monkeypatch):
    c, app = client
    calls = []

    async def fake_run_poll_cycle(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(routes, "run_poll_cycle", fake_run_poll_cycle)

    response = await c.post("/poll-cycle")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["forced"] is False
    assert len(calls) == 1
    assert calls[0]["broker"] is app.state.broker
    assert calls[0]["llm_client"] is app.state.llm_client
    assert calls[0]["bypass_window"] is False


async def test_poll_cycle_trigger_force_bypasses_window(client, monkeypatch):
    c, _ = client
    calls = []

    async def fake_run_poll_cycle(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(routes, "run_poll_cycle", fake_run_poll_cycle)

    response = await c.post("/poll-cycle?force=true")

    assert response.status_code == 200
    assert response.json()["forced"] is True
    assert calls[0]["bypass_window"] is True


async def test_poll_cycle_trigger_refuses_when_halted(client, monkeypatch):
    c, app = client
    app.state.halted = True

    async def fake_run_poll_cycle(**kwargs):
        raise AssertionError("must not run a poll cycle while halted")

    monkeypatch.setattr(routes, "run_poll_cycle", fake_run_poll_cycle)

    response = await c.post("/poll-cycle")

    assert response.status_code == 409
