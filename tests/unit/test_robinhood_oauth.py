"""Tests for the Robinhood OAuth bridging endpoints (api/robinhood_oauth.py).

The actual OAuth/MCP wire protocol is exercised for real by
scripts/bootstrap_mcp_oauth.py (see its own manual verification against the live
server) -- these tests instead prove the new HTTP-level bridging logic works:
GET /authorize blocks until a redirect URL is available (or the flow completes
without needing one), and GET /callback unblocks it and returns a result.
`_run_authorization_flow` (the thing that actually talks to the MCP) is replaced
with a fake per test so no network/OAuth machinery is involved.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from agentic_trading.api import robinhood_oauth

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_pending_state():
    robinhood_oauth._pending = None
    yield
    robinhood_oauth._pending = None


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(robinhood_oauth.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _fake_flow_needing_redirect(pending: robinhood_oauth._PendingAuthorization) -> None:
    pending.auth_url = "https://robinhood.example/oauth?state=xyz"
    pending.auth_url_ready.set()
    await pending.callback_received.wait()
    pending.tool_names = ["get_equity_positions", "place_equity_order"]
    pending.done.set()


async def _fake_flow_already_authorized(pending: robinhood_oauth._PendingAuthorization) -> None:
    # No redirect needed -- a valid cached token meant the flow completed immediately.
    pending.done.set()


async def test_authorize_redirects_to_the_generated_auth_url(client, monkeypatch):
    monkeypatch.setattr(robinhood_oauth, "_run_authorization_flow", _fake_flow_needing_redirect)

    response = await client.get("/oauth/robinhood/authorize", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "https://robinhood.example/oauth?state=xyz"


async def test_authorize_redirects_to_status_when_already_authorized(client, monkeypatch):
    monkeypatch.setattr(robinhood_oauth, "_run_authorization_flow", _fake_flow_already_authorized)

    response = await client.get("/oauth/robinhood/authorize", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/oauth/robinhood/status"


async def test_callback_without_a_flow_in_progress_is_rejected(client):
    response = await client.get("/oauth/robinhood/callback", params={"code": "abc"})
    assert response.status_code == 400


async def test_callback_completes_the_pending_flow_and_returns_tool_names(client, monkeypatch):
    monkeypatch.setattr(robinhood_oauth, "_run_authorization_flow", _fake_flow_needing_redirect)
    await client.get("/oauth/robinhood/authorize", follow_redirects=False)

    response = await client.get(
        "/oauth/robinhood/callback", params={"code": "abc123", "state": "xyz"}
    )

    assert response.status_code == 200
    assert "get_equity_positions" in response.text
    assert "place_equity_order" in response.text

    status = (await client.get("/oauth/robinhood/status")).json()
    assert status == {
        "in_progress": False,
        "completed": True,
        "error": None,
        "tool_names": ["get_equity_positions", "place_equity_order"],
        "accounts": None,
    }


async def test_callback_with_error_param_marks_flow_failed(client, monkeypatch):
    monkeypatch.setattr(robinhood_oauth, "_run_authorization_flow", _fake_flow_needing_redirect)
    await client.get("/oauth/robinhood/authorize", follow_redirects=False)

    response = await client.get(
        "/oauth/robinhood/callback",
        params={"error": "access_denied", "error_description": "User denied access"},
    )

    assert response.status_code == 400
    assert "User denied access" in response.text

    status = (await client.get("/oauth/robinhood/status")).json()
    assert status["completed"] is False
    assert status["error"] == "User denied access"


async def test_callback_without_code_or_error_is_rejected(client, monkeypatch):
    monkeypatch.setattr(robinhood_oauth, "_run_authorization_flow", _fake_flow_needing_redirect)
    await client.get("/oauth/robinhood/authorize", follow_redirects=False)

    response = await client.get("/oauth/robinhood/callback")

    assert response.status_code == 400


async def test_status_before_any_authorization_attempt(client):
    response = await client.get("/oauth/robinhood/status")
    assert response.json() == {"in_progress": False, "completed": False}


async def test_underlying_flow_failure_surfaces_as_502_on_callback(client, monkeypatch):
    async def _failing_flow(pending: robinhood_oauth._PendingAuthorization) -> None:
        pending.auth_url = "https://robinhood.example/oauth?state=xyz"
        pending.auth_url_ready.set()
        await pending.callback_received.wait()
        pending.error = "token exchange failed: 400 Bad Request"
        pending.done.set()

    monkeypatch.setattr(robinhood_oauth, "_run_authorization_flow", _failing_flow)
    await client.get("/oauth/robinhood/authorize", follow_redirects=False)

    response = await client.get("/oauth/robinhood/callback", params={"code": "abc"})

    assert response.status_code == 502
    assert "token exchange failed" in response.text
