"""In-app alternative to scripts/bootstrap_mcp_oauth.py's standalone local callback
server: complete the Robinhood MCP's OAuth authorization by visiting
GET /oauth/robinhood/authorize on the running app instead of running a separate
script. Useful when there's no convenient way to run a local script against the
environment the app is deployed in (e.g. you'd rather hit a URL on your already
publicly-reachable production instance than open a port on it for a one-off script).

Use either this or the script -- not both against the same MCP_OAUTH_REDIRECT_URI at
once (see config.mcp_oauth_redirect_uri). Switching from one to the other after
already completing a flow may require deleting MCP_TOKEN_STORE_PATH first, since the
cached OAuth client registration is tied to whichever redirect_uri it was created
with.

OAuth's authorization-code flow is inherently stateful across two separate HTTP
round trips: the browser hits /authorize, gets redirected to Robinhood, grants
consent, and Robinhood redirects the browser back to /callback with a code. A single
module-level `_pending` object bridges mcp.client.auth.OAuthClientProvider's
redirect_handler/callback_handler pair across those two requests -- deliberately not
designed for concurrent/multi-user flows, since this is a rare, operator-driven
action (there's only ever one Robinhood account being linked).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from mcp.shared.auth import AuthorizationCodeResult

from agentic_trading.config import get_settings
from agentic_trading.execution.broker_mcp_client import open_mcp_session, unwrap_tool_result

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/oauth/robinhood", tags=["robinhood-oauth"])

_AUTH_WAIT_TIMEOUT_SECONDS = 20.0


class _PendingAuthorization:
    def __init__(self) -> None:
        self.auth_url: str | None = None
        self.auth_url_ready = asyncio.Event()
        self.callback_result: AuthorizationCodeResult | None = None
        self.callback_received = asyncio.Event()
        self.done = asyncio.Event()
        self.error: str | None = None
        self.tool_names: list[str] | None = None
        self.accounts: list[dict] | None = None
        self.task: asyncio.Task | None = None


_pending: _PendingAuthorization | None = None


async def _run_authorization_flow(pending: _PendingAuthorization) -> None:
    """Runs the whole first-authenticated-request flow in the background: this is
    what actually drives OAuthClientProvider's redirect_handler (fires as soon as
    Robinhood needs the user to consent) and callback_handler (fires once
    /callback below hands it a code) -- see module docstring for why this can't be
    two fully independent, stateless requests.
    """
    settings = get_settings()

    async def redirect_handler(auth_url: str) -> None:
        pending.auth_url = auth_url
        pending.auth_url_ready.set()

    async def callback_handler() -> AuthorizationCodeResult:
        await pending.callback_received.wait()
        if pending.callback_result is None:
            raise RuntimeError("callback_received was set without a callback_result")
        return pending.callback_result

    try:
        async with open_mcp_session(
            redirect_handler=redirect_handler, callback_handler=callback_handler
        ) as session:
            tools = await session.list_tools()
            pending.tool_names = [tool.name for tool in tools.tools]

            accounts_result = await session.call_tool("get_accounts", {})
            pending.accounts = unwrap_tool_result(accounts_result).get("accounts", [])
        logger.info(
            "Robinhood MCP authorization completed; token saved to %s",
            settings.mcp_token_store_path,
        )
    except Exception as exc:
        pending.error = str(exc)
        logger.exception("Robinhood MCP authorization failed")
    finally:
        # Whichever of authorize()/callback() is waiting needs to stop waiting even
        # if we errored out before ever reaching a redirect (or a token already on
        # disk meant no redirect was needed at all).
        pending.auth_url_ready.set()
        pending.done.set()


@router.get("/authorize")
async def authorize() -> RedirectResponse:
    """Visit this directly in a browser. Starts (or resumes, if one is already
    running) the OAuth flow and redirects to Robinhood's consent screen -- unless a
    still-valid token is already cached, in which case this just confirms that.
    """
    global _pending
    if _pending is None or _pending.done.is_set():
        _pending = _PendingAuthorization()
        _pending.task = asyncio.create_task(_run_authorization_flow(_pending))

    pending = _pending
    done_task = asyncio.ensure_future(pending.done.wait())
    redirect_task = asyncio.ensure_future(pending.auth_url_ready.wait())
    try:
        await asyncio.wait(
            {done_task, redirect_task},
            timeout=_AUTH_WAIT_TIMEOUT_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for task in (done_task, redirect_task):
            if not task.done():
                task.cancel()

    if pending.auth_url is not None:
        return RedirectResponse(pending.auth_url)
    if pending.done.is_set():
        if pending.error:
            raise HTTPException(status_code=502, detail=f"Authorization failed: {pending.error}")
        # Completed without ever needing a redirect -- a valid token was already cached.
        return RedirectResponse("/oauth/robinhood/status")
    raise HTTPException(
        status_code=504, detail="Timed out waiting for the Robinhood authorization URL"
    )


@router.get("/callback")
async def callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    iss: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
) -> HTMLResponse:
    """Robinhood redirects the browser here after the user approves (or denies)
    access. Must match MCP_OAUTH_REDIRECT_URI exactly -- that's what got registered
    as this app's redirect_uri when the flow started.
    """
    if _pending is None or _pending.done.is_set():
        raise HTTPException(
            status_code=400, detail="No authorization flow is currently in progress"
        )

    if error:
        _pending.error = error_description or error
        _pending.done.set()
        raise HTTPException(
            status_code=400, detail=f"Robinhood denied authorization: {_pending.error}"
        )

    if not code:
        raise HTTPException(
            status_code=400, detail="Callback did not include an authorization code"
        )

    _pending.callback_result = AuthorizationCodeResult(code=code, state=state, iss=iss)
    _pending.callback_received.set()

    try:
        await asyncio.wait_for(_pending.done.wait(), timeout=_AUTH_WAIT_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="Timed out completing the token exchange"
        ) from exc

    if _pending.error:
        raise HTTPException(status_code=502, detail=f"Authorization failed: {_pending.error}")

    tool_list = "".join(f"<li>{name}</li>" for name in _pending.tool_names or [])
    account_list = "".join(
        f"<li>{a.get('account_number')} ({a.get('type')}, "
        f"agentic_allowed={a.get('agentic_allowed')})</li>"
        for a in _pending.accounts or []
    )
    return HTMLResponse(f"""
        <html><body>
          <h3>Robinhood authorization successful.</h3>
          <p>Token saved. You can close this tab.</p>
          <p>Live MCP tools (compare against _TOOL_NAMES in broker_mcp_client.py):</p>
          <ul>{tool_list}</ul>
          <p>Your accounts (use the agentic_allowed=true one as
             ROBINHOOD_AGENTIC_ACCOUNT_NUMBER in .env):</p>
          <ul>{account_list}</ul>
        </body></html>
    """)


@router.get("/status")
async def oauth_status() -> dict:
    if _pending is None:
        return {"in_progress": False, "completed": False}
    return {
        "in_progress": not _pending.done.is_set(),
        "completed": _pending.done.is_set() and _pending.error is None,
        "error": _pending.error,
        "tool_names": _pending.tool_names,
        "accounts": _pending.accounts,
    }
