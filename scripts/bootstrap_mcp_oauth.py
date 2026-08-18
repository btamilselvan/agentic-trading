#!/usr/bin/env python
"""One-time interactive OAuth bootstrap for the Robinhood Trading MCP.

Run this once, locally, before starting the service:

    python scripts/bootstrap_mcp_oauth.py

It opens a browser to Robinhood's OAuth consent screen, catches the redirect on a
local callback server (bound to whatever host/port MCP_OAUTH_REDIRECT_URI specifies,
default http://localhost:8765/callback), and persists the resulting token to
MCP_TOKEN_STORE_PATH (default .secrets/mcp_token.json). execution/broker_mcp_client.py's
headless handlers deliberately refuse to run this flow themselves -- it only ever
happens here, interactively, on demand, or via the alternative in-app
GET /oauth/robinhood/{authorize,callback} endpoints (api/robinhood_oauth.py). Use one
or the other, not both against the same MCP_OAUTH_REDIRECT_URI at once.

It also prints the live server's tool list and your accounts at the end -- use the
former to confirm the tool names in execution/broker_mcp_client.py's _TOOL_NAMES
actually match, and the latter (the account with agentic_allowed=true) as the value
for ROBINHOOD_AGENTIC_ACCOUNT_NUMBER in .env.
"""

from __future__ import annotations

import asyncio
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from mcp.shared.auth import AuthorizationCodeResult

from agentic_trading.config import get_settings
from agentic_trading.execution.broker_mcp_client import open_mcp_session, unwrap_tool_result


class _CallbackResult:
    code: str | None = None
    state: str | None = None
    iss: str | None = None


def _run_callback_server(
    host: str, port: int, result: _CallbackResult, ready: threading.Event, done: threading.Event
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 -- http.server's required method name
            params = parse_qs(urlparse(self.path).query)
            result.code = params.get("code", [None])[0]
            result.state = params.get("state", [None])[0]
            result.iss = params.get("iss", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>Authorized. You can close this tab.</body></html>")
            done.set()

        def log_message(self, log_format: str, *args: object) -> None:
            return  # silence http.server's default request logging to stderr

    server = HTTPServer((host, port), Handler)
    ready.set()
    while not done.is_set():
        server.handle_request()
    server.server_close()


async def main() -> None:
    settings = get_settings()
    redirect = urlparse(settings.mcp_oauth_redirect_uri)
    callback_host = redirect.hostname or "localhost"
    callback_port = redirect.port or (443 if redirect.scheme == "https" else 80)

    print(f"Authorizing against {settings.mcp_server_url}")
    print(f"Token will be saved to {settings.mcp_token_store_path}")
    print(f"Listening for the callback on {callback_host}:{callback_port}\n")

    result = _CallbackResult()
    ready = threading.Event()
    done = threading.Event()
    server_thread = threading.Thread(
        target=_run_callback_server,
        args=(callback_host, callback_port, result, ready, done),
        daemon=True,
    )
    server_thread.start()
    ready.wait(timeout=5)

    async def redirect_handler(auth_url: str) -> None:
        print(f"Opening browser for authorization:\n  {auth_url}\n")
        print("If it doesn't open automatically, paste that URL into a browser.")
        webbrowser.open(auth_url)

    async def callback_handler() -> AuthorizationCodeResult:
        # The callback server runs on its own OS thread -- poll rather than block
        # this coroutine's event loop while waiting for the browser redirect.
        while not done.is_set():
            await asyncio.sleep(0.25)
        if not result.code:
            raise RuntimeError("OAuth callback did not include an authorization code")
        return AuthorizationCodeResult(code=result.code, state=result.state, iss=result.iss)

    async with open_mcp_session(
        redirect_handler=redirect_handler, callback_handler=callback_handler
    ) as session:
        print("\nAuthorization successful, token saved.\n")

        tools = await session.list_tools()
        print("Live server tools (compare against _TOOL_NAMES in broker_mcp_client.py):")
        for tool in tools.tools:
            print(f"  - {tool.name}")

        accounts_result = await session.call_tool("get_accounts", {})
        accounts = unwrap_tool_result(accounts_result).get("accounts", [])
        print("\nYour accounts (use the agentic_allowed=true one as "
              "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER in .env):")
        for account in accounts:
            allowed = account.get("agentic_allowed")
            marker = "agentic_allowed=true" if allowed else "agentic_allowed=false"
            print(f"  - {account.get('account_number')} ({account.get('type')}, {marker})")


if __name__ == "__main__":
    asyncio.run(main())
