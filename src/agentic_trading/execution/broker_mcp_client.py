"""Robinhood Trading MCP client -- account state + order execution.

This is the ONLY module that talks to the MCP; market data never comes from here
(see market_data/robinhood_client.py) -- keeping the two brokerage integrations
separate is what lets each be swapped independently later.

CONFIRMED against the live server on 2026-08-17 (via a real authenticated session --
see scripts/bootstrap_mcp_oauth.py / api/robinhood_oauth.py):
  - `_TOOL_NAMES` are all correct as originally guessed from public docs.
  - get_equity_positions / review_equity_order / place_equity_order /
    cancel_equity_order all REQUIRE an explicit `account_number` -- the MCP never
    auto-selects one (see config.robinhood_agentic_account_number).
  - review_equity_order / place_equity_order also require `type` (we always send
    "limit"), and want `quantity` / `limit_price` as decimal STRINGS, not JSON
    numbers.
  - Tool responses are wrapped as {"data": ..., "guide": "<agent-facing prose>"} --
    confirmed for get_accounts and get_equity_positions.

CONFIRMED against the live server on 2026-08-21 (a real order placed in the funded
Agentic account -- CLOV, 1 share, $4.11 limit):
  - place_equity_order's response, after the {"data": ..., "guide": ...} envelope,
    has one MORE level of nesting: {"order": {...}} -- not the flat order dict
    originally guessed. The order dict's id is under `id` (a UUID string, not
    `order_id`); its lifecycle state is under `state` (observed: "unconfirmed"
    immediately after placement -- NOT "filled"; real orders fill asynchronously,
    unlike DryRunBrokerClient's instant simulated fill), and its price is under
    `average_price` (decimal string, null until actually filled).
  - `place_order` unwraps `data["order"]` (falling back to `data` itself if some
    other tool/response doesn't nest this way) before reading `id`/`state`/
    `average_price`, and still fails loudly with the real keys it saw if `id` is
    missing -- same "don't silently mis-parse" posture as before, now with a
    confirmed shape to check against instead of a guess.

Fill detection deliberately does NOT depend on an order-status tool name: order_manager
detects fills by polling `get_open_position_quantity` for an increase, which only
depends on `get_equity_positions` -- confirmed above.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import httpx2
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from agentic_trading.config import get_settings

logger = logging.getLogger(__name__)

_TOOL_NAMES = {
    "positions": "get_equity_positions",
    "review_order": "review_equity_order",
    "place_order": "place_equity_order",
    "cancel_order": "cancel_equity_order",
}


def _require_account_number() -> str:
    settings = get_settings()
    if not settings.robinhood_agentic_account_number:
        raise RuntimeError(
            "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER is not set. Call the MCP's get_accounts "
            "tool once (the entry with agentic_allowed=true) and set it in .env before "
            "any McpBrokerClient call."
        )
    return settings.robinhood_agentic_account_number


def _to_amount_string(value: float) -> str:
    """The order tools want quantity/limit_price as plain decimal strings, not JSON
    numbers. Converting via Decimal(str(value)) (not Decimal(value) directly) avoids
    that conversion itself adding artifacts -- Decimal(0.1) expands to the full
    binary-float value ('0.1000000000000000055511151231257827...'), whereas
    Decimal(str(0.1)) gives a clean '0.1'. This does NOT retroactively fix a value
    that was already imprecise before reaching here (0.1 + 0.2 == 0.30000000000000004
    as a float, independent of how it's later stringified) -- callers should avoid
    doing float arithmetic on prices/quantities in the first place.
    """
    return format(Decimal(str(value)), "f")


def _parse_price(value: Any) -> float | None:
    """Reverse of `_to_amount_string` for reading order-tool responses back: price
    fields (`average_price`, etc.) come back as decimal strings, or null when not
    yet applicable (e.g. an unfilled order has no average_price yet).
    """
    return float(value) if value is not None else None


class FileTokenStorage(TokenStorage):
    """JSON-file persistence for the OAuth tokens/client registration used to talk to
    the MCP server, so the headless service can reuse + refresh the session
    established once, interactively, by scripts/bootstrap_mcp_oauth.py.
    """

    def __init__(self, path: str):
        self._path = Path(path)

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read().get("tokens")
        return OAuthToken.model_validate(data) if data else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json")
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)


async def _headless_redirect_handler(auth_url: str) -> None:
    raise RuntimeError(
        "No valid MCP OAuth token on disk. Run `python scripts/bootstrap_mcp_oauth.py` "
        "once, interactively, before starting the service."
    )


async def _headless_callback_handler():
    raise RuntimeError("MCP OAuth callback handling must happen via scripts/bootstrap_mcp_oauth.py")


def build_oauth_provider(*, redirect_handler=None, callback_handler=None) -> OAuthClientProvider:
    """Shared by the headless service (uses the handlers above, which fail loudly if
    no cached token is usable), scripts/bootstrap_mcp_oauth.py, and
    api/robinhood_oauth.py (the latter two pass real handlers to perform the
    authorization flow -- via a standalone local server or in-app HTTP endpoints,
    respectively; see config.mcp_oauth_redirect_uri for how they stay in sync).
    """
    settings = get_settings()
    return OAuthClientProvider(
        server_url=settings.mcp_server_url,
        client_metadata=OAuthClientMetadata(
            client_name="agentic-trading",
            redirect_uris=[settings.mcp_oauth_redirect_uri],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=FileTokenStorage(settings.mcp_token_store_path),
        redirect_handler=redirect_handler or _headless_redirect_handler,
        callback_handler=callback_handler or _headless_callback_handler,
    )


@dataclass(frozen=True)
class OrderReview:
    warnings: list[str]
    estimated_price: float | None


@dataclass(frozen=True)
class PlacedOrder:
    broker_order_id: str
    status: str
    fill_price: float | None = None


class BrokerExecutionClient(Protocol):
    """What order_manager depends on. Implemented by McpBrokerClient for LIVE mode and
    by DryRunBrokerClient (see order_manager.py) for DRY_RUN -- same interface either
    way, so order_manager never branches on mode itself.
    """

    async def get_open_position_quantity(self, ticker: str) -> float: ...

    async def review_order(
        self, *, ticker: str, side: str, quantity: float, limit_price: float
    ) -> OrderReview: ...

    async def place_order(
        self, *, ticker: str, side: str, quantity: float, limit_price: float
    ) -> PlacedOrder: ...

    async def cancel_order(self, broker_order_id: str) -> None: ...


async def _safe_aexit(cm: Any, label: str) -> None:
    """Best-effort cleanup.

    Confirmed against the live server: session/transport teardown can itself raise
    (its DELETE-based session termination has been observed to return 400) even when
    the actual tool call already succeeded. A raised __aexit__ replaces whatever the
    caller's `async with` block was in the middle of returning -- left unguarded,
    every successful call would come back looking like a failure. So cleanup errors
    are logged and swallowed here; only errors from setup or the actual call (which
    happen before this point, inside the `try` blocks below) propagate normally.
    """
    try:
        await cm.__aexit__(None, None, None)
    except Exception:
        logger.warning(
            "MCP %s cleanup failed (any call result already obtained is unaffected)",
            label,
            exc_info=True,
        )


@asynccontextmanager
async def open_mcp_session(*, redirect_handler=None, callback_handler=None):
    """Opens one authenticated MCP session, cleaning it up without letting teardown
    errors mask whatever the caller's block already produced (see _safe_aexit).

    Reused by McpBrokerClient's methods (headless -- default handlers refuse to
    perform a fresh authorization, see build_oauth_provider), and by
    scripts/bootstrap_mcp_oauth.py / api/robinhood_oauth.py (interactive handlers,
    to actually perform the OAuth flow) -- one implementation of the
    session-open/cleanup dance, not three.
    """
    settings = get_settings()
    oauth = build_oauth_provider(
        redirect_handler=redirect_handler, callback_handler=callback_handler
    )

    http_client = httpx2.AsyncClient(auth=oauth)
    await http_client.__aenter__()
    try:
        transport = streamable_http_client(settings.mcp_server_url, http_client=http_client)
        read, write = await transport.__aenter__()
        try:
            session = ClientSession(read, write)
            await session.__aenter__()
            try:
                await session.initialize()
                yield session
            finally:
                await _safe_aexit(session, "ClientSession")
        finally:
            await _safe_aexit(transport, "streamable_http_client transport")
    finally:
        await _safe_aexit(http_client, "httpx2.AsyncClient")


def unwrap_tool_result(result) -> dict[str, Any]:
    """CallToolResult -> dict: prefer structured_content when the tool provides it,
    else fall back to parsing the first text content block as JSON. Then unwrap the
    {"data": ..., "guide": "..."} envelope every tool response has been observed to
    use, falling back to the raw payload if a future/other tool doesn't wrap.

    `ClientSession.call_tool` does NOT raise when a tool call fails business-logic-
    wise (e.g. market closed, insufficient buying power, bad symbol) -- it just
    returns a CallToolResult with `is_error=True` and the error described in prose
    inside `content`, not JSON. Left unchecked, that prose then hits the JSON-parsing
    fallback below and blows up with an opaque JSONDecodeError instead of the actual
    reason -- check `is_error` first and fail loudly with the real message.
    """
    if result.is_error:
        message = "; ".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        raise RuntimeError(f"MCP tool call failed: {message or 'no error detail returned'}")

    if result.structured_content:
        payload = result.structured_content
    else:
        payload = None
        for block in result.content:
            if getattr(block, "type", None) == "text":
                payload = json.loads(block.text)
                break
        if payload is None:
            raise ValueError(f"MCP tool call returned no parseable content: {result!r}")

    data = payload.get("data")
    return data if isinstance(data, dict) else payload


class McpBrokerClient:
    """BrokerExecutionClient backed by the real Robinhood Trading MCP. Opens a fresh
    MCP session per call -- call volume here is a handful of requests per 5-minute
    poll, not a hot path worth holding a persistent session open for.
    """

    async def get_open_position_quantity(self, ticker: str) -> float:
        account_number = _require_account_number()
        async with open_mcp_session() as session:
            result = await session.call_tool(
                _TOOL_NAMES["positions"], {"account_number": account_number}
            )
            data = unwrap_tool_result(result)
            for position in data.get("positions", []):
                if position.get("symbol") == ticker:
                    return float(position.get("quantity", 0))
            return 0.0

    async def review_order(
        self, *, ticker: str, side: str, quantity: float, limit_price: float
    ) -> OrderReview:
        account_number = _require_account_number()
        async with open_mcp_session() as session:
            result = await session.call_tool(
                _TOOL_NAMES["review_order"],
                {
                    "account_number": account_number,
                    "symbol": ticker,
                    "side": side,
                    "type": "limit",
                    "quantity": _to_amount_string(quantity),
                    "limit_price": _to_amount_string(limit_price),
                },
            )
            data = unwrap_tool_result(result)
            return OrderReview(
                warnings=data.get("warnings", []),
                estimated_price=data.get("estimated_price"),
            )

    async def place_order(
        self, *, ticker: str, side: str, quantity: float, limit_price: float
    ) -> PlacedOrder:
        account_number = _require_account_number()
        async with open_mcp_session() as session:
            result = await session.call_tool(
                _TOOL_NAMES["place_order"],
                {
                    "account_number": account_number,
                    "symbol": ticker,
                    "side": side,
                    "type": "limit",
                    "quantity": _to_amount_string(quantity),
                    "limit_price": _to_amount_string(limit_price),
                },
            )
            data = unwrap_tool_result(result)
            logger.info("Placed order %s", data)
            # CONFIRMED 2026-08-21 (see module docstring): the order lives one level
            # deeper than the {"data": ...} envelope, under "order".
            order = data["order"] if isinstance(data.get("order"), dict) else data
            order_id = order.get("id") or order.get("order_id")
            if order_id is None:
                # Fail loudly with what we actually got rather than silently
                # mis-record a real order.
                raise ValueError(
                    "place_equity_order response has no order.id/order_id field -- "
                    f"update McpBrokerClient.place_order's parsing; got keys: {list(order.keys())}"
                )
            return PlacedOrder(
                broker_order_id=str(order_id),
                status=str(order.get("state") or order.get("status", "pending")),
                fill_price=_parse_price(order.get("average_price") or order.get("fill_price")),
            )

    async def cancel_order(self, broker_order_id: str) -> None:
        account_number = _require_account_number()
        async with open_mcp_session() as session:
            result = await session.call_tool(
                _TOOL_NAMES["cancel_order"],
                {"account_number": account_number, "order_id": broker_order_id},
            )
            # Same is_error check as every other tool call -- previously discarded
            # unchecked, so a rejected cancel (e.g. already filled/already cancelled)
            # silently looked like a success.
            unwrap_tool_result(result)
