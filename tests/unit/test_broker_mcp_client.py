"""Unit tests for the fixes made to execution/broker_mcp_client.py after testing
against the live Robinhood MCP on 2026-08-17: the required account_number, the
{"data": ..., "guide": ...} response envelope, and cleanup errors no longer masking
a successful call's result. The full request/response round trip against the real
MCP is verified manually (see module docstring); these test the pure logic pieces.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agentic_trading.config import Settings
from agentic_trading.execution import broker_mcp_client as bmc
from agentic_trading.execution.broker_mcp_client import (
    _parse_price,
    _require_account_number,
    _safe_aexit,
    _to_amount_string,
    unwrap_tool_result,
)


def _fake_result(structured_content=None, content=None, is_error=False):
    return SimpleNamespace(
        structured_content=structured_content, content=content or [], is_error=is_error
    )


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


def test_require_account_number_raises_when_unset(monkeypatch):
    # Settings() reads from the real .env file regardless of os.environ, so we
    # inject a controlled instance directly rather than fight that via monkeypatch.
    monkeypatch.setattr(
        bmc, "get_settings", lambda: Settings(robinhood_agentic_account_number=None)
    )
    with pytest.raises(RuntimeError, match="ROBINHOOD_AGENTIC_ACCOUNT_NUMBER"):
        _require_account_number()


def test_require_account_number_returns_configured_value(monkeypatch):
    monkeypatch.setattr(
        bmc, "get_settings", lambda: Settings(robinhood_agentic_account_number="TEST-ACCT-0001")
    )
    assert _require_account_number() == "TEST-ACCT-0001"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (100.5, "100.5"),
        (5.0, "5.0"),
        (0.1, "0.1"),
        (2, "2"),
    ],
)
def test_to_amount_string_avoids_float_artifacts(value, expected):
    assert _to_amount_string(value) == expected


def test_to_amount_string_avoids_decimal_conversion_artifacts():
    # Decimal(0.1) directly (skipping the str() round-trip) produces a long,
    # ugly binary-float expansion like '0.1000000000000000055511151231257827...'.
    # _to_amount_string must not leak that -- it can't fix a float that's already
    # imprecise by the time it gets here (0.1 + 0.2 == 0.30000000000000004 as a
    # float, full stop), only avoid adding a second layer of artifacts on top.
    assert _to_amount_string(0.1) == "0.1"


def test_unwrap_tool_result_unwraps_data_envelope():
    result = _fake_result(structured_content={"data": {"positions": []}, "guide": "..."})
    assert unwrap_tool_result(result) == {"positions": []}


def test_unwrap_tool_result_falls_back_to_raw_payload_without_data_key():
    result = _fake_result(structured_content={"positions": []})
    assert unwrap_tool_result(result) == {"positions": []}


def test_unwrap_tool_result_parses_text_block_when_no_structured_content():
    result = _fake_result(content=[_TextBlock('{"data": {"order_id": "abc"}}')])
    assert unwrap_tool_result(result) == {"order_id": "abc"}


def test_unwrap_tool_result_raises_when_nothing_parseable():
    result = _fake_result()
    with pytest.raises(ValueError, match="no parseable content"):
        unwrap_tool_result(result)


def test_unwrap_tool_result_raises_with_the_real_message_on_tool_error():
    # ClientSession.call_tool does not raise on a business-logic failure (market
    # closed, insufficient buying power, bad symbol, ...) -- it returns
    # is_error=True with a plain-English explanation in content, not JSON. Before
    # this check existed, that text hit the JSON-parsing fallback and blew up with
    # an opaque JSONDecodeError instead of the real reason (see 2026-08-21 CLOV
    # review_order failure).
    result = _fake_result(
        content=[_TextBlock("Market is closed for regular trading hours")], is_error=True
    )
    with pytest.raises(RuntimeError, match="Market is closed for regular trading hours"):
        unwrap_tool_result(result)


def test_unwrap_tool_result_raises_even_with_no_error_text():
    result = _fake_result(content=[], is_error=True)
    with pytest.raises(RuntimeError, match="no error detail returned"):
        unwrap_tool_result(result)


def test_parse_price_reads_a_decimal_string():
    assert _parse_price("4.110000") == 4.11


def test_parse_price_is_none_for_an_unfilled_order():
    assert _parse_price(None) is None


class _FakeSession:
    """Stands in for mcp.ClientSession -- records the tool/arguments it was called
    with and returns a canned CallToolResult-shaped object."""

    def __init__(self, result):
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._result


def _stub_open_mcp_session(monkeypatch, session):
    @asynccontextmanager
    async def _fake(*args, **kwargs):
        yield session

    monkeypatch.setattr(bmc, "open_mcp_session", _fake)
    monkeypatch.setattr(bmc, "_require_account_number", lambda: "ACCT-1")


async def test_place_order_parses_the_nested_order_envelope(monkeypatch):
    # CONFIRMED 2026-08-21 against the live server: place_equity_order's response
    # nests the order one level deeper than the {"data": ...} envelope, under
    # "order" -- and immediately after placement the order is "unconfirmed", not
    # "filled" (real orders fill asynchronously; order_manager's sweep detects the
    # eventual fill via position-quantity polling, not this response).
    session = _FakeSession(
        _fake_result(
            structured_content={
                "data": {
                    "order": {
                        "id": "6a888c54-318f-4568-9a63-75750ff98220",
                        "state": "unconfirmed",
                        "average_price": None,
                    }
                },
                "guide": "...",
            }
        )
    )
    _stub_open_mcp_session(monkeypatch, session)

    placed = await bmc.McpBrokerClient().place_order(
        ticker="CLOV", side="buy", quantity=1.0, limit_price=4.11
    )

    assert placed.broker_order_id == "6a888c54-318f-4568-9a63-75750ff98220"
    assert placed.status == "unconfirmed"
    assert placed.fill_price is None
    assert session.calls == [
        (
            "place_equity_order",
            {
                "account_number": "ACCT-1",
                "symbol": "CLOV",
                "side": "buy",
                "type": "limit",
                "quantity": "1.0",
                "limit_price": "4.11",
            },
        )
    ]


async def test_place_order_parses_fill_price_once_filled(monkeypatch):
    session = _FakeSession(
        _fake_result(
            structured_content={
                "data": {"order": {"id": "abc-123", "state": "filled", "average_price": "4.1100"}}
            }
        )
    )
    _stub_open_mcp_session(monkeypatch, session)

    placed = await bmc.McpBrokerClient().place_order(
        ticker="CLOV", side="buy", quantity=1.0, limit_price=4.11
    )

    assert placed.status == "filled"
    assert placed.fill_price == 4.11


async def test_place_order_raises_loudly_when_order_id_still_missing(monkeypatch):
    session = _FakeSession(
        _fake_result(structured_content={"data": {"order": {"state": "unconfirmed"}}})
    )
    _stub_open_mcp_session(monkeypatch, session)

    with pytest.raises(ValueError, match=r"no order\.id/order_id field"):
        await bmc.McpBrokerClient().place_order(
            ticker="CLOV", side="buy", quantity=1.0, limit_price=4.11
        )


async def test_cancel_order_surfaces_tool_errors(monkeypatch):
    # Previously the cancel_order response was discarded unchecked, so a rejected
    # cancel (e.g. already filled/already cancelled) silently looked like success.
    session = _FakeSession(
        _fake_result(content=[_TextBlock("Order already filled")], is_error=True)
    )
    _stub_open_mcp_session(monkeypatch, session)

    with pytest.raises(RuntimeError, match="Order already filled"):
        await bmc.McpBrokerClient().cancel_order("abc-123")


async def test_safe_aexit_swallows_cleanup_exceptions():
    class _BoomOnExit:
        async def __aexit__(self, *exc_info):
            raise RuntimeError("Session termination failed: 400")

    await _safe_aexit(_BoomOnExit(), "test-resource")  # must not raise


async def test_safe_aexit_is_a_noop_when_cleanup_succeeds():
    calls = []

    class _CleanExit:
        async def __aexit__(self, *exc_info):
            calls.append(exc_info)

    await _safe_aexit(_CleanExit(), "test-resource")
    assert calls == [(None, None, None)]
