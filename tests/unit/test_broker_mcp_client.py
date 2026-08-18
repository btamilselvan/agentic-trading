"""Unit tests for the fixes made to execution/broker_mcp_client.py after testing
against the live Robinhood MCP on 2026-08-17: the required account_number, the
{"data": ..., "guide": ...} response envelope, and cleanup errors no longer masking
a successful call's result. The full request/response round trip against the real
MCP is verified manually (see module docstring); these test the pure logic pieces.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_trading.config import Settings
from agentic_trading.execution import broker_mcp_client as bmc
from agentic_trading.execution.broker_mcp_client import (
    _require_account_number,
    _safe_aexit,
    _to_amount_string,
    unwrap_tool_result,
)


def _fake_result(structured_content=None, content=None):
    return SimpleNamespace(structured_content=structured_content, content=content or [])


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
