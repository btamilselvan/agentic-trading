import json

import httpx
import pytest
import respx

from agentic_trading.llm.errors import LLMDecisionError
from agentic_trading.llm.gemini_client import GeminiClient
from agentic_trading.llm.schema import TickerState

_VALID_DECISION_JSON = json.dumps(
    {
        "decision": "BUY",
        "confidence_score": 0.9,
        "buy_limit_price": 100.0,
        "target_sell_price": 102.0,
        "stop_loss_price": 98.0,
        "max_holding_time_minutes": 20,
        "pattern_reasoning": "breakout",
        "thesis_continuity_flag": True,
    }
)

_URL = "http://gemini.local/v1beta/models/gemini-3.6-flash:generateContent"


def _generate_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]},
    )


@pytest.mark.asyncio
@respx.mock
async def test_decide_returns_parsed_decision_on_first_success():
    respx.post(_URL).mock(return_value=_generate_response(_VALID_DECISION_JSON))
    client = GeminiClient(model="gemini-3.6-flash", api_key=None, api_base="http://gemini.local")

    decision, prompt, raw = await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert decision.decision == "BUY"
    assert decision.buy_limit_price == 100.0
    assert "AAPL" in prompt
    assert raw == _VALID_DECISION_JSON


@pytest.mark.asyncio
@respx.mock
async def test_decide_retries_after_malformed_response_then_succeeds():
    route = respx.post(_URL)
    route.side_effect = [
        _generate_response("not valid json"),
        _generate_response(_VALID_DECISION_JSON),
    ]
    client = GeminiClient(model="gemini-3.6-flash", api_key=None, api_base="http://gemini.local")
    client.max_retries = 1

    decision, _, _ = await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert decision.decision == "BUY"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_decide_raises_after_exhausting_retries():
    respx.post(_URL).mock(return_value=_generate_response("still not json"))
    client = GeminiClient(model="gemini-3.6-flash", api_key=None, api_base="http://gemini.local")
    client.max_retries = 1

    with pytest.raises(LLMDecisionError):
        await client.decide("AAPL", [], TickerState(0, 0, 0.0))


@pytest.mark.asyncio
@respx.mock
async def test_decide_raises_when_response_has_no_candidates():
    # e.g. a safety-blocked prompt -- Gemini returns 200 with an empty/absent
    # candidates list rather than an HTTP error.
    respx.post(_URL).mock(return_value=httpx.Response(200, json={"candidates": []}))
    client = GeminiClient(model="gemini-3.6-flash", api_key=None, api_base="http://gemini.local")
    client.max_retries = 0

    with pytest.raises(LLMDecisionError):
        await client.decide("AAPL", [], TickerState(0, 0, 0.0))


@pytest.mark.asyncio
@respx.mock
async def test_decide_sends_api_key_header_when_set():
    route = respx.post(_URL).mock(return_value=_generate_response(_VALID_DECISION_JSON))
    client = GeminiClient(
        model="gemini-3.6-flash", api_key="secret-key", api_base="http://gemini.local"
    )

    await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert route.calls.last.request.headers["x-goog-api-key"] == "secret-key"


@pytest.mark.asyncio
@respx.mock
async def test_decide_omits_api_key_header_when_unset():
    route = respx.post(_URL).mock(return_value=_generate_response(_VALID_DECISION_JSON))
    client = GeminiClient(model="gemini-3.6-flash", api_key=None, api_base="http://gemini.local")

    await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert "x-goog-api-key" not in route.calls.last.request.headers


@pytest.mark.asyncio
@respx.mock
async def test_decide_strips_markdown_fence_around_json():
    fenced = f"```json\n{_VALID_DECISION_JSON}\n```"
    respx.post(_URL).mock(return_value=_generate_response(fenced))
    client = GeminiClient(model="gemini-3.6-flash", api_key=None, api_base="http://gemini.local")

    decision, _, raw = await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert decision.decision == "BUY"
    assert raw == fenced


@pytest.mark.asyncio
@respx.mock
async def test_decide_sends_gemini_compatible_response_schema():
    route = respx.post(_URL).mock(return_value=_generate_response(_VALID_DECISION_JSON))
    client = GeminiClient(model="gemini-3.6-flash", api_key=None, api_base="http://gemini.local")

    await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    schema = json.loads(route.calls.last.request.content)["generationConfig"]["responseSchema"]
    # No pydantic-only keywords Gemini's Schema object doesn't understand.
    assert "anyOf" not in json.dumps(schema)
    assert "exclusiveMinimum" not in json.dumps(schema)
    # Optional numeric fields translated to nullable rather than a union type.
    assert schema["properties"]["buy_limit_price"]["nullable"] is True
    assert schema["properties"]["buy_limit_price"]["type"] == "number"
    # pattern_reasoning has a Python-side default ("") for cross-provider parsing
    # tolerance, so pydantic's own required list omits it -- but Gemini's request
    # forces it anyway (observed live: flash-lite returns it blank/omitted
    # otherwise). See gemini_client._build_response_schema.
    assert "pattern_reasoning" in schema["required"]
    assert "decision" in schema["required"]  # still there -- not replaced, only added to
