import json

import httpx
import pytest
import respx

from agentic_trading.llm.ollama_client import LLMDecisionError, OllamaClient
from agentic_trading.llm.schema import TickerState

_VALID_DECISION_JSON = json.dumps(
    {
        "decision": "BUY",
        "confidence_score": 0.9,
        "buy_limit_price": 100.0,
        "target_sell_price": 102.0,
        "max_holding_time_minutes": 20,
        "pattern_reasoning": "breakout",
    }
)


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


@pytest.mark.asyncio
@respx.mock
async def test_decide_returns_parsed_decision_on_first_success():
    respx.post("http://ollama.local/api/chat").mock(return_value=_chat_response(_VALID_DECISION_JSON))
    client = OllamaClient(host="http://ollama.local", model="gemma3", api_key=None)

    decision, prompt, raw = await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert decision.decision == "BUY"
    assert decision.buy_limit_price == 100.0
    assert "AAPL" in prompt
    assert raw == _VALID_DECISION_JSON


@pytest.mark.asyncio
@respx.mock
async def test_decide_retries_after_malformed_response_then_succeeds():
    route = respx.post("http://ollama.local/api/chat")
    route.side_effect = [
        _chat_response("not valid json"),
        _chat_response(_VALID_DECISION_JSON),
    ]
    client = OllamaClient(host="http://ollama.local", model="gemma3", api_key=None)
    client.max_retries = 1

    decision, _, _ = await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert decision.decision == "BUY"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_decide_raises_after_exhausting_retries():
    respx.post("http://ollama.local/api/chat").mock(return_value=_chat_response("still not json"))
    client = OllamaClient(host="http://ollama.local", model="gemma3", api_key=None)
    client.max_retries = 1

    with pytest.raises(LLMDecisionError):
        await client.decide("AAPL", [], TickerState(0, 0, 0.0))


@pytest.mark.asyncio
@respx.mock
async def test_decide_sends_bearer_token_when_api_key_set():
    route = respx.post("http://ollama.local/api/chat").mock(
        return_value=_chat_response(_VALID_DECISION_JSON)
    )
    client = OllamaClient(host="http://ollama.local", model="gemma3", api_key="secret-key")

    await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert route.calls.last.request.headers["Authorization"] == "Bearer secret-key"


@pytest.mark.asyncio
@respx.mock
async def test_decide_omits_auth_header_when_no_api_key():
    route = respx.post("http://ollama.local/api/chat").mock(
        return_value=_chat_response(_VALID_DECISION_JSON)
    )
    client = OllamaClient(host="http://ollama.local", model="gemma3", api_key=None)

    await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert "Authorization" not in route.calls.last.request.headers


@pytest.mark.asyncio
@respx.mock
async def test_decide_strips_markdown_fence_around_json():
    # Observed live from gemma4:31b on Ollama Cloud: valid JSON wrapped in a
    # ```json fence despite the `format` structured-output constraint.
    fenced = f"```json\n{_VALID_DECISION_JSON}\n```"
    respx.post("http://ollama.local/api/chat").mock(return_value=_chat_response(fenced))
    client = OllamaClient(host="http://ollama.local", model="gemma4:31b", api_key=None)

    decision, _, raw = await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert decision.decision == "BUY"
    # raw_text preserves exactly what the model returned, fence and all, for the audit trail.
    assert raw == fenced
