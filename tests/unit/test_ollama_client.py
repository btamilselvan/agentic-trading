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


@pytest.mark.asyncio
@respx.mock
async def test_decide_returns_parsed_decision_on_first_success():
    respx.post("http://ollama.local/api/generate").mock(
        return_value=httpx.Response(200, json={"response": _VALID_DECISION_JSON})
    )
    client = OllamaClient(host="http://ollama.local", model="gemma3")

    decision, prompt, raw = await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert decision.decision == "BUY"
    assert decision.buy_limit_price == 100.0
    assert "AAPL" in prompt
    assert raw == _VALID_DECISION_JSON


@pytest.mark.asyncio
@respx.mock
async def test_decide_retries_after_malformed_response_then_succeeds():
    route = respx.post("http://ollama.local/api/generate")
    route.side_effect = [
        httpx.Response(200, json={"response": "not valid json"}),
        httpx.Response(200, json={"response": _VALID_DECISION_JSON}),
    ]
    client = OllamaClient(host="http://ollama.local", model="gemma3")
    client.max_retries = 1

    decision, _, _ = await client.decide("AAPL", [], TickerState(0, 0, 0.0))

    assert decision.decision == "BUY"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_decide_raises_after_exhausting_retries():
    respx.post("http://ollama.local/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "still not json"})
    )
    client = OllamaClient(host="http://ollama.local", model="gemma3")
    client.max_retries = 1

    with pytest.raises(LLMDecisionError):
        await client.decide("AAPL", [], TickerState(0, 0, 0.0))
