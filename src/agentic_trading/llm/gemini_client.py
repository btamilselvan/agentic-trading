"""LLMClient implementation for Google's Gemini API (generativelanguage.googleapis.com),
including the free tier of Google AI Studio. Plain REST over httpx -- no google-genai
SDK dependency -- mirroring ollama_client.py's shape so the two are easy to compare.

Uses Gemini's structured-output support (`generationConfig.responseSchema`) so the
model is constrained to emit a parseable TradeDecision. Gemini's responseSchema only
accepts an OpenAPI-3.0 subset of JSON Schema -- notably no `exclusiveMinimum`/
`exclusiveMaximum` (which pydantic's `Field(gt=0)` emits) and no `anyOf` unions (which
`float | None` emits, as a `{...}` / `{"type": "null"}` pair) -- so
`_to_gemini_schema` translates TradeDecision's pydantic schema into that subset rather
than passing `TradeDecision.model_json_schema()` through as-is. TradeDecision is flat
(no nested $defs), so this translation only ever needs to recurse one level deep.

Retries on malformed/invalid output for the same reason ollama_client.py does:
structured-output constraints aren't a hard guarantee, and `responseMimeType` set to
"application/json" is not a guarantee this hasn't been observed to break either, so
`strip_markdown_fence` is applied defensively even though Gemini shouldn't need it.

Deliberately targets `generateContent`, not the newer `Interactions API` (GA'd June
2026, now Google's recommended default for new projects) -- `generateContent` remains
fully supported, and its request/response shape (including temperature control, which
this module depends on for deterministic decisions) is verified against docs and this
module's own tests. Docs for the Interactions API's exact wire format were internally
inconsistent (endpoint version, response shape, and whether `generation_config` even
exposes temperature) as of this writing; revisit only after confirming the real
request/response shape live, the same way broker_mcp_client.py was confirmed against
the live Robinhood MCP server rather than trusting docs alone.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from agentic_trading.config import get_settings
from agentic_trading.llm.errors import LLMDecisionError, strip_markdown_fence
from agentic_trading.llm.prompt import build_prompt
from agentic_trading.llm.schema import TickerState, TradeDecision
from agentic_trading.market_data.bucket_builder import BucketLike

logger = logging.getLogger(__name__)

# Keys Gemini's Schema object actually understands (OpenAPI 3.0 subset) -- anything
# else (title, default, exclusiveMinimum, ...) from pydantic's JSON Schema output is
# dropped by _to_gemini_schema rather than passed through.
_GEMINI_SCHEMA_KEYS = {
    "type",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "format",
    "nullable",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
}


def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Translate a pydantic-generated JSON Schema dict into the subset Gemini's
    responseSchema accepts. See module docstring for why this is needed."""
    any_of = schema.get("anyOf")
    if any_of is not None:
        # pydantic's shape for `X | None`: [{...the real type...}, {"type": "null"}].
        # Collapse that into "the real type, but nullable" -- Gemini has no anyOf.
        non_null = [branch for branch in any_of if branch.get("type") != "null"]
        base = _to_gemini_schema(non_null[0]) if non_null else {"type": "string"}
        base["nullable"] = True
        return base

    result: dict[str, Any] = {k: v for k, v in schema.items() if k in _GEMINI_SCHEMA_KEYS}
    if "properties" in schema:
        result["properties"] = {
            name: _to_gemini_schema(sub) for name, sub in schema["properties"].items()
        }
    if "items" in schema:
        result["items"] = _to_gemini_schema(schema["items"])
    return result


def _build_response_schema() -> dict[str, Any]:
    """TradeDecision's `pattern_reasoning` has a Python-side default (`= ""`) so
    every OTHER provider's response still parses even when it's omitted -- but
    that also means pydantic's own `required` list (which _to_gemini_schema just
    passes through) doesn't include it, so Gemini's structured-output constraint
    never actually forces a model to fill it in. Observed live: gemini's lighter/
    cheaper flash-lite tier happily returns it blank once the schema allows that.
    Force it into Gemini's required list regardless of what pydantic requires on
    the way back in -- we still want every response to explain itself.
    """
    schema = _to_gemini_schema(TradeDecision.model_json_schema())
    required = list(schema.get("required", []))
    if "pattern_reasoning" not in required:
        required.append("pattern_reasoning")
    schema["required"] = required
    return schema


_UNSET: str | None = "__unset__"  # sentinel distinct from None, which is a valid api_key override


class GeminiClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = _UNSET,
        api_base: str | None = None,
    ):
        settings = get_settings()
        self.model = model or settings.llm_model
        # api_key=None must mean "explicitly no auth" (e.g. tests), distinct from the
        # argument being omitted entirely (defer to settings.gemini_api_key) -- see
        # OllamaClient for why a plain `api_key or settings...` default can't do this.
        self.api_key = settings.gemini_api_key if api_key is _UNSET else api_key
        self.api_base = api_base or settings.gemini_api_base
        self.timeout = settings.llm_request_timeout_seconds
        self.max_retries = settings.llm_max_retries
        self.temperature = settings.llm_temperature

    async def decide(
        self, ticker: str, bucket_history: Sequence[BucketLike], ticker_state: TickerState
    ) -> tuple[TradeDecision, str, str]:
        prompt = build_prompt(ticker, bucket_history, ticker_state)

        logger.debug("Calling Gemini (%s) with prompt: %s", self.model, prompt)

        headers = {"x-goog-api-key": self.api_key} if self.api_key else {}
        url = f"{self.api_base}/v1beta/models/{self.model}:generateContent"
        response_schema = _build_response_schema()

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            for attempt in range(1, self.max_retries + 2):
                raw_text = ""
                try:
                    response = await client.post(
                        url,
                        json={
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": self.temperature,
                                "responseMimeType": "application/json",
                                "responseSchema": response_schema,
                            },
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                    raw_text = body["candidates"][0]["content"]["parts"][0]["text"]
                    decision = TradeDecision.model_validate_json(strip_markdown_fence(raw_text))
                    return decision, prompt, raw_text
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    last_error = exc
                    logger.warning(
                        "Gemini decision attempt %d/%d failed for %s: %s (raw=%r)",
                        attempt,
                        self.max_retries + 1,
                        ticker,
                        exc,
                        raw_text,
                    )

        raise LLMDecisionError(
            f"Gemini produced no valid TradeDecision for {ticker} after "
            f"{self.max_retries + 1} attempts"
        ) from last_error
