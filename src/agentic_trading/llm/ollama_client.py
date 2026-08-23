"""Default LLMClient implementation: an Ollama server, local or cloud.

Uses the /api/chat endpoint (rather than /api/generate) because that's the one
verified to work against Ollama Cloud (ollama.com) with a Bearer API key; it also
works unchanged against a local Ollama daemon, so one code path covers both --
switching between them is purely a `ollama_host`/`ollama_api_key` settings change,
see config.py. Uses Ollama's structured-output support (`format` set to the
TradeDecision JSON schema) so the model is constrained to emit a parseable
response. Still retries on malformed/invalid output since structured-output
constraints aren't a hard guarantee across all models -- confirmed live:
gemma4:31b on Ollama Cloud wraps its otherwise-valid JSON in a ```json fence
even under the `format` constraint (a local gemma daemon does not); see
`_strip_markdown_fence`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from agentic_trading.config import get_settings
from agentic_trading.llm.prompt import build_prompt
from agentic_trading.llm.schema import TickerState, TradeDecision
from agentic_trading.market_data.bucket_builder import BucketLike

logger = logging.getLogger(__name__)


class LLMDecisionError(Exception):
    """Raised when no valid TradeDecision could be obtained after all retries."""


def _strip_markdown_fence(text: str) -> str:
    """Some models (observed: gemma4:31b on Ollama Cloud) wrap structured-output JSON
    in a ```json ... ``` fence despite the `format` constraint, even though the same
    request against a local Ollama daemon returns bare JSON. Strip one if present;
    a no-op on already-bare JSON."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    stripped = stripped[first_newline + 1 :] if first_newline != -1 else stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


_UNSET: str | None = "__unset__"  # sentinel distinct from None, which is a valid api_key override


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        api_key: str | None = _UNSET,
    ):
        settings = get_settings()
        self.host = host or settings.ollama_host
        self.model = model or settings.llm_model
        # api_key=None must mean "explicitly no auth" (e.g. tests), distinct from the
        # argument being omitted entirely (defer to settings.ollama_api_key) -- a plain
        # `api_key or settings...` default can't tell those apart, since None and "not
        # passed" would otherwise collapse to the same fallback.
        self.api_key = settings.ollama_api_key if api_key is _UNSET else api_key
        self.timeout = settings.llm_request_timeout_seconds
        self.max_retries = settings.llm_max_retries
        self.temperature = settings.llm_temperature

    async def decide(
        self, ticker: str, bucket_history: Sequence[BucketLike], ticker_state: TickerState
    ) -> tuple[TradeDecision, str, str]:
        prompt = build_prompt(ticker, bucket_history, ticker_state)

        logger.debug("Calling Ollama (%s) with prompt: %s", self.model, prompt)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            for attempt in range(1, self.max_retries + 2):
                raw_text = ""
                try:
                    response = await client.post(
                        f"{self.host}/api/chat",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "format": TradeDecision.model_json_schema(),
                            "options": {"temperature": self.temperature},
                        },
                    )
                    response.raise_for_status()
                    raw_text = response.json()["message"]["content"]
                    decision = TradeDecision.model_validate_json(_strip_markdown_fence(raw_text))
                    return decision, prompt, raw_text
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    last_error = exc
                    logger.warning(
                        "Ollama decision attempt %d/%d failed for %s: %s (raw=%r)",
                        attempt,
                        self.max_retries + 1,
                        ticker,
                        exc,
                        raw_text,
                    )

        raise LLMDecisionError(
            f"Ollama produced no valid TradeDecision for {ticker} after "
            f"{self.max_retries + 1} attempts"
        ) from last_error
