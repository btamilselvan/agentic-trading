"""Default LLMClient implementation: a local Ollama server.

Uses Ollama's structured-output support (`format` set to the TradeDecision JSON
schema) so the model is constrained to emit a parseable response. Still retries on
malformed/invalid output since structured-output constraints aren't a hard guarantee
across all models.
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


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        settings = get_settings()
        self.host = host or settings.ollama_host
        self.model = model or settings.llm_model
        self.timeout = settings.llm_request_timeout_seconds
        self.max_retries = settings.llm_max_retries

    async def decide(
        self, ticker: str, bucket_history: Sequence[BucketLike], ticker_state: TickerState
    ) -> tuple[TradeDecision, str, str]:
        prompt = build_prompt(ticker, bucket_history, ticker_state)
        
        logger.debug("Calling Ollama (%s) with prompt: %s", self.model, prompt)

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 2):
                raw_text = ""
                try:
                    response = await client.post(
                        f"{self.host}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "format": TradeDecision.model_json_schema(),
                        },
                    )
                    response.raise_for_status()
                    raw_text = response.json()["response"]
                    decision = TradeDecision.model_validate_json(raw_text)
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
