"""LLMClient interface -- the only thing the rest of the app depends on for the
decision engine. Swap the provider (Ollama, Gemini, OpenAI, Claude, ...) by
implementing this Protocol and registering it in `get_llm_client`; no other module
needs to change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agentic_trading.config import LlmProvider, get_settings
from agentic_trading.llm.schema import TickerState, TradeDecision
from agentic_trading.market_data.bucket_builder import BucketLike


class LLMClient(Protocol):
    async def decide(
        self, ticker: str, bucket_history: Sequence[BucketLike], ticker_state: TickerState
    ) -> tuple[TradeDecision, str, str]:
        """Returns (decision, prompt_text, raw_response_text).

        prompt_text/raw_response_text are returned purely for audit persistence
        (llm_decisions.prompt / llm_decisions.raw_response) -- callers should not
        need to re-derive them.
        """
        ...


def get_llm_client() -> LLMClient:
    """Provider selection driven entirely by config.llm_provider -- this is the one
    place that needs a new branch when a new LLMClient implementation is added.
    """
    settings = get_settings()
    if settings.llm_provider == LlmProvider.OLLAMA:
        from agentic_trading.llm.ollama_client import OllamaClient

        return OllamaClient()
    if settings.llm_provider == LlmProvider.GEMINI:
        from agentic_trading.llm.gemini_client import GeminiClient

        return GeminiClient()
    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} has no LLMClient implementation yet. "
        "Add one under agentic_trading/llm/ and register it here."
    )
