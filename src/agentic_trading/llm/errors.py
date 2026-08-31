"""Shared bits between LLMClient implementations that aren't part of the Protocol
itself: the exception raised when no provider could produce a valid TradeDecision
after retries, and defensive parsing of "structured output" responses that some
models wrap in a Markdown code fence despite the format constraint (observed live
with both Ollama Cloud's gemma4:31b and worth guarding for any new provider until
proven unnecessary).
"""

from __future__ import annotations


class LLMDecisionError(Exception):
    """Raised when no valid TradeDecision could be obtained after all retries."""


def strip_markdown_fence(text: str) -> str:
    """Strip a leading/trailing ```-fence around JSON if present; a no-op on
    already-bare JSON. See module docstring."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    stripped = stripped[first_newline + 1 :] if first_newline != -1 else stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()
