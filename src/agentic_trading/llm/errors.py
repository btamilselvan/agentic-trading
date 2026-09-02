"""Shared bits between LLMClient implementations that aren't part of the Protocol
itself: the exception raised when no provider could produce a valid TradeDecision
after retries, and defensive parsing of "structured output" responses that some
models wrap in a Markdown code fence despite the format constraint (observed live
with both Ollama Cloud's gemma4:31b and worth guarding for any new provider until
proven unnecessary).

Also observed live (Gemini's gemma-4-31b-it, 2026-09-02): a lone trailing ``` with
no matching leading fence -- bare JSON followed by a stray closing fence marker,
apparently a habit left over from the model's fenced-code-block training even
though nothing opened one here. `strip_markdown_fence` strips a trailing fence
unconditionally (not only when a leading one was also found), so this asymmetric
case doesn't fail JSON parsing with "trailing characters" the way it used to.
"""

from __future__ import annotations


class LLMDecisionError(Exception):
    """Raised when no valid TradeDecision could be obtained after all retries."""


def strip_markdown_fence(text: str) -> str:
    """Strip a leading and/or trailing ```-fence around JSON if present; a no-op on
    already-bare JSON. The two ends are stripped independently (see module
    docstring for why) rather than requiring both or neither."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline != -1 else stripped[3:]
        stripped = stripped.strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()
