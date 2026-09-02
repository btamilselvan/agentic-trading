from agentic_trading.llm.errors import strip_markdown_fence

_JSON = '{\n  "decision": "HOLD"\n}'


def test_strip_markdown_fence_is_a_noop_on_bare_json():
    assert strip_markdown_fence(_JSON) == _JSON


def test_strip_markdown_fence_strips_leading_and_trailing_fence():
    assert strip_markdown_fence(f"```json\n{_JSON}\n```") == _JSON


def test_strip_markdown_fence_strips_leading_and_trailing_fence_no_language_tag():
    assert strip_markdown_fence(f"```\n{_JSON}\n```") == _JSON


def test_strip_markdown_fence_strips_trailing_only_fence():
    # Observed live from Gemini's gemma-4-31b-it (2026-09-02): bare JSON followed
    # by a stray closing fence with no matching opening one.
    assert strip_markdown_fence(f"{_JSON}\n```") == _JSON


def test_strip_markdown_fence_strips_leading_only_fence():
    assert strip_markdown_fence(f"```json\n{_JSON}") == _JSON
