import httpx
import respx

from agentic_trading.alerts.base import NullNotifier
from agentic_trading.alerts.webhook_notifier import WebhookNotifier


async def test_null_notifier_does_nothing():
    await NullNotifier().notify("title", {"a": 1})  # just must not raise


@respx.mock
async def test_webhook_notifier_posts_title_and_fields_as_text():
    route = respx.post("https://hooks.example.com/webhook").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    notifier = WebhookNotifier("https://hooks.example.com/webhook")

    await notifier.notify("Trade Filled", {"ticker": "AAPL", "fill_price": 100.5})

    assert route.called
    body = route.calls.last.request.content.decode()
    assert "Trade Filled" in body
    assert "AAPL" in body
    assert "100.5" in body


@respx.mock
async def test_webhook_notifier_swallows_http_errors():
    respx.post("https://hooks.example.com/webhook").mock(
        return_value=httpx.Response(500, text="boom")
    )
    notifier = WebhookNotifier("https://hooks.example.com/webhook")

    await notifier.notify("Trade Filled", {"ticker": "AAPL"})  # must not raise


def test_get_notifier_picks_webhook_when_url_configured(monkeypatch):
    from agentic_trading import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("WEBHOOK_URL", "https://hooks.example.com/webhook")
    from agentic_trading.alerts.base import get_notifier

    notifier = get_notifier()
    assert isinstance(notifier, WebhookNotifier)
    config.get_settings.cache_clear()


def test_get_notifier_picks_null_when_no_url_configured(monkeypatch):
    from agentic_trading import config

    config.get_settings.cache_clear()
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    from agentic_trading.alerts.base import get_notifier

    notifier = get_notifier()
    assert isinstance(notifier, NullNotifier)
    config.get_settings.cache_clear()
