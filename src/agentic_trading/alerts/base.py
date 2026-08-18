"""Notifier interface -- decouples "something happened" from "how it's announced".
Swap the implementation via config (WEBHOOK_URL set or not) without touching callers.
"""

from __future__ import annotations

from typing import Protocol


class Notifier(Protocol):
    async def notify(self, title: str, fields: dict[str, object]) -> None:
        """Sends a title + key/value fields alert. Implementations decide how to
        render this (Slack/Discord/Telegram markdown, or nothing at all)."""
        ...


class NullNotifier:
    """No-op notifier used when WEBHOOK_URL isn't configured -- keeps callers from
    needing an `if webhook_configured` check at every call site."""

    async def notify(self, title: str, fields: dict[str, object]) -> None:
        return None


def get_notifier() -> Notifier:
    from agentic_trading.config import get_settings

    settings = get_settings()
    if settings.webhook_url:
        from agentic_trading.alerts.webhook_notifier import WebhookNotifier

        return WebhookNotifier(settings.webhook_url)
    return NullNotifier()
