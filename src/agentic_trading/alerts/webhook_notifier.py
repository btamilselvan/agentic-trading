"""Generic webhook notifier -- posts a JSON payload shaped to work with Slack,
Discord, and Telegram's generic incoming-webhook formats: all three render a
top-level "text" string, so one payload shape covers all three without per-provider
branching. If you need provider-specific formatting (Slack blocks, Discord embeds),
add a new Notifier implementation rather than special-casing this one.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class WebhookNotifier:
    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self.webhook_url = webhook_url
        self.timeout = timeout

    async def notify(self, title: str, fields: dict[str, object]) -> None:
        lines = [f"*{title}*"] + [f"{key}: {value}" for key, value in fields.items()]
        text = "\n".join(lines)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.webhook_url, json={"text": text})
                response.raise_for_status()
        except httpx.HTTPError:
            # A failed alert should never take down the trading loop -- log and move on.
            logger.exception("Webhook notification failed (title=%r)", title)
