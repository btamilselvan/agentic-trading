#!/usr/bin/env python
"""One-time (well, periodic -- see below) interactive OAuth bootstrap for Schwab's
Market Data Production API.

Run this once, locally, before starting the service:

    uv run scripts/bootstrap_schwab_oauth.py

Unlike the Robinhood MCP flow (scripts/bootstrap_mcp_oauth.py), schwab-py's
`easy_client` handles the whole dance itself: it opens a browser to Schwab's OAuth
consent screen, spins up its own local callback server on SCHWAB_CALLBACK_URL (must
exactly match the callback URL configured in your Schwab developer app -- Schwab
requires HTTPS here), and writes the resulting token to SCHWAB_TOKEN_PATH (default
.secrets/schwab_token.json). market_data/schwab_client.py's runtime code never
performs this interactive step itself -- it only ever reads that cached token file
and lets schwab-py refresh it silently in the background.

Schwab refresh tokens go stale after roughly 7 days of disuse (schwab-py's
easy_client proactively discards anything older than `max_token_age`, ~6.5 days,
and re-runs this same flow) -- re-run this script periodically to keep Schwab as
the live primary market-data source. There's no harm in it running past the token's
mid-life; the Robinhood fallback (market_data/market_data_client.py) covers any gap
automatically in the meantime, so this is an availability optimization, not a hard
dependency for the service to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agentic_trading.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.schwab_client_id or not settings.schwab_client_secret:
        print(
            "SCHWAB_CLIENT_ID / SCHWAB_CLIENT_SECRET are not set in .env -- register "
            "an app at https://developer.schwab.com first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    token_path = Path(settings.schwab_token_path)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Callback URL: {settings.schwab_callback_url}")
    print(f"Token will be saved to {token_path}\n")

    from schwab.auth import easy_client

    client = easy_client(
        api_key=settings.schwab_client_id,
        app_secret=settings.schwab_client_secret,
        callback_url=settings.schwab_callback_url,
        token_path=str(token_path),
    )

    print("\nAuthorization successful, token saved.\n")

    # Sanity-check the session end-to-end, same spirit as bootstrap_mcp_oauth.py
    # printing the live tool list/accounts.
    response = client.get_quote("AAPL")
    if response.status_code == 200:
        print("Sanity check -- AAPL quote:")
        print(response.json())
    else:
        print(
            f"Sanity check quote call returned HTTP {response.status_code}: "
            f"{response.text[:500]}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
