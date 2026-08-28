#!/usr/bin/env python
"""Force-refresh the Schwab OAuth token on disk using its refresh token -- no
browser needed. Run this periodically (e.g. a daily cron) so the token
market_data/schwab_client.py reads is never far from a fresh access token:

    uv run scripts/refresh_schwab_token.py

Unlike scripts/bootstrap_schwab_oauth.py (which needs a browser for the *initial*
consent), this only exercises the refresh_token grant against the token already
saved at SCHWAB_TOKEN_PATH. It doesn't strictly need to run on a schedule --
client_from_token_file's session already auto-refreshes the access token as needed
on every real API call (see schwab_client.py) -- but running it proactively keeps
the on-disk token from ever going stale during long gaps with no market-data calls
(e.g. over a weekend), and gives an explicit, scriptable way to confirm the refresh
token itself is still valid without waiting for the next live trading session.

This does NOT extend the refresh token's own ~7-day lifetime (Schwab's refresh
tokens have a fixed absolute expiry from issuance, not a sliding one renewed by
use) -- once that's gone, only scripts/bootstrap_schwab_oauth.py's full interactive
browser flow gets a new one. This script fails clearly when that's the case rather
than leaving a stale token in place silently.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from agentic_trading.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.schwab_client_id or not settings.schwab_client_secret:
        print(
            "SCHWAB_CLIENT_ID / SCHWAB_CLIENT_SECRET are not set in .env.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    token_path = Path(settings.schwab_token_path)
    if not token_path.exists():
        print(
            f"No token file at {token_path} -- run "
            "`uv run scripts/bootstrap_schwab_oauth.py` first (one-time browser consent).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from schwab.auth import client_from_token_file

    client = client_from_token_file(
        token_path=str(token_path),
        api_key=settings.schwab_client_id,
        app_secret=settings.schwab_client_secret,
    )

    try:
        token = client.session.refresh_token()
    except Exception as exc:
        print(
            f"Refresh failed ({exc}) -- the refresh token itself has likely expired "
            "(~7 days of disuse). Re-run `uv run scripts/bootstrap_schwab_oauth.py` "
            "to get a new one via the interactive browser flow.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    expires_at = token.get("expires_at")
    expires_str = (
        datetime.fromtimestamp(expires_at, tz=UTC).isoformat() if expires_at else "unknown"
    )
    print(f"Token refreshed and saved to {token_path}. New access token expires: {expires_str}")


if __name__ == "__main__":
    main()
