"""Central runtime configuration, loaded from environment / .env.

Every external dependency the app has an opinion about (LLM backend, DB, broker
mode) is a config value here rather than hardcoded, so swapping providers later
is a settings change, not a code change.
"""

from __future__ import annotations

import enum
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class TradingMode(enum.StrEnum):
    DRY_RUN = "DRY_RUN"
    # Phase 1: real market data + real LLM decisions, zero order interaction of any
    # kind -- not even a simulated fill. BUY signals are alerted, not acted on.
    # Needs no MCP/OAuth setup at all, since the broker's write path is never
    # touched. Promote to LIVE (Phase 2) once you trust what it's been reporting.
    OBSERVE = "OBSERVE"
    LIVE = "LIVE"


class LlmProvider(enum.StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLAUDE = "claude"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Trading mode -------------------------------------------------
    mode: TradingMode = TradingMode.DRY_RUN

    # --- Database (Supabase-hosted Postgres by default; any Postgres works) ---
    # Plain SQLAlchemy + asyncpg against the connection string, no supabase-py SDK,
    # so pointing this at a different Postgres host later is a one-line change.
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_trading",
        description="Async SQLAlchemy connection string (Supabase Postgres by default).",
    )

    # --- LLM decision engine -------------------------------------------
    # Kept swappable: llm_provider selects the LLMClient implementation (see llm/base.py),
    # llm_model is provider-specific. Default is local Ollama running gemma.
    llm_provider: LlmProvider = LlmProvider.OLLAMA
    llm_model: str = "gemma4:e4b"
    # Same Ollama API shape locally and on Ollama Cloud -- switch between them purely
    # by settings: local default is a bare localhost daemon with no key; for Ollama
    # Cloud, point ollama_host at https://ollama.com and set ollama_api_key (sent as
    # a Bearer token). ollama_api_key being set is what triggers the auth header --
    # there's no separate "mode" flag to keep in sync with the host.
    ollama_host: str = "http://localhost:11434"
    ollama_api_key: str | None = None
    llm_request_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # 0.0 for deterministic, repeatable trade decisions -- this is a trading agent, not a
    # creative one, so we don't want sampling variance changing BUY/HOLD calls run to run.
    llm_temperature: float = 0.0

    # --- Robinhood market data (robin_stocks) ---------------------------
    robinhood_username: str | None = None
    robinhood_password: str | None = None
    robinhood_token_path: str = ".secrets/robinhood_token.pickle"

    # --- Robinhood Trading MCP (account state + order execution) --------
    mcp_server_url: str = "https://agent.robinhood.com/mcp/trading"
    mcp_token_store_path: str = ".secrets/mcp_token.json"
    # Where Robinhood redirects the browser after the user grants consent. Whichever
    # of scripts/bootstrap_mcp_oauth.py's own local server (default, unchanged) or
    # the app's GET /oauth/robinhood/{authorize,callback} endpoints (see api/) you
    # use, this must point at wherever that listener actually is -- the two are
    # alternatives, not both-at-once. In production this needs to be a publicly
    # reachable URL for the API-endpoint path (e.g.
    # https://your-host/oauth/robinhood/callback), since Robinhood's redirect goes to
    # this address directly, not through an internal network.
    mcp_oauth_redirect_uri: str = "http://localhost:8765/callback"
    # The Robinhood "Agentic" account's account_number (NOT your main brokerage
    # account) -- confirmed 2026-08 that get_equity_positions/review_equity_order/
    # place_equity_order/cancel_equity_order all require this explicitly; the MCP
    # deliberately never auto-selects an account. Find it by calling the MCP's
    # get_accounts tool once (e.g. via `uv run scripts/bootstrap_mcp_oauth.py`, or
    # any MCP client) and taking the entry with agentic_allowed=true. Required for
    # every McpBrokerClient call -- unset is fine in MODE=DRY_RUN, which never
    # constructs one.
    robinhood_agentic_account_number: str | None = None

    # --- Strategy / watchlist -------------------------------------------
    # NoDecode opts this field out of pydantic-settings' default behavior of
    # JSON-decoding list-typed env vars (which would require e.g.
    # WATCHLIST=["AAPL","TSLA"]) -- .env.example documents the friendlier plain
    # WATCHLIST=AAPL,TSLA,NVDA, parsed by the validator below instead.
    watchlist: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "TSLA", "NVDA"]
    )
    confidence_threshold: float = 0.7
    # Broad-market benchmark the LLM is shown alongside each ticker's own bucket
    # history, so a ticker-specific breakout can be weighed against whether the
    # whole market is trending with it or against it (see llm/prompt.py's
    # market_context). Empty string disables it -- the poll loop then skips the
    # extra fetch and the LLM sees no market_context section at all.
    market_benchmark_ticker: str = "SPY"
    # RSI lookback period (bars), Wilder-smoothed -- spec section 6 says "RSI-14 or
    # RSI-9"; 14 is the conventional default. Computed intraday from 5-min closes,
    # so this many bars (70 minutes at RSI-14) must accumulate before RSI reads
    # non-null (see bucket_builder.compute_rsi).
    rsi_period: int = 14

    @field_validator("watchlist", mode="before")
    @classmethod
    def _parse_comma_separated_watchlist(cls, value: object) -> object:
        if isinstance(value, str):
            return [ticker.strip().upper() for ticker in value.split(",") if ticker.strip()]
        return value

    # --- Guardrails (spec section 4) -------------------------------------
    max_open_positions_per_ticker: int = 1
    daily_trade_cap_per_ticker: int = 3
    max_capital_per_trade_usd: float = 500.0
    max_daily_drawdown_usd: float = 1000.0
    order_timeout_minutes: int = 15

    # --- Schedule (America/New_York) --------------------------------------
    timezone: str = "America/New_York"
    market_open_time: str = "09:30"
    evaluation_window_end_time: str = "11:30"
    eod_liquidation_time: str = "15:45"
    poll_interval_minutes: int = 5

    # --- Alerts -------------------------------------------------------------
    webhook_url: str | None = None

    # --- Stateful decision engine (Phase 3) ----------------------------------
    # Redis holds per-ticker *working* evaluation state (status, active thesis,
    # recent decision log) across poll cycles -- Postgres remains the durable
    # audit trail (buckets/llm_decisions/orders/trades); Redis is deliberately
    # ephemeral, keyed by (ticker, trade_date) and TTL'd so it can never leak
    # across trading sessions even if explicit clearing is missed. See
    # state/ticker_state_store.py.
    redis_url: str = "redis://localhost:6379/0"
    # How many past decisions (this ticker, today) are replayed into the prompt
    # alongside the active thesis -- spec section 8 says 3-5.
    decision_history_length: int = 5
    ticker_state_ttl_hours: int = 24
    # Once a position is open, a favorable trailing stop/target ratchet can be
    # applied by cancelling and replacing the resting sell order at the new,
    # more favorable price (never a less favorable one -- see
    # execution/invalidation.py's compute_trailing_stop). The stop-loss/
    # momentum-break forced-exit check itself is NOT gated by this flag; only
    # the one-way ratchet's order replacement is.
    trailing_stop_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
