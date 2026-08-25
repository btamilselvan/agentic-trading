# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An autonomous intraday momentum trading service for Robinhood (FastAPI + APScheduler): it polls market
microstructure data in 5-minute buckets during market hours, sends the accumulated time series to a
local LLM for BUY/HOLD/SELL pattern decisions, and executes paired buy→sell limit orders within
guardrails — entering and exiting positions within the same trading session only. Once a position is
open, it's actively re-evaluated every cycle rather than left to passively wait on its resting sell order
(Phase 3: code-enforced stop-loss/momentum invalidation, LLM-judged thesis continuity, one-way trailing
stops — see `execution/invalidation.py` and `state/ticker_state_store.py`). `requirements.md` is the
authoritative product spec; read it before making behavioral changes.

## Commands

This is a `uv` project (`pyproject.toml` + `uv.lock`) — use `uv run ...` for everything below rather than
activating a venv by hand; `uv` provisions Python 3.14+ and `.venv` automatically. See README.md for the
full command reference across local/Docker/production modes; the essentials:

```bash
# Setup
uv sync --extra dev
cp .env.example .env   # fill in DATABASE_URL, REDIS_URL, ROBINHOOD_*, WEBHOOK_URL, etc.

# Lint
uv run ruff check src/ tests/ scripts/
uv run ruff check --fix ...     # autofix

# Unit tests (no external services needed)
uv run pytest tests/unit -q
uv run pytest tests/unit/test_guardrails.py -q          # single file
uv run pytest tests/unit/test_guardrails.py::test_position_cap_blocks_when_at_or_above_cap -q  # single test

# Integration tests (need a real Postgres -- these exercise the DB, and one test suite
# also spins up real local HTTP servers standing in for Ollama and a webhook receiver, plus
# talks to a real Redis at REDIS_URL for Phase 3's ticker-state store; everywhere else in
# tests/integration uses an in-memory fake instead, so Redis is only required for that file)
docker run -d --name agentic-trading-test-pg -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=agentic_trading -p 55432:5432 postgres:16-alpine
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/agentic_trading
uv run alembic upgrade head
docker run -d --name agentic-trading-test-redis -p 6379:6379 redis:7-alpine
uv run pytest tests/integration -q

# Run everything
uv run pytest tests/ -q

# Run the app locally (MODE=DRY_RUN by default -- see .env.example)
uv run uvicorn agentic_trading.main:app --reload

# Docker (app + local Ollama + local Redis; point DATABASE_URL in .env at a real Postgres, e.g.
# Supabase -- this path uses plain pip inside the container, not uv)
docker compose up --build

# One-time interactive step before MODE=LIVE ever talks to the real Robinhood MCP --
# alternative: GET /oauth/robinhood/authorize on the running app (api/robinhood_oauth.py)
uv run scripts/bootstrap_mcp_oauth.py
```

Migrations live in `alembic/versions/`; `alembic/env.py` reads `DATABASE_URL` from `config.Settings`, not
from `alembic.ini` — the `sqlalchemy.url` in `alembic.ini` is unused.

## Architecture

Settled design decisions (see `requirements.md` §6 for the question this resolves): brokerage access is
**split across two integrations**, and both the LLM backend and the database are wrapped behind small
interfaces specifically so they're swappable later without touching callers.

- **`execution/broker_mcp_client.py`** — the official **Robinhood Trading MCP**
  (`agent.robinhood.com/mcp/trading`) handles account state and order placement. It isolates real trading
  capital to a separate, self-funded "Agentic" account and is reached via an OAuth-authenticated MCP
  client session (the `mcp` + `httpx2` packages — note `httpx2` is a distinct package from `httpx`, only
  used here because `OAuthClientProvider` subclasses `httpx2.Auth`). Confirmed against the live server on
  2026-08-17 (see module docstring for the full list): `_TOOL_NAMES` are all correct;
  `get_equity_positions`/`review_equity_order`/`place_equity_order`/`cancel_equity_order` all require an
  explicit `account_number` (`config.robinhood_agentic_account_number` — never auto-selected, per the
  MCP's own tool descriptions); order tools want `type` + string-encoded `quantity`/`limit_price`, not
  JSON numbers; responses are wrapped as `{"data": ..., "guide": "..."}` (see `unwrap_tool_result`).
  **Still unconfirmed:** `place_equity_order`'s response shape — placing a real order needs real money,
  which wasn't done without explicit authorization; `place_order` fails loudly with the actual keys if
  its field-name guesses are wrong rather than silently mis-parsing.

  **Safety-critical fix, applies everywhere in this module:** MCP session/transport teardown against the
  live server can itself raise (its DELETE-based session termination has been observed to return 400)
  *even when the actual tool call already succeeded* — left unhandled, that turns every successful call
  into an apparent failure (in `MODE=LIVE`, a real order could get placed but recorded as failed, leaving
  an untracked position). `open_mcp_session()` + `_safe_aexit()` log and swallow cleanup-phase errors
  while still propagating errors from setup or the actual call. `open_mcp_session()` is the **one, shared**
  implementation of "open a session, clean it up safely" — used by `McpBrokerClient`'s methods
  (headless handlers) and by both ways of completing the one-time OAuth authorization (interactive
  handlers): `scripts/bootstrap_mcp_oauth.py` (standalone script, its own local callback server) and
  `api/robinhood_oauth.py` (`GET /oauth/robinhood/{authorize,callback,status}`, for completing auth
  against an already-deployed instance without a local script). Pick one, not both, against the same
  `config.mcp_oauth_redirect_uri` (see README's "Going live"). The API-endpoint path bridges OAuth's
  two-request redirect/callback dance across separate HTTP requests via a single in-process `_pending`
  object — deliberately not built for concurrent flows, since linking a Robinhood account is a rare,
  one-at-a-time, operator-driven action. Both paths print/return the live tool list AND your accounts
  (via `get_accounts`) at the end, for `_TOOL_NAMES` verification and `ROBINHOOD_AGENTIC_ACCOUNT_NUMBER`
  discovery respectively.
- **`api/routes.py`** also has two read-only connectivity checks with no order-placement involved:
  `GET /market-data/{ticker}` (robin_stocks) and `GET /broker/positions/{ticker}` (always the real MCP
  via a fresh `McpBrokerClient`, regardless of `MODE` — the point is to test that connection
  independent of the trading pipeline's own DRY_RUN/LIVE broker selection).
- **`market_data/robinhood_client.py`** — `robin_stocks` (unofficial) handles market data polling
  (5-minute historicals, quotes). This is the only module that imports `robin_stocks`.
- **`market_data/bucket_builder.py`** — turns raw bars/quotes into a `MetricBucket`. Neither data source
  exposes a trade-by-trade tape, so "buy volume vs. sell volume" is an *estimate* (a Chaikin-style
  money-flow-multiplier proxy from where the bar closed in its range), and "order book depth imbalance"
  is approximated from top-of-book bid/ask size only — both are called out in that module's docstring,
  not silently presented as exact reads.
- **`llm/`** — `base.py` defines the `LLMClient` protocol and `get_llm_client()` factory switched on
  `config.llm_provider`; `ollama_client.py` is the only implementation today (Ollama running `gemma`,
  per `LLM_MODEL`; `LLM_TEMPERATURE` defaults to `0.0` for repeatable decisions). It talks to Ollama's
  `/api/chat` endpoint, which is identical whether `OLLAMA_HOST` points at a local daemon or Ollama
  Cloud (`https://ollama.com`) — switching between them is purely an `OLLAMA_HOST`/`OLLAMA_API_KEY`
  settings change (the key, if set, is sent as a Bearer token), not a code change. Add a new provider
  by implementing the protocol and adding one branch to `get_llm_client()` — nothing else should need
  to change. `schema.py` holds the structured `TradeDecision` contract (spec §3.2, extended by §8):
  `decision` is `BUY`/`HOLD`/`SELL` (SELL is only meaningful for an already-open position); BUY requires
  `buy_limit_price`/`target_sell_price`/`stop_loss_price`/`max_holding_time_minutes` (target above entry,
  stop below entry); `thesis_continuity_flag` is required on every response, not just BUY. `prompt.py`
  builds the prompt from the *complete* bucket history for the day plus `position_context` — the
  continuity state (`status`/`active_thesis`/stop/target/`recent_decisions`) `scheduler.py` loaded from
  Redis before this cycle — and instructs the model on hysteresis (don't flip on single-bar noise) and
  the one-way trailing-stop rule.
- **`execution/guardrails.py`** — every safety control from spec §4 (position cap, daily trade cap,
  capital limits, circuit breaker, order timeout, EOD cutoff) as pure, dependency-free functions.
  `execution/order_manager.py` calls `evaluate_buy_guardrails` independently before every order
  submission — an LLM BUY decision is necessary but never sufficient to place an order. Treat these as
  invariants to re-check at the point of action, not something to trust from upstream state.
- **`execution/invalidation.py`** (Phase 3, spec §8) — the inverse of `guardrails.py`: instead of
  blocking a would-be BUY, `evaluate_exit_guardrails` can *force* an exit on an already-open position,
  even against an LLM HOLD. Checks the two invalidation criteria that are cheap/deterministic without an
  LLM call — price crossing `stop_loss` (`is_stop_loss_breached`) and a momentum break
  (`is_momentum_broken`: RSI centerline or VWAP crossing "down", both already computed per bucket by
  `market_data/bucket_builder.py`). The third criterion (a negative catalyst headline) has no code-side
  check — there's no sentiment classifier here — so it's left entirely to the LLM's own
  `thesis_continuity_flag`/`SELL` response. `compute_trailing_stop` enforces the one-way ratchet (stop/
  target can only move in the position's favor, never back down) that `execution/order_manager.py`'s
  `apply_trailing_stop` then applies.
- **`state/`** — SQLAlchemy async models + a `repository.py` that's the only thing touching ORM
  sessions/queries directly (everything else goes through it). `db.py`'s engine/session factory is
  `lru_cache`d — this assumes one long-lived event loop (true for the running app); tests instead pin
  `asyncio_default_fixture_loop_scope`/`asyncio_default_test_loop_scope` to `"session"` in `pyproject.toml`
  so the cached engine isn't reused across a closed event loop. The DB is Supabase-hosted Postgres by
  default, reached via plain SQLAlchemy+asyncpg (not the `supabase-py` SDK) so pointing `DATABASE_URL` at
  a different Postgres host is a one-line change. This remains the durable audit trail; `Trade` gained
  `stop_loss_price`/`exit_reason` and `LlmDecision` gained `stop_loss_price`/`thesis_continuity_flag` in
  Phase 3 (migration `0005_add_stop_loss_exit`).
- **`state/ticker_state_store.py`** (Phase 3, spec §8) — per-ticker *working* evaluation state
  (`status`/`active_thesis`/stop/target/`decision_history`, the last 3–5 decisions) that persists across
  poll cycles, backed by a real Redis (`REDIS_URL`) via `RedisTickerStateStore`, behind a
  `TickerStateStore` protocol (same swappable-interface pattern as `LLMClient`/`Notifier`) so it's
  fakeable (`InMemoryTickerStateStore`, used by every integration test except
  `test_end_to_end_dry_run.py`, which exercises the real client). Deliberately ephemeral and separate
  from Postgres: keyed by `(ticker, trade_date)` and TTL'd (`ticker_state_ttl_hours`) so it can never leak
  into a later trading session even if explicit clearing (on trade close / EOD) is missed. Postgres stays
  the source of truth for `stop_loss_price` — `scheduler.py` rebuilds Redis state from the `Trade` row if
  it ever finds them disagreeing (Redis lost/expired mid-position).
- **`scheduler.py`** — wires three APScheduler jobs (poll cycle, order-management sweep, EOD
  liquidation) around `market_data` → `llm` → `execution` → `alerts`, matching spec §3.1/3.3. Job bodies
  take their dependencies as arguments rather than importing concrete implementations, so
  `run_poll_cycle`/`run_order_management_sweep`/`run_eod_liquidation` are unit-testable with fakes;
  `build_scheduler` is the only place real dependencies get wired in. Trading-day detection is a plain
  Mon–Fri check, not a real market-holiday calendar. `_poll_ticker` branches on
  `config.TradingMode.OBSERVE` (Phase 1: real data + real LLM, zero order interaction of any kind) to
  report a BUY signal via the notifier and skip `order_manager.try_enter_position` entirely rather than
  calling it with a "don't actually trade" flag -- the broker is never touched in that mode, so it needs
  no MCP/OAuth setup at all. `DRY_RUN` → `OBSERVE` → `LIVE` is the intended promotion path (see README's
  "Going live"). Phase 3: `_poll_ticker` also branches on whether the ticker already has an OPEN `Trade`
  (checked via `repo.get_open_trade_for_ticker`, *before* the entry-eligibility guardrails, since an open
  position is exactly what makes `check_position_cap` block entry) — if so, `_manage_open_position` takes
  over: `execution.invalidation.evaluate_exit_guardrails` first (no LLM call on a forced exit), then, if
  clear, an LLM call for continuity/trailing-target, then `execution.order_manager.try_exit_position_early`
  or `.apply_trailing_stop` (the latter gated by `settings.trailing_stop_enabled`) as appropriate.
- **`alerts/`** — `Notifier` protocol + a generic `WebhookNotifier` (one JSON shape works for Slack/
  Discord/Telegram incoming webhooks) with a `NullNotifier` fallback when `WEBHOOK_URL` is unset.
  `order_manager.py` fires alerts on fills/trade-close; `scheduler.py` fires one on each BUY signal that
  clears the confidence threshold (not on every HOLD, to avoid spam from a 5-minute poll loop).
- **`main.py`** — `MODE=LIVE` picks `McpBrokerClient`; anything else (including the default
  `MODE=DRY_RUN`) picks `execution.order_manager.DryRunBrokerClient`, an in-memory broker simulator that
  exercises the exact same state machine and DB writes with zero network calls and zero real money.

## Testing conventions

- `tests/unit/` — pure logic (guardrails, bucket math, LLM schema/prompt, alerts) plus anything else that
  needs neither a real DB nor real network, including `test_robinhood_oauth.py` (the OAuth redirect/
  callback bridging logic in `api/robinhood_oauth.py`, tested with `_run_authorization_flow` faked out --
  the real OAuth/MCP wire protocol is only exercised by actually running the flow, see above).
- `tests/integration/` — needs `DATABASE_URL` pointed at a real Postgres (see Commands above);
  `tests/integration/conftest.py`'s `db_session` fixture truncates the relevant tables before/after each
  test rather than requiring a fresh database per test. `test_scheduler.py`'s `_fake_ticker_state_store`
  autouse fixture patches `scheduler.get_ticker_state_store` to an `InMemoryTickerStateStore` for every
  test in that file, so none of them need (or pollute) a real Redis.
- `tests/integration/test_end_to_end_dry_run.py` runs the real `OllamaClient` and `WebhookNotifier` HTTP
  code paths against real local FastAPI/uvicorn servers standing in for Ollama and a webhook receiver, and
  the real `RedisTickerStateStore` against `REDIS_URL` (its `_ticker_state_cleanup` autouse fixture clears
  that key before/after) — only market data is stubbed at the function level (it needs real Robinhood
  credentials this environment doesn't have). Treat this as the standing DRY_RUN validation rather than a
  manual step.
