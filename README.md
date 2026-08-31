# Agentic Trading

An autonomous intraday momentum trading service for Robinhood. It polls market microstructure data in
5-minute buckets during market hours, sends the accumulated time series to a local LLM for BUY/HOLD
pattern decisions, and executes paired buy→sell limit orders within a set of hard safety guardrails —
entering and exiting positions within the same trading session only.

See [`requirements.md`](requirements.md) for the full product spec, [`CLAUDE.md`](CLAUDE.md) for an
architecture tour of the codebase, and [`docs/architecture.html`](docs/architecture.html) for system,
component, logical, and deployment architecture diagrams.

> ⚠️ **This system places real orders with real money when `MODE=LIVE`.** Always validate a change in
> `MODE=DRY_RUN` first (the default). See [Going live](#going-live) before ever flipping the switch.

## Contents

- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [LLM backend: local vs. Ollama Cloud](#llm-backend-local-vs-ollama-cloud)
- [Schwab market data (primary source)](#schwab-market-data-primary-source)
- [Run locally (no Docker)](#run-locally-no-docker)
- [Run with Docker Compose](#run-with-docker-compose)
- [Run in production](#run-in-production)
- [Going live](#going-live)
- [Running tests](#running-tests)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Prerequisites

| Mode | You need |
|---|---|
| Local | [uv](https://docs.astral.sh/uv/) (manages the Python 3.14+ install and venv for you), a Postgres database (e.g. [Supabase](https://supabase.com)), a Redis instance (e.g. `docker run -d -p 6379:6379 redis:7-alpine`), [Ollama](https://ollama.com) running locally **or** an [Ollama Cloud](https://ollama.com) API key — see [LLM backend](#llm-backend-local-vs-ollama-cloud) |
| Docker Compose | Docker + Docker Compose, a Postgres database (remote/Supabase — not containerized here; Redis *is* containerized, see below) |
| Production | A container host/orchestrator, a production Postgres, a production Redis, a Robinhood account, a funded Robinhood **Agentic** account for `MODE=LIVE` |

This repo is set up as a `uv` project (`pyproject.toml` + `uv.lock`) — `uv` resolves and pins exact
dependency versions in `uv.lock` and provisions an isolated `.venv` automatically; you don't create or
activate a venv by hand. All local commands below are `uv run ...`, which transparently uses that venv.
`uv` isn't required for Docker/production, since the image is built with plain `pip` inside the
container — see [Run in production](#run-in-production).

All modes need a Robinhood account for market data (`robin_stocks`, unofficial — used read-only) and,
for `MODE=LIVE` only, the official Robinhood Trading MCP linked via a one-time OAuth step (see
[Going live](#going-live)). A Schwab developer app is optional but recommended in every mode: it's the
primary quote/5-min-historicals source (Phase 4), with Robinhood as the automatic fallback — see
[Schwab market data](#schwab-market-data-primary-source).

## Configuration

Every mode reads its config from `.env` (via [`src/agentic_trading/config.py`](src/agentic_trading/config.py)):

```bash
cp .env.example .env
```

Then fill in at least:

- `DATABASE_URL` — an async Postgres connection string (`postgresql+asyncpg://...`). Defaults to a local
  Postgres; point it at your Supabase project's connection string (Session Pooler recommended) or any
  other Postgres-compatible host.
- `REDIS_URL` — a Redis connection string. Defaults to a local Redis (`redis://localhost:6379/0`); point
  it at any Redis-compatible host (Upstash, Redis Cloud, ElastiCache, ...) the same way `DATABASE_URL`
  points at any Postgres-compatible host. Holds Phase 3's per-ticker continuity state (active thesis,
  decision history, stop/target levels) across poll cycles — ephemeral working memory, not the audit
  trail (that's still Postgres); see `state/ticker_state_store.py`.
- `ROBINHOOD_USERNAME` / `ROBINHOOD_PASSWORD` — for market-data polling only. On first run, `robin_stocks`
  prompts for an MFA code on stdin unless a cached session already exists at `ROBINHOOD_TOKEN_PATH`.
- `WATCHLIST` — comma-separated tickers to trade.
- `MARKET_BENCHMARK_TICKER` — broad-market ETF (default `SPY`) shown to the LLM alongside each ticker's
  own bucket history, so it can weigh a ticker-specific setup against the day's overall market trend.
  Set to empty to disable.
- `RSI_PERIOD` — Wilder-smoothed RSI lookback period in bars (default `14`), computed intraday from
  5-min closes. Reads `null` until this many bars have accumulated.
- `WEBHOOK_URL` — optional; a Slack/Discord/Telegram-compatible incoming webhook URL for trade alerts.
  Leave blank to disable alerting.
- `ROBINHOOD_AGENTIC_ACCOUNT_NUMBER` — required for any real MCP call (`MODE=LIVE`, or the
  `/broker/positions/{ticker}` connectivity check): every position/order tool requires this explicitly,
  it's never auto-selected. `scripts/bootstrap_mcp_oauth.py` prints your accounts (with
  `agentic_allowed`) at the end of the OAuth flow — use the one with `agentic_allowed=true`, not your
  main brokerage account. Unset is fine in `MODE=DRY_RUN`.
- `SCHWAB_CLIENT_ID` / `SCHWAB_CLIENT_SECRET` — optional; the primary market-data source when set (falls
  back to Robinhood automatically otherwise, or on any Schwab failure). See
  [Schwab market data](#schwab-market-data-primary-source).

Everything else (guardrail thresholds, schedule, LLM provider/model) has a sensible default — see
`.env.example` for the full list and `config.py` for descriptions.

`MODE` defaults to `DRY_RUN`: the app runs its full pipeline (polling, LLM decisions, guardrails, paired
buy/sell "fills") against an in-memory simulated broker — no network calls to Robinhood for orders, no
real money at risk. Leave it there until you've validated the strategy.

There are three modes, meant to be moved through in order:

1. **`DRY_RUN`** (default) — fully simulated, as above. Good for validating the pipeline itself works
   (buckets get built, the LLM gets called, orders/fills/guardrails behave) without needing real
   Robinhood credentials for anything beyond market data. Meant for local/dev testing at any time.
2. **`PAPER_TRADING`** — the exact same simulated in-memory broker and order lifecycle as `DRY_RUN`
   (real market data, real LLM decisions, simulated buy/sell fills, trailing stops, PnL — all written to
   the DB same as a real trade), meant to be run during real market hours as the final, capital-free
   rehearsal of what `LIVE` would actually do. Trades are tagged `PAPER_TRADING` (not `DRY_RUN`) so this
   run's performance can be reviewed apart from ad hoc dev testing. This needs no MCP/OAuth setup at all,
   since the broker's write path never leaves the in-memory simulator.
3. **`LIVE`** — real orders through the Robinhood MCP, real money, in the isolated Agentic account. See
   [Going live](#going-live).

## LLM backend: local vs. Ollama Cloud

By default the app talks to a local Ollama daemon (`ollama serve`) — free, fully offline, no data leaves
your machine. You can instead point it at [Ollama Cloud](https://ollama.com) if you want a larger model
than your hardware can run, or don't want to keep a local Ollama process alive. Both use the exact same
`OllamaClient` code path (`/api/chat`) — switching between them is a `.env` change, not a code change.

| | Local (default) | Ollama Cloud |
|---|---|---|
| `.env` | `OLLAMA_HOST=http://localhost:11434`, `OLLAMA_API_KEY` unset | `OLLAMA_HOST=https://ollama.com`, `OLLAMA_API_KEY=<your key>` |
| `LLM_MODEL` | whatever you've `ollama pull`ed (default `gemma4:e4b`) | whatever's available on Ollama Cloud (e.g. `gemma4:31b`) |
| Needs | `ollama serve` running | an [Ollama account](https://ollama.com) with an API key generated |
| Cost | free, offline | usage-based — check current Ollama Cloud pricing |

To switch to Ollama Cloud:

1. Sign in at [ollama.com](https://ollama.com) and generate an API key (Settings → API keys).
2. Set in `.env`:

   ```bash
   OLLAMA_HOST=https://ollama.com
   OLLAMA_API_KEY=<your key>
   LLM_MODEL=gemma4:31b   # or whichever cloud model you want
   ```

3. Skip step 2 of [Run locally](#run-locally-no-docker) (`ollama serve` / `ollama pull`) entirely — no
   local Ollama process is needed for LLM calls.

`OLLAMA_API_KEY` being set is what triggers the `Authorization: Bearer` header on every request — there's
no separate mode flag to keep in sync with the host. Leave it unset (and `OLLAMA_HOST` at its local
default) to use a local daemon instead.

Some cloud models (observed with `gemma4:31b`) wrap their JSON response in a Markdown code fence even
under the structured-output `format` constraint, where a local daemon returns bare JSON for the same
request; `OllamaClient` strips this automatically before parsing, so no extra config is needed for that.

Docker Compose forces `OLLAMA_HOST` to its own local `ollama` container regardless of `.env` (see
[Run with Docker Compose](#run-with-docker-compose)) — to use Ollama Cloud under Compose, remove that
override from `docker-compose.yml` and keep `OLLAMA_API_KEY` set in `.env`.

### Using Gemini instead of Ollama

`LLM_PROVIDER=gemini` (`llm/gemini_client.py`) talks to Google's Gemini API — including the free tier of
[Google AI Studio](https://aistudio.google.com/apikey) — over plain REST, no extra SDK. Set:

```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.6-flash   # or whichever exact model id your API key has access to
GEMINI_API_KEY=<your Google AI Studio API key>
```

No Ollama process needed in this mode — skip step 2 of [Run locally](#run-locally-no-docker) entirely.
Like Ollama, this uses the model's native structured-output support to get back a schema-constrained
`TradeDecision`, so no prompt changes are needed; `_to_gemini_schema` in `gemini_client.py` translates
`TradeDecision`'s pydantic schema into the JSON Schema subset Gemini's `responseSchema` accepts. See
[Switching LLM providers](#switching-llm-providers) for the general pattern this follows.

## Schwab market data (primary source)

Phase 4: [Schwab's Market Data Production API](https://developer.schwab.com) (via `schwab-py`) is the
primary source for quotes and 5-minute historicals — `robin_stocks` becomes the automatic fallback
(`market_data/market_data_client.py`), used whenever Schwab is unconfigured, unauthorized, or a call
fails. News and float shares stay Robinhood-only either way; Schwab has no equivalent feed for those.

1. Register an app at [developer.schwab.com](https://developer.schwab.com) and request Market Data
   Production access. Set a callback URL — Schwab requires HTTPS, e.g. `https://127.0.0.1:8182`.
2. Set `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET`, and `SCHWAB_CALLBACK_URL` (matching the app's
   configured callback URL exactly) in `.env`.
3. Complete the one-time browser consent:

   ```bash
   uv run scripts/bootstrap_schwab_oauth.py
   ```

   This uses `schwab-py`'s `easy_client`, which opens a browser, catches its own callback, and writes the
   token to `SCHWAB_TOKEN_PATH` (default `.secrets/schwab_token.json`). The running app never performs
   this interactive step itself — it only reads the cached token and lets `schwab-py` refresh it silently.

The access token underneath is short-lived, but `schwab-py`'s session auto-refreshes it (via the refresh
token) on every real API call and rewrites `SCHWAB_TOKEN_PATH` in place — no extra steps needed in normal
operation. To force that refresh explicitly (e.g. a daily cron, so the token never goes long-idle over a
quiet weekend) without waiting for a live market-data call:

```bash
uv run scripts/refresh_schwab_token.py
```

This doesn't need a browser — it just exercises the refresh token already on disk. It does **not**
extend the refresh token's own ~7-day absolute lifetime, though: once that's gone (roughly a week of
disuse), only step 3 above (the full interactive browser flow) gets a new one — re-run
`bootstrap_schwab_oauth.py` periodically for that. Either way, a stale/expired token just means
`market_data_client.py` falls back to Robinhood until it's refreshed, not an outage. Check
`GET /market-data/schwab/{ticker}` at any time to verify Schwab connectivity directly, independent of
which provider the trading pipeline actually used on its last poll.

If `SCHWAB_CLIENT_ID`/`SCHWAB_CLIENT_SECRET` are left unset, the app runs exactly as it did before Phase
4 — Robinhood market data only, no Schwab calls attempted.

## Run locally (no Docker)

1. **Install `uv`** (if you don't have it) and sync the project's dependencies, including dev tools:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh   # or: brew install uv
   uv sync --extra dev
   ```

   This downloads Python 3.14 if you don't already have it, creates `.venv`, and installs everything
   pinned in `uv.lock`. `cp .env.example .env` and fill it in first (see [Configuration](#configuration))
   if you haven't already.

2. **Start Ollama and pull the model** configured in `.env` as `LLM_MODEL` (default: `gemma4:e4b`):

   ```bash
   ollama serve &          # if not already running
   ollama pull gemma4:e4b  # match whatever LLM_MODEL is set to
   ```

   Using [Ollama Cloud](#llm-backend-local-vs-ollama-cloud) instead? Skip this step entirely — set
   `OLLAMA_HOST`/`OLLAMA_API_KEY` in `.env` and there's no local model to pull or serve.

3. **Point `DATABASE_URL` at a real Postgres** (a free Supabase project works well) and apply migrations:

   ```bash
   uv run alembic upgrade head
   ```

4. **Run the app:**

   ```bash
   uv run uvicorn agentic_trading.main:app --reload
   ```

   This starts the FastAPI server on `http://localhost:8000` and, via its lifespan, the APScheduler jobs
   (poll cycle, order-management sweep, EOD liquidation — see `scheduler.py`). In `MODE=DRY_RUN` this is
   fully safe to leave running; it'll poll real market data (if the market's open) and log simulated
   decisions/trades to the database and configured webhook.

5. **Check it's alive:**

   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/status
   ```

Any other one-off command in the project (a script, a REPL, etc.) should be run the same way —
`uv run <command>` — rather than activating the venv yourself; it keeps `uv.lock` and `.venv` in sync
automatically on every invocation.

## Run with Docker Compose

Docker Compose runs the app, a local Ollama container, **and** a local Redis container together; Postgres
stays external (point `DATABASE_URL` at Supabase or another remote instance — it isn't containerized
here, since it's meant to be a durable store you don't want tied to `docker compose down`). Redis *is*
containerized by default since it's ephemeral working memory, not a durable store — see `REDIS_URL` above.

```bash
cp .env.example .env   # fill in DATABASE_URL, ROBINHOOD_*, WATCHLIST, etc.
docker compose up --build
```

Notes:

- The compose file overrides `OLLAMA_HOST` to `http://ollama:11434` and `REDIS_URL` to
  `redis://redis:6379/0` regardless of what's in `.env`, since inside the compose network `localhost`
  refers to the app container itself, not the `ollama`/`redis` services.
- `./.secrets` is bind-mounted into the container so the Robinhood session pickle and MCP OAuth token
  (see [Going live](#going-live)) persist across container restarts.
- First run needs the model pulled into the `ollama` container once:

  ```bash
  docker compose exec ollama ollama pull gemma4:e4b
  ```

- Apply migrations against your external Postgres before or after bringing the stack up — from the host
  with `uv` (`DATABASE_URL` exported, per the [local](#run-locally-no-docker) section):

  ```bash
  uv run alembic upgrade head
  ```

  or from inside the container (which uses plain `pip`, not `uv` — see
  [Run in production](#run-in-production)):

  ```bash
  docker compose exec app alembic upgrade head
  ```

- Tear down with `docker compose down`; add `-v` to also drop the `ollama_data`/`redis_data` volumes
  (re-pull the model next time; Redis's volume only matters for surviving a container restart mid-day,
  since it's ephemeral working memory anyway).

## Run in production

The `Dockerfile` builds a standalone image (no Ollama bundled) — use this for a real deployment behind
whatever orchestrator you use (a single VM with `docker run`, ECS, Cloud Run, Kubernetes, etc.):

```bash
docker build -t agentic-trading .
docker run -d \
  --name agentic-trading \
  --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/.secrets:/app/.secrets \
  agentic-trading
```

Production-specific points:

- **LLM backend:** point `OLLAMA_HOST` at a reachable Ollama instance (a separate managed/self-hosted
  server, or a sidecar container) — this image does not run one itself. Or use
  [Ollama Cloud](#llm-backend-local-vs-ollama-cloud) instead so there's no Ollama instance to run/manage
  at all — set `OLLAMA_HOST=https://ollama.com` and `OLLAMA_API_KEY`. For a non-Ollama provider entirely,
  see [Switching LLM providers](#switching-llm-providers).
- **Database:** use your real Postgres (Supabase or otherwise) via `DATABASE_URL`; run
  `alembic upgrade head` as a release step before starting new containers, not automatically on boot.
- **Redis:** use a real Redis (Upstash, Redis Cloud, ElastiCache, or self-hosted) via `REDIS_URL` — it's
  working memory, not the audit trail, so it doesn't need the same backup/durability rigor as Postgres,
  but it does need to actually be reachable (Phase 3's continuity/hysteresis logic reads/writes it every
  poll cycle).
- **Secrets:** `.secrets/` holds the Robinhood session pickle and the MCP OAuth token — treat it like any
  other credential store (mounted volume backed by your platform's secret storage, not baked into the
  image). Never commit it; it's already in `.gitignore`.
- **Process supervision:** run behind whatever restarts the container on crash — the scheduler jobs are
  in-process (APScheduler), so a crashed process means missed polls until it's restarted. There's no
  built-in leader election, so **run exactly one instance**; a second instance would double-poll and
  double-decide (guardrails still prevent double *orders* per ticker, since they check DB state, but
  running twice is wasted LLM calls and noise at best).
- **Monitoring:** `GET /status` reports mode, watchlist, and each scheduled job's next run time — poll it
  from your uptime checker. Configure `WEBHOOK_URL` so fills/closes/BUY signals reach you in real time
  without needing to watch logs.
- **Kill switch:** `POST /kill-switch` immediately pauses all scheduled jobs (no new polls, no new
  orders) without touching already-open positions or pending orders — use it if something looks wrong.
  `POST /resume` restarts them. This is a manual override on top of the automatic circuit breaker (new
  BUYs already self-block once the daily drawdown cap trips — see `execution/guardrails.py`).

## Going live

`MODE=LIVE` routes real orders through the official Robinhood Trading MCP into a separate, self-funded
**Agentic** account (this isolation is by design — your main brokerage account is never touched).
Before flipping `MODE=LIVE` in `.env`:

1. **Fund the Robinhood Agentic account** you intend to trade with, separately from your main account.

2. **Complete the one-time OAuth authorization**, via **one** of two equivalent paths (don't use both
   against the same `MCP_OAUTH_REDIRECT_URI` — see the note below):

   **Option A — standalone script.** Simplest for local use; needs a browser on the same machine, so
   don't run this on a headless server (copy the resulting `.secrets/mcp_token.json` to production
   afterward instead):

   ```bash
   uv run scripts/bootstrap_mcp_oauth.py
   ```

   **Option B — in-app endpoints.** Useful when there's no convenient way to run a local script against
   wherever the service is deployed (e.g. you'd rather hit a URL on your already-reachable production
   instance). With the app running, visit in a browser:

   ```
   GET /oauth/robinhood/authorize
   ```

   which redirects to Robinhood's consent screen and, once approved, redirects back to
   `GET /oauth/robinhood/callback` (handled automatically — that's just the redirect target). Check
   `GET /oauth/robinhood/status` at any point for progress/errors.

   Either way, `MCP_OAUTH_REDIRECT_URI` in `.env` must match wherever the callback is actually being
   caught: the script's own local server (default `http://localhost:8765/callback`, unchanged) for
   Option A, or this app's own address for Option B (e.g. `http://localhost:8000/oauth/robinhood/callback`
   locally, or your real public URL in production — Robinhood redirects the browser there directly, not
   through an internal network). If you switch from one option to the other after already completing a
   flow, delete `.secrets/mcp_token.json` first — the cached OAuth client registration is tied to
   whichever redirect URI it was created with.

   Both paths end by printing/showing the live MCP server's tool list. **Compare that output against
   `_TOOL_NAMES` in
   [`execution/broker_mcp_client.py`](src/agentic_trading/execution/broker_mcp_client.py)**: those names
   were written from public documentation of the MCP without a live server available to confirm them
   against, and a mismatch there is a one-line fix once you know the real names.

3. **Work through `DRY_RUN` → `PAPER_TRADING` → `LIVE` in order**, not straight to `LIVE`:
   - `MODE=DRY_RUN` first, to confirm the pipeline mechanics work (buckets, LLM calls, simulated
     fills/guardrails) without needing steps 1–2 above at all.
   - `MODE=PAPER_TRADING` next, run during real market hours — real market data and real LLM decisions,
     the same simulated buy/sell/trailing-stop lifecycle as `DRY_RUN`, but with no real broker call
     (still no MCP/OAuth setup needed). Run this for a real session or two and review the simulated
     trades/PnL it actually produced before trusting it with money.
   - Only then `MODE=LIVE`.

4. **Start small.** Lower `MAX_CAPITAL_PER_TRADE_USD` and `MAX_DAILY_DRAWDOWN_USD` in `.env` for your
   first live sessions, and watch the first few cycles live rather than walking away.

5. Set `MODE=LIVE` and restart the service.

### Switching LLM providers

The default is Ollama (`LLM_PROVIDER=ollama`), which itself supports both a local daemon and Ollama
Cloud — see [LLM backend: local vs. Ollama Cloud](#llm-backend-local-vs-ollama-cloud) if that's all you
need; it's a `.env` change, not a code change. `LLM_PROVIDER=gemini` is also implemented — see
[Using Gemini instead of Ollama](#using-gemini-instead-of-ollama) — likewise just a `.env` change. To add
an entirely new provider (OpenAI, Claude, etc.), implement the `LLMClient` protocol in `llm/base.py` (see
`llm/ollama_client.py` or `llm/gemini_client.py` for the shape) and add one branch to `get_llm_client()`
in the same file — no other module needs to change.

## Running tests

```bash
# Unit tests -- pure logic, no network or DB
uv run pytest tests/unit -q

# Integration tests -- need a real Postgres, and (for one test suite) a real Redis
docker run -d --name agentic-trading-test-pg -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=agentic_trading -p 55432:5432 postgres:16-alpine
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/agentic_trading
uv run alembic upgrade head
docker run -d --name agentic-trading-test-redis -p 6379:6379 redis:7-alpine
uv run pytest tests/integration -q

# Everything
uv run pytest tests/ -q

# Lint
uv run ruff check src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/   # autofix
```

`tests/integration/test_end_to_end_dry_run.py` is the closest thing to an automated version of the
"run a real `MODE=DRY_RUN` session" check: it exercises the real Ollama client, real webhook notifier, and
real Redis-backed ticker-state store over real local HTTP servers / a real Redis at `REDIS_URL` (only
market data is stubbed, since that needs live Robinhood credentials) — it cleans up its own Redis key
before and after. Everywhere else, integration tests use `InMemoryTickerStateStore` (an in-memory fake,
same role as `DryRunBrokerClient`) instead of talking to Redis at all, so `tests/integration` as a whole
only *requires* Redis to be reachable for that one file; the rest run fine without it.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/status` | GET | Mode, watchlist, halted flag, next run time per scheduled job |
| `/decisions` | GET | Recent LLM decisions (`?limit=`) |
| `/trades` | GET | Recent trades (`?limit=`) |
| `/poll-cycle` | POST | Manually trigger one poll cycle across the watchlist, using the real broker/LLM client (`MODE=LIVE` can place real orders here) — `?force=true` bypasses the market-hours window; refuses to run while halted |
| `/orders/manual-entry` | POST | Debug/test hook: directly call `order_manager.try_enter_position` with a synthetic BUY decision (`ticker`, `buy_limit_price`, `target_sell_price`, `max_holding_time_minutes`), bypassing market data + the LLM entirely — uses the real broker (`MODE=LIVE` can place real orders here, requires `?confirm=true`); refuses to run while halted |
| `/kill-switch` | POST | Pause all scheduled jobs immediately |
| `/resume` | POST | Resume scheduled jobs after a kill-switch |
| `/oauth/robinhood/authorize` | GET | Start (or resume) Robinhood MCP authorization — visit in a browser; see [Going live](#going-live) Option B |
| `/oauth/robinhood/callback` | GET | Robinhood's OAuth redirect target — not visited directly |
| `/oauth/robinhood/status` | GET | Progress/result of the most recent authorization attempt |
| `/market-data/{ticker}` | GET | Read-only robin_stocks connectivity check — live quote + latest 5-min bar |
| `/market-data/schwab/{ticker}` | GET | Read-only Schwab connectivity check (Phase 4) — same shape as above, but always hits Schwab directly, bypassing the primary/fallback logic |
| `/broker/positions/{ticker}` | GET | Read-only Robinhood MCP connectivity check — always hits the real MCP regardless of `MODE` |

## Troubleshooting

- **`robin_stocks` login hangs / asks for MFA repeatedly** — delete `ROBINHOOD_TOKEN_PATH` and log in
  again interactively; the cached session may have expired or been invalidated.
- **`No valid MCP OAuth token on disk`** — complete (or redo) authorization via either
  `uv run scripts/bootstrap_mcp_oauth.py` or `GET /oauth/robinhood/authorize` (see
  [Going live](#going-live)); this only matters for `MODE=LIVE` / `/broker/positions`. It can also
  surface on a previously-working setup if the MCP's refresh-token rotation ends up in a state the
  cached token can't recover from on its own — re-running the OAuth flow always fixes it.
- **`invalid params: ... missing properties: ["account_number"]`** — set
  `ROBINHOOD_AGENTIC_ACCOUNT_NUMBER` (see [Configuration](#configuration)); every MCP position/order tool
  requires it explicitly.
- **`GET /oauth/robinhood/authorize` times out / 504s** — check `GET /oauth/robinhood/status` for the
  underlying error, and confirm `MCP_OAUTH_REDIRECT_URI` matches wherever you're actually completing the
  flow (see [Going live](#going-live) Option A vs. B) — a mismatch there means Robinhood's redirect never
  reaches the listener this app (or the script) is waiting on.
- **`400: No authorization flow is currently in progress` on `/oauth/robinhood/callback`** — you (or
  Robinhood) hit `/callback` without a prior `/authorize` call in this process, or the previous flow
  already finished/timed out. Start over from `GET /oauth/robinhood/authorize`.
- **Ollama connection refused** — confirm `ollama serve` is running and `OLLAMA_HOST` is reachable from
  wherever the app is running (in Docker Compose, that's `http://ollama:11434`, not `localhost`). If
  you're using [Ollama Cloud](#llm-backend-local-vs-ollama-cloud), this instead means `OLLAMA_HOST` is
  still set to a local address — check for a stale `http://localhost:11434` in `.env`.
- **Ollama Cloud returns 401/403** — `OLLAMA_API_KEY` is missing, expired, or wrong; regenerate one at
  [ollama.com](https://ollama.com) (Settings → API keys). A response that comes back but fails to parse
  as a `TradeDecision` is a different problem — check the logged raw response — not an auth issue.
- **`GET /market-data/schwab/{ticker}` returns null quote/bars** — this endpoint never raises (Schwab
  failures degrade to null fields, not a 502) — check the app logs for the underlying
  `market_data.schwab_client` warning. Usually either `SCHWAB_CLIENT_ID`/`SCHWAB_CLIENT_SECRET` are unset,
  or the token at `SCHWAB_TOKEN_PATH` is missing/stale — re-run
  `uv run scripts/bootstrap_schwab_oauth.py`. This is informational, not an outage: the trading pipeline
  itself (`market_data_client.py`) falls back to Robinhood automatically whenever Schwab is unavailable.
- **Migrations fail against Supabase** — use the Session Pooler connection string (port `5432`, or the
  pooled `6543` variant per Supabase's docs), not a direct connection that may not be reachable from
  wherever you're running `alembic`.
- **`pydantic_settings.exceptions.SettingsError` parsing a field from `.env`** — list-typed settings
  (currently just `WATCHLIST`) need either the plain comma-separated form documented in `.env.example`
  (`WATCHLIST=AAPL,TSLA,NVDA`) or a real JSON array; anything else fails fast at startup with this error.
  If you add a new list-typed setting in `config.py`, give it the same `NoDecode` + validator treatment as
  `watchlist`, or document that it must be JSON.
- **`uv sync` re-downloads a lot / seems to ignore `uv.lock`** — make sure you're not passing `--upgrade`;
  a plain `uv sync --extra dev` respects the committed lockfile and should be fast and reproducible.
- **`uv run` uses the wrong Python version** — `uv` provisions its own interpreter per
  `requires-python` in `pyproject.toml` (3.14+); it doesn't use `python3`/`pyenv`/etc. on your `PATH`
  unless you've pinned it otherwise with `uv python pin`.
