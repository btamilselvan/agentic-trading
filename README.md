# Agentic Trading

An autonomous intraday momentum trading service for Robinhood. It polls market microstructure data in
5-minute buckets during market hours, sends the accumulated time series to a local LLM for BUY/HOLD
pattern decisions, and executes paired buy→sell limit orders within a set of hard safety guardrails —
entering and exiting positions within the same trading session only.

See [`requirements.md`](requirements.md) for the full product spec and [`CLAUDE.md`](CLAUDE.md) for an
architecture tour of the codebase.

> ⚠️ **This system places real orders with real money when `MODE=LIVE`.** Always validate a change in
> `MODE=DRY_RUN` first (the default). See [Going live](#going-live) before ever flipping the switch.

## Contents

- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
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
| Local | [uv](https://docs.astral.sh/uv/) (manages the Python 3.14+ install and venv for you), a Postgres database (e.g. [Supabase](https://supabase.com)), [Ollama](https://ollama.com) running locally |
| Docker Compose | Docker + Docker Compose, a Postgres database (remote/Supabase — not containerized here) |
| Production | A container host/orchestrator, a production Postgres, a Robinhood account, a funded Robinhood **Agentic** account for `MODE=LIVE` |

This repo is set up as a `uv` project (`pyproject.toml` + `uv.lock`) — `uv` resolves and pins exact
dependency versions in `uv.lock` and provisions an isolated `.venv` automatically; you don't create or
activate a venv by hand. All local commands below are `uv run ...`, which transparently uses that venv.
`uv` isn't required for Docker/production, since the image is built with plain `pip` inside the
container — see [Run in production](#run-in-production).

All modes need a Robinhood account for market data (`robin_stocks`, unofficial — used read-only) and,
for `MODE=LIVE` only, the official Robinhood Trading MCP linked via a one-time OAuth step (see
[Going live](#going-live)).

## Configuration

Every mode reads its config from `.env` (via [`src/agentic_trading/config.py`](src/agentic_trading/config.py)):

```bash
cp .env.example .env
```

Then fill in at least:

- `DATABASE_URL` — an async Postgres connection string (`postgresql+asyncpg://...`). Defaults to a local
  Postgres; point it at your Supabase project's connection string (Session Pooler recommended) or any
  other Postgres-compatible host.
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

Everything else (guardrail thresholds, schedule, LLM provider/model) has a sensible default — see
`.env.example` for the full list and `config.py` for descriptions.

`MODE` defaults to `DRY_RUN`: the app runs its full pipeline (polling, LLM decisions, guardrails, paired
buy/sell "fills") against an in-memory simulated broker — no network calls to Robinhood for orders, no
real money at risk. Leave it there until you've validated the strategy.

There are three modes, meant to be moved through in order:

1. **`DRY_RUN`** (default) — fully simulated, as above. Good for validating the pipeline itself works
   (buckets get built, the LLM gets called, orders/fills/guardrails behave) without needing real
   Robinhood credentials for anything beyond market data.
2. **`OBSERVE`** — real market data, real LLM decisions, but **zero order interaction of any kind**, not
   even a simulated one. When a BUY signal clears the confidence threshold, it's alerted (webhook) and
   recorded in `/decisions` with `acted_on: false` — nothing is ever bought or sold. This needs no MCP/
   OAuth setup at all, since the broker's write path is never reached; it's the natural "watch it call
   real signals for a while before trusting it with money" phase.
3. **`LIVE`** — real orders through the Robinhood MCP, real money, in the isolated Agentic account. See
   [Going live](#going-live).

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

Docker Compose runs the app **and** a local Ollama container together; Postgres stays external (point
`DATABASE_URL` at Supabase or another remote instance — it isn't containerized here, since it's meant to
be a durable store you don't want tied to `docker compose down`).

```bash
cp .env.example .env   # fill in DATABASE_URL, ROBINHOOD_*, WATCHLIST, etc.
docker compose up --build
```

Notes:

- The compose file overrides `OLLAMA_HOST` to `http://ollama:11434` regardless of what's in `.env`, since
  inside the compose network `localhost` refers to the app container itself, not the `ollama` service.
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

- Tear down with `docker compose down`; add `-v` to also drop the `ollama_data` volume (re-pull the model
  next time).

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
  server, or a sidecar container) — this image does not run one itself. If you'd rather use a cloud LLM
  provider, see [Switching LLM providers](#switching-llm-providers).
- **Database:** use your real Postgres (Supabase or otherwise) via `DATABASE_URL`; run
  `alembic upgrade head` as a release step before starting new containers, not automatically on boot.
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

3. **Work through `DRY_RUN` → `OBSERVE` → `LIVE` in order**, not straight to `LIVE`:
   - `MODE=DRY_RUN` first, to confirm the pipeline mechanics work (buckets, LLM calls, simulated
     fills/guardrails) without needing steps 1–2 above at all.
   - `MODE=OBSERVE` next — real market data and real LLM decisions against the account you just
     authorized, but it never places an order, not even a simulated one; BUY signals only get reported
     (webhook + `/decisions` with `acted_on: false`). Run this for a real session or two and read what
     it's actually reporting before trusting it with money.
   - Only then `MODE=LIVE`.

4. **Start small.** Lower `MAX_CAPITAL_PER_TRADE_USD` and `MAX_DAILY_DRAWDOWN_USD` in `.env` for your
   first live sessions, and watch the first few cycles live rather than walking away.

5. Set `MODE=LIVE` and restart the service.

### Switching LLM providers

The default is local Ollama (`LLM_PROVIDER=ollama`, `LLM_MODEL` set to whatever you've pulled). To use a
different provider, implement the `LLMClient` protocol in `llm/base.py` (see `llm/ollama_client.py` for
the shape) and add one branch to `get_llm_client()` in the same file — no other module needs to change.

## Running tests

```bash
# Unit tests -- pure logic, no network or DB
uv run pytest tests/unit -q

# Integration tests -- need a real Postgres
docker run -d --name agentic-trading-test-pg -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=agentic_trading -p 55432:5432 postgres:16-alpine
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/agentic_trading
uv run alembic upgrade head
uv run pytest tests/integration -q

# Everything
uv run pytest tests/ -q

# Lint
uv run ruff check src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/   # autofix
```

`tests/integration/test_end_to_end_dry_run.py` is the closest thing to an automated version of the
"run a real `MODE=DRY_RUN` session" check: it exercises the real Ollama client and real webhook notifier
over real local HTTP servers (only market data is stubbed, since that needs live Robinhood credentials).

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/status` | GET | Mode, watchlist, halted flag, next run time per scheduled job |
| `/decisions` | GET | Recent LLM decisions (`?limit=`) |
| `/trades` | GET | Recent trades (`?limit=`) |
| `/poll-cycle` | POST | Manually trigger one poll cycle across the watchlist, using the real broker/LLM client (`MODE=LIVE` can place real orders here) — `?force=true` bypasses the market-hours window; refuses to run while halted |
| `/kill-switch` | POST | Pause all scheduled jobs immediately |
| `/resume` | POST | Resume scheduled jobs after a kill-switch |
| `/oauth/robinhood/authorize` | GET | Start (or resume) Robinhood MCP authorization — visit in a browser; see [Going live](#going-live) Option B |
| `/oauth/robinhood/callback` | GET | Robinhood's OAuth redirect target — not visited directly |
| `/oauth/robinhood/status` | GET | Progress/result of the most recent authorization attempt |
| `/market-data/{ticker}` | GET | Read-only robin_stocks connectivity check — live quote + latest 5-min bar |
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
  wherever the app is running (in Docker Compose, that's `http://ollama:11434`, not `localhost`).
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
