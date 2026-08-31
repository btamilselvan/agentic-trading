# Requirements Document: Robinhood Autonomous Intraday Momentum Trader

## Change Log & Revision History
> **Instructions for Claude Code:** Always inspect this log first to see what was modified recently.

* **v1.5.0 (2026-08-28):** Added Phase 5 requirements (New).
* **v1.4.0 (2026-08-28):** Added Phase 4 requirements (Completed).
* **v1.3.0 (2026-08-24):** Added Phase 3 requirements (Completed).
* **v1.1.0 (2026-08-19):** Added Phase 2 requirements (Completed).
* **v1.0.0 (2026-08-15):** Initial Phase 1 release (Completed).

## Phase 1: Core MVP (Completed)

## 1. Executive Summary
An autonomous, lightweight intraday trading service built in Python (FastAPI) that polls real-time market microstructure data during market open. It processes 5-minute time-series buckets (buy/sell volume, price velocity, bid-ask dynamics), passes the complete metric stream to an LLM agent for intraday pattern analysis, and automatically executes bracket/paired intraday orders (Buy Limit + Target Sell Limit) designed to enter and exit within the same trading session.

---

## 2. System Architecture & Tech Stack

* **Core Runtime:** Python 3.14+
* **Framework:** FastAPI (REST endpoints, background tasks, cron/scheduler)
* **Data & Indicator Engine:** Pandas / Polars (Time-series aggregation, metric calculation)
* **Brokerage Integration:** Robinhood API (`robin_stocks` or official SDK)
* **AI Engine:** LLM Agent (via OpenAI API or local Ollama instance)
* **Data Storage:** SQLite / PostgreSQL (Historical trade logs, 5-min bucket snapshots, LLM decision reasoning)
* **Deployment:** Local containerized setup (Docker / Docker Compose)

---

## 3. Workflow & Functional Requirements

### 3.1 Data Collection Phase (Intraday Microstructure)
* **Schedule:** Launches automatically at 09:30 AM EST on market trading days.
* **Interval:** Polling occurs every 5 minutes during active evaluation windows (e.g., 09:30 AM – 11:30 AM EST).
* **Metrics Ingested per 5-Minute Bucket:**
  * **5-Minute OHLC Prices:** Exact 5-minute Open, High, Low, and Close prices (e.g., 9:30 Open $200.00 → 9:35 Close $205.00, net +$5.00 / +2.5% delta).
  * **Volume Dynamics:** Estimated Buy Volume vs. Sell Volume (tick trade classification at Bid vs. Ask).
  * **Bid-Ask Dynamics:** Bid-ask ratio, spread width, and order book depth imbalance.
  * **Price Velocity & Volatility:** Candle body size (`Close - Open`), wick lengths, and intraday range expansion.
  * **Relative Volume (RVOL):** 5-minute volume relative to historical opening averages.

### 3.2 LLM Analysis & Intraday Decision Engine
* **Context Payload:** The engine feeds the complete array of collected 5-minute metric buckets to the LLM agent, alongside the ticker's current trade state (e.g., completed trades today, total daily PnL on ticker).
* **Intraday Strategy Mandate:** The LLM evaluates the time series strictly for **same-day intraday setups** (e.g., morning breakout, volume absorption, quick mean reversion, or momentum continuation).
* **Structured LLM Output:**
  * `decision`: `BUY` or `HOLD`
  * `confidence_score`: 0.0 – 1.0 (threshold required to trigger order submission)
  * `buy_limit_price`: Calculated entry limit price based on current bid/ask
  * `target_sell_price`: Calculated intraday profit target limit price (sized for intraday exit)
  * `max_holding_time_minutes`: Estimated max intraday duration before mandatory exit
  * `pattern_reasoning`: Concise breakdown of the order flow setup detected

### 3.3 Intraday Order Execution & Management
* **Paired Execution Flow:** Once a `BUY` signal is generated and the buy order fills on Robinhood:
  1. The system confirms the buy fill price and size.
  2. It immediately places a matching **Limit Sell Order** at `target_sell_price`.
* **Sequential Multi-Trade Support:**
  * Tickers can undergo **multiple sequential trades** within the same day.
  * A ticker becomes eligible for a new `BUY` signal **only after** its previous sell order executes and the active position count returns to **0**.
* **Day-End Liquidation Rule:** Because this is strictly an **intraday** strategy, any open positions or unfilled limit orders are automatically cancelled/closed prior to market close (e.g., 3:45 PM EST) to eliminate overnight holding risk.

---

## 4. Guardrails & Safety Controls

* **Zero Concurrent Stacking:** Maximum of 1 active/open position per ticker at any given moment.
* **Daily Trade Cap Per Ticker:** Optional safety limit on maximum total completed trades per ticker per day (e.g., max 3 round-trip trades) to prevent over-trading in choppy markets.
* **No Overnight Exposure:** Mandatory cancellation of open orders and liquidation of remaining positions before market close.
* **Capital Allocation Limits:** Hard caps on total dollar allocation per trade and aggregate daily risk.
* **Order Timeout:** Unfilled Buy limit orders automatically cancel after a configurable threshold (e.g., 10–15 minutes).
* **Circuit Breakers:** Immediate engine shutdown if daily realized drawdown limit is reached or if market data feeds drop.

---

## 5. Telemetry & Audit Logs

* **Trade Telemetry:** Logs all 5-minute raw metrics, calculated ratios, exact LLM prompts, agent JSON responses, and order ID status updates.
* **Real-Time Webhook Alerts:** Sends instant notifications (Telegram / Slack / Discord) containing trade execution details, LLM confidence scores, and order fill prices.

---

## Phase 2: Enhanced Data Collection enhancements (Completed)

## 6. Data Collection (Completed)
* **Metrics Ingested per 5-Minute Bucket:**
  * **Qualitative Catalyst & Metadata:** Real-time news headline flag/summary, Float size (<20M shares indication), and short interest %.
  * **Session Time Context:** `minutes_since_open` and session classification (`Opening Volatility`, `Morning Trend`, or `Midday Chop`).
  * **Relative Strength Index (RSI):** 5-minute intraday RSI (calculated locally via Pandas-TA using RSI-14 or RSI-9) to evaluate overbought/oversold boundaries, centerline 50 crossovers, and price/RSI momentum divergence.
---

**Implementation notes (Claude Code, 2026-08-19):**
* News headline/summary and float size are implemented (`market_data/robinhood_client.get_latest_news`/`get_float_shares`, fetched per-ticker each poll cycle and threaded into the LLM prompt as `catalyst_context`).
* **Short interest % is NOT implemented** -- neither robin_stocks nor the Robinhood API expose it (no field, no endpoint), and unlike buy/sell volume or book depth there's no reasonable OHLCV-only proxy for it. Would need a third-party data provider to add later.
* RSI is implemented in plain Python (Wilder's smoothing) rather than Pandas-TA, since nothing else in this project uses pandas and the algorithm doesn't justify a first-time dependency for one indicator.
* Session Time Context boundaries (30 / 120 minutes) aren't specified above, so they default to the existing `MARKET_OPEN_TIME`/`EVALUATION_WINDOW_END_TIME` poll window (09:30-11:30) rather than being separately configurable.
* **(2026-08-31) LLM payload trim:** `minutes_since_open` is still computed per bucket (needed to derive session classification) but is no longer sent to the LLM in `llm/prompt.py` -- it's redundant with the session_phase label the LLM is already told how to weigh, and dropping it (along with folding `est_buy_volume`/`est_sell_volume` into a single normalized `buy_pressure_pct`, and sending raw bid/ask quote fields only on the most recent bucket) reduces prompt size/noise for the small local model without losing information. `est_buy_volume`/`est_sell_volume` and the full per-bucket bid/ask are still computed and persisted to Postgres (`state/models.py`'s `Bucket`) for audit/backtesting -- only the LLM-facing payload is trimmed.


## Phase 3: Stateful Decision Engine & Memory Persistence (Completed)

### 7. Problem Statement & Objective
In Phase 1 and 2, evaluating isolated 5-minute market metrics caused decision oscillation (e.g., flipping between `BUY` and `HOLD` across consecutive bars due to minor price noise). Phase 3 introduces stateful context persistence to ensure decisions maintain continuity, respect an active thesis, and enforce strict invalidation thresholds.

### 8. Functional Requirements

* **State Persistence & Storage:**
  * Implement a lightweight local state store (Redis) to track active evaluation state per ticker across background scanning cycles.
  * Maintain persistent state metadata including: `status` (`SELL`, `HOLD`, `BUY`, `IN_POSITION`), `active_thesis`, `initial_entry_price`, `target_price`, `stop_loss`, and `decision_history`.

* **Contextual Payload Architecture:**
  * For each 5-minute evaluation cycle, append the previous 3 to 5 decision logs and active thesis parameters into the LLM input prompt payload.
  * Require the LLM to output both the updated signal/decision and a `thesis_continuity_flag` indicating whether the original trade rationale remains intact.

* **State Transition & Invalidation Logic:**
  * **Hysteresis Enforcement:** The LLM shall not abandon an active `BUY` or `IN_POSITION` or `HOLD` state due to single-bar noise unless an explicit invalidation criterion is met.
  * **Invalidation Criteria:** State downgrades (e.g., `BUY` to `FLAT` or `EXIT`) require one of the following:
    1. Underlying price crosses the calculated stop-loss boundary.
    2. Primary momentum alignment breaks (e.g., RSI crosses below key support or loss of VWAP).
    3. Major high-impact negative catalyst headline is detected.

* **Trailing Target & Exit Management:**
  * Once a position is established, the agent shall retain fixed baseline profit and stop targets.
  * Target adjustments are strictly limited to one-way trailing stops (ratcheting upward for long positions); downward adjustments to profit targets on noise are prohibited.

### Implementation Notes

* The "lightweight local state store" is Redis (`REDIS_URL`, `state/ticker_state_store.py`), holding only the ephemeral working state this section describes (`status`/`active_thesis`/`decision_history`/stop/target) -- it is deliberately *not* the audit trail. Postgres remains that (`Trade.stop_loss_price`/`exit_reason`, `LlmDecision.stop_loss_price`/`thesis_continuity_flag`, migration `0005_add_stop_loss_exit`); if Redis state is ever lost or expired mid-position, `scheduler.py` rebuilds it from the `Trade` row rather than treating the ticker as `FLAT`. Redis state is keyed by `(ticker, trade_date)` and TTL'd (`TICKER_STATE_TTL_HOURS`, default 24h) as a belt-and-suspenders guard against ever leaking into a later session, in addition to being explicitly cleared on trade close.
* `decision_history` length is configurable (`DECISION_HISTORY_LENGTH`, default 5, within the spec's 3-5 range).
* Invalidation criteria 1 (stop-loss) and 2 (momentum break) are code-enforced (`execution/invalidation.py`'s `evaluate_exit_guardrails`), checked *before* any LLM call each cycle and able to force an exit even against an LLM `HOLD` -- consistent with this codebase's existing guardrail philosophy that safety-critical checks are never merely advisory. Criterion 3 (a negative catalyst headline) has no code-side check, since there is no sentiment classifier available (see `market_data/robinhood_client.py`'s news integration) -- it is judged entirely by the LLM via `thesis_continuity_flag`/`SELL`.
* The spec's `SELL`/`FLAT`/`EXIT` state-downgrade language is implemented as: `TradeDecision.decision` gains a third value, `SELL` -- meaningful only when a position is already `IN_POSITION` -- and a downgrade is any of (a) `evaluate_exit_guardrails` forcing an exit, or (b) the LLM responding `SELL` or `thesis_continuity_flag=false`. Either path calls `execution/order_manager.py`'s `try_exit_position_early` (cancels the resting target-sell order, places an immediate marketable exit) with the triggering reason recorded on `Trade.exit_reason`.
* The one-way trailing-stop ratchet (`execution/invalidation.py`'s `compute_trailing_stop`, applied via `execution/order_manager.py`'s `apply_trailing_stop`) is on by default (`TRAILING_STOP_ENABLED=true`) but operator-configurable off, in which case a position's stop/target stay exactly as set at entry for its whole lifetime -- the deterministic stop-loss/momentum invalidation check above still runs regardless of this flag either way.

---

## Phase 4: Stock quotes using Schwab Market Data Production API (Completed)
### 9. Problem Statement & Objective
* The Robinhood API is unofficial and subject to rate limits, downtime, and potential future deprecation. To ensure continuity of market data feeds, the system will integrate Schwab's Market Data Production API for stock quotes.

### 10. Functional Requirements
* **Schwab API Integration:** Implement a Schwab client module to fetch real-time stock quotes and market data.
* **Fallback Mechanism:** If the Schwab client module fails to fetch real-stime stock quotes, use Robinhood API.

### Implementation Notes

* Scope was widened slightly beyond the literal "stock quotes" wording, in consultation with the
  operator: Schwab became the primary source for **both** quotes and 5-minute historicals, not quotes
  only. Schwab's `pricehistory` endpoint natively supports 5-minute candles (`schwab-py`'s
  `get_price_history_every_five_minutes`), the same granularity `robin_stocks.get_stock_historicals
  (interval="5minute")` already provided, so no separate 1-minute-poll-and-aggregate scheme was needed.
  News (`get_latest_news`) and float shares (`get_float_shares`) stay Robinhood-only -- Schwab's Market
  Data API has no confirmed equivalent for either.
* `market_data/schwab_client.py` is the only module that imports `schwab-py`, mirroring
  `market_data/robinhood_client.py`'s isolation of robin_stocks -- same pattern
  `execution/broker_mcp_client.py`'s module docstring calls out for brokerage integrations generally.
  `market_data/market_data_client.py` is the new orchestrator: tries Schwab first, falls back to
  Robinhood on any empty/failed result. Both provider modules fail closed (log a warning, return
  `None`/`[]`) rather than raise, so the fallback layer never needs exception handling of its own --
  `scheduler.py` and `api/routes.py` call `market_data_client` instead of either provider module
  directly (except for the two Robinhood-only calls above, and the dedicated
  `GET /market-data/schwab/{ticker}` connectivity check, which bypasses the fallback on purpose to test
  Schwab specifically).
* `HistoricalBar`/`Quote`/`NewsItem` moved out of `robinhood_client.py` into a new
  `market_data/models.py`, so `schwab_client.py` doesn't need to import `robinhood_client.py` (or
  duplicate the dataclasses) to produce the same shapes; `robinhood_client.py` re-exports them for
  backward compatibility.
* Auth: Schwab's OAuth needs a real, refreshable session for every call (unlike robin_stocks, whose
  market-data calls work unauthenticated) -- `scripts/bootstrap_schwab_oauth.py` is the one-time
  (practically: roughly weekly, see below) interactive step using `schwab-py`'s `easy_client`, which
  handles the whole browser-consent + local-callback-server + token-write dance internally, unlike the
  Robinhood MCP flow's hand-rolled callback server (`scripts/bootstrap_mcp_oauth.py`). The running app
  never performs this interactive step itself -- `schwab_client.py` only ever reads the cached token via
  `client_from_token_file` and lets `schwab-py` refresh it silently in the background. Chose the
  single-script pattern over also adding in-app `GET /oauth/schwab/{authorize,callback,status}` endpoints
  (as Robinhood's MCP flow has, for completing auth on a headless deployed instance) -- not worth the
  extra surface area unless/until there's a concrete headless-deploy need, per operator decision.
* Schwab refresh tokens go stale after roughly a week of disuse (`schwab-py`'s `easy_client` proactively
  discards anything older than `max_token_age`, ~6.5 days) -- this is a soft dependency, not a hard one:
  a stale/missing/unconfigured token just means `market_data_client.py` falls back to Robinhood for every
  call until the bootstrap script is re-run, never an outage.
* `scripts/refresh_schwab_token.py` (added after initial Phase 4 delivery, per operator request) forces
  an access-token refresh via the stored refresh token, no browser needed -- unlike
  `bootstrap_schwab_oauth.py`, it's meant for unattended/cron use (e.g. daily) so the on-disk token never
  goes long-idle. It calls `client.session.refresh_token()` directly (the same authlib `OAuth2Client`
  underneath `client_from_token_file`, whose `update_token` callback already persists to
  `SCHWAB_TOKEN_PATH` on every refresh -- both the transparent per-call refresh `schwab_client.py` relies
  on and this script's explicit one go through the identical write path). It does NOT reset the refresh
  token's own ~7-day absolute expiry -- Schwab's is a fixed lifetime from issuance, not sliding-on-use --
  so it fails clearly (rather than leaving a stale token in place) once that's elapsed, pointing back at
  `bootstrap_schwab_oauth.py` for the interactive re-auth that actually is needed at that point.
* The Schwab token path moved under `.secrets/` (`SCHWAB_TOKEN_PATH=.secrets/schwab_token.json`,
  consistent with `ROBINHOOD_TOKEN_PATH`/`MCP_TOKEN_STORE_PATH`) rather than the repo root the original
  spike (`scripts/schwab_auth_demo.py`, not merged) used -- `.secrets/` is already gitignored.

---

## Phase 5: Stock screener (New)
### 11. Problem Statement & Objective
* To enhance the system's ability to identify potential intraday trading opportunities, a stock screener will be implemented to filter stocks based on specific criteria such as price, volume, volatility, and other technical indicators.
### 12. Functional Requirements
* **Screener Criteria:** The screener should allow filtering based on:
  * Price range (e.g., $10 - $100)
  * Average daily volume (e.g., > 1M shares)
  * Volatility (e.g., ATR or standard deviation)
  * Technical indicators (e.g., RSI, MACD, moving averages)

---

## References
* [Robinhood API Documentation](https://robinhood.com/us/en/support/articles/robinhood-api/)
* [Unofficial Robinhood API Documentation](https://github.com/sanko/Robinhood)
