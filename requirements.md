# Requirements Document: Robinhood Autonomous Intraday Momentum Trader

## Change Log & Revision History
> **Instructions for Claude Code:** Always inspect this log first to see what was modified recently.

* **v1.1.0 (2026-08-19):** Added Phase 2 requirements (In progress).
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

## Phase 2: Enhanced Data Collection enhancements (WIP)

## 6. Data Collection (continued...)
* **Metrics Ingested per 5-Minute Bucket:**
  * **Qualitative Catalyst & Metadata:** Real-time news headline flag/summary, Float size (<20M shares indication), and short interest %.
  * **Session Time Context:** `minutes_since_open` and session classification (`Opening Volatility`, `Morning Trend`, or `Midday Chop`).
  * **Relative Strength Index (RSI):** 5-minute intraday RSI (calculated locally via Pandas-TA using RSI-14 or RSI-9) to evaluate overbought/oversold boundaries, centerline 50 crossovers, and price/RSI momentum divergence.
---

## References
* [Robinhood API Documentation](https://robinhood.com/us/en/support/articles/robinhood-api/)
* [Unofficial Robinhood API Documentation](https://github.com/sanko/Robinhood)
