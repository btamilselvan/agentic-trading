"""Structured LLM output contract (spec section 3.2, extended by section 8's
stateful decision engine), plus the per-ticker trade-state DTO fed alongside the
bucket history so the LLM knows what's already happened today.

TickerState has broadened beyond strictly "trade state" (completed_trades_today etc.)
to also carry same-day context the LLM otherwise has no way to see: prior_close (gap
detection), market_* (broad-market conditions, see
market_data.bucket_builder.MarketContext), news_*/float_shares (qualitative
catalyst & metadata, requirements.md section 6 -- see
market_data.robinhood_client.get_latest_news/get_float_shares), and now
status/active_thesis/decision_history (continuity context, requirements.md section 8
-- see state.ticker_state_store.TickerEvaluationState, which scheduler.py loads this
from every cycle) -- kept flat here rather than as nested objects to match the
existing prior_close precedent, and because build_prompt renders each group under
its own section regardless of how it's carried here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agentic_trading.state.ticker_state_store import DecisionLogEntry, TickerStatus


@dataclass(frozen=True)
class TickerState:
    completed_trades_today: int
    open_positions: int
    realized_pnl_today: float
    prior_close: float | None = None
    market_benchmark_ticker: str | None = None
    market_change_pct: float | None = None
    market_vwap_deviation_pct: float | None = None
    market_range_pct: float | None = None
    news_headline: str | None = None
    news_summary: str | None = None
    news_published_at: datetime | None = None
    float_shares: int | None = None
    # Continuity context (requirements.md section 8) -- the evaluation state Redis
    # has on record for this ticker going into this cycle, so the LLM can maintain
    # an active thesis rather than re-deriving one from scratch each cycle (the
    # oscillation problem the phase exists to fix). status defaults to "FLAT" (no
    # prior state) rather than None, since build_prompt always renders this section.
    status: TickerStatus = "FLAT"
    active_thesis: str | None = None
    initial_entry_price: float | None = None
    current_target_price: float | None = None
    current_stop_loss: float | None = None
    decision_history: list[DecisionLogEntry] = field(default_factory=list)


class TradeDecision(BaseModel):
    decision: Literal["BUY", "HOLD", "SELL"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    buy_limit_price: float | None = Field(default=None, gt=0)
    target_sell_price: float | None = Field(default=None, gt=0)
    # Required alongside buy_limit_price/target_sell_price on BUY -- the protective
    # exit level execution.invalidation checks every cycle once a position is open
    # (requirements.md section 8: "Underlying price crosses the calculated
    # stop-loss boundary" is a hard invalidation criterion, code-enforced, not
    # merely advisory -- see execution/invalidation.py).
    stop_loss_price: float | None = Field(default=None, gt=0)
    max_holding_time_minutes: int | None = Field(default=None, gt=0)
    pattern_reasoning: str = ""
    # Required on every response, not just BUY -- requirements.md section 8:
    # "Require the LLM to output both the updated signal/decision and a
    # thesis_continuity_flag indicating whether the original trade rationale
    # remains intact." No default: the point is to force the model to always
    # explicitly reason about continuity rather than silently defaulting to
    # "still fine". When there's no active thesis yet (a fresh FLAT ticker),
    # true is the natural answer -- nothing to have broken.
    thesis_continuity_flag: bool = Field(
        description="Whether the active thesis (if any) from prior cycles still holds."
    )

    @model_validator(mode="after")
    def _validate_buy_fields(self) -> TradeDecision:
        if self.decision == "BUY":
            if self.buy_limit_price is None or self.target_sell_price is None:
                raise ValueError("BUY decisions require buy_limit_price and target_sell_price")
            if self.target_sell_price <= self.buy_limit_price:
                raise ValueError("target_sell_price must be greater than buy_limit_price")
            if self.stop_loss_price is None:
                raise ValueError("BUY decisions require stop_loss_price")
            if self.stop_loss_price >= self.buy_limit_price:
                raise ValueError("stop_loss_price must be below buy_limit_price")
            if self.max_holding_time_minutes is None:
                raise ValueError("BUY decisions require max_holding_time_minutes")
        return self
