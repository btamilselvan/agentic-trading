"""Structured LLM output contract (spec section 3.2), plus the per-ticker trade-state
DTO fed alongside the bucket history so the LLM knows what's already happened today.

TickerState has broadened beyond strictly "trade state" (completed_trades_today etc.)
to also carry same-day context the LLM otherwise has no way to see: prior_close (gap
detection) and now the market_* fields (broad-market conditions, see
market_data.bucket_builder.MarketContext) -- kept flat here rather than as a nested
object to match the existing prior_close precedent, and because build_prompt renders
them under their own "market_context" section regardless of how they're carried here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class TradeDecision(BaseModel):
    decision: Literal["BUY", "HOLD"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    buy_limit_price: float | None = Field(default=None, gt=0)
    target_sell_price: float | None = Field(default=None, gt=0)
    max_holding_time_minutes: int | None = Field(default=None, gt=0)
    pattern_reasoning: str = ""

    @model_validator(mode="after")
    def _validate_buy_fields(self) -> TradeDecision:
        if self.decision == "BUY":
            if self.buy_limit_price is None or self.target_sell_price is None:
                raise ValueError("BUY decisions require buy_limit_price and target_sell_price")
            if self.target_sell_price <= self.buy_limit_price:
                raise ValueError("target_sell_price must be greater than buy_limit_price")
            if self.max_holding_time_minutes is None:
                raise ValueError("BUY decisions require max_holding_time_minutes")
        return self
