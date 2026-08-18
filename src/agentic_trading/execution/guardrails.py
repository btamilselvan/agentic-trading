"""Guardrails: spec section 4 safety controls, implemented as pure functions.

These are enforced at the execution layer regardless of what the LLM decided --
defense in depth, not just prompt instructions. No I/O here; callers (order_manager,
scheduler) fetch the current counts/prices from the repository and pass them in,
which keeps this module trivially unit-testable and the single place guardrail
*logic* lives, rather than scattering checks across order_manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None

    @staticmethod
    def ok() -> GuardrailResult:
        return GuardrailResult(allowed=True)

    @staticmethod
    def blocked(reason: str) -> GuardrailResult:
        return GuardrailResult(allowed=False, reason=reason)


def check_position_cap(
    open_positions_count: int, max_open_positions_per_ticker: int
) -> GuardrailResult:
    """Spec 4: Zero Concurrent Stacking -- max 1 (configurable) open position/ticker."""
    if open_positions_count >= max_open_positions_per_ticker:
        return GuardrailResult.blocked(
            f"{open_positions_count} open position(s) already at/above cap of "
            f"{max_open_positions_per_ticker}"
        )
    return GuardrailResult.ok()


def check_daily_trade_cap(
    completed_trades_today: int, daily_trade_cap_per_ticker: int
) -> GuardrailResult:
    """Spec 4: Daily Trade Cap Per Ticker."""
    if completed_trades_today >= daily_trade_cap_per_ticker:
        return GuardrailResult.blocked(
            f"{completed_trades_today} completed trade(s) today at/above daily cap of "
            f"{daily_trade_cap_per_ticker}"
        )
    return GuardrailResult.ok()


def check_capital_allocation(
    order_notional: float, max_capital_per_trade_usd: float
) -> GuardrailResult:
    """Spec 4: Capital Allocation Limits -- hard cap on dollar allocation per trade."""
    if order_notional > max_capital_per_trade_usd:
        return GuardrailResult.blocked(
            f"order notional ${order_notional:,.2f} exceeds per-trade cap of "
            f"${max_capital_per_trade_usd:,.2f}"
        )
    return GuardrailResult.ok()


def check_daily_drawdown_circuit_breaker(
    realized_pnl_today_all_tickers: float, max_daily_drawdown_usd: float
) -> GuardrailResult:
    """Spec 4: Circuit Breaker -- immediate shutdown if daily realized drawdown limit hit.

    `realized_pnl_today_all_tickers` is the account-wide realized PnL for the day
    (negative = loss). The breaker trips once the loss magnitude reaches the cap.
    """
    if realized_pnl_today_all_tickers <= -abs(max_daily_drawdown_usd):
        return GuardrailResult.blocked(
            f"realized daily drawdown ${-realized_pnl_today_all_tickers:,.2f} has reached "
            f"the ${max_daily_drawdown_usd:,.2f} circuit breaker limit"
        )
    return GuardrailResult.ok()


def is_order_timed_out(submitted_at: datetime, now: datetime, order_timeout_minutes: int) -> bool:
    """Spec 4: Order Timeout -- unfilled buy limit orders auto-cancel after N minutes."""
    return (now - submitted_at).total_seconds() >= order_timeout_minutes * 60


def is_past_liquidation_time(now_local: datetime, eod_liquidation_time: time) -> bool:
    """Spec 4 / 3.3: Day-End Liquidation Rule -- true once `now_local` (already
    converted to market-local time by the caller) is at/after the liquidation cutoff.
    """
    return now_local.time() >= eod_liquidation_time


def evaluate_buy_guardrails(
    *,
    open_positions_count: int,
    max_open_positions_per_ticker: int,
    completed_trades_today: int,
    daily_trade_cap_per_ticker: int,
    order_notional: float,
    max_capital_per_trade_usd: float,
    realized_pnl_today_all_tickers: float,
    max_daily_drawdown_usd: float,
) -> GuardrailResult:
    """Runs every BUY-eligibility guardrail, short-circuiting on the first violation.

    This is what order_manager calls before ever submitting a buy order, independent
    of the LLM's own confidence-threshold gate -- the LLM saying BUY is necessary but
    not sufficient.
    """
    checks = (
        check_position_cap(open_positions_count, max_open_positions_per_ticker),
        check_daily_trade_cap(completed_trades_today, daily_trade_cap_per_ticker),
        check_capital_allocation(order_notional, max_capital_per_trade_usd),
        check_daily_drawdown_circuit_breaker(
            realized_pnl_today_all_tickers, max_daily_drawdown_usd
        ),
    )
    for result in checks:
        if not result.allowed:
            return result
    return GuardrailResult.ok()
