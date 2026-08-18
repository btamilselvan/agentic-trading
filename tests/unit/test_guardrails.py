from datetime import datetime, time

from agentic_trading.execution.guardrails import (
    check_capital_allocation,
    check_daily_drawdown_circuit_breaker,
    check_daily_trade_cap,
    check_position_cap,
    evaluate_buy_guardrails,
    is_order_timed_out,
    is_past_liquidation_time,
)


def test_position_cap_blocks_when_at_or_above_cap():
    assert check_position_cap(0, max_open_positions_per_ticker=1).allowed
    assert not check_position_cap(1, max_open_positions_per_ticker=1).allowed


def test_daily_trade_cap_blocks_when_at_or_above_cap():
    assert check_daily_trade_cap(2, daily_trade_cap_per_ticker=3).allowed
    assert not check_daily_trade_cap(3, daily_trade_cap_per_ticker=3).allowed


def test_capital_allocation_blocks_when_notional_exceeds_cap():
    assert check_capital_allocation(499.99, max_capital_per_trade_usd=500).allowed
    assert check_capital_allocation(500.0, max_capital_per_trade_usd=500).allowed
    assert not check_capital_allocation(500.01, max_capital_per_trade_usd=500).allowed


def test_circuit_breaker_trips_at_drawdown_limit():
    assert check_daily_drawdown_circuit_breaker(-999.0, max_daily_drawdown_usd=1000).allowed
    assert not check_daily_drawdown_circuit_breaker(-1000.0, max_daily_drawdown_usd=1000).allowed
    assert not check_daily_drawdown_circuit_breaker(-1500.0, max_daily_drawdown_usd=1000).allowed


def test_circuit_breaker_ignores_positive_pnl():
    assert check_daily_drawdown_circuit_breaker(5000.0, max_daily_drawdown_usd=1000).allowed


def test_order_timeout():
    submitted = datetime(2026, 8, 17, 9, 30)
    assert not is_order_timed_out(submitted, datetime(2026, 8, 17, 9, 44), order_timeout_minutes=15)
    assert is_order_timed_out(submitted, datetime(2026, 8, 17, 9, 45), order_timeout_minutes=15)


def test_liquidation_time_cutoff():
    cutoff = time(15, 45)
    assert not is_past_liquidation_time(datetime(2026, 8, 17, 15, 44), cutoff)
    assert is_past_liquidation_time(datetime(2026, 8, 17, 15, 45), cutoff)
    assert is_past_liquidation_time(datetime(2026, 8, 17, 16, 0), cutoff)


def _buy_kwargs(**overrides):
    kwargs = dict(
        open_positions_count=0,
        max_open_positions_per_ticker=1,
        completed_trades_today=0,
        daily_trade_cap_per_ticker=3,
        order_notional=500.0,
        max_capital_per_trade_usd=1000.0,
        realized_pnl_today_all_tickers=0.0,
        max_daily_drawdown_usd=1000.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_evaluate_buy_guardrails_allows_when_all_checks_pass():
    assert evaluate_buy_guardrails(**_buy_kwargs()).allowed


def test_evaluate_buy_guardrails_blocks_on_open_position():
    result = evaluate_buy_guardrails(**_buy_kwargs(open_positions_count=1))
    assert not result.allowed
    assert "position" in result.reason


def test_evaluate_buy_guardrails_blocks_on_trade_cap():
    result = evaluate_buy_guardrails(**_buy_kwargs(completed_trades_today=3))
    assert not result.allowed
    assert "daily cap" in result.reason


def test_evaluate_buy_guardrails_blocks_on_capital_cap():
    result = evaluate_buy_guardrails(**_buy_kwargs(order_notional=1500.0))
    assert not result.allowed
    assert "notional" in result.reason


def test_evaluate_buy_guardrails_blocks_on_circuit_breaker():
    result = evaluate_buy_guardrails(**_buy_kwargs(realized_pnl_today_all_tickers=-1000.0))
    assert not result.allowed
    assert "circuit breaker" in result.reason


def test_evaluate_buy_guardrails_short_circuits_on_first_violation():
    # Both position cap AND trade cap are violated -- should report the position cap
    # (checked first) rather than the trade cap.
    result = evaluate_buy_guardrails(
        **_buy_kwargs(open_positions_count=1, completed_trades_today=3)
    )
    assert "position" in result.reason
