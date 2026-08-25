import pytest
from pydantic import ValidationError

from agentic_trading.llm.schema import TradeDecision


def test_hold_decision_needs_no_price_fields():
    decision = TradeDecision(
        decision="HOLD",
        confidence_score=0.2,
        pattern_reasoning="chop",
        thesis_continuity_flag=True,
    )
    assert decision.decision == "HOLD"
    assert decision.buy_limit_price is None


def test_valid_buy_decision():
    decision = TradeDecision(
        decision="BUY",
        confidence_score=0.85,
        buy_limit_price=100.0,
        target_sell_price=102.0,
        stop_loss_price=98.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
        thesis_continuity_flag=True,
    )
    assert decision.target_sell_price > decision.buy_limit_price
    assert decision.stop_loss_price < decision.buy_limit_price


def test_buy_decision_missing_prices_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(decision="BUY", confidence_score=0.85, thesis_continuity_flag=True)


def test_buy_decision_with_target_below_entry_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(
            decision="BUY",
            confidence_score=0.85,
            buy_limit_price=100.0,
            target_sell_price=99.0,
            stop_loss_price=98.0,
            max_holding_time_minutes=30,
            thesis_continuity_flag=True,
        )


def test_buy_decision_missing_holding_time_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(
            decision="BUY",
            confidence_score=0.85,
            buy_limit_price=100.0,
            target_sell_price=102.0,
            stop_loss_price=98.0,
            thesis_continuity_flag=True,
        )


def test_buy_decision_missing_stop_loss_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(
            decision="BUY",
            confidence_score=0.85,
            buy_limit_price=100.0,
            target_sell_price=102.0,
            max_holding_time_minutes=30,
            thesis_continuity_flag=True,
        )


def test_buy_decision_with_stop_loss_above_entry_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(
            decision="BUY",
            confidence_score=0.85,
            buy_limit_price=100.0,
            target_sell_price=102.0,
            stop_loss_price=100.5,  # >= buy_limit_price
            max_holding_time_minutes=30,
            thesis_continuity_flag=True,
        )


def test_decision_missing_thesis_continuity_flag_is_rejected():
    """Required on every response, not just BUY -- requirements.md section 8."""
    with pytest.raises(ValidationError):
        TradeDecision(decision="HOLD", confidence_score=0.2)


def test_sell_decision_needs_no_price_fields():
    """SELL is the early-exit signal for an IN_POSITION ticker -- it doesn't carry
    entry-side pricing, just the continuity verdict."""
    decision = TradeDecision(
        decision="SELL",
        confidence_score=0.8,
        thesis_continuity_flag=False,
        pattern_reasoning="momentum broke down",
    )
    assert decision.decision == "SELL"


def test_confidence_score_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(decision="HOLD", confidence_score=1.5, thesis_continuity_flag=True)
