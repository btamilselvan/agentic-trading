import pytest
from pydantic import ValidationError

from agentic_trading.llm.schema import TradeDecision


def test_hold_decision_needs_no_price_fields():
    decision = TradeDecision(decision="HOLD", confidence_score=0.2, pattern_reasoning="chop")
    assert decision.decision == "HOLD"
    assert decision.buy_limit_price is None


def test_valid_buy_decision():
    decision = TradeDecision(
        decision="BUY",
        confidence_score=0.85,
        buy_limit_price=100.0,
        target_sell_price=102.0,
        max_holding_time_minutes=30,
        pattern_reasoning="breakout",
    )
    assert decision.target_sell_price > decision.buy_limit_price


def test_buy_decision_missing_prices_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(decision="BUY", confidence_score=0.85)


def test_buy_decision_with_target_below_entry_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(
            decision="BUY",
            confidence_score=0.85,
            buy_limit_price=100.0,
            target_sell_price=99.0,
            max_holding_time_minutes=30,
        )


def test_buy_decision_missing_holding_time_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(
            decision="BUY",
            confidence_score=0.85,
            buy_limit_price=100.0,
            target_sell_price=102.0,
        )


def test_confidence_score_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        TradeDecision(decision="HOLD", confidence_score=1.5)
