from agentic_trading.execution.invalidation import (
    compute_trailing_stop,
    evaluate_exit_guardrails,
    is_momentum_broken,
    is_stop_loss_breached,
)


def test_stop_loss_breached_at_or_below_the_stop():
    assert not is_stop_loss_breached(current_price=99.0, stop_loss=98.0)
    assert is_stop_loss_breached(current_price=98.0, stop_loss=98.0)  # exact touch
    assert is_stop_loss_breached(current_price=97.5, stop_loss=98.0)


def test_momentum_broken_on_rsi_centerline_down_cross():
    assert is_momentum_broken(rsi_centerline_cross="down", vwap_cross=None)
    assert not is_momentum_broken(rsi_centerline_cross="up", vwap_cross=None)
    assert not is_momentum_broken(rsi_centerline_cross=None, vwap_cross=None)


def test_momentum_broken_on_vwap_down_cross():
    assert is_momentum_broken(rsi_centerline_cross=None, vwap_cross="down")
    assert not is_momentum_broken(rsi_centerline_cross=None, vwap_cross="up")


def _exit_kwargs(**overrides):
    kwargs = dict(
        current_price=100.0,
        stop_loss=98.0,
        rsi_centerline_cross=None,
        vwap_cross=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_evaluate_exit_guardrails_holds_when_nothing_broken():
    result = evaluate_exit_guardrails(**_exit_kwargs())
    assert not result.should_exit
    assert result.reason is None


def test_evaluate_exit_guardrails_forces_exit_on_stop_loss_breach():
    result = evaluate_exit_guardrails(**_exit_kwargs(current_price=97.0))
    assert result.should_exit
    assert result.reason == "STOP_LOSS"


def test_evaluate_exit_guardrails_forces_exit_on_momentum_break():
    result = evaluate_exit_guardrails(**_exit_kwargs(vwap_cross="down"))
    assert result.should_exit
    assert result.reason == "MOMENTUM_BREAK"


def test_evaluate_exit_guardrails_short_circuits_on_stop_loss_first():
    # Both a stop-loss breach AND a momentum break are true -- should report
    # STOP_LOSS (checked first), same short-circuit shape as
    # guardrails.evaluate_buy_guardrails.
    result = evaluate_exit_guardrails(
        **_exit_kwargs(current_price=97.0, vwap_cross="down")
    )
    assert result.reason == "STOP_LOSS"


def test_evaluate_exit_guardrails_skips_stop_loss_check_when_missing():
    # Defensive path: stop_loss somehow unset -- criterion 2 still applies.
    result = evaluate_exit_guardrails(**_exit_kwargs(stop_loss=None, vwap_cross="down"))
    assert result.should_exit
    assert result.reason == "MOMENTUM_BREAK"


def test_compute_trailing_stop_ratchets_up_on_better_proposal():
    new_stop, new_target = compute_trailing_stop(
        current_stop_loss=98.0,
        current_target=103.0,
        proposed_stop_loss=99.5,
        proposed_target=105.0,
    )
    assert new_stop == 99.5
    assert new_target == 105.0


def test_compute_trailing_stop_ignores_a_worse_proposal():
    new_stop, new_target = compute_trailing_stop(
        current_stop_loss=98.0,
        current_target=103.0,
        proposed_stop_loss=95.0,  # worse than current -- must not be adopted
        proposed_target=101.0,  # worse than current -- must not be adopted
    )
    assert new_stop == 98.0
    assert new_target == 103.0


def test_compute_trailing_stop_keeps_current_when_nothing_proposed():
    new_stop, new_target = compute_trailing_stop(
        current_stop_loss=98.0,
        current_target=103.0,
        proposed_stop_loss=None,
        proposed_target=None,
    )
    assert new_stop == 98.0
    assert new_target == 103.0


def test_compute_trailing_stop_never_moves_below_current_even_mixed():
    # A better stop but a worse target proposed in the same cycle -- each ratchets
    # independently.
    new_stop, new_target = compute_trailing_stop(
        current_stop_loss=98.0,
        current_target=103.0,
        proposed_stop_loss=99.0,
        proposed_target=102.0,
    )
    assert new_stop == 99.0
    assert new_target == 103.0
