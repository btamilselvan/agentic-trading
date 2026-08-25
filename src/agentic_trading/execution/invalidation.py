"""Code-enforced position-invalidation checks (requirements.md section 8) plus the
one-way trailing-stop ratchet -- pure, dependency-free functions, same philosophy
as execution/guardrails.py: no I/O, callers (scheduler.py) fetch the current
price/RSI-cross/VWAP-cross and pass them in.

guardrails.py's checks only ever BLOCK a would-be BUY; this module is the inverse --
it can FORCE an exit on an already-open position even against an LLM HOLD, which is
why it's a separate module rather than an addition to guardrails.py. scheduler.py
calls evaluate_exit_guardrails for every IN_POSITION ticker BEFORE calling the LLM
at all each cycle -- a stop-loss breach or momentum break exits immediately, no LLM
round-trip needed (see execution/order_manager.py's try_exit_position_early).

Criterion 3 (requirements.md section 8: "major high-impact negative catalyst
headline") is deliberately NOT implemented here -- there's no sentiment classifier
in this codebase, so that judgment is left entirely to the LLM's own
thesis_continuity_flag/SELL response (see llm/prompt.py's HYSTERESIS section).
Criteria 1 and 2 are hard, numeric, and cheap to check without an LLM call at all,
so they're code-enforced here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExitReason = Literal["STOP_LOSS", "MOMENTUM_BREAK"]


@dataclass(frozen=True)
class ExitGuardrailResult:
    should_exit: bool
    reason: ExitReason | None = None
    detail: str | None = None

    @staticmethod
    def hold() -> ExitGuardrailResult:
        return ExitGuardrailResult(should_exit=False)

    @staticmethod
    def force_exit(reason: ExitReason, detail: str) -> ExitGuardrailResult:
        return ExitGuardrailResult(should_exit=True, reason=reason, detail=detail)


def is_stop_loss_breached(current_price: float, stop_loss: float) -> bool:
    """Spec section 8, invalidation criterion 1: "underlying price crosses the
    calculated stop-loss boundary". Long-only: breached at or below the stop, not
    strictly below, so an exact touch still triggers -- a resting stop order would
    fill at that price too.
    """
    return current_price <= stop_loss


def is_momentum_broken(rsi_centerline_cross: str | None, vwap_cross: str | None) -> bool:
    """Spec section 8, invalidation criterion 2: "primary momentum alignment breaks
    (e.g. RSI crosses below key support or loss of VWAP)". Both arguments are
    already-computed discrete crossing events for the latest bucket (see
    market_data.bucket_builder.rsi_centerline_cross/detect_vwap_cross) -- "down" on
    either is a momentum break for a long position; anything else (including no
    crossing at all this bucket, which is the common case) is not.
    """
    return rsi_centerline_cross == "down" or vwap_cross == "down"


def evaluate_exit_guardrails(
    *,
    current_price: float,
    stop_loss: float | None,
    rsi_centerline_cross: str | None,
    vwap_cross: str | None,
) -> ExitGuardrailResult:
    """Runs every hard, code-enforced invalidation check for an open position,
    short-circuiting on the first violation -- criterion 1 (stop-loss) before
    criterion 2 (momentum break), same shape as
    guardrails.evaluate_buy_guardrails. A missing stop_loss (shouldn't happen for a
    real open trade, but defensive -- e.g. Redis state was lost and Postgres's
    Trade.stop_loss_price is also somehow unset) skips criterion 1 rather than
    raising; criterion 2 still applies regardless.
    """
    if stop_loss is not None and is_stop_loss_breached(current_price, stop_loss):
        return ExitGuardrailResult.force_exit(
            "STOP_LOSS", f"price {current_price} crossed stop_loss {stop_loss}"
        )
    if is_momentum_broken(rsi_centerline_cross, vwap_cross):
        return ExitGuardrailResult.force_exit(
            "MOMENTUM_BREAK",
            f"rsi_centerline_cross={rsi_centerline_cross} vwap_cross={vwap_cross}",
        )
    return ExitGuardrailResult.hold()


def compute_trailing_stop(
    *,
    current_stop_loss: float,
    current_target: float,
    proposed_stop_loss: float | None,
    proposed_target: float | None,
) -> tuple[float, float]:
    """One-way ratchet (requirements.md section 8: "Target adjustments are strictly
    limited to one-way trailing stops ... downward adjustments ... are
    prohibited"). A long position's stop_loss/target may only move in the
    position's favor -- stop_loss up, target up -- never down, regardless of what
    is proposed. `proposed_*=None` (nothing better offered this cycle) is treated
    the same as a proposal equal to the current value: no change.

    Returns (new_stop_loss, new_target), both >= the corresponding current value.
    """
    stop_candidate = proposed_stop_loss if proposed_stop_loss is not None else current_stop_loss
    target_candidate = proposed_target if proposed_target is not None else current_target
    return max(current_stop_loss, stop_candidate), max(current_target, target_candidate)
