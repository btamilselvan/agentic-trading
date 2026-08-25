"""HTTP surface: health/status for ops, read endpoints over the audit trail (spec
section 5) for inspecting recent decisions/trades without a DB client, a manual
kill-switch (spec section 4's circuit breaker is otherwise fully automatic --
new BUYs already self-block once the daily drawdown cap trips, see
execution/guardrails.py -- this is the operator's manual override on top of that),
and two read-only connectivity checks (market-data via robin_stocks, broker state
via the Robinhood MCP) -- neither places or touches any order.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_trading.alerts.base import get_notifier
from agentic_trading.config import TradingMode, get_settings
from agentic_trading.execution import order_manager as om
from agentic_trading.execution.broker_mcp_client import McpBrokerClient
from agentic_trading.llm.schema import TradeDecision
from agentic_trading.market_data import robinhood_client as rh_market
from agentic_trading.scheduler import run_poll_cycle
from agentic_trading.state import repository as repo
from agentic_trading.state.db import get_session, session_scope
from agentic_trading.state.models import LlmDecision, Trade

router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/status")
async def status(request: Request) -> dict:
    settings = get_settings()
    scheduler = request.app.state.scheduler
    jobs = [
        {
            "id": job.id,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
        }
        for job in scheduler.get_jobs()
    ]
    return {
        "mode": settings.mode.value,
        "watchlist": settings.watchlist,
        "halted": request.app.state.halted,
        "jobs": jobs,
    }


@router.get("/decisions")
async def recent_decisions(session: SessionDep, limit: int = 50) -> list[dict]:
    stmt = select(LlmDecision).order_by(LlmDecision.created_at.desc()).limit(limit)
    rows = (await session.scalars(stmt)).all()
    return [
        {
            "id": r.id,
            "ticker": r.ticker,
            "decision": r.decision.value,
            "confidence_score": float(r.confidence_score),
            "buy_limit_price": float(r.buy_limit_price) if r.buy_limit_price is not None else None,
            "target_sell_price": (
                float(r.target_sell_price) if r.target_sell_price is not None else None
            ),
            "acted_on": r.acted_on,
            "pattern_reasoning": r.pattern_reasoning,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@router.get("/trades")
async def recent_trades(session: SessionDep, limit: int = 50) -> list[dict]:
    stmt = select(Trade).order_by(Trade.opened_at.desc()).limit(limit)
    rows = (await session.scalars(stmt)).all()
    return [
        {
            "id": t.id,
            "ticker": t.ticker,
            "status": t.status.value,
            "entry_price": float(t.entry_price) if t.entry_price is not None else None,
            "exit_price": float(t.exit_price) if t.exit_price is not None else None,
            "quantity": float(t.quantity) if t.quantity is not None else None,
            "pnl": float(t.pnl) if t.pnl is not None else None,
            "opened_at": t.opened_at.isoformat(),
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        }
        for t in rows
    ]


@router.get("/market-data/{ticker}")
async def market_data(ticker: str) -> dict:
    """Read-only smoke test for the robin_stocks (Robinhood API) integration --
    fetches a live quote and today's 5-minute bars for `ticker`, the same calls the
    poll cycle makes. robin_stocks is synchronous, so these run in a worker thread
    (asyncio.to_thread) rather than blocking the event loop the scheduler shares.

    Note: if ROBINHOOD_USERNAME/PASSWORD aren't set, robin_stocks' first login
    attempt prompts for MFA on stdin -- in a headless deployment that means this
    request (and the scheduler's own polling) will hang rather than fail cleanly.
    """
    ticker = ticker.upper()
    try:
        quote = await asyncio.to_thread(rh_market.get_quote, ticker)
        bars = await asyncio.to_thread(rh_market.get_5min_historicals, ticker, span="day")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"robin_stocks call failed: {exc}") from exc

    return {
        "ticker": ticker,
        "quote": asdict(quote) if quote else None,
        "latest_bar": asdict(bars[-1]) if bars else None,
        "bars_today": len(bars),
    }


@router.get("/broker/positions/{ticker}")
async def broker_position(ticker: str) -> dict:
    """Read-only smoke test for the Robinhood Trading MCP integration. Always talks
    to the real MCP via a fresh McpBrokerClient, regardless of MODE -- unlike the
    trading pipeline (which uses the simulated DryRunBrokerClient outside
    MODE=LIVE), the whole point here is to verify the OAuth token + MCP session +
    get_equity_positions tool call independent of the rest of the app. Requires
    completed OAuth authorization (see /oauth/robinhood/authorize).
    """
    ticker = ticker.upper()
    try:
        quantity = await McpBrokerClient().get_open_position_quantity(ticker)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Robinhood MCP call failed: {exc}") from exc

    return {"ticker": ticker, "open_position_quantity": quantity}


@router.post("/poll-cycle")
async def trigger_poll_cycle(request: Request, force: bool = False) -> dict:
    """Manually triggers one run_poll_cycle across the whole watchlist, using the
    same broker/llm_client instances the scheduler itself uses (app.state, wired in
    main.py's lifespan) -- not fresh ones -- so this exercises the real pipeline
    (MODE=LIVE really can place orders here) rather than a side simulation.

    By default this still enforces the MARKET_OPEN_TIME-EVALUATION_WINDOW_END_TIME
    poll window and does nothing outside it, same as the scheduled job; pass
    ?force=true to bypass that for manual/debugging use. Refuses to run at all while
    halted (see /kill-switch) -- an on-demand trigger must not be a way around a
    halt.
    """
    if request.app.state.halted:
        raise HTTPException(status_code=409, detail="Halted via kill-switch -- resume first")

    await run_poll_cycle(
        broker=request.app.state.broker,
        llm_client=request.app.state.llm_client,
        bypass_window=force,
    )
    return {"status": "completed", "watchlist": get_settings().watchlist, "forced": force}


class ManualEntryRequest(BaseModel):
    ticker: str
    buy_limit_price: float = Field(gt=0)
    target_sell_price: float = Field(gt=0)
    # Required now that TradeDecision validates BUY decisions carry a protective
    # stop level (requirements.md section 8) -- see llm/schema.py.
    stop_loss_price: float = Field(gt=0)
    max_holding_time_minutes: int = Field(gt=0, default=15)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    pattern_reasoning: str = "Manual test entry via POST /orders/manual-entry"


@router.post("/orders/manual-entry")
async def manual_order_entry(
    request: Request, body: ManualEntryRequest, confirm: bool = False
) -> dict:
    """Debug/test hook: builds a synthetic BUY `TradeDecision` from the request body
    and calls `order_manager.try_enter_position` directly, bypassing market-data
    collection and the LLM entirely. Exists to live-test try_enter_position itself
    (guardrails, sizing, broker calls, paired-sell placement, DB writes) against a
    real broker connection without waiting for an actual BUY signal out of the poll
    cycle.

    Uses the SAME broker instance the scheduler uses (app.state.broker, wired in
    main.py's lifespan) -- in MODE=LIVE that's a real McpBrokerClient, so a call here
    CAN place a REAL order with real money if try_enter_position's guardrails allow
    it (position cap, daily trade cap, capital limits, circuit breaker -- all
    independently re-checked there regardless of what's passed here). Nothing here
    validates that buy_limit_price/target_sell_price make sense against the market
    -- that's the caller's job. Requires `?confirm=true` whenever MODE=LIVE, so a
    routine or accidental call can't place real money by mistake; DRY_RUN/OBSERVE
    need no confirmation since their broker is the in-memory simulator. Note this
    deliberately bypasses OBSERVE mode's normal "zero order interaction" guarantee
    (see scheduler._poll_ticker) -- that guarantee is about the automated poll
    cycle, not this manual/operator-invoked debug endpoint. Refuses to run at all
    while halted (see /kill-switch), same as /poll-cycle.
    """
    settings = get_settings()
    if request.app.state.halted:
        raise HTTPException(status_code=409, detail="Halted via kill-switch -- resume first")
    if settings.mode == TradingMode.LIVE and not confirm:
        raise HTTPException(
            status_code=400,
            detail="MODE=LIVE -- this can place a REAL order. Pass ?confirm=true to proceed.",
        )

    ticker = body.ticker.upper()
    decision = TradeDecision(
        decision="BUY",
        confidence_score=body.confidence_score,
        buy_limit_price=body.buy_limit_price,
        target_sell_price=body.target_sell_price,
        stop_loss_price=body.stop_loss_price,
        max_holding_time_minutes=body.max_holding_time_minutes,
        pattern_reasoning=body.pattern_reasoning,
        # No active thesis to break for a manual/operator-invoked test entry.
        thesis_continuity_flag=True,
    )
    today = datetime.now(UTC).date()

    async with session_scope() as session:
        llm_decision = await repo.save_llm_decision(
            session,
            ticker=ticker,
            bucket_id=None,
            prompt="(manual entry -- no LLM prompt, see POST /orders/manual-entry)",
            raw_response="(manual entry -- no LLM call, see POST /orders/manual-entry)",
            decision=decision.decision,
            confidence_score=decision.confidence_score,
            buy_limit_price=decision.buy_limit_price,
            target_sell_price=decision.target_sell_price,
            stop_loss_price=decision.stop_loss_price,
            max_holding_time_minutes=decision.max_holding_time_minutes,
            pattern_reasoning=decision.pattern_reasoning,
            thesis_continuity_flag=decision.thesis_continuity_flag,
        )
        realized_pnl_all = await repo.realized_pnl_today_all_tickers(
            session, settings.watchlist, today
        )
        try:
            outcome = await om.try_enter_position(
                session,
                request.app.state.broker,
                ticker=ticker,
                decision=decision,
                llm_decision_id=llm_decision.id,
                settings=settings,
                today=today,
                realized_pnl_today_all_tickers=realized_pnl_all,
                notifier=get_notifier(),
            )
        except Exception as exc:
            # Surface the real broker/MCP error instead of a bare 500 -- e.g. a
            # rejected order (market closed, insufficient buying power, bad
            # symbol) raises here (see broker_mcp_client.unwrap_tool_result). The
            # whole transaction (including the llm_decision row above) rolls back
            # via session_scope, same as any other exception raised in this block.
            raise HTTPException(
                status_code=502, detail=f"try_enter_position failed: {exc}"
            ) from exc
        llm_decision.acted_on = outcome.opened

    return {
        "ticker": ticker,
        "mode": settings.mode.value,
        "opened": outcome.opened,
        "reason": outcome.reason,
        "trade_id": outcome.trade_id,
        "order_id": outcome.order_id,
        "broker_order_id": outcome.broker_order_id,
    }


@router.post("/kill-switch")
async def kill_switch(request: Request) -> dict:
    """Pauses all scheduled jobs immediately. Does NOT cancel/liquidate existing
    orders on its own -- this stops new activity; handle open positions manually or
    wait for the next scheduled eod_liquidation run (or resume once safe)."""
    request.app.state.scheduler.pause()
    request.app.state.halted = True
    return {"status": "halted"}


@router.post("/resume")
async def resume(request: Request) -> dict:
    request.app.state.scheduler.resume()
    request.app.state.halted = False
    return {"status": "resumed"}
