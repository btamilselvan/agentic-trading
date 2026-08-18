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
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_trading.config import get_settings
from agentic_trading.execution.broker_mcp_client import McpBrokerClient
from agentic_trading.market_data import robinhood_client as rh_market
from agentic_trading.scheduler import run_poll_cycle
from agentic_trading.state.db import get_session
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
