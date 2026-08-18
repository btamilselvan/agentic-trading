"""Scheduling: the 5-minute poll loop, the order-management sweep, and the
end-of-day liquidation job (spec sections 3.1 / 3.3).

Job bodies (`run_poll_cycle`, `run_order_management_sweep`, `run_eod_liquidation`)
take their dependencies (broker, llm_client) as arguments rather than importing
concrete implementations themselves, so they're unit-testable with fakes; only
`build_scheduler` wires the real APScheduler + real dependencies together.

Known limitation: trading-day detection is a plain Mon-Fri weekday check, not a real
market-holiday calendar -- acceptable for a first cut, called out here rather than
silently assumed correct.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agentic_trading.alerts.base import Notifier, get_notifier
from agentic_trading.config import Settings, TradingMode, get_settings
from agentic_trading.execution import order_manager as om
from agentic_trading.execution.broker_mcp_client import BrokerExecutionClient
from agentic_trading.execution.guardrails import check_daily_trade_cap, check_position_cap
from agentic_trading.llm.base import LLMClient
from agentic_trading.llm.schema import TickerState
from agentic_trading.market_data import robinhood_client as rh
from agentic_trading.market_data.bucket_builder import build_bucket
from agentic_trading.state import repository as repo
from agentic_trading.state.db import session_scope
from agentic_trading.state.models import TradingModeEnum

logger = logging.getLogger(__name__)

# A week of 5-min bars per ticker, used only to compute the RVOL baseline (spec
# 3.1). Refreshed once per process lifetime -- it doesn't need to be re-fetched every
# 5-minute poll, only rebuilt if the process restarts mid-day.
_rvol_lookback_cache: dict[str, list] = {}


async def _get_lookback_bars(ticker: str) -> list:
    if ticker not in _rvol_lookback_cache:
        _rvol_lookback_cache[ticker] = await asyncio.to_thread(
            rh.get_5min_historicals, ticker, span="week"
        )
    return _rvol_lookback_cache[ticker]


async def _realized_pnl_today_all_tickers(session, today) -> float:
    total = 0.0
    for ticker in get_settings().watchlist:
        state = await repo.get_or_create_daily_state(session, ticker, today)
        total += float(state.realized_pnl or 0)
    return total


async def _poll_ticker(
    ticker: str,
    *,
    broker: BrokerExecutionClient,
    llm_client: LLMClient,
    settings: Settings,
    notifier: Notifier,
) -> None:
    # robin_stocks is synchronous -- run it in a worker thread rather than block the
    # event loop this scheduler (and the whole FastAPI app) shares. Left unwrapped,
    # a slow call (or an interactive MFA prompt on first login) freezes everything,
    # including subsequent scheduled poll cycles.
    bars = await asyncio.to_thread(rh.get_5min_historicals, ticker, span="day")
    if not bars:
        return
    latest_bar = bars[-1]
    quote = await asyncio.to_thread(rh.get_quote, ticker)
    lookback = await _get_lookback_bars(ticker)
    bucket_data = build_bucket(latest_bar, quote, lookback)
    today = bucket_data.bucket_start.date()

    rvol_pct = f"{bucket_data.rvol * 100:.2f}%" if bucket_data.rvol is not None else "n/a"
    logger.info("Polling %s: %s (RVOL=%s)", ticker, latest_bar.begins_at, rvol_pct)

    async with session_scope() as session:
        already_recorded = await repo.get_buckets_for_ticker(
            session, ticker, since=bucket_data.bucket_start
        )
        if any(b.bucket_start == bucket_data.bucket_start for b in already_recorded):
            return  # already polled this 5-minute bucket

        bucket = await repo.save_bucket(
            session,
            ticker=bucket_data.ticker,
            bucket_start=bucket_data.bucket_start,
            bucket_end=bucket_data.bucket_end,
            open=bucket_data.open,
            high=bucket_data.high,
            low=bucket_data.low,
            close=bucket_data.close,
            volume=bucket_data.volume,
            est_buy_volume=bucket_data.est_buy_volume,
            est_sell_volume=bucket_data.est_sell_volume,
            bid_price=bucket_data.bid_price,
            ask_price=bucket_data.ask_price,
            bid_size=bucket_data.bid_size,
            ask_size=bucket_data.ask_size,
            spread=bucket_data.spread,
            candle_body=bucket_data.candle_body,
            upper_wick=bucket_data.upper_wick,
            lower_wick=bucket_data.lower_wick,
            rvol=bucket_data.rvol,
        )

        daily_state = await repo.get_or_create_daily_state(session, ticker, today)
        eligible = (
            check_position_cap(
                daily_state.open_positions_count, settings.max_open_positions_per_ticker
            ).allowed
            and check_daily_trade_cap(
                daily_state.completed_trades_count, settings.daily_trade_cap_per_ticker
            ).allowed
        )
        if not eligible:
            logger.info("Skipping LLM call for %s -- not eligible for a new trade today", ticker)
            return

        history = await repo.get_buckets_for_ticker(
            session, ticker, since=datetime.combine(today, time.min, tzinfo=UTC)
        )
        ticker_state = TickerState(
            completed_trades_today=daily_state.completed_trades_count,
            open_positions=daily_state.open_positions_count,
            realized_pnl_today=float(daily_state.realized_pnl or 0),
        )

        decision, prompt, raw = await llm_client.decide(ticker, history, ticker_state)
        logger.debug("llm decision (%s), raw_response (%s)", decision, raw)
        llm_decision = await repo.save_llm_decision(
            session,
            ticker=ticker,
            bucket_id=bucket.id,
            prompt=prompt,
            raw_response=raw,
            decision=decision.decision,
            confidence_score=decision.confidence_score,
            buy_limit_price=decision.buy_limit_price,
            target_sell_price=decision.target_sell_price,
            max_holding_time_minutes=decision.max_holding_time_minutes,
            pattern_reasoning=decision.pattern_reasoning,
        )
        
        logger.info(
            "LLM decision for %s: %s (confidence: %.2f) in bucket %s",
            ticker,
            decision.decision,
            decision.confidence_score,
            bucket_data.bucket_start,
        )

        is_buy_signal = (
            decision.decision == "BUY"
            and decision.confidence_score >= settings.confidence_threshold
        )
        if is_buy_signal:
            # Spec 5: alert on LLM confidence scores -- only for signals that clear
            # the threshold, since a 5-minute poll across a watchlist would otherwise
            # spam a HOLD/low-confidence notification every cycle.
            is_observe_only = settings.mode == TradingMode.OBSERVE
            await notifier.notify(
                "Buying opportunity (observation only)" if is_observe_only else "BUY signal",
                {
                    "ticker": ticker,
                    "confidence_score": decision.confidence_score,
                    "buy_limit_price": decision.buy_limit_price,
                    "target_sell_price": decision.target_sell_price,
                    "pattern_reasoning": decision.pattern_reasoning,
                },
            )

            if is_observe_only:
                # Phase 1: report only -- no broker call of any kind, not even a
                # simulated one. llm_decision.acted_on stays False (its default).
                logger.info(
                    "OBSERVE mode: BUY signal for %s reported, not acted on", ticker
                )
            else:
                realized_pnl_all = await _realized_pnl_today_all_tickers(session, today)
                outcome = await om.try_enter_position(
                    session,
                    broker,
                    ticker=ticker,
                    decision=decision,
                    llm_decision_id=llm_decision.id,
                    settings=settings,
                    today=today,
                    realized_pnl_today_all_tickers=realized_pnl_all,
                    notifier=notifier,
                )
                llm_decision.acted_on = outcome.opened
                if not outcome.opened:
                    logger.info("BUY decision for %s not acted on: %s", ticker, outcome.reason)


def _parse_hhmm(value: str) -> time:
    hour, minute = (int(p) for p in value.split(":"))
    return time(hour, minute)


def _is_within_poll_window(now_local: time, settings: Settings) -> bool:
    """The CronTrigger in build_scheduler only has hour-level granularity (it fires
    every poll_interval_minutes across the relevant *hours*, which is coarser than
    MARKET_OPEN_TIME/EVALUATION_WINDOW_END_TIME's exact HH:MM) -- this is the precise
    check, so a 09:30 open doesn't actually start firing at 09:00.
    """
    window_start = _parse_hhmm(settings.market_open_time)
    window_end = _parse_hhmm(settings.evaluation_window_end_time)
    return window_start <= now_local <= window_end


async def run_poll_cycle(
    *,
    broker: BrokerExecutionClient,
    llm_client: LLMClient,
    settings: Settings | None = None,
    notifier: Notifier | None = None,
    now: datetime | None = None,
    bypass_window: bool = False,
) -> None:
    """One 5-minute poll across the whole watchlist (spec 3.1 + 3.2). Each ticker is
    isolated -- one ticker's failure (bad data, LLM error) must not block the rest.
    `now` is injectable for tests; real callers (the scheduler) always let it default
    to the actual current time. `bypass_window` skips the market-hours check -- only
    the on-demand /poll-cycle endpoint sets it, for manually triggering a cycle
    outside 09:30-11:30 (e.g. to debug); the scheduled job never does.
    """
    settings = settings or get_settings()
    notifier = notifier or get_notifier()

    tz = ZoneInfo(settings.timezone)
    now_local = (now.astimezone(tz) if now else datetime.now(tz)).time()
    if not bypass_window and not _is_within_poll_window(now_local, settings):
        logger.debug(
            "Outside the %s-%s poll window (now=%s local) -- skipping this cycle",
            settings.market_open_time,
            settings.evaluation_window_end_time,
            now_local,
        )
        return

    for ticker in settings.watchlist:
        try:
            await _poll_ticker(
                ticker, broker=broker, llm_client=llm_client, settings=settings, notifier=notifier
            )
        except Exception:
            logger.exception("Poll cycle failed for %s", ticker)


async def run_order_management_sweep(
    *,
    broker: BrokerExecutionClient,
    settings: Settings | None = None,
    notifier: Notifier | None = None,
) -> None:
    """Runs between poll cycles too (fills can happen anytime, not just at the
    5-minute mark): detects buy/sell fills and enforces the order-timeout guardrail.
    """
    settings = settings or get_settings()
    notifier = notifier or get_notifier()
    now = datetime.now(UTC)
    async with session_scope() as session:
        await om.poll_pending_buy_orders(
            session,
            broker,
            now=now,
            order_timeout_minutes=settings.order_timeout_minutes,
            notifier=notifier,
        )
        await om.poll_pending_sell_orders(session, broker, notifier=notifier)


async def run_eod_liquidation(
    *,
    broker: BrokerExecutionClient,
    settings: Settings | None = None,
    notifier: Notifier | None = None,
) -> None:
    """Spec 4 / 3.3: Day-End Liquidation Rule. Runs once at eod_liquidation_time --
    cancels every pending order and force-sells every open position at the current
    bid so nothing carries overnight risk.
    """
    settings = settings or get_settings()
    notifier = notifier or get_notifier()
    now = datetime.now(UTC)
    liquidation_prices: dict[str, float] = {}
    async with session_scope() as session:
        for trade in await repo.get_open_trades(session):
            quote = await asyncio.to_thread(rh.get_quote, trade.ticker)
            if quote and quote.bid_price is not None:
                liquidation_prices[trade.ticker] = quote.bid_price
        await om.liquidate_all_open_positions(
            session,
            broker,
            now=now,
            liquidation_prices=liquidation_prices,
            mode=TradingModeEnum(settings.mode.value),
            notifier=notifier,
        )


def build_scheduler(*, broker: BrokerExecutionClient, llm_client: LLMClient) -> AsyncIOScheduler:
    """Wires the real APScheduler jobs. Called once from main.py's FastAPI lifespan."""
    settings = get_settings()
    tz = ZoneInfo(settings.timezone)
    # Hour-only granularity here is deliberately coarse -- it just needs to bracket
    # the real window; run_poll_cycle's own _is_within_poll_window check enforces
    # the exact MARKET_OPEN_TIME/EVALUATION_WINDOW_END_TIME minute precision.
    open_hour = _parse_hhmm(settings.market_open_time).hour
    window_end_hour = _parse_hhmm(settings.evaluation_window_end_time).hour
    eod_hour, eod_minute = (int(p) for p in settings.eod_liquidation_time.split(":"))

    scheduler = AsyncIOScheduler(timezone=tz)

    scheduler.add_job(
        run_poll_cycle,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=f"{open_hour}-{window_end_hour}",
            minute=f"*/{settings.poll_interval_minutes}",
            timezone=tz,
        ),
        kwargs={"broker": broker, "llm_client": llm_client},
        id="poll_cycle",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_order_management_sweep,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=f"{open_hour}-{eod_hour}",
            minute="*",
            timezone=tz,
        ),
        kwargs={"broker": broker},
        id="order_management_sweep",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_eod_liquidation,
        trigger=CronTrigger(
            day_of_week="mon-fri", hour=eod_hour, minute=eod_minute, timezone=tz
        ),
        kwargs={"broker": broker},
        id="eod_liquidation",
        max_instances=1,
    )

    logger.info(
        "Scheduler configured: poll every %dmin %s-%s, sweep every minute until %s, "
        "liquidation at %s (%s), watchlist=%s",
        settings.poll_interval_minutes,
        settings.market_open_time,
        settings.evaluation_window_end_time,
        settings.eod_liquidation_time,
        settings.eod_liquidation_time,
        settings.timezone,
        settings.watchlist,
    )
    if settings.mode == TradingMode.LIVE:
        logger.warning("MODE=LIVE -- this scheduler will place REAL orders with real money.")
    return scheduler
