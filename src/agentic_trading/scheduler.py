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
from agentic_trading.execution.invalidation import compute_trailing_stop, evaluate_exit_guardrails
from agentic_trading.llm.base import LLMClient
from agentic_trading.llm.schema import TickerState, TradeDecision
from agentic_trading.market_data import robinhood_client as rh
from agentic_trading.market_data.bucket_builder import (
    MarketContext,
    build_bucket,
    build_market_context,
    detect_vwap_cross,
    find_prior_close,
    rsi_centerline_cross,
    to_float,
)
from agentic_trading.state import repository as repo
from agentic_trading.state.db import session_scope
from agentic_trading.state.models import Trade, TradeStatus, TradingModeEnum
from agentic_trading.state.ticker_state_store import (
    DecisionLogEntry,
    TickerEvaluationState,
    TickerStateStore,
    get_ticker_state_store,
)

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


# Float shares barely move intraday, so unlike bars/quotes this is fetched once per
# ticker per process lifetime rather than every 5-minute poll -- same treatment as
# _rvol_lookback_cache above.
_float_shares_cache: dict[str, int | None] = {}


async def _get_float_shares(ticker: str) -> int | None:
    if ticker not in _float_shares_cache:
        try:
            _float_shares_cache[ticker] = await asyncio.to_thread(rh.get_float_shares, ticker)
        except Exception:
            logger.exception(
                "Failed to fetch float shares for %s -- proceeding without it", ticker
            )
            _float_shares_cache[ticker] = None
    return _float_shares_cache[ticker]


async def _get_latest_news(ticker: str) -> rh.NewsItem | None:
    """Unlike float, re-fetched every poll cycle -- a news story can land at any
    point in the session, unlike float which is effectively static intraday.
    Isolated with its own try/except so a news-feed hiccup degrades to "no catalyst
    context" for this ticker this cycle rather than blocking the poll.
    """
    try:
        return await asyncio.to_thread(rh.get_latest_news, ticker)
    except Exception:
        logger.exception("Failed to fetch news for %s -- proceeding without it", ticker)
        return None


async def _get_market_context(settings: Settings) -> MarketContext | None:
    """Fetched once per poll cycle (not once per ticker -- it's the same broad-market
    snapshot for every ticker being polled this cycle), from settings.market_
    benchmark_ticker (SPY by default; empty string disables it). Isolated with its
    own try/except so a benchmark-fetch hiccup degrades to "no market context" for
    this cycle rather than taking down the whole poll cycle -- same isolation
    principle as each ticker's own try/except in run_poll_cycle.
    """
    benchmark = settings.market_benchmark_ticker
    if not benchmark:
        return None
    try:
        bars_today = await asyncio.to_thread(rh.get_5min_historicals, benchmark, span="day")
        lookback = await _get_lookback_bars(benchmark)
    except Exception:
        logger.exception(
            "Failed to fetch market context (%s) -- proceeding without it", benchmark
        )
        return None
    return build_market_context(benchmark, bars_today, lookback)


async def _realized_pnl_today_all_tickers(session, today) -> float:
    return await repo.realized_pnl_today_all_tickers(session, get_settings().watchlist, today)


def _build_ticker_state(
    *,
    daily_state,
    prior_close: float | None,
    market_context: MarketContext | None,
    news: rh.NewsItem | None,
    float_shares: int | None,
    redis_state: TickerEvaluationState,
) -> TickerState:
    """Assembles the TickerState DTO shared by both _poll_ticker branches (entry
    evaluation and open-position management) -- the same-day counters/market/
    catalyst fields are identical either way; only the continuity fields
    (status/active_thesis/decision_history/...) come from a different source
    (Redis, per requirements.md section 8) depending on which branch calls this.
    """
    return TickerState(
        completed_trades_today=daily_state.completed_trades_count,
        open_positions=daily_state.open_positions_count,
        realized_pnl_today=float(daily_state.realized_pnl or 0),
        prior_close=prior_close,
        market_benchmark_ticker=market_context.ticker if market_context else None,
        market_change_pct=market_context.change_pct if market_context else None,
        market_vwap_deviation_pct=(market_context.vwap_deviation_pct if market_context else None),
        market_range_pct=market_context.range_pct if market_context else None,
        news_headline=news.title if news else None,
        news_summary=news.summary if news else None,
        news_published_at=news.published_at if news else None,
        float_shares=float_shares,
        status=redis_state.status,
        active_thesis=redis_state.active_thesis,
        initial_entry_price=redis_state.initial_entry_price,
        current_target_price=redis_state.target_price,
        current_stop_loss=redis_state.stop_loss,
        decision_history=redis_state.decision_history,
    )


def _exit_price_from_quote(quote: rh.Quote | None, fallback: float) -> float:
    """A marketable price to exit at -- the current bid if a fresh quote is
    available this cycle, else the latest bucket's close as a same-cycle
    approximation (better than skipping a forced exit outright; EOD liquidation's
    own liquidation_prices has the luxury of a dedicated quote fetch per open
    trade, but a forced intracycle exit shouldn't wait for that).
    """
    if quote is not None and quote.bid_price is not None:
        return quote.bid_price
    return fallback


async def _manage_open_position(
    session,
    ticker: str,
    trade: Trade,
    *,
    broker: BrokerExecutionClient,
    llm_client: LLMClient,
    settings: Settings,
    notifier: Notifier,
    ticker_state_store: TickerStateStore,
    bucket_data,
    history: list,
    quote: rh.Quote | None,
    market_context: MarketContext | None,
    redis_state: TickerEvaluationState,
) -> None:
    """Phase 3 (requirements.md section 8): re-evaluates an already-OPEN trade every
    poll cycle instead of leaving it to passively wait on its resting sell order.

    Two layers, in order:
    1. Code-enforced invalidation (execution.invalidation.evaluate_exit_guardrails)
       -- stop-loss breach or momentum break force an immediate exit with NO LLM
       call at all (fast, deterministic, and cannot be overridden by an LLM HOLD).
    2. If clear, the LLM is consulted for thesis_continuity_flag / a possible SELL
       (the one invalidation criterion -- a negative catalyst -- that has no
       code-side check) and for optional trailing stop/target levels, applied via
       execution.invalidation.compute_trailing_stop and, if
       settings.trailing_stop_enabled, execution.order_manager.apply_trailing_stop.
    """
    today = bucket_data.bucket_start.date()
    # Redis is ephemeral working memory (see state.ticker_state_store's module
    # docstring) -- if it was lost or expired mid-position, Postgres's Trade row
    # is still the source of truth for entry/target/stop, so rebuild from there
    # rather than silently treating this ticker as FLAT while a real position is
    # still open.
    if redis_state.status != "IN_POSITION":
        redis_state = TickerEvaluationState(
            ticker=ticker,
            trade_date=today,
            status="IN_POSITION",
            active_thesis=redis_state.active_thesis or "(recovered -- no thesis on record)",
            initial_entry_price=to_float(trade.entry_price),
            target_price=to_float(trade.target_sell_price),
            stop_loss=to_float(trade.stop_loss_price),
        )

    previous_bucket = history[-2] if len(history) >= 2 else None
    rsi_cross = rsi_centerline_cross(
        to_float(previous_bucket.rsi) if previous_bucket else None, bucket_data.rsi
    )
    vwap_cross = detect_vwap_cross(
        to_float(previous_bucket.close) if previous_bucket else None,
        to_float(previous_bucket.vwap) if previous_bucket else None,
        bucket_data.close,
        bucket_data.vwap,
    )

    exit_check = evaluate_exit_guardrails(
        current_price=bucket_data.close,
        stop_loss=to_float(trade.stop_loss_price),
        rsi_centerline_cross=rsi_cross,
        vwap_cross=vwap_cross,
    )
    if exit_check.should_exit:
        logger.info(
            "Forced exit check tripped for %s: %s (%s) -- exiting without an LLM call",
            ticker,
            exit_check.reason,
            exit_check.detail,
        )
        exit_price = _exit_price_from_quote(quote, bucket_data.close)
        await om.try_exit_position_early(
            session,
            broker,
            trade=trade,
            exit_price=exit_price,
            exit_reason=exit_check.reason,
            notifier=notifier,
        )
        await _record_position_decision(
            ticker_state_store,
            redis_state,
            bucket_start=bucket_data.bucket_start,
            decision="SELL",
            confidence_score=1.0,
            thesis_continuity_flag=False,
            pattern_reasoning=f"Forced exit ({exit_check.reason}): {exit_check.detail}",
            trade_closed=trade.status == TradeStatus.CLOSED,
            max_history=settings.decision_history_length,
        )
        return

    daily_state = await repo.get_or_create_daily_state(session, ticker, today)
    news = await _get_latest_news(ticker)
    float_shares = await _get_float_shares(ticker)
    ticker_state = _build_ticker_state(
        daily_state=daily_state,
        prior_close=None,  # not relevant once already in a position today
        market_context=market_context,
        news=news,
        float_shares=float_shares,
        redis_state=redis_state,
    )

    decision, prompt, raw = await llm_client.decide(ticker, history, ticker_state)
    logger.info("llm continuity decision for %s (%s): %s", ticker, trade.id, decision)
    await repo.save_llm_decision(
        session,
        ticker=ticker,
        bucket_id=None,
        prompt=prompt,
        raw_response=raw,
        decision=decision.decision,
        confidence_score=decision.confidence_score,
        buy_limit_price=decision.buy_limit_price,
        target_sell_price=decision.target_sell_price,
        stop_loss_price=decision.stop_loss_price,
        max_holding_time_minutes=decision.max_holding_time_minutes,
        pattern_reasoning=decision.pattern_reasoning,
        thesis_continuity_flag=decision.thesis_continuity_flag,
    )

    llm_judged_exit = decision.decision == "SELL" or not decision.thesis_continuity_flag
    if llm_judged_exit:
        exit_price = _exit_price_from_quote(quote, bucket_data.close)
        await om.try_exit_position_early(
            session,
            broker,
            trade=trade,
            exit_price=exit_price,
            exit_reason="LLM_THESIS_BREAK",
            notifier=notifier,
        )
        await _record_position_decision(
            ticker_state_store,
            redis_state,
            bucket_start=bucket_data.bucket_start,
            decision=decision.decision,
            confidence_score=decision.confidence_score,
            thesis_continuity_flag=decision.thesis_continuity_flag,
            pattern_reasoning=decision.pattern_reasoning,
            trade_closed=trade.status == TradeStatus.CLOSED,
            max_history=settings.decision_history_length,
        )
        return

    new_stop = to_float(trade.stop_loss_price)
    new_target = to_float(trade.target_sell_price)
    if settings.trailing_stop_enabled:
        new_stop, new_target = compute_trailing_stop(
            current_stop_loss=new_stop,
            current_target=new_target,
            proposed_stop_loss=decision.stop_loss_price,
            proposed_target=decision.target_sell_price,
        )
        await om.apply_trailing_stop(
            session,
            broker,
            trade=trade,
            new_target=new_target,
            new_stop=new_stop,
            notifier=notifier,
        )

    updated_state = redis_state.with_decision_appended(
        DecisionLogEntry(
            bucket_start=bucket_data.bucket_start,
            decision=decision.decision,
            confidence_score=decision.confidence_score,
            thesis_continuity_flag=decision.thesis_continuity_flag,
            pattern_reasoning=decision.pattern_reasoning,
        ),
        max_history=settings.decision_history_length,
    )
    updated_state = updated_state.model_copy(
        update={"status": "IN_POSITION", "target_price": new_target, "stop_loss": new_stop}
    )
    await ticker_state_store.save(updated_state)


async def _record_position_decision(
    ticker_state_store: TickerStateStore,
    redis_state: TickerEvaluationState,
    *,
    bucket_start: datetime,
    decision: str,
    confidence_score: float,
    thesis_continuity_flag: bool,
    pattern_reasoning: str,
    trade_closed: bool,
    max_history: int,
) -> None:
    """Shared tail of both exit paths in _manage_open_position: records the
    decision into the ticker's history and either clears its Redis state (the
    trade actually closed this cycle -- instant fill, e.g. DryRunBrokerClient) or
    keeps it IN_POSITION (an exit order is now resting but hasn't filled yet --
    LIVE mode commonly won't fill instantly; the next cycle's exit-guardrail check
    will keep re-pricing/re-attempting the exit at the then-current bid until it
    does).
    """
    if trade_closed:
        await ticker_state_store.clear(redis_state.ticker, redis_state.trade_date)
        return
    updated_state = redis_state.with_decision_appended(
        DecisionLogEntry(
            bucket_start=bucket_start,
            decision=decision,
            confidence_score=confidence_score,
            thesis_continuity_flag=thesis_continuity_flag,
            pattern_reasoning=pattern_reasoning,
        ),
        max_history=max_history,
    )
    updated_state = updated_state.model_copy(update={"status": "IN_POSITION"})
    await ticker_state_store.save(updated_state)


async def _record_entry_decision(
    ticker_state_store: TickerStateStore,
    redis_state: TickerEvaluationState,
    *,
    bucket_start: datetime,
    decision: TradeDecision,
    opened: bool,
    max_history: int,
) -> None:
    """Tail of the entry-evaluation branch (a ticker with no open trade yet):
    records this cycle's decision into the ticker's Redis history so a future
    cycle's HYSTERESIS check (llm/prompt.py) has continuity to work from even
    before any position exists -- and, only if a BUY actually opened a real
    position this cycle (`opened`), seeds the position-tracking fields
    (initial_entry_price/target_price/stop_loss/active_thesis) that
    _manage_open_position relies on from the next cycle onward.
    """
    updated_state = redis_state.with_decision_appended(
        DecisionLogEntry(
            bucket_start=bucket_start,
            decision=decision.decision,
            confidence_score=decision.confidence_score,
            thesis_continuity_flag=decision.thesis_continuity_flag,
            pattern_reasoning=decision.pattern_reasoning,
        ),
        max_history=max_history,
    )
    if opened:
        updated_state = updated_state.model_copy(
            update={
                "status": "IN_POSITION",
                "active_thesis": decision.pattern_reasoning,
                "initial_entry_price": decision.buy_limit_price,
                "target_price": decision.target_sell_price,
                "stop_loss": decision.stop_loss_price,
            }
        )
    else:
        # Covers HOLD, a BUY that didn't clear the confidence threshold, and a
        # BUY that was blocked by a guardrail or reported-only in OBSERVE mode --
        # none of these actually opened a position, so status stays HOLD, not
        # IN_POSITION/BUY (see TickerStatus's docstring in state.ticker_state_store).
        updated_state = updated_state.model_copy(update={"status": "HOLD"})
    await ticker_state_store.save(updated_state)


async def _poll_ticker(
    ticker: str,
    *,
    broker: BrokerExecutionClient,
    llm_client: LLMClient,
    settings: Settings,
    notifier: Notifier,
    ticker_state_store: TickerStateStore,
    market_context: MarketContext | None = None,
) -> None:
    # robin_stocks is synchronous -- run it in a worker thread rather than block the
    # event loop this scheduler (and the whole FastAPI app) shares. Left unwrapped,
    # a slow call (or an interactive MFA prompt on first login) freezes everything,
    # including subsequent scheduled poll cycles.

    # get 5 mins bars for today (begin time, open/close price, high/low and volume)
    bars = await asyncio.to_thread(rh.get_5min_historicals, ticker, span="day")
    if not bars:
        return
    latest_bar = bars[-1]

    #get current quote (bid/ask price + size)
    quote = await asyncio.to_thread(rh.get_quote, ticker)

    #get the bars for the last one week
    lookback = await _get_lookback_bars(ticker)

    #current metric bucket with RVOL + session VWAP + RSI (today_bars=bars gives
    #VWAP/RSI the full open-through-now series they need to accumulate correctly)
    bucket_data = build_bucket(
        latest_bar, quote, lookback, today_bars=bars, rsi_period=settings.rsi_period
    )
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
            book_imbalance=bucket_data.book_imbalance,
            candle_body=bucket_data.candle_body,
            upper_wick=bucket_data.upper_wick,
            lower_wick=bucket_data.lower_wick,
            rvol=bucket_data.rvol,
            vwap=bucket_data.vwap,
            rsi=bucket_data.rsi,
        )
        # Full day-so-far history -- needed by both branches below (the LLM call
        # either way, plus _manage_open_position's momentum-cross check).
        history = await repo.get_buckets_for_ticker(
            session, ticker, since=datetime.combine(today, time.min, tzinfo=UTC)
        )
        redis_state = await ticker_state_store.get(ticker, today) or TickerEvaluationState.fresh(
            ticker, today
        )

        # An already-OPEN trade takes a completely different path (requirements.md
        # section 8) -- checked BEFORE the entry-eligibility guardrails below,
        # since an open position is exactly what makes check_position_cap block
        # entry-eligibility; that block must not also swallow position management.
        open_trade = await repo.get_open_trade_for_ticker(session, ticker)
        if open_trade is not None:
            await _manage_open_position(
                session,
                ticker,
                open_trade,
                broker=broker,
                llm_client=llm_client,
                settings=settings,
                notifier=notifier,
                ticker_state_store=ticker_state_store,
                bucket_data=bucket_data,
                history=history,
                quote=quote,
                market_context=market_context,
                redis_state=redis_state,
            )
            return

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

        # Catalyst context (requirements.md section 6) -- fetched here, not
        # up-front like market_context, since it's per-ticker (no benefit to
        # sharing across the watchlist) and only worth the extra calls once we
        # know this ticker is actually eligible for a new trade this cycle.
        news = await _get_latest_news(ticker)
        float_shares = await _get_float_shares(ticker)
        ticker_state = _build_ticker_state(
            daily_state=daily_state,
            prior_close=find_prior_close(latest_bar, lookback),
            market_context=market_context,
            news=news,
            float_shares=float_shares,
            redis_state=redis_state,
        )

        # Get insights from LLM using today's Metrics
        decision, prompt, raw = await llm_client.decide(ticker, history, ticker_state)
        logger.info("llm decision (%s), raw_response (%s)", decision, raw)
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
            stop_loss_price=decision.stop_loss_price,
            max_holding_time_minutes=decision.max_holding_time_minutes,
            pattern_reasoning=decision.pattern_reasoning,
            thesis_continuity_flag=decision.thesis_continuity_flag,
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
        opened = False
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
                # DryRunBrokerClient fills both legs instantly, so a BUY can open
                # AND close again within this same call (paired sell fills right
                # away) -- opened=True alone doesn't mean still holding by now.
                # Re-check the actual DB state rather than trusting the outcome
                # flag, so Redis doesn't end up claiming IN_POSITION for a trade
                # that's already round-tripped closed.
                opened = outcome.opened and (
                    await repo.get_open_trade_for_ticker(session, ticker) is not None
                )

        await _record_entry_decision(
            ticker_state_store,
            redis_state,
            bucket_start=bucket_data.bucket_start,
            decision=decision,
            opened=opened,
            max_history=settings.decision_history_length,
        )


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
    ticker_state_store: TickerStateStore | None = None,
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
    ticker_state_store = ticker_state_store or get_ticker_state_store()

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

    market_context = await _get_market_context(settings)

    for ticker in settings.watchlist:
        try:
            await _poll_ticker(
                ticker,
                broker=broker,
                llm_client=llm_client,
                settings=settings,
                notifier=notifier,
                ticker_state_store=ticker_state_store,
                market_context=market_context,
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

    retry_missing_paired_sells runs after poll_pending_buy_orders on every tick too
    -- covers a buy fill whose paired sell placement failed (see that function's
    docstring), so it doesn't sit unmanaged until EOD liquidation.
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
        await om.retry_missing_paired_sells(session, broker, notifier=notifier)


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
