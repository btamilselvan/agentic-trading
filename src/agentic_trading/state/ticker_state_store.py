"""Redis-backed per-ticker evaluation state -- Phase 3's "working memory" that
carries BUY/HOLD/IN_POSITION continuity (active thesis, stop/target levels,
recent decision log) across 5-minute poll cycles, distinct from Postgres's
durable audit trail (buckets/llm_decisions/orders/trades -- see state/models.py).

Deliberately ephemeral: keyed by (ticker, trade_date) and TTL'd (see
config.ticker_state_ttl_hours) so state can never silently leak into a later
trading session even if explicit clearing (on trade close / EOD) is missed --
spec requires entering/exiting positions within the same session only.

TickerStateStore is a small Protocol (same pattern as llm/base.py's LLMClient,
alerts/base.py's Notifier) so callers (scheduler.py) don't depend on Redis
directly. InMemoryTickerStateStore is the fake used by unit tests and anywhere
Redis isn't available -- same role DryRunBrokerClient plays for the broker.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Literal, Protocol

from pydantic import BaseModel, Field

TickerStatus = Literal["FLAT", "HOLD", "BUY", "IN_POSITION"]


class DecisionLogEntry(BaseModel):
    """One past cycle's decision, replayed into the prompt for continuity
    (requirements.md section 8: "append the previous 3 to 5 decision logs").
    Deliberately a plain string for `decision` rather than importing
    llm.schema's literal -- this module sits below llm/ in the dependency
    graph (llm/prompt.py reads TickerEvaluationState, not the other way
    around), so it stays decoupled from the LLM contract's exact shape.
    """

    bucket_start: datetime
    decision: str  # "BUY" | "HOLD" | "SELL"
    confidence_score: float
    thesis_continuity_flag: bool
    pattern_reasoning: str = ""


class TickerEvaluationState(BaseModel):
    """Per-ticker, per-day working state (requirements.md section 8).
    `active_thesis`/`initial_entry_price`/`target_price`/`stop_loss` are only
    populated once a position is open (status == IN_POSITION), but live on the
    same model rather than a separate type since a HOLD/BUY-pending ticker can
    still be carrying a thesis forward from its own decision history.
    """

    ticker: str
    trade_date: date
    status: TickerStatus = "FLAT"
    active_thesis: str | None = None
    initial_entry_price: float | None = None
    target_price: float | None = None
    stop_loss: float | None = None
    thesis_continuity_flag: bool = True
    decision_history: list[DecisionLogEntry] = Field(default_factory=list)
    updated_at: datetime | None = None

    @classmethod
    def fresh(cls, ticker: str, trade_date: date) -> TickerEvaluationState:
        """Default state for a ticker Redis has never seen (or that expired) --
        callers should fall back to this rather than treating a cache miss as
        an error."""
        return cls(ticker=ticker, trade_date=trade_date)

    def with_decision_appended(
        self, entry: DecisionLogEntry, *, max_history: int
    ) -> TickerEvaluationState:
        """Returns a copy with `entry` appended and trimmed to the most recent
        `max_history` entries -- doesn't mutate in place, so callers always
        pass the returned value to save() explicitly."""
        history = [*self.decision_history, entry][-max_history:]
        return self.model_copy(update={"decision_history": history})


def _redis_key(ticker: str, trade_date: date) -> str:
    return f"ticker_state:{ticker}:{trade_date.isoformat()}"


class TickerStateStore(Protocol):
    async def get(self, ticker: str, trade_date: date) -> TickerEvaluationState | None:
        """None means "no state on record" (never seen today, or expired) --
        callers should treat that the same as TickerEvaluationState.fresh(...),
        not as an error."""
        ...

    async def save(self, state: TickerEvaluationState) -> None: ...

    async def clear(self, ticker: str, trade_date: date) -> None:
        """Called on trade close / EOD liquidation so a ticker that re-enters
        later the same day doesn't inherit a closed trade's stale thesis --
        belt-and-suspenders alongside the TTL."""
        ...


class RedisTickerStateStore:
    """Real implementation, backed by redis.asyncio. One JSON value per
    (ticker, trade_date) key; TTL refreshed on every save so an actively
    traded ticker's state doesn't expire mid-session.
    """

    def __init__(self, redis_url: str, ttl_hours: int) -> None:
        # Local import: keeps `redis` from being imported at module load for
        # every caller of this file (e.g. tests that only need the Protocol /
        # InMemoryTickerStateStore) -- mirrors ollama_client.py/webhook
        # deferring their own network client imports similarly.
        import redis.asyncio as redis

        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl_hours = ttl_hours

    async def get(self, ticker: str, trade_date: date) -> TickerEvaluationState | None:
        raw = await self._client.get(_redis_key(ticker, trade_date))
        if raw is None:
            return None
        return TickerEvaluationState.model_validate_json(raw)

    async def save(self, state: TickerEvaluationState) -> None:
        state = state.model_copy(update={"updated_at": datetime.now(UTC)})
        await self._client.set(
            _redis_key(state.ticker, state.trade_date),
            state.model_dump_json(),
            ex=timedelta(hours=self._ttl_hours),
        )

    async def clear(self, ticker: str, trade_date: date) -> None:
        await self._client.delete(_redis_key(ticker, trade_date))


class InMemoryTickerStateStore:
    """Test double implementing the same Protocol as RedisTickerStateStore --
    no network. Used by unit tests, and anywhere a real Redis isn't available.
    """

    def __init__(self) -> None:
        self._data: dict[str, TickerEvaluationState] = {}

    async def get(self, ticker: str, trade_date: date) -> TickerEvaluationState | None:
        return self._data.get(_redis_key(ticker, trade_date))

    async def save(self, state: TickerEvaluationState) -> None:
        state = state.model_copy(update={"updated_at": datetime.now(UTC)})
        self._data[_redis_key(state.ticker, state.trade_date)] = state

    async def clear(self, ticker: str, trade_date: date) -> None:
        self._data.pop(_redis_key(ticker, trade_date), None)


@lru_cache
def get_ticker_state_store() -> TickerStateStore:
    """Provider selection point, same shape as llm/base.py's get_llm_client()
    and alerts/base.py's get_notifier() -- today there's only one real
    implementation (Redis), but callers depend on the Protocol, not this
    function's return type, so swapping backends later doesn't ripple out.
    lru_cache'd like state/db.py's engine -- one client per process lifetime.
    """
    from agentic_trading.config import get_settings

    settings = get_settings()
    return RedisTickerStateStore(settings.redis_url, settings.ticker_state_ttl_hours)
