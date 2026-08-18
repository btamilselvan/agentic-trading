"""Integration tests need a real Postgres (DATABASE_URL env var) -- run migrations
against a throwaway database before running these, e.g.:

    docker run -d --name agentic-trading-test-pg -e POSTGRES_PASSWORD=postgres \\
        -e POSTGRES_DB=agentic_trading -p 55432:5432 postgres:16-alpine
    export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/agentic_trading
    alembic upgrade head
    pytest tests/integration
"""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy import delete

from agentic_trading.state.db import session_scope
from agentic_trading.state.models import Bucket, LlmDecision, Order, TickerDailyState, Trade


async def _truncate_all(session) -> None:
    # Deletion order respects FKs: orders/trades -> llm_decisions -> buckets.
    for model in (Order, Trade, LlmDecision, Bucket, TickerDailyState):
        await session.execute(delete(model))
    await session.commit()


@pytest_asyncio.fixture
async def db_session():
    async with session_scope() as session:
        await _truncate_all(session)
        yield session
        await _truncate_all(session)
