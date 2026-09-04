"""FastAPI app factory. Wires the broker/LLM implementations chosen by config (spec
sections 2 + 6) into the scheduler on startup, and tears the scheduler down cleanly
on shutdown.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_trading.api.robinhood_oauth import router as robinhood_oauth_router
from agentic_trading.api.routes import router
from agentic_trading.config import TradingMode, get_settings
from agentic_trading.execution.broker_mcp_client import BrokerExecutionClient, McpBrokerClient
from agentic_trading.execution.order_manager import DryRunBrokerClient
from agentic_trading.llm.base import get_llm_client
from agentic_trading.logging_config import configure_logging
from agentic_trading.scheduler import build_scheduler

configure_logging()
logger = logging.getLogger(__name__)


def build_broker() -> BrokerExecutionClient:
    """MODE=LIVE talks to the real Robinhood Trading MCP (real money, in the
    isolated Agentic account). DRY_RUN and PAPER_TRADING both get the in-memory
    DryRunBrokerClient -- scheduler.py runs the identical order-entry/management
    path for both, so this simulator is genuinely exercised, not just wired to
    satisfy build_scheduler's signature.
    """
    settings = get_settings()
    if settings.mode == TradingMode.LIVE:
        return McpBrokerClient()
    return DryRunBrokerClient()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    broker = build_broker()
    llm_client = get_llm_client()
    scheduler = build_scheduler(broker=broker, llm_client=llm_client)
    scheduler.start()

    app.state.scheduler = scheduler
    app.state.broker = broker
    app.state.llm_client = llm_client
    app.state.halted = False

    logger.info(
        "agentic-trading started (mode=%s, llm_provider=%s, watchlist=%s)",
        settings.mode.value,
        settings.llm_provider.value,
        settings.watchlist,
    )
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        logger.info("agentic-trading stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic Trading", lifespan=lifespan)
    app.include_router(router)
    app.include_router(robinhood_oauth_router)
    return app


app = create_app()
