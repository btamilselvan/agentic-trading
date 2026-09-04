"""Central logging setup.

Third-party libraries in this stack (httpx/httpx2 for the MCP + Ollama/webhook HTTP
calls, robin_stocks, apscheduler, uvicorn, ...) log plenty of their own INFO-level
noise -- most visibly httpx's `logging.getLogger("httpx")` in `_client.py`, which
logs one line per HTTP request/response. That's useful for debugging but drowns out
this project's own log lines in the console during normal operation.

Split the two: everything (our logs + all third-party noise), DEBUG and up, goes to a
rotating file under LOG_DIR so it's still available if something needs debugging,
while the console only shows INFO-and-up records from loggers under the
`agentic_trading` package -- i.e. the lines this codebase actually emits.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(filename)s %(lineno)d] [Thread-%(thread)d] %(message)s"

# Package name of this project's own loggers (every module does
# `logger = logging.getLogger(__name__)`, so they all fall under this prefix).
_APP_LOGGER_PREFIX = __name__.rsplit(".", 1)[0]


class _AppOnlyFilter(logging.Filter):
    """Passes only log records from this project's own loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == _APP_LOGGER_PREFIX or record.name.startswith(_APP_LOGGER_PREFIX + ".")


def configure_logging() -> None:
    """Wire up root logging: everything to a file, only our own logs to the console.

    Idempotent -- safe to call more than once (e.g. under `--reload`), since it clears
    any handlers it previously attached to the root logger first.
    """
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    root = logging.getLogger()
    # DEBUG here, not INFO -- the level filtering actually happens per-handler below.
    # A root level of INFO would drop DEBUG records before they ever reach a handler.
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_AppOnlyFilter())
    root.addHandler(console_handler)
