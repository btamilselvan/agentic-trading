"""add PAPER_TRADING to tradingmodeenum

TradingMode.OBSERVE (report-only, zero order interaction) was renamed to
TradingMode.PAPER_TRADING and now runs the exact same simulated buy/sell/
trailing-stop order lifecycle DRY_RUN does -- so, like DRY_RUN, its Order/Trade
rows need a valid TradingModeEnum member to tag themselves with
(`orders.mode` is a NOT NULL Postgres enum column). This was actually a
pre-existing latent bug for the old OBSERVE mode too: TradingModeEnum only ever
had DRY_RUN/LIVE, so anything constructing TradingModeEnum(settings.mode.value)
under MODE=OBSERVE (e.g. POST /orders/manual-entry) would raise ValueError.

Revision ID: 0006_add_paper_trading_mode
Revises: 0005_add_stop_loss_exit
Create Date: 2026-08-24

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006_add_paper_trading_mode"
down_revision: str | None = "0005_add_stop_loss_exit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Same ADD VALUE-inside-a-transaction pattern as 0005's SELL addition to
    # decisiontype -- fine as long as the new value isn't used in this same
    # transaction, which it isn't here.
    op.execute("ALTER TYPE tradingmodeenum ADD VALUE IF NOT EXISTS 'PAPER_TRADING'")


def downgrade() -> None:
    # Removing 'PAPER_TRADING' from tradingmodeenum is deliberately not
    # implemented -- Postgres has no "ALTER TYPE ... DROP VALUE", only a full
    # type rebuild, and that would fail outright if any orders row already has
    # mode='PAPER_TRADING'. Same tradeoff as 0005's SELL downgrade.
    pass
