"""add stateful exit management columns (Phase 3, requirements.md section 8)

Adds the durable (Postgres-side) half of Phase 3's stop-loss/trailing-target
tracking -- Redis (state.ticker_state_store) holds the richer ephemeral working
state (active thesis, decision history), but the protective stop level and why a
trade eventually closed belong in the audit trail too, same as target_sell_price
already does.

- trades.stop_loss_price: the (possibly trailed-up) protective exit level.
- trades.exit_reason: why the trade closed -- TARGET_HIT/STOP_LOSS/MOMENTUM_BREAK/
  CATALYST/TIMEOUT/EOD/LLM_THESIS_BREAK. Plain string, not a Postgres enum -- this
  is descriptive audit metadata, not something enforced/validated like decision.
- llm_decisions.stop_loss_price / .thesis_continuity_flag: the new TradeDecision
  fields (llm/schema.py), persisted alongside the existing buy_limit_price/
  target_sell_price for the same audit-trail reason.
- decisiontype gains a SELL value: the early-exit signal for an IN_POSITION
  ticker (llm/schema.py's TradeDecision.decision now allows "BUY" | "HOLD" |
  "SELL").

Revision ID: 0005_add_stop_loss_exit
Revises: 0004_add_rsi
Create Date: 2026-08-24

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005_add_stop_loss_exit"
down_revision: str | None = "0004_add_rsi"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Postgres allows ADD VALUE inside a transaction block since v12, as long as
    # the new value isn't used within that same transaction -- true here, nothing
    # in this migration inserts a SELL row.
    op.execute("ALTER TYPE decisiontype ADD VALUE IF NOT EXISTS 'SELL'")

    op.add_column("trades", sa.Column("stop_loss_price", sa.Numeric(12, 4), nullable=True))
    op.add_column("trades", sa.Column("exit_reason", sa.String(), nullable=True))

    op.add_column(
        "llm_decisions", sa.Column("stop_loss_price", sa.Numeric(12, 4), nullable=True)
    )
    op.add_column(
        "llm_decisions", sa.Column("thesis_continuity_flag", sa.Boolean(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("llm_decisions", "thesis_continuity_flag")
    op.drop_column("llm_decisions", "stop_loss_price")
    op.drop_column("trades", "exit_reason")
    op.drop_column("trades", "stop_loss_price")
    # Removing 'SELL' from decisiontype is deliberately not implemented: Postgres
    # has no "ALTER TYPE ... DROP VALUE", only a full type rebuild, and that would
    # fail outright if any llm_decisions row already has decision='SELL' -- not
    # worth the complexity for a downgrade path.
