"""add rsi to buckets

Phase 2 (requirements.md section 6): Wilder's RSI, computed intraday from 5-min
closes (see market_data/bucket_builder.compute_rsi) -- plain Python, not
pandas-ta, since nothing else in the project uses pandas.

Revision ID: 0004_add_rsi
Revises: 0003_add_vwap
Create Date: 2026-08-19

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_add_rsi"
down_revision: str | None = "0003_add_vwap"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("buckets", sa.Column("rsi", sa.Numeric(5, 2), nullable=True))


def downgrade() -> None:
    op.drop_column("buckets", "rsi")
