"""add vwap to buckets

Session VWAP (spec 3.1's "price velocity" context is close, but the standard
intraday momentum reference line was missing entirely) -- volume-weighted typical
price accumulated from market open through each bucket, inclusive.

Revision ID: 0003_add_vwap
Revises: 0002_add_book_imbalance
Create Date: 2026-08-18

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_add_vwap"
down_revision: str | None = "0002_add_book_imbalance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("buckets", sa.Column("vwap", sa.Numeric(12, 4), nullable=True))


def downgrade() -> None:
    op.drop_column("buckets", "vwap")
