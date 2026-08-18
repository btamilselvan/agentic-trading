"""add book_imbalance to buckets

bid_size/ask_size were already fetched and persisted, but the order-book depth
imbalance derived from them (spec 3.1) was never computed or exposed to the LLM
prompt -- this adds the computed column; llm/prompt.py now includes it (and the raw
sizes) in the payload.

Revision ID: 0002_add_book_imbalance
Revises: 0001_initial
Create Date: 2026-08-18

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_add_book_imbalance"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("buckets", sa.Column("book_imbalance", sa.Numeric(6, 4), nullable=True))


def downgrade() -> None:
    op.drop_column("buckets", "book_imbalance")
