"""initial schema: buckets, trades, llm_decisions, orders, ticker_daily_state

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-16

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "buckets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("bucket_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bucket_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(12, 4), nullable=False),
        sa.Column("high", sa.Numeric(12, 4), nullable=False),
        sa.Column("low", sa.Numeric(12, 4), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("volume", sa.Integer(), nullable=False),
        sa.Column("est_buy_volume", sa.Integer(), nullable=False),
        sa.Column("est_sell_volume", sa.Integer(), nullable=False),
        sa.Column("bid_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("ask_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("bid_size", sa.Integer(), nullable=True),
        sa.Column("ask_size", sa.Integer(), nullable=True),
        sa.Column("spread", sa.Numeric(12, 4), nullable=True),
        sa.Column("candle_body", sa.Numeric(12, 4), nullable=False),
        sa.Column("upper_wick", sa.Numeric(12, 4), nullable=False),
        sa.Column("lower_wick", sa.Numeric(12, 4), nullable=False),
        sa.Column("rvol", sa.Numeric(12, 4), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("ticker", "bucket_start", name="uq_bucket_ticker_start"),
    )
    op.create_index("ix_buckets_ticker", "buckets", ["ticker"])
    op.create_index("ix_buckets_bucket_start", "buckets", ["bucket_start"])

    op.create_table(
        "llm_decisions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("bucket_id", sa.Integer(), sa.ForeignKey("buckets.id"), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=False),
        sa.Column("decision", sa.Enum("BUY", "HOLD", name="decisiontype"), nullable=False),
        sa.Column("confidence_score", sa.Numeric(4, 3), nullable=False),
        sa.Column("buy_limit_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("target_sell_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("max_holding_time_minutes", sa.Integer(), nullable=True),
        sa.Column("pattern_reasoning", sa.Text(), nullable=True),
        sa.Column("acted_on", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_llm_decisions_ticker", "llm_decisions", ["ticker"])
    op.create_index("ix_llm_decisions_created_at", "llm_decisions", ["created_at"])

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("OPEN", "CLOSED", name="tradestatus"),
            nullable=False,
        ),
        sa.Column(
            "llm_decision_id", sa.Integer(), sa.ForeignKey("llm_decisions.id"), nullable=True
        ),
        sa.Column("target_sell_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("max_holding_time_minutes", sa.Integer(), nullable=True),
        sa.Column("entry_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("exit_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("quantity", sa.Numeric(14, 6), nullable=True),
        sa.Column("pnl", sa.Numeric(14, 4), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_trades_ticker", "trades", ["ticker"])
    op.create_index("ix_trades_trade_date", "trades", ["trade_date"])

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("side", sa.Enum("BUY", "SELL", name="orderside"), nullable=False),
        sa.Column("limit_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("quantity", sa.Numeric(14, 6), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "FILLED", "CANCELLED", "TIMED_OUT", name="orderstatus"),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("mode", sa.Enum("DRY_RUN", "LIVE", name="tradingmodeenum"), nullable=False),
        sa.Column("broker_order_id", sa.String(), nullable=True),
        sa.Column("trade_id", sa.Integer(), sa.ForeignKey("trades.id"), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("filled_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_orders_ticker", "orders", ["ticker"])

    op.create_table(
        "ticker_daily_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("completed_trades_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("open_positions_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("realized_pnl", sa.Numeric(14, 4), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("ticker", "trade_date", name="uq_ticker_daily_state"),
    )
    op.create_index("ix_ticker_daily_state_ticker", "ticker_daily_state", ["ticker"])
    op.create_index("ix_ticker_daily_state_trade_date", "ticker_daily_state", ["trade_date"])


def downgrade() -> None:
    op.drop_table("ticker_daily_state")
    op.drop_table("orders")
    op.drop_table("trades")
    op.drop_table("llm_decisions")
    op.drop_table("buckets")
    for enum_name in ("tradingmodeenum", "orderstatus", "orderside", "decisiontype", "tradestatus"):
        sa.Enum(name=enum_name).drop(op.get_bind(), checkfirst=True)
