"""Add idempotency keys table for deterministic retry handling.

Revision ID: 002_add_idempotency_keys
Revises: 001_persistence_foundation
Create Date: 2026-09-21 15:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_add_idempotency_keys"
down_revision: Union[str, None] = "001_persistence_foundation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "idempotency_keys",
        sa.Column("idempotency_key", sa.String(length=128), primary_key=True),
        sa.Column("resume_id", sa.String(length=64), nullable=True),
        sa.Column("document_id", sa.String(length=64), nullable=True),
        sa.Column("response_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'IN_PROGRESS'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("idx_idempotency_keys_created_at", "idempotency_keys", [sa.text("created_at DESC")])


def downgrade() -> None:
    op.drop_table("idempotency_keys")
