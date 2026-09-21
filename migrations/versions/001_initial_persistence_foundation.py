"""Initial PostgreSQL persistence foundation migration.

Revision ID: 001_persistence_foundation
Revises: None
Create Date: 2026-09-21 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_persistence_foundation"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable pg_trgm extension for text search
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # 2. Create candidates table
    op.create_table(
        "candidates",
        sa.Column("candidate_id", sa.String(length=64), primary_key=True),
        sa.Column("external_id", sa.String(length=128), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("primary_email", sa.String(length=255), nullable=True),
        sa.Column("primary_phone", sa.String(length=64), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_candidates_external_id",
        "candidates",
        ["external_id"],
        unique=True,
        postgresql_where=sa.text("external_id IS NOT NULL AND deleted_at IS NULL"),
    )
    op.create_index(
        "idx_candidates_email",
        "candidates",
        ["primary_email"],
        postgresql_where=sa.text("primary_email IS NOT NULL"),
    )
    op.create_index("idx_candidates_deleted_at", "candidates", ["deleted_at"])

    # 3. Create documents table
    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(length=64), primary_key=True),
        sa.Column("candidate_id", sa.String(length=64), sa.ForeignKey("candidates.candidate_id", ondelete="SET NULL"), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=64), nullable=False, server_default=sa.text("'application/pdf'")),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("storage_uri", sa.String(length=512), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "uq_documents_active_content_hash",
        "documents",
        ["content_hash"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_documents_candidate_id",
        "documents",
        ["candidate_id"],
        postgresql_where=sa.text("candidate_id IS NOT NULL"),
    )
    op.create_index("idx_documents_created_at", "documents", [sa.text("created_at DESC")])

    # 4. Create resume_snapshots table
    op.create_table(
        "resume_snapshots",
        sa.Column("resume_id", sa.String(length=64), primary_key=True),
        sa.Column("document_id", sa.String(length=64), sa.ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("candidate_id", sa.String(length=64), sa.ForeignKey("candidates.candidate_id", ondelete="SET NULL"), nullable=True),
        sa.Column("parse_status", sa.String(length=32), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("parser_version", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("representation", sa.String(length=64), nullable=False),
        sa.Column("parse_config_hash", sa.String(length=64), nullable=False),
        sa.Column("is_latest", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("resume_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("violations", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("candidate_name", sa.String(length=255), nullable=True),
        sa.Column("candidate_email", sa.String(length=255), nullable=True),
        sa.Column("candidate_phone", sa.String(length=64), nullable=True),
        sa.Column("candidate_location", sa.String(length=255), nullable=True),
        sa.Column("skills", postgresql.ARRAY(sa.Text()), nullable=False, server_default=sa.text("'{}'::text[]")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_snapshots_reusable_lookup",
        "resume_snapshots",
        ["document_id", "parse_config_hash"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index("idx_snapshots_document_id", "resume_snapshots", ["document_id"])
    op.create_index(
        "idx_snapshots_candidate_id",
        "resume_snapshots",
        ["candidate_id"],
        postgresql_where=sa.text("candidate_id IS NOT NULL"),
    )
    op.create_index("idx_snapshots_status", "resume_snapshots", ["parse_status"])
    op.create_index("idx_snapshots_created_at", "resume_snapshots", [sa.text("created_at DESC")])
    op.create_index(
        "idx_snapshots_is_latest",
        "resume_snapshots",
        ["document_id", "is_latest"],
        postgresql_where=sa.text("is_latest = true AND deleted_at IS NULL"),
    )
    op.create_index("idx_snapshots_skills_gin", "resume_snapshots", ["skills"], postgresql_using="gin")
    op.create_index(
        "idx_snapshots_name_trgm",
        "resume_snapshots",
        ["candidate_name"],
        postgresql_using="gin",
        postgresql_ops={"candidate_name": "gin_trgm_ops"},
    )
    op.create_index(
        "idx_snapshots_email",
        "resume_snapshots",
        ["candidate_email"],
        postgresql_where=sa.text("candidate_email IS NOT NULL"),
    )
    op.create_index(
        "idx_snapshots_resume_data_gin",
        "resume_snapshots",
        ["resume_data"],
        postgresql_using="gin",
        postgresql_ops={"resume_data": "jsonb_path_ops"},
    )

    # 5. Create resume_provenance table (1:1 with resume_snapshots)
    op.create_table(
        "resume_provenance",
        sa.Column("resume_id", sa.String(length=64), sa.ForeignKey("resume_snapshots.resume_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("records", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("resume_provenance")
    op.drop_table("resume_snapshots")
    op.drop_table("documents")
    op.drop_table("candidates")
