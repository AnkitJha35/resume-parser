"""PostgreSQL schema definitions using SQLAlchemy Core tables."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

# Shared metadata for all tables in the persistence foundation
metadata = MetaData()

# 1. Candidates table
candidates_table = Table(
    "candidates",
    metadata,
    Column("candidate_id", String(64), primary_key=True),
    Column("external_id", String(128), nullable=True),
    Column("name", String(255), nullable=True),
    Column("primary_email", String(255), nullable=True),
    Column("primary_phone", String(64), nullable=True),
    Column("metadata", JSONB(astext_type=Text()), nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
    Index(
        "idx_candidates_external_id",
        "external_id",
        unique=True,
        postgresql_where=text("external_id IS NOT NULL AND deleted_at IS NULL"),
    ),
    Index("idx_candidates_email", "primary_email", postgresql_where=text("primary_email IS NOT NULL")),
    Index("idx_candidates_deleted_at", "deleted_at"),
)

# 2. Documents table
documents_table = Table(
    "documents",
    metadata,
    Column("document_id", String(64), primary_key=True),
    Column("candidate_id", String(64), ForeignKey("candidates.candidate_id", ondelete="SET NULL"), nullable=True),
    Column("content_hash", String(64), nullable=False),
    Column("original_filename", String(255), nullable=False),
    Column("mime_type", String(64), nullable=False, server_default=text("'application/pdf'")),
    Column("file_size_bytes", BigInteger, nullable=False),
    Column("storage_uri", String(512), nullable=False),
    Column("page_count", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
    Index(
        "uq_documents_active_content_hash",
        "content_hash",
        unique=True,
        postgresql_where=text("deleted_at IS NULL"),
    ),
    Index("idx_documents_candidate_id", "candidate_id", postgresql_where=text("candidate_id IS NOT NULL")),
    Index("idx_documents_created_at", text("created_at DESC")),
)

# 3. Resume Snapshots table
resume_snapshots_table = Table(
    "resume_snapshots",
    metadata,
    Column("resume_id", String(64), primary_key=True),
    Column("document_id", String(64), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
    Column("candidate_id", String(64), ForeignKey("candidates.candidate_id", ondelete="SET NULL"), nullable=True),
    Column("parse_status", String(32), nullable=False),
    Column("success", Boolean, nullable=False),
    Column("parser_version", String(32), nullable=False),
    Column("schema_version", String(32), nullable=False),
    Column("provider", String(32), nullable=False),
    Column("model", String(64), nullable=False),
    Column("representation", String(64), nullable=False),
    Column("parse_config_hash", String(64), nullable=False),
    Column("is_latest", Boolean, nullable=False, server_default=text("true")),
    Column("resume_data", JSONB(astext_type=Text()), nullable=False),
    Column("violations", JSONB(astext_type=Text()), nullable=False, server_default=text("'[]'::jsonb")),
    Column("metadata", JSONB(astext_type=Text()), nullable=False, server_default=text("'{}'::jsonb")),
    Column("latency_ms", Float, nullable=False),
    Column("request_count", Integer, nullable=False, server_default=text("1")),
    Column("candidate_name", String(255), nullable=True),
    Column("candidate_email", String(255), nullable=True),
    Column("candidate_phone", String(64), nullable=True),
    Column("candidate_location", String(255), nullable=True),
    Column("skills", ARRAY(Text()), nullable=False, server_default=text("'{}'::text[]")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
    Index(
        "idx_snapshots_reusable_lookup",
        "document_id",
        "parse_config_hash",
        postgresql_where=text("deleted_at IS NULL"),
    ),
    Index("idx_snapshots_document_id", "document_id"),
    Index("idx_snapshots_candidate_id", "candidate_id", postgresql_where=text("candidate_id IS NOT NULL")),
    Index("idx_snapshots_status", "parse_status"),
    Index("idx_snapshots_created_at", text("created_at DESC")),
    Index(
        "idx_snapshots_is_latest",
        "document_id",
        "is_latest",
        postgresql_where=text("is_latest = true AND deleted_at IS NULL"),
    ),
    Index("idx_snapshots_skills_gin", "skills", postgresql_using="gin"),
    Index(
        "idx_snapshots_name_trgm",
        "candidate_name",
        postgresql_using="gin",
        postgresql_ops={"candidate_name": "gin_trgm_ops"},
    ),
    Index("idx_snapshots_email", "candidate_email", postgresql_where=text("candidate_email IS NOT NULL")),
    Index(
        "idx_snapshots_resume_data_gin",
        "resume_data",
        postgresql_using="gin",
        postgresql_ops={"resume_data": "jsonb_path_ops"},
    ),
)

# 4. Resume Provenance table (1:1 with resume_snapshots)
resume_provenance_table = Table(
    "resume_provenance",
    metadata,
    Column("resume_id", String(64), ForeignKey("resume_snapshots.resume_id", ondelete="CASCADE"), primary_key=True),
    Column("records", JSONB(astext_type=Text()), nullable=False, server_default=text("'[]'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# 5. Idempotency Keys table (minimal primitive for deterministic retry handling)
idempotency_keys_table = Table(
    "idempotency_keys",
    metadata,
    Column("idempotency_key", String(128), primary_key=True),
    Column("resume_id", String(64), nullable=True),
    Column("document_id", String(64), nullable=True),
    Column("response_data", JSONB(astext_type=Text()), nullable=True),
    Column("status", String(32), nullable=False, server_default=text("'IN_PROGRESS'")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    Index("idx_idempotency_keys_created_at", text("created_at DESC")),
)

