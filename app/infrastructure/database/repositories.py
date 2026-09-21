"""SQLAlchemy Core repositories for Candidate, Document, ResumeSnapshot, and Provenance."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Connection, and_, desc, func, insert, select, update

from app.infrastructure.database.schema import (
    candidates_table,
    documents_table,
    idempotency_keys_table,
    resume_provenance_table,
    resume_snapshots_table,
)

logger = logging.getLogger(__name__)


class CandidateRepository:
    """Repository managing Candidate records."""

    @staticmethod
    def insert(
        conn: Connection,
        candidate_id: str,
        external_id: str | None = None,
        name: str | None = None,
        primary_email: str | None = None,
        primary_phone: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stmt = (
            insert(candidates_table)
            .values(
                candidate_id=candidate_id,
                external_id=external_id,
                name=name,
                primary_email=primary_email,
                primary_phone=primary_phone,
                metadata=metadata or {},
            )
            .returning(candidates_table)
        )
        result = conn.execute(stmt)
        return dict(result.mappings().one())

    @staticmethod
    def get_by_id(conn: Connection, candidate_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        conditions = [candidates_table.c.candidate_id == candidate_id]
        if not include_deleted:
            conditions.append(candidates_table.c.deleted_at.is_(None))
        stmt = select(candidates_table).where(and_(*conditions))
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    @staticmethod
    def get_by_external_id(conn: Connection, external_id: str) -> dict[str, Any] | None:
        stmt = select(candidates_table).where(
            and_(
                candidates_table.c.external_id == external_id,
                candidates_table.c.deleted_at.is_(None),
            )
        )
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    @staticmethod
    def soft_delete(conn: Connection, candidate_id: str) -> bool:
        stmt = (
            update(candidates_table)
            .where(
                and_(
                    candidates_table.c.candidate_id == candidate_id,
                    candidates_table.c.deleted_at.is_(None),
                )
            )
            .values(deleted_at=func.now(), updated_at=func.now())
        )
        result = conn.execute(stmt)
        return result.rowcount > 0


class DocumentRepository:
    """Repository managing Document records."""

    @staticmethod
    def insert(
        conn: Connection,
        document_id: str,
        content_hash: str,
        original_filename: str,
        file_size_bytes: int,
        storage_uri: str,
        page_count: int,
        candidate_id: str | None = None,
        mime_type: str = "application/pdf",
    ) -> dict[str, Any]:
        stmt = (
            insert(documents_table)
            .values(
                document_id=document_id,
                candidate_id=candidate_id,
                content_hash=content_hash,
                original_filename=original_filename,
                mime_type=mime_type,
                file_size_bytes=file_size_bytes,
                storage_uri=storage_uri,
                page_count=page_count,
            )
            .returning(documents_table)
        )
        result = conn.execute(stmt)
        return dict(result.mappings().one())

    @staticmethod
    def get_by_id(conn: Connection, document_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        conditions = [documents_table.c.document_id == document_id]
        if not include_deleted:
            conditions.append(documents_table.c.deleted_at.is_(None))
        stmt = select(documents_table).where(and_(*conditions))
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    @staticmethod
    def get_by_content_hash(conn: Connection, content_hash: str) -> dict[str, Any] | None:
        """Lookup active document by content hash (SHA-256)."""
        stmt = select(documents_table).where(
            and_(
                documents_table.c.content_hash == content_hash,
                documents_table.c.deleted_at.is_(None),
            )
        )
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    @staticmethod
    def soft_delete(conn: Connection, document_id: str) -> bool:
        stmt = (
            update(documents_table)
            .where(
                and_(
                    documents_table.c.document_id == document_id,
                    documents_table.c.deleted_at.is_(None),
                )
            )
            .values(deleted_at=func.now())
        )
        result = conn.execute(stmt)
        return result.rowcount > 0


class ResumeSnapshotRepository:
    """Repository managing ResumeSnapshot records."""

    @staticmethod
    def insert(
        conn: Connection,
        resume_id: str,
        document_id: str,
        parse_status: str,
        success: bool,
        parser_version: str,
        schema_version: str,
        provider: str,
        model: str,
        representation: str,
        parse_config_hash: str,
        resume_data: dict[str, Any],
        violations: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
        request_count: int = 1,
        candidate_id: str | None = None,
        candidate_name: str | None = None,
        candidate_email: str | None = None,
        candidate_phone: str | None = None,
        candidate_location: str | None = None,
        skills: list[str] | None = None,
        is_latest: bool = True,
    ) -> dict[str, Any]:
        # If marked as latest, demote any existing latest snapshot for this document
        if is_latest:
            conn.execute(
                update(resume_snapshots_table)
                .where(
                    and_(
                        resume_snapshots_table.c.document_id == document_id,
                        resume_snapshots_table.c.is_latest.is_(True),
                    )
                )
                .values(is_latest=False)
            )

        stmt = (
            insert(resume_snapshots_table)
            .values(
                resume_id=resume_id,
                document_id=document_id,
                candidate_id=candidate_id,
                parse_status=parse_status,
                success=success,
                parser_version=parser_version,
                schema_version=schema_version,
                provider=provider,
                model=model,
                representation=representation,
                parse_config_hash=parse_config_hash,
                is_latest=is_latest,
                resume_data=resume_data,
                violations=violations or [],
                metadata=metadata or {},
                latency_ms=latency_ms,
                request_count=request_count,
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                candidate_phone=candidate_phone,
                candidate_location=candidate_location,
                skills=skills or [],
            )
            .returning(resume_snapshots_table)
        )
        result = conn.execute(stmt)
        return dict(result.mappings().one())

    @staticmethod
    def get_by_id(conn: Connection, resume_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        conditions = [resume_snapshots_table.c.resume_id == resume_id]
        if not include_deleted:
            conditions.append(resume_snapshots_table.c.deleted_at.is_(None))
        stmt = select(resume_snapshots_table).where(and_(*conditions))
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    @staticmethod
    def get_reusable_snapshot(
        conn: Connection,
        document_id: str,
        parse_config_hash: str,
    ) -> dict[str, Any] | None:
        """Lookup reusable snapshot matching document_id and parse_config_hash."""
        stmt = (
            select(resume_snapshots_table)
            .where(
                and_(
                    resume_snapshots_table.c.document_id == document_id,
                    resume_snapshots_table.c.parse_config_hash == parse_config_hash,
                    resume_snapshots_table.c.deleted_at.is_(None),
                    resume_snapshots_table.c.parse_status.in_(["SUCCESS", "PARTIAL"]),
                )
            )
            .order_by(desc(resume_snapshots_table.c.created_at))
        )
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    @staticmethod
    def update_is_latest(conn: Connection, resume_id: str, document_id: str) -> None:
        """Atomically set resume_id as the latest snapshot for document_id."""
        conn.execute(
            update(resume_snapshots_table)
            .where(resume_snapshots_table.c.document_id == document_id)
            .values(is_latest=False)
        )
        conn.execute(
            update(resume_snapshots_table)
            .where(resume_snapshots_table.c.resume_id == resume_id)
            .values(is_latest=True)
        )

    @staticmethod
    def soft_delete(conn: Connection, resume_id: str) -> bool:
        stmt = (
            update(resume_snapshots_table)
            .where(
                and_(
                    resume_snapshots_table.c.resume_id == resume_id,
                    resume_snapshots_table.c.deleted_at.is_(None),
                )
            )
            .values(deleted_at=func.now())
        )
        result = conn.execute(stmt)
        return result.rowcount > 0


class ResumeProvenanceRepository:
    """Repository managing ResumeProvenance records."""

    @staticmethod
    def insert(conn: Connection, resume_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        stmt = (
            insert(resume_provenance_table)
            .values(
                resume_id=resume_id,
                records=records,
            )
            .returning(resume_provenance_table)
        )
        result = conn.execute(stmt)
        return dict(result.mappings().one())

    @staticmethod
    def get_by_resume_id(conn: Connection, resume_id: str) -> dict[str, Any] | None:
        stmt = select(resume_provenance_table).where(resume_provenance_table.c.resume_id == resume_id)
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None


class IdempotencyRepository:
    """Repository managing IdempotencyKey records for deterministic retry handling."""

    @staticmethod
    def get(conn: Connection, idempotency_key: str) -> dict[str, Any] | None:
        stmt = select(idempotency_keys_table).where(idempotency_keys_table.c.idempotency_key == idempotency_key)
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    @staticmethod
    def set_completed(
        conn: Connection,
        idempotency_key: str,
        resume_id: str,
        document_id: str,
        response_data: dict[str, Any],
    ) -> None:
        stmt = (
            update(idempotency_keys_table)
            .where(idempotency_keys_table.c.idempotency_key == idempotency_key)
            .values(
                resume_id=resume_id,
                document_id=document_id,
                response_data=response_data,
                status="COMPLETED",
                updated_at=func.now(),
            )
        )
        res = conn.execute(stmt)
        if res.rowcount == 0:
            insert_stmt = insert(idempotency_keys_table).values(
                idempotency_key=idempotency_key,
                resume_id=resume_id,
                document_id=document_id,
                response_data=response_data,
                status="COMPLETED",
            )
            conn.execute(insert_stmt)

