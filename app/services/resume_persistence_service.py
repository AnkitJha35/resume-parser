"""Persistence orchestration service connecting upload flow to document storage, frozen parser, and PostgreSQL persistence."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import fitz
from pydantic import BaseModel, Field
from sqlalchemy import Engine

from app.api.v1.endpoints import _run_parse_pipeline
from app.api.v1.models import ParseResponse, ParseStatus
from app.core.config import Settings
from app.core.exceptions import ResumeParserError
from app.infrastructure.database.connection import (
    get_engine,
    try_acquire_document_advisory_lock,
)
from app.infrastructure.database.hashing import compute_parse_config_hash
from app.infrastructure.database.repositories import (
    CandidateRepository,
    DocumentRepository,
    IdempotencyRepository,
    ResumeProvenanceRepository,
    ResumeSnapshotRepository,
)

import threading

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

_inflight_events: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()


class UploadValidationError(ResumeParserError):
    """Validation error for upload payloads."""

    def __init__(self, message: str, code: str = "UPLOAD_VALIDATION_ERROR", status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ResumePersistenceResult(BaseModel):
    """Structured internal service result for uploaded and persisted resume snapshots."""

    resume_id: str
    document_id: str
    candidate_id: str | None = None
    status: ParseStatus
    success: bool
    is_latest: bool
    is_idempotent_hit: bool
    created_at: datetime
    resume: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] | None = None


class ResumePersistenceService:
    """Orchestrates document registration, idempotent lookup, frozen parser invocation, and snapshot persistence."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: Engine | None = None,
        parser_runner: Callable[[bytes, str, Settings], ParseResponse] | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._engine = engine
        self._parser_runner = parser_runner or _run_parse_pipeline

    def _get_engine(self) -> Engine:
        if self._engine is not None:
            return self._engine
        return get_engine(self._settings)

    def _validate_payload(self, pdf_bytes: bytes, filename: str) -> None:
        """Validate upload byte size and PDF signature."""
        if filename and not filename.lower().endswith(".pdf"):
            raise UploadValidationError(
                f"Expected a PDF file, but received: '{filename}'",
                code="INVALID_FILE_TYPE",
                status_code=400,
            )

        if len(pdf_bytes) > MAX_UPLOAD_SIZE:
            raise UploadValidationError(
                f"File size ({len(pdf_bytes)} bytes) exceeds max limit of {MAX_UPLOAD_SIZE} bytes (10MB)",
                code="PAYLOAD_TOO_LARGE",
                status_code=413,
            )

        if not pdf_bytes.startswith(b"%PDF-"):
            raise UploadValidationError(
                "File header does not match valid PDF magic bytes (%PDF-).",
                code="INVALID_PDF_HEADER",
                status_code=400,
            )

    def _resolve_model_name(self) -> str:
        provider = self._settings.semantic_provider.lower()
        if provider == "gemini":
            return self._settings.gemini_model
        if provider == "nvidia":
            return self._settings.nvidia_model
        if provider == "openrouter":
            return self._settings.openrouter_model
        if provider == "ollama":
            return self._settings.ollama_model
        return "unknown"

    def _extract_page_count(self, pdf_bytes: bytes) -> int:
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                return doc.page_count
        except Exception:
            return 1

    def process_upload(
        self,
        pdf_bytes: bytes,
        filename: str = "uploaded_resume.pdf",
        candidate_id: str | None = None,
        force_reparse: bool = False,
        idempotency_key: str | None = None,
    ) -> ResumePersistenceResult:
        """Execute the persistence orchestration workflow.

        Sequence:
        1. Validate payload.
        2. Compute content_hash and parse_config_hash.
        3. Short transaction 1: Check IdempotencyKey / find or create Document / check reusable snapshot cache.
        4. (Outside DB transaction) Invoke frozen parser pipeline (1 LLM request).
        5. Short transaction 2: Re-check cache, acquire advisory lock, persist ResumeSnapshot, Provenance, and IdempotencyKey.
        """
        # Step 1: Validate payload
        self._validate_payload(pdf_bytes, filename)

        # Step 2: Calculate deterministic hashes
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        parser_version = self._settings.parser_version
        schema_version = "1.0"
        provider = self._settings.semantic_provider
        model = self._resolve_model_name()
        representation = "candidate_b_compact"

        parse_config_hash = compute_parse_config_hash(
            parser_version, schema_version, provider, model, representation
        )

        engine = self._get_engine()

        # Step 3: Short transaction 1 (Check Idempotency-Key & Cache)
        with engine.connect() as conn:
            with conn.begin():
                # 3a. Idempotency-Key lookup
                if idempotency_key:
                    cached_entry = IdempotencyRepository.get(conn, idempotency_key)
                    if cached_entry and cached_entry.get("response_data"):
                        resp_data = cached_entry["response_data"]
                        return ResumePersistenceResult.model_validate(resp_data)

                # 3b. Ensure Candidate exists if client supplied candidate_id
                if candidate_id:
                    existing_cand = CandidateRepository.get_by_id(conn, candidate_id)
                    if not existing_cand:
                        CandidateRepository.insert(conn, candidate_id=candidate_id)

                # 3c. Find or create Document
                doc = DocumentRepository.get_by_content_hash(conn, content_hash)
                if not doc:
                    document_id = f"doc_{content_hash[:24]}"
                    page_count = self._extract_page_count(pdf_bytes)
                    doc = DocumentRepository.insert(
                        conn,
                        document_id=document_id,
                        content_hash=content_hash,
                        original_filename=filename,
                        file_size_bytes=len(pdf_bytes),
                        storage_uri=f"resumes/{content_hash}.pdf",
                        page_count=page_count,
                        candidate_id=candidate_id,
                    )
                document_id = doc["document_id"]

                # 3d. Check reusable snapshot if force_reparse is False
                if not force_reparse:
                    reusable = ResumeSnapshotRepository.get_reusable_snapshot(
                        conn, document_id, parse_config_hash
                    )
                    if reusable:
                        prov_record = ResumeProvenanceRepository.get_by_resume_id(conn, reusable["resume_id"])
                        provenance_list = prov_record["records"] if prov_record else None

                        result = ResumePersistenceResult(
                            resume_id=reusable["resume_id"],
                            document_id=document_id,
                            candidate_id=reusable["candidate_id"] or candidate_id,
                            status=ParseStatus(reusable["parse_status"]),
                            success=reusable["success"],
                            is_latest=reusable["is_latest"],
                            is_idempotent_hit=True,
                            created_at=reusable["created_at"],
                            resume=reusable["resume_data"],
                            metadata=reusable["metadata"],
                            violations=reusable["violations"],
                            provenance=provenance_list,
                        )
                        if idempotency_key:
                            IdempotencyRepository.set_completed(
                                conn,
                                idempotency_key=idempotency_key,
                                resume_id=reusable["resume_id"],
                                document_id=document_id,
                                response_data=result.model_dump(mode="json"),
                            )
                        return result

        # In-flight deduplication coordination: avoid concurrent duplicate parses for same doc + config
        coord_key = f"{document_id}:{parse_config_hash}"
        is_leader = False
        wait_event = None

        if not force_reparse:
            with _inflight_lock:
                if coord_key in _inflight_events:
                    wait_event = _inflight_events[coord_key]
                else:
                    wait_event = threading.Event()
                    _inflight_events[coord_key] = wait_event
                    is_leader = True

            if not is_leader and wait_event is not None:
                # Another concurrent thread is currently parsing this exact document + config.
                # Wait for leader to finish and commit outside any DB transaction.
                wait_event.wait(timeout=60.0)

        parse_response: ParseResponse | None = None
        try:
            if is_leader or force_reparse or wait_event is None:
                # Step 4: Execute parser OUTSIDE of any database transaction or lock
                logger.info("Executing parser pipeline for doc=%s config=%s", document_id, parse_config_hash)
                parse_response = self._parser_runner(pdf_bytes, filename, self._settings)

            # Step 5: Short transaction 2 (Re-check cache under advisory lock & persist)
            with engine.connect() as conn:
                with conn.begin():
                    # Acquire advisory lock on content_hash to coordinate concurrent identical uploads
                    try_acquire_document_advisory_lock(conn, content_hash)

                    # Re-check cache if force_reparse is False
                    if not force_reparse:
                        reusable = ResumeSnapshotRepository.get_reusable_snapshot(
                            conn, document_id, parse_config_hash
                        )
                        if reusable:
                            prov_record = ResumeProvenanceRepository.get_by_resume_id(conn, reusable["resume_id"])
                            provenance_list = prov_record["records"] if prov_record else None

                            result = ResumePersistenceResult(
                                resume_id=reusable["resume_id"],
                                document_id=document_id,
                                candidate_id=reusable["candidate_id"] or candidate_id,
                                status=ParseStatus(reusable["parse_status"]),
                                success=reusable["success"],
                                is_latest=reusable["is_latest"],
                                is_idempotent_hit=True,
                                created_at=reusable["created_at"],
                                resume=reusable["resume_data"],
                                metadata=reusable["metadata"],
                                violations=reusable["violations"],
                                provenance=provenance_list,
                            )
                            if idempotency_key:
                                IdempotencyRepository.set_completed(
                                    conn,
                                    idempotency_key=idempotency_key,
                                    resume_id=reusable["resume_id"],
                                    document_id=document_id,
                                    response_data=result.model_dump(mode="json"),
                                )
                            return result

                    # If parse_response is None (e.g. concurrent leader did not persist a reusable snapshot), run parser
                    if parse_response is None:
                        parse_response = self._parser_runner(pdf_bytes, filename, self._settings)

                    # Extract denormalized search fields
                    resume_dict = parse_response.resume or {}
                    personal = resume_dict.get("personal") or {}
                    candidate_name = personal.get("name")
                    candidate_email = personal.get("email")
                    candidate_phone = personal.get("phone")
                    candidate_location = personal.get("location")
                    skills = resume_dict.get("skills") or []

                    new_resume_id = f"res_{uuid.uuid4().hex}"
                    provenance_data = parse_response.diagnostics.get("provenance") or []
                    meta_dict = parse_response.metadata.model_dump() if parse_response.metadata else {}

                    # Persist snapshot
                    created_snapshot = ResumeSnapshotRepository.insert(
                        conn,
                        resume_id=new_resume_id,
                        document_id=document_id,
                        parse_status=parse_response.status.value,
                        success=parse_response.success,
                        parser_version=parser_version,
                        schema_version=schema_version,
                        provider=provider,
                        model=model,
                        representation=representation,
                        parse_config_hash=parse_config_hash,
                        resume_data=resume_dict,
                        violations=parse_response.violations or [],
                        metadata=meta_dict,
                        latency_ms=parse_response.metadata.latencyMs if parse_response.metadata else 0.0,
                        request_count=parse_response.metadata.requestCount if parse_response.metadata else 1,
                        candidate_id=candidate_id,
                        candidate_name=candidate_name,
                        candidate_email=candidate_email,
                        candidate_phone=candidate_phone,
                        candidate_location=candidate_location,
                        skills=skills,
                        is_latest=True,
                    )

                    # Persist provenance (1:1 with snapshot)
                    ResumeProvenanceRepository.insert(conn, new_resume_id, provenance_data)

                    result = ResumePersistenceResult(
                        resume_id=new_resume_id,
                        document_id=document_id,
                        candidate_id=candidate_id,
                        status=parse_response.status,
                        success=parse_response.success,
                        is_latest=True,
                        is_idempotent_hit=False,
                        created_at=created_snapshot["created_at"],
                        resume=resume_dict,
                        metadata=meta_dict,
                        violations=parse_response.violations or [],
                        provenance=provenance_data,
                    )

                    # Record idempotency key if provided
                    if idempotency_key:
                        IdempotencyRepository.set_completed(
                            conn,
                            idempotency_key=idempotency_key,
                            resume_id=new_resume_id,
                            document_id=document_id,
                            response_data=result.model_dump(mode="json"),
                        )

                    return result
        finally:
            if is_leader and wait_event is not None:
                with _inflight_lock:
                    _inflight_events.pop(coord_key, None)
                wait_event.set()
