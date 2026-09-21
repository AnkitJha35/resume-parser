"""Focused tests for Product v1 persistence orchestration service."""

from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Any

import fitz
import pytest
from sqlalchemy import create_engine, text

from app.api.v1.models import ParseMetadata, ParseResponse, ParseStatus
from app.core.config import Settings
from app.domain.resume import Resume
from app.infrastructure.database.repositories import (
    CandidateRepository,
    DocumentRepository,
    ResumeProvenanceRepository,
    ResumeSnapshotRepository,
)
from app.services.resume_persistence_service import (
    ResumePersistenceService,
    UploadValidationError,
)

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:probus%40220706@localhost:5432/resume_parser_test",
)


def make_test_pdf(text_content: str = "Test Candidate") -> bytes:
    """Generate minimal valid PDF bytes with text using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text_content)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class MockParserRunner:
    """Controllable parser runner that tracks call counts and returns valid ParseResponse."""

    def __init__(
        self,
        status: ParseStatus = ParseStatus.SUCCESS,
        success: bool = True,
        resume_data: dict[str, Any] | None = None,
        violations: list[str] | None = None,
        delay_sec: float = 0.0,
    ) -> None:
        self.call_count = 0
        self.status = status
        self.success = success
        self.violations = violations or []
        self.delay_sec = delay_sec
        self.resume_data = resume_data or {
            "schemaVersion": "1.0",
            "parserVersion": "1.0.0",
            "personal": {
                "name": "Jane Candidate",
                "email": "jane@example.com",
                "phone": "+1-555-0100",
                "location": "New York, NY",
            },
            "skills": ["Python", "SQL", "Docker"],
            "experience": [
                {
                    "company": "Tech Corp",
                    "designation": "Software Engineer",
                    "startDate": "2020-01",
                    "endDate": "2023-01",
                    "current": False,
                }
            ],
            "education": [],
            "projects": [],
            "certifications": [],
        }

    def __call__(self, pdf_bytes: bytes, filename: str, settings: Settings) -> ParseResponse:
        self.call_count += 1
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)

        meta = ParseMetadata(
            filename=filename,
            pageCount=1,
            ocrUsed=False,
            provider=settings.semantic_provider,
            model=settings.gemini_model,
            representation="candidate_b_compact",
            latencyMs=120.0,
            requestCount=1,
        )
        prov = [
            {
                "canonicalField": "personal.name",
                "extractedValue": "Jane Candidate",
                "rawValue": "Jane Candidate",
                "sourceBlockIds": ["b_p1_1"],
                "sourceTexts": ["Jane Candidate"],
                "pageNumbers": [1],
                "confidence": 1.0,
            },
            {
                "canonicalField": "personal.email",
                "extractedValue": "jane@example.com",
                "rawValue": "jane@example.com",
                "sourceBlockIds": ["b_p1_2"],
                "sourceTexts": ["jane@example.com"],
                "pageNumbers": [1],
                "confidence": 1.0,
            },
        ]
        return ParseResponse(
            success=self.success,
            status=self.status,
            resume=self.resume_data,
            violations=self.violations,
            metadata=meta,
            diagnostics={"provenance": prov},
        )


@pytest.fixture(scope="module")
def db_engine():
    """Ensure PostgreSQL test database is available and return engine."""
    try:
        engine = create_engine(TEST_DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL test database not available at {TEST_DB_URL}: {exc}")

    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_db(db_engine):
    """Truncate tables before each test to guarantee test isolation."""
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE idempotency_keys, resume_provenance, resume_snapshots, documents, candidates CASCADE;"
            )
        )
    yield


# =====================================================================
# A. New Upload Tests
# =====================================================================


def test_new_upload_persists_document_snapshot_and_provenance(db_engine):
    """A fresh upload must execute parser exactly once and persist all three layers."""
    mock_parser = MockParserRunner()
    service = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=mock_parser,
    )

    pdf_bytes = make_test_pdf("Jane Candidate Resume Content")
    result = service.process_upload(pdf_bytes, filename="jane_resume.pdf")

    assert result.is_idempotent_hit is False
    assert result.success is True
    assert result.status == ParseStatus.SUCCESS
    assert result.is_latest is True
    assert mock_parser.call_count == 1

    # Verify Document in DB
    with db_engine.connect() as conn:
        doc = DocumentRepository.get_by_id(conn, result.document_id)
        assert doc is not None
        assert doc["original_filename"] == "jane_resume.pdf"
        assert doc["page_count"] == 1

        # Verify Snapshot in DB
        snap = ResumeSnapshotRepository.get_by_id(conn, result.resume_id)
        assert snap is not None
        assert snap["is_latest"] is True
        assert snap["candidate_name"] == "Jane Candidate"
        assert snap["candidate_email"] == "jane@example.com"
        assert "Python" in snap["skills"]

        # Verify Provenance in DB
        prov = ResumeProvenanceRepository.get_by_resume_id(conn, result.resume_id)
        assert prov is not None
        assert len(prov["records"]) == 2


# =====================================================================
# B. Duplicate Upload Tests (Zero Extra Parser Executions)
# =====================================================================


def test_duplicate_upload_reuses_snapshot_with_zero_parser_calls(db_engine):
    """Subsequent upload of identical bytes with identical config must return cached snapshot with 0 parser executions."""
    mock_parser = MockParserRunner()
    service = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=mock_parser,
    )

    pdf_bytes = make_test_pdf("Deduplication Candidate Resume")

    # 1. First upload
    res1 = service.process_upload(pdf_bytes, filename="first.pdf")
    assert res1.is_idempotent_hit is False
    assert mock_parser.call_count == 1

    # 2. Second upload (same bytes, same configuration)
    res2 = service.process_upload(pdf_bytes, filename="second.pdf")
    assert res2.is_idempotent_hit is True
    assert res2.resume_id == res1.resume_id
    assert res2.document_id == res1.document_id
    # Call count must strictly remain 1! Zero extra LLM/parser calls.
    assert mock_parser.call_count == 1


# =====================================================================
# C. Changed Parser Configuration Tests
# =====================================================================


def test_changed_parser_configuration_triggers_new_snapshot(db_engine):
    """Same document under a modified parser configuration must execute parser and create new snapshot."""
    mock_parser = MockParserRunner()
    settings_v1 = Settings(database_url=TEST_DB_URL, parser_version="1.0.0")
    settings_v2 = Settings(database_url=TEST_DB_URL, parser_version="1.1.0")

    service_v1 = ResumePersistenceService(
        settings=settings_v1,
        engine=db_engine,
        parser_runner=mock_parser,
    )
    service_v2 = ResumePersistenceService(
        settings=settings_v2,
        engine=db_engine,
        parser_runner=mock_parser,
    )

    pdf_bytes = make_test_pdf("Version Upgrade Resume")

    # Parse with v1.0.0
    res_v1 = service_v1.process_upload(pdf_bytes)
    assert res_v1.is_idempotent_hit is False
    assert mock_parser.call_count == 1

    # Parse with v1.1.0 (configuration changed)
    res_v2 = service_v2.process_upload(pdf_bytes)
    assert res_v2.is_idempotent_hit is False
    assert res_v2.document_id == res_v1.document_id  # Same physical document
    assert res_v2.resume_id != res_v1.resume_id      # Different snapshot
    assert mock_parser.call_count == 2               # Re-parsed for new config

    # Atomicity check on is_latest
    with db_engine.connect() as conn:
        old_snap = ResumeSnapshotRepository.get_by_id(conn, res_v1.resume_id)
        new_snap = ResumeSnapshotRepository.get_by_id(conn, res_v2.resume_id)
        assert old_snap["is_latest"] is False
        assert new_snap["is_latest"] is True


# =====================================================================
# D. force_reparse Tests
# =====================================================================


def test_force_reparse_triggers_new_snapshot(db_engine):
    """Setting force_reparse=True must execute a new parse even if matching config snapshot exists."""
    mock_parser = MockParserRunner()
    service = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=mock_parser,
    )

    pdf_bytes = make_test_pdf("Forced Reparse Resume")

    # First parse
    res1 = service.process_upload(pdf_bytes)
    assert res1.is_idempotent_hit is False
    assert mock_parser.call_count == 1

    # Force reparse
    res2 = service.process_upload(pdf_bytes, force_reparse=True)
    assert res2.is_idempotent_hit is False
    assert res2.document_id == res1.document_id
    assert res2.resume_id != res1.resume_id
    assert mock_parser.call_count == 2

    # Verify is_latest updated
    with db_engine.connect() as conn:
        old_snap = ResumeSnapshotRepository.get_by_id(conn, res1.resume_id)
        new_snap = ResumeSnapshotRepository.get_by_id(conn, res2.resume_id)
        assert old_snap["is_latest"] is False
        assert new_snap["is_latest"] is True


# =====================================================================
# E. Concurrent Normal Uploads (Advisory Lock Coordination)
# =====================================================================


def test_concurrent_uploads_coordinate_and_execute_parser_once(db_engine):
    """Simultaneous uploads of the same PDF must coordinate so only one actual parse executes."""
    mock_parser = MockParserRunner(delay_sec=0.08)  # simulate brief parse duration
    service = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=mock_parser,
    )

    pdf_bytes = make_test_pdf("Concurrent Race Condition Candidate")

    def worker():
        return service.process_upload(pdf_bytes, filename="race.pdf")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(worker)
        f2 = executor.submit(worker)
        res1 = f1.result()
        res2 = f2.result()

    # Both results must reference the exact same document
    assert res1.document_id == res2.document_id

    # Exactly one parser execution occurred across both threads
    assert mock_parser.call_count == 1

    # Exactly one latest snapshot exists
    with db_engine.connect() as conn:
        doc_id = res1.document_id
        latest_snaps = conn.execute(
            text(
                "SELECT resume_id FROM resume_snapshots WHERE document_id = :doc_id AND is_latest = true"
            ),
            {"doc_id": doc_id},
        ).fetchall()
        assert len(latest_snaps) == 1


# =====================================================================
# F. Candidate ID Handling (No Automatic Inference)
# =====================================================================


def test_candidate_id_persists_when_supplied_and_remains_null_when_absent(db_engine):
    """candidate_id must persist when supplied, remain null when absent, and never infer from email."""
    mock_parser = MockParserRunner()
    service = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=mock_parser,
    )

    pdf_bytes1 = make_test_pdf("Candidate A with email test@example.com")
    pdf_bytes2 = make_test_pdf("Candidate B with email test@example.com")  # Same email!

    # 1. Supplied candidate_id
    res_supplied = service.process_upload(pdf_bytes1, candidate_id="cand_custom_99")
    assert res_supplied.candidate_id == "cand_custom_99"

    with db_engine.connect() as conn:
        doc = DocumentRepository.get_by_id(conn, res_supplied.document_id)
        snap = ResumeSnapshotRepository.get_by_id(conn, res_supplied.resume_id)
        assert doc["candidate_id"] == "cand_custom_99"
        assert snap["candidate_id"] == "cand_custom_99"

    # 2. Absent candidate_id (even though email is identical to candidate A)
    res_absent = service.process_upload(pdf_bytes2, candidate_id=None)
    assert res_absent.candidate_id is None

    with db_engine.connect() as conn:
        doc_b = DocumentRepository.get_by_id(conn, res_absent.document_id)
        snap_b = ResumeSnapshotRepository.get_by_id(conn, res_absent.resume_id)
        # Must strictly remain None; NEVER auto-linked to cand_custom_99 based on email!
        assert doc_b["candidate_id"] is None
        assert snap_b["candidate_id"] is None


# =====================================================================
# G. Failure Handling (Failures Never Become Reusable SUCCESS)
# =====================================================================


def test_failed_parse_is_not_reusable_and_preserves_previous_success(db_engine):
    """Validation failures must not be reused as SUCCESS and must not overwrite valid existing snapshots."""
    # 1. Successful parse first
    success_parser = MockParserRunner(status=ParseStatus.SUCCESS, success=True)
    service_success = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=success_parser,
    )
    pdf_bytes = make_test_pdf("Valid Candidate")
    res_success = service_success.process_upload(pdf_bytes)
    assert res_success.status == ParseStatus.SUCCESS

    # 2. Forced reparse that fails validation
    fail_parser = MockParserRunner(
        status=ParseStatus.VALIDATION_FAILED,
        success=False,
        violations=["HALLUCINATION: ungrounded field"],
    )
    service_fail = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=fail_parser,
    )
    res_fail = service_fail.process_upload(pdf_bytes, force_reparse=True)
    assert res_fail.status == ParseStatus.VALIDATION_FAILED
    assert res_fail.success is False

    # 3. Third normal upload with force_reparse=False must NOT reuse the failed snapshot!
    # It must find the valid SUCCESS snapshot
    third_parser = MockParserRunner()
    service_third = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=third_parser,
    )
    res_third = service_third.process_upload(pdf_bytes, force_reparse=False)
    assert res_third.status == ParseStatus.SUCCESS
    assert res_third.resume_id == res_success.resume_id
    assert third_parser.call_count == 0  # Reused the valid one without calling parser!


# =====================================================================
# H. Idempotency-Key Support
# =====================================================================


def test_idempotency_key_deduplicates_retries(db_engine):
    """Sending the same Idempotency-Key must return the exact completed response without re-parsing."""
    mock_parser = MockParserRunner()
    service = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=mock_parser,
    )
    pdf_bytes = make_test_pdf("Idempotency Candidate")

    res1 = service.process_upload(pdf_bytes, idempotency_key="idemp_key_001")
    assert mock_parser.call_count == 1

    # Immediate retry with same key (even if force_reparse=True)
    res2 = service.process_upload(
        pdf_bytes, force_reparse=True, idempotency_key="idemp_key_001"
    )
    assert res2.resume_id == res1.resume_id
    assert mock_parser.call_count == 1


# =====================================================================
# I. Validation Guards (Invalid files rejected before DB/parser)
# =====================================================================


def test_invalid_upload_payloads_rejected_early(db_engine):
    """Oversized files, non-PDF extensions, and invalid headers must raise UploadValidationError."""
    mock_parser = MockParserRunner()
    service = ResumePersistenceService(
        settings=Settings(database_url=TEST_DB_URL),
        engine=db_engine,
        parser_runner=mock_parser,
    )

    # Invalid extension
    with pytest.raises(UploadValidationError) as exc:
        service.process_upload(b"%PDF-1.4...", filename="resume.docx")
    assert exc.value.code == "INVALID_FILE_TYPE"

    # Invalid magic bytes
    with pytest.raises(UploadValidationError) as exc:
        service.process_upload(b"NOT_A_PDF_FILE", filename="resume.pdf")
    assert exc.value.code == "INVALID_PDF_HEADER"

    # Oversized payload
    oversized = b"%PDF-" + b"0" * (10 * 1024 * 1024 + 1)
    with pytest.raises(UploadValidationError) as exc:
        service.process_upload(oversized, filename="resume.pdf")
    assert exc.value.code == "PAYLOAD_TOO_LARGE"

    # Parser was never invoked
    assert mock_parser.call_count == 0
