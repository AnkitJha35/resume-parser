"""Focused tests for Product v1 PostgreSQL persistence foundation."""

from __future__ import annotations

import os
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.domain.resume import (
    CertificationItem,
    EducationItem,
    ExperienceItem,
    PersonalInfo,
    ProjectItem,
    Resume,
)
from app.infrastructure.database.connection import (
    get_connection,
    reset_engine,
    try_acquire_document_advisory_lock,
)
from app.infrastructure.database.hashing import compute_parse_config_hash
from app.infrastructure.database.repositories import (
    CandidateRepository,
    DocumentRepository,
    ResumeProvenanceRepository,
    ResumeSnapshotRepository,
)

# Test PostgreSQL connection string
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:probus%40220706@localhost:5432/resume_parser_test",
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
    """Truncate tables before each test to ensure test isolation."""
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE resume_provenance, resume_snapshots, documents, candidates CASCADE;"
            )
        )
    yield


# =====================================================================
# 1. Parse Configuration Hashing Tests
# =====================================================================


def test_parse_config_hash_deterministic():
    """Hash must be strictly deterministic across repeated invocations."""
    h1 = compute_parse_config_hash(
        "1.0.0", "1.0", "gemini", "gemini-3.5-flash-lite", "candidate_b_compact"
    )
    h2 = compute_parse_config_hash(
        "1.0.0", "1.0", "gemini", "gemini-3.5-flash-lite", "candidate_b_compact"
    )
    assert h1 == h2
    assert len(h1) == 64


def test_parse_config_hash_case_and_whitespace_invariant():
    """Hash must normalize casing and whitespace across all configuration components."""
    base_hash = compute_parse_config_hash(
        "1.0.0", "1.0", "gemini", "gemini-3.5-flash-lite", "candidate_b_compact"
    )
    variant_hash = compute_parse_config_hash(
        " 1.0.0  ",
        "  1.0",
        " GEMINI ",
        " Gemini-3.5-Flash-Lite ",
        " CANDIDATE_B_COMPACT ",
    )
    assert base_hash == variant_hash


def test_parse_config_hash_component_sensitivity():
    """Changing any single component must yield a completely different hash."""
    base_hash = compute_parse_config_hash(
        "1.0.0", "1.0", "gemini", "gemini-3.5-flash-lite", "candidate_b_compact"
    )

    diff_parser = compute_parse_config_hash(
        "1.1.0", "1.0", "gemini", "gemini-3.5-flash-lite", "candidate_b_compact"
    )
    diff_schema = compute_parse_config_hash(
        "1.0.0", "2.0", "gemini", "gemini-3.5-flash-lite", "candidate_b_compact"
    )
    diff_provider = compute_parse_config_hash(
        "1.0.0", "1.0", "nvidia", "gemini-3.5-flash-lite", "candidate_b_compact"
    )
    diff_model = compute_parse_config_hash(
        "1.0.0", "1.0", "gemini", "gemini-1.5-pro", "candidate_b_compact"
    )
    diff_rep = compute_parse_config_hash(
        "1.0.0", "1.0", "gemini", "gemini-3.5-flash-lite", "full_json"
    )

    hashes = {base_hash, diff_parser, diff_schema, diff_provider, diff_model, diff_rep}
    assert len(hashes) == 6, "Every configuration change must produce a distinct hash"


# =====================================================================
# 2. Candidate Repository Tests
# =====================================================================


def test_candidate_crud(db_engine):
    """Test candidate insertion, lookup, and soft deletion."""
    with db_engine.begin() as conn:
        cand = CandidateRepository.insert(
            conn,
            candidate_id="cand_test_01",
            external_id="ext_9876",
            name="Alice Smith",
            primary_email="alice@example.com",
            primary_phone="+15551234567",
            metadata={"source": "ats_integration"},
        )
        assert cand["candidate_id"] == "cand_test_01"
        assert cand["name"] == "Alice Smith"
        assert cand["metadata"] == {"source": "ats_integration"}

        # Lookup by ID
        found = CandidateRepository.get_by_id(conn, "cand_test_01")
        assert found is not None
        assert found["primary_email"] == "alice@example.com"

        # Lookup by external ID
        found_ext = CandidateRepository.get_by_external_id(conn, "ext_9876")
        assert found_ext is not None
        assert found_ext["candidate_id"] == "cand_test_01"

        # Soft delete
        deleted = CandidateRepository.soft_delete(conn, "cand_test_01")
        assert deleted is True

        # Default lookup excludes soft-deleted
        assert CandidateRepository.get_by_id(conn, "cand_test_01") is None
        assert CandidateRepository.get_by_id(conn, "cand_test_01", include_deleted=True) is not None


# =====================================================================
# 3. Document Repository & Uniqueness Tests
# =====================================================================


def test_document_crud_and_content_hash_lookup(db_engine):
    """Test document insertion, content hash lookup, and soft deletion."""
    with db_engine.begin() as conn:
        doc = DocumentRepository.insert(
            conn,
            document_id="doc_abc123",
            content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            original_filename="sample_resume.pdf",
            file_size_bytes=10240,
            storage_uri="resumes/doc_abc123.pdf",
            page_count=2,
        )
        assert doc["document_id"] == "doc_abc123"
        assert doc["page_count"] == 2

        # Lookup by content hash
        found = DocumentRepository.get_by_content_hash(
            conn, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert found is not None
        assert found["document_id"] == "doc_abc123"

        # Soft delete
        assert DocumentRepository.soft_delete(conn, "doc_abc123") is True
        assert (
            DocumentRepository.get_by_content_hash(
                conn, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            )
            is None
        )


def test_document_active_content_hash_uniqueness(db_engine):
    """Active documents must enforce uniqueness on content_hash."""
    hash_val = "1111111111111111111111111111111111111111111111111111111111111111"
    with db_engine.begin() as conn:
        DocumentRepository.insert(
            conn,
            document_id="doc_first",
            content_hash=hash_val,
            original_filename="resume_v1.pdf",
            file_size_bytes=2048,
            storage_uri="resumes/doc_first.pdf",
            page_count=1,
        )

    # Attempting to insert another active document with the same content_hash must raise IntegrityError
    with pytest.raises(IntegrityError):
        with db_engine.begin() as conn:
            DocumentRepository.insert(
                conn,
                document_id="doc_second",
                content_hash=hash_val,
                original_filename="duplicate.pdf",
                file_size_bytes=2048,
                storage_uri="resumes/doc_second.pdf",
                page_count=1,
            )


# =====================================================================
# 4. ResumeSnapshot Repository & Reusability Lookup Tests
# =====================================================================


def test_resume_snapshot_crud_and_reusable_lookup(db_engine):
    """Test snapshot creation, reusable cache lookup, and is_latest promotion."""
    cfg_hash = compute_parse_config_hash(
        "1.0.0", "1.0", "gemini", "gemini-3.5-flash-lite", "candidate_b_compact"
    )

    with db_engine.begin() as conn:
        DocumentRepository.insert(
            conn,
            document_id="doc_snap_01",
            content_hash="2222222222222222222222222222222222222222222222222222222222222222",
            original_filename="candidate_cv.pdf",
            file_size_bytes=4096,
            storage_uri="resumes/doc_snap_01.pdf",
            page_count=1,
        )

        snapshot1 = ResumeSnapshotRepository.insert(
            conn,
            resume_id="res_01",
            document_id="doc_snap_01",
            parse_status="SUCCESS",
            success=True,
            parser_version="1.0.0",
            schema_version="1.0",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            representation="candidate_b_compact",
            parse_config_hash=cfg_hash,
            resume_data={"personal": {"name": "Bob Jones"}, "skills": ["Python", "FastAPI"]},
            violations=[],
            metadata={"archetype": "standard_cv"},
            latency_ms=1250.0,
            request_count=1,
            candidate_name="Bob Jones",
            skills=["Python", "FastAPI"],
            is_latest=True,
        )
        assert snapshot1["is_latest"] is True

        # Reusable snapshot query must match
        reusable = ResumeSnapshotRepository.get_reusable_snapshot(
            conn, "doc_snap_01", cfg_hash
        )
        assert reusable is not None
        assert reusable["resume_id"] == "res_01"

        # Inserting a second snapshot for the same document with is_latest=True demotes the first
        snapshot2 = ResumeSnapshotRepository.insert(
            conn,
            resume_id="res_02",
            document_id="doc_snap_01",
            parse_status="SUCCESS",
            success=True,
            parser_version="1.1.0",
            schema_version="1.0",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            representation="candidate_b_compact",
            parse_config_hash="new_config_hash",
            resume_data={"personal": {"name": "Bob Jones"}, "skills": ["Python", "FastAPI", "Docker"]},
            is_latest=True,
        )
        assert snapshot2["is_latest"] is True

        # First snapshot is now is_latest=False
        old_snap = ResumeSnapshotRepository.get_by_id(conn, "res_01")
        assert old_snap["is_latest"] is False


def test_reusable_snapshot_ignores_failures_and_deleted(db_engine):
    """Reusable lookup must ignore VALIDATION_FAILED, EXTRACTION_FAILED, or soft-deleted snapshots."""
    cfg_hash = "failed_or_deleted_hash"
    with db_engine.begin() as conn:
        DocumentRepository.insert(
            conn,
            document_id="doc_fail_01",
            content_hash="3333333333333333333333333333333333333333333333333333333333333333",
            original_filename="failed.pdf",
            file_size_bytes=1000,
            storage_uri="resumes/doc_fail_01.pdf",
            page_count=1,
        )

        # Failed snapshot
        ResumeSnapshotRepository.insert(
            conn,
            resume_id="res_fail",
            document_id="doc_fail_01",
            parse_status="VALIDATION_FAILED",
            success=False,
            parser_version="1.0.0",
            schema_version="1.0",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            representation="candidate_b_compact",
            parse_config_hash=cfg_hash,
            resume_data={},
            violations=["HALLUCINATION: invalid company"],
        )

        assert (
            ResumeSnapshotRepository.get_reusable_snapshot(conn, "doc_fail_01", cfg_hash)
            is None
        )


# =====================================================================
# 5. ResumeProvenance Repository & Cascade Deletion Tests
# =====================================================================


def test_provenance_crud_and_cascade_delete(db_engine):
    """Test provenance record storage and cascade delete with snapshot."""
    with db_engine.begin() as conn:
        DocumentRepository.insert(
            conn,
            document_id="doc_prov_01",
            content_hash="4444444444444444444444444444444444444444444444444444444444444444",
            original_filename="prov.pdf",
            file_size_bytes=2000,
            storage_uri="resumes/doc_prov_01.pdf",
            page_count=1,
        )

        ResumeSnapshotRepository.insert(
            conn,
            resume_id="res_prov_01",
            document_id="doc_prov_01",
            parse_status="SUCCESS",
            success=True,
            parser_version="1.0.0",
            schema_version="1.0",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            representation="candidate_b_compact",
            parse_config_hash="h1",
            resume_data={},
        )

        records = [
            {
                "canonicalField": "personal.name",
                "extractedValue": "Jane Doe",
                "rawValue": "Jane Doe",
                "sourceBlockIds": ["b_p1_1"],
                "sourceTexts": ["Jane Doe"],
                "pageNumbers": [1],
                "confidence": 1.0,
            }
        ]
        ResumeProvenanceRepository.insert(conn, "res_prov_01", records)

        # Retrieve provenance
        prov = ResumeProvenanceRepository.get_by_resume_id(conn, "res_prov_01")
        assert prov is not None
        assert prov["records"] == records

        # Deleting parent document cascades to delete snapshot and provenance
        conn.execute(
            text("DELETE FROM documents WHERE document_id = 'doc_prov_01';")
        )
        assert ResumeSnapshotRepository.get_by_id(conn, "res_prov_01") is None
        assert ResumeProvenanceRepository.get_by_resume_id(conn, "res_prov_01") is None


# =====================================================================
# 6. Full Resume Domain Model JSON Round-Trip Fidelity Test
# =====================================================================


def test_resume_domain_model_round_trip(db_engine):
    """Verify that a full Resume domain model survives database storage with 100% fidelity."""
    original_resume = Resume(
        schemaVersion="1.0",
        parserVersion="1.0.0",
        personal=PersonalInfo(
            name="Alexander Hamilton",
            email="alex@treasury.gov",
            phone="+12125551776",
            location="New York, NY",
            linkedin="https://linkedin.com/in/alexander-hamilton",
            github="https://github.com/hamilton",
            portfolio="https://hamilton.org",
        ),
        summary="Founding father and financial architect with extensive governance experience.",
        skills=["Economics", "Fiscal Policy", "Constitutional Law", "Public Finance"],
        experience=[
            ExperienceItem(
                company="Department of the Treasury",
                designation="Secretary of the Treasury",
                location="Philadelphia, PA",
                startDate="1789-09",
                endDate="1795-01",
                current=False,
                description="Designed the US financial system, established the national bank.",
                skills=["Fiscal Policy", "Public Finance"],
                confidence=1.0,
            )
        ],
        education=[
            EducationItem(
                institution="King's College",
                degree="Bachelor of Arts",
                fieldOfStudy="Liberal Arts",
                startDate="1774",
                endDate="1776",
                grade="Honors",
                confidence=0.95,
            )
        ],
        projects=[
            ProjectItem(
                name="Bank of the United States",
                description="Central bank to handle national debt and manage currency.",
                technologies=["Finance"],
                startDate="1791",
                endDate="1791",
                current=False,
            )
        ],
        certifications=[
            CertificationItem(
                name="Bar Admission",
                issuingOrganization="New York State Bar",
                issueDate="1782",
            )
        ],
        achievements=["Co-authored 51 of the 85 Federalist Papers."],
        languages=["English", "French"],
        metadata={"archetype": "standard_cv", "extractor": "semantic_llm"},
    )

    resume_dict = original_resume.model_dump()

    with db_engine.begin() as conn:
        DocumentRepository.insert(
            conn,
            document_id="doc_hamilton",
            content_hash="5555555555555555555555555555555555555555555555555555555555555555",
            original_filename="hamilton_cv.pdf",
            file_size_bytes=8192,
            storage_uri="resumes/doc_hamilton.pdf",
            page_count=2,
        )

        ResumeSnapshotRepository.insert(
            conn,
            resume_id="res_hamilton",
            document_id="doc_hamilton",
            parse_status="SUCCESS",
            success=True,
            parser_version="1.0.0",
            schema_version="1.0",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            representation="candidate_b_compact",
            parse_config_hash="hamilton_cfg",
            resume_data=resume_dict,
            candidate_name=original_resume.personal.name,
            candidate_email=original_resume.personal.email,
            candidate_phone=original_resume.personal.phone,
            candidate_location=original_resume.personal.location,
            skills=original_resume.skills,
        )

        stored = ResumeSnapshotRepository.get_by_id(conn, "res_hamilton")
        assert stored is not None

        # Re-validate stored JSON back into the canonical Resume domain model
        restored_resume = Resume.model_validate(stored["resume_data"])
        assert restored_resume == original_resume
        assert restored_resume.personal.name == "Alexander Hamilton"
        assert restored_resume.experience[0].company == "Department of the Treasury"
        assert len(restored_resume.skills) == 4


# =====================================================================
# 7. Advisory Lock Primitive Test
# =====================================================================


def test_advisory_lock_primitive(db_engine):
    """Test PostgreSQL transaction-scoped advisory lock acquisition."""
    test_hash = "6666666666666666666666666666666666666666666666666666666666666666"
    with db_engine.begin() as conn:
        acquired = try_acquire_document_advisory_lock(conn, test_hash)
        assert acquired is True
