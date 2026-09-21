"""Focused tests for Product v1 Resume Read/Search APIs (Phase 11E-1).

Tests:
A. Fetch SUCCESS snapshot
B. Fetch PARTIAL snapshot
C. Fetch nonexistent snapshot -> 404
D. Deleted snapshot -> not returned
E. List pagination
F. Deterministic ordering
G. Status filter
H. Candidate filter
I. Skills filter (ALL semantics, case-insensitive)
J. Company filter
K. Location filter
L. Date range
M. Candidate-specific listing
N. include_provenance behavior
O. Empty result behavior
P. Pagination bounds
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, update

from app.core.config import Settings
from app.infrastructure.database.connection import reset_engine
from app.infrastructure.database.repositories import (
    CandidateRepository,
    DocumentRepository,
    ResumeProvenanceRepository,
    ResumeSnapshotRepository,
)
from app.infrastructure.database.schema import resume_snapshots_table
from app.main import app

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:probus%40220706@localhost:5432/resume_parser_test",
)


@pytest.fixture(scope="module")
def db_engine():
    """Ensure PostgreSQL test database is available."""
    try:
        engine = create_engine(TEST_DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL test database not available at {TEST_DB_URL}: {exc}")

    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_db(db_engine, monkeypatch):
    """Truncate tables before each test and configure settings."""
    monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)
    reset_engine()

    with db_engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE idempotency_keys, resume_provenance, resume_snapshots, documents, candidates CASCADE;"
            )
        )
    yield
    reset_engine()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)
    return TestClient(app)


# Helpers to populate test database records


def insert_candidate(
    engine,
    candidate_id: str = "cand_01",
    name: str = "Alice Smith",
    email: str = "alice@example.com",
    phone: str = "+15551234567",
) -> dict:
    with engine.begin() as conn:
        return CandidateRepository.insert(
            conn,
            candidate_id=candidate_id,
            name=name,
            primary_email=email,
            primary_phone=phone,
        )


def insert_doc(
    engine,
    doc_id: str = "doc_01",
    content_hash: str = "hash_01",
    filename: str = "resume.pdf",
    candidate_id: str | None = None,
) -> dict:
    with engine.begin() as conn:
        return DocumentRepository.insert(
            conn,
            document_id=doc_id,
            content_hash=content_hash,
            original_filename=filename,
            file_size_bytes=2048,
            storage_uri=f"resumes/{doc_id}.pdf",
            page_count=1,
            candidate_id=candidate_id,
        )


def insert_snapshot(
    engine,
    resume_id: str = "res_01",
    document_id: str = "doc_01",
    candidate_id: str | None = None,
    parse_status: str = "SUCCESS",
    success: bool = True,
    is_latest: bool = True,
    candidate_name: str | None = "Alice Smith",
    candidate_email: str | None = "alice@example.com",
    candidate_phone: str | None = "+15551234567",
    candidate_location: str | None = "San Francisco, CA",
    skills: list[str] | None = None,
    resume_data: dict | None = None,
    metadata: dict | None = None,
    violations: list[str] | None = None,
    created_at: datetime | None = None,
) -> dict:
    if resume_data is None:
        resume_data = {
            "personal": {
                "name": candidate_name,
                "email": candidate_email,
                "phone": candidate_phone,
                "location": candidate_location,
            },
            "summary": "Experienced software architect",
            "skills": skills or ["Python", "FastAPI"],
            "experience": [
                {
                    "company": "Tech Solutions Inc",
                    "designation": "Staff Software Engineer",
                    "location": candidate_location,
                    "startDate": "2020-01",
                    "endDate": None,
                    "current": True,
                    "description": "Leading distributed systems development.",
                }
            ],
            "education": [
                {
                    "institution": "University of California, Berkeley",
                    "degree": "B.S.",
                    "fieldOfStudy": "Computer Science",
                }
            ],
        }

    with engine.begin() as conn:
        snap = ResumeSnapshotRepository.insert(
            conn,
            resume_id=resume_id,
            document_id=document_id,
            candidate_id=candidate_id,
            parse_status=parse_status,
            success=success,
            parser_version="1.0.0",
            schema_version="1.0",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            representation="candidate_b_compact",
            parse_config_hash="cfg_hash_test",
            resume_data=resume_data,
            violations=violations or [],
            metadata=metadata or {"archetype": "standard_cv"},
            latency_ms=1000.0,
            request_count=1,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            candidate_phone=candidate_phone,
            candidate_location=candidate_location,
            skills=skills or ["Python", "FastAPI"],
            is_latest=is_latest,
        )

        if created_at is not None:
            conn.execute(
                update(resume_snapshots_table)
                .where(resume_snapshots_table.c.resume_id == resume_id)
                .values(created_at=created_at)
            )

        return snap


# =====================================================================
# Test Cases A through P
# =====================================================================


def test_a_fetch_success_snapshot(db_engine, client):
    """A. Fetch SUCCESS snapshot: returns snapshot details and omits heavy provenance by default."""
    insert_doc(db_engine, doc_id="doc_a", content_hash="hash_a")
    insert_snapshot(db_engine, resume_id="res_a", document_id="doc_a", parse_status="SUCCESS", success=True)
    with db_engine.begin() as conn:
        ResumeProvenanceRepository.insert(
            conn,
            "res_a",
            [{"canonicalField": "personal.name", "extractedValue": "Alice Smith"}],
        )

    resp = client.get("/api/v1/resumes/res_a")
    assert resp.status_code == 200
    data = resp.json()

    assert data["resume_id"] == "res_a"
    assert data["document_id"] == "doc_a"
    assert data["status"] == "SUCCESS"
    assert data["success"] is True
    assert data["is_latest"] is True
    assert data["candidate_name"] == "Alice Smith"
    assert data["resume"]["personal"]["name"] == "Alice Smith"
    assert data["provenance"] is None  # Default omits heavy provenance


def test_b_fetch_partial_snapshot(db_engine, client):
    """B. Fetch PARTIAL snapshot: returns 200 with partial status and canonical data."""
    insert_doc(db_engine, doc_id="doc_b", content_hash="hash_b")
    insert_snapshot(
        db_engine,
        resume_id="res_b",
        document_id="doc_b",
        parse_status="PARTIAL",
        success=True,
        candidate_phone=None,
    )

    resp = client.get("/api/v1/resumes/res_b")
    assert resp.status_code == 200
    data = resp.json()
    assert data["resume_id"] == "res_b"
    assert data["status"] == "PARTIAL"
    assert data["success"] is True


def test_c_fetch_nonexistent_snapshot_404(client):
    """C. Fetch nonexistent snapshot -> 404."""
    resp = client.get("/api/v1/resumes/res_nonexistent_12345")
    assert resp.status_code == 404
    err = resp.json()
    assert err["detail"]["error"] == "RESUME_NOT_FOUND"


def test_d_soft_deleted_snapshot_not_returned(db_engine, client):
    """D. Deleted snapshot -> not returned (404 and excluded from search)."""
    # 1. Snapshot itself deleted
    insert_doc(db_engine, doc_id="doc_d1", content_hash="hash_d1")
    insert_snapshot(db_engine, resume_id="res_d1", document_id="doc_d1")
    with db_engine.begin() as conn:
        ResumeSnapshotRepository.soft_delete(conn, "res_d1")

    resp1 = client.get("/api/v1/resumes/res_d1")
    assert resp1.status_code == 404

    # 2. Underlying document deleted
    insert_doc(db_engine, doc_id="doc_d2", content_hash="hash_d2")
    insert_snapshot(db_engine, resume_id="res_d2", document_id="doc_d2")
    with db_engine.begin() as conn:
        DocumentRepository.soft_delete(conn, "doc_d2")

    resp2 = client.get("/api/v1/resumes/res_d2")
    assert resp2.status_code == 404

    # 3. Linked candidate deleted
    insert_candidate(db_engine, candidate_id="cand_d3")
    insert_doc(db_engine, doc_id="doc_d3", content_hash="hash_d3", candidate_id="cand_d3")
    insert_snapshot(db_engine, resume_id="res_d3", document_id="doc_d3", candidate_id="cand_d3")
    with db_engine.begin() as conn:
        CandidateRepository.soft_delete(conn, "cand_d3")

    resp3 = client.get("/api/v1/resumes/res_d3")
    assert resp3.status_code == 404

    # None of the deleted items appear in list
    list_resp = client.get("/api/v1/resumes")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 0


def test_e_list_pagination(db_engine, client):
    """E. List pagination: verify limit, offset, total, and has_more."""
    for i in range(5):
        insert_doc(db_engine, doc_id=f"doc_e_{i}", content_hash=f"hash_e_{i}")
        insert_snapshot(db_engine, resume_id=f"res_e_{i}", document_id=f"doc_e_{i}")

    # Page 1 (limit=2, offset=0)
    p1 = client.get("/api/v1/resumes?limit=2&offset=0").json()
    assert len(p1["items"]) == 2
    assert p1["total"] == 5
    assert p1["limit"] == 2
    assert p1["offset"] == 0
    assert p1["has_more"] is True

    # Page 2 (limit=2, offset=2)
    p2 = client.get("/api/v1/resumes?limit=2&offset=2").json()
    assert len(p2["items"]) == 2
    assert p2["total"] == 5
    assert p2["has_more"] is True

    # Page 3 (limit=2, offset=4)
    p3 = client.get("/api/v1/resumes?limit=2&offset=4").json()
    assert len(p3["items"]) == 1
    assert p3["total"] == 5
    assert p3["has_more"] is False


def test_f_deterministic_ordering(db_engine, client):
    """F. Deterministic ordering: created_at DESC, resume_id DESC."""
    base_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Insert 3 snapshots with distinct timestamps
    insert_doc(db_engine, doc_id="doc_f1", content_hash="hash_f1")
    insert_snapshot(db_engine, resume_id="res_f1", document_id="doc_f1", created_at=base_time)

    insert_doc(db_engine, doc_id="doc_f2", content_hash="hash_f2")
    insert_snapshot(db_engine, resume_id="res_f2", document_id="doc_f2", created_at=base_time + timedelta(hours=2))

    insert_doc(db_engine, doc_id="doc_f3", content_hash="hash_f3")
    insert_snapshot(db_engine, resume_id="res_f3", document_id="doc_f3", created_at=base_time + timedelta(hours=1))

    resp = client.get("/api/v1/resumes")
    assert resp.status_code == 200
    ids = [item["resume_id"] for item in resp.json()["items"]]
    # Should be ordered newest first: res_f2 (+2h), res_f3 (+1h), res_f1 (base)
    assert ids == ["res_f2", "res_f3", "res_f1"]


def test_g_status_filter(db_engine, client):
    """G. Status filter: exact ParseStatus matching."""
    insert_doc(db_engine, doc_id="doc_g1", content_hash="hash_g1")
    insert_snapshot(db_engine, resume_id="res_g_success", document_id="doc_g1", parse_status="SUCCESS")

    insert_doc(db_engine, doc_id="doc_g2", content_hash="hash_g2")
    insert_snapshot(db_engine, resume_id="res_g_partial", document_id="doc_g2", parse_status="PARTIAL")

    # Filter SUCCESS
    resp_s = client.get("/api/v1/resumes?status=SUCCESS").json()
    assert resp_s["total"] == 1
    assert resp_s["items"][0]["resume_id"] == "res_g_success"

    # Filter PARTIAL
    resp_p = client.get("/api/v1/resumes?status=PARTIAL").json()
    assert resp_p["total"] == 1
    assert resp_p["items"][0]["resume_id"] == "res_g_partial"


def test_h_candidate_filter(db_engine, client):
    """H. Candidate filter: filter snapshots by candidate_id."""
    insert_candidate(db_engine, candidate_id="cand_h1", name="Candidate H1")
    insert_candidate(db_engine, candidate_id="cand_h2", name="Candidate H2")

    insert_doc(db_engine, doc_id="doc_h1", content_hash="hash_h1", candidate_id="cand_h1")
    insert_snapshot(db_engine, resume_id="res_h1", document_id="doc_h1", candidate_id="cand_h1")

    insert_doc(db_engine, doc_id="doc_h2", content_hash="hash_h2", candidate_id="cand_h2")
    insert_snapshot(db_engine, resume_id="res_h2", document_id="doc_h2", candidate_id="cand_h2")

    resp = client.get("/api/v1/resumes?candidate_id=cand_h1").json()
    assert resp["total"] == 1
    assert resp["items"][0]["resume_id"] == "res_h1"


def test_i_skills_filter_all_semantics(db_engine, client):
    """I. Skills filter: ALL semantics (conjunctive matching) and case-insensitivity."""
    insert_doc(db_engine, doc_id="doc_i1", content_hash="hash_i1")
    insert_snapshot(db_engine, resume_id="res_i1", document_id="doc_i1", skills=["Python", "FastAPI", "Docker"])

    insert_doc(db_engine, doc_id="doc_i2", content_hash="hash_i2")
    insert_snapshot(db_engine, resume_id="res_i2", document_id="doc_i2", skills=["Python", "Django"])

    insert_doc(db_engine, doc_id="doc_i3", content_hash="hash_i3")
    insert_snapshot(db_engine, resume_id="res_i3", document_id="doc_i3", skills=["Java", "Spring"])

    # Single skill, case-insensitive (matches i1 and i2)
    resp_py = client.get("/api/v1/resumes?skills=python").json()
    assert resp_py["total"] == 2
    matched_ids = {item["resume_id"] for item in resp_py["items"]}
    assert matched_ids == {"res_i1", "res_i2"}

    # Multiple skills (ALL semantics): Python AND FastAPI (matches only i1)
    resp_both = client.get("/api/v1/resumes?skills=Python&skills=FastAPI").json()
    assert resp_both["total"] == 1
    assert resp_both["items"][0]["resume_id"] == "res_i1"

    # Comma-separated query: Python,Docker (matches only i1)
    resp_comma = client.get("/api/v1/resumes?skills=python,docker").json()
    assert resp_comma["total"] == 1
    assert resp_comma["items"][0]["resume_id"] == "res_i1"


def test_j_company_filter(db_engine, client):
    """J. Company filter: case-insensitive experience company matching."""
    insert_doc(db_engine, doc_id="doc_j1", content_hash="hash_j1")
    resume_data_j1 = {
        "personal": {"name": "Dev One"},
        "experience": [{"company": "Acme Global Solutions", "designation": "Engineer"}],
    }
    insert_snapshot(db_engine, resume_id="res_j1", document_id="doc_j1", resume_data=resume_data_j1)

    insert_doc(db_engine, doc_id="doc_j2", content_hash="hash_j2")
    resume_data_j2 = {
        "personal": {"name": "Dev Two"},
        "experience": [{"company": "Starlight Industries", "designation": "Manager"}],
    }
    insert_snapshot(db_engine, resume_id="res_j2", document_id="doc_j2", resume_data=resume_data_j2)

    resp = client.get("/api/v1/resumes?company=acme").json()
    assert resp["total"] == 1
    assert resp["items"][0]["resume_id"] == "res_j1"


def test_k_location_filter(db_engine, client):
    """K. Location filter: case-insensitive matching."""
    insert_doc(db_engine, doc_id="doc_k1", content_hash="hash_k1")
    insert_snapshot(db_engine, resume_id="res_k1", document_id="doc_k1", candidate_location="San Francisco, CA")

    insert_doc(db_engine, doc_id="doc_k2", content_hash="hash_k2")
    insert_snapshot(db_engine, resume_id="res_k2", document_id="doc_k2", candidate_location="New York, NY")

    resp = client.get("/api/v1/resumes?location=san+francisco").json()
    assert resp["total"] == 1
    assert resp["items"][0]["resume_id"] == "res_k1"


def test_l_date_range(db_engine, client):
    """L. Date range: inclusive created_after and created_before filtering."""
    t1 = datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 20, 10, 0, 0, tzinfo=timezone.utc)

    insert_doc(db_engine, doc_id="doc_l1", content_hash="hash_l1")
    insert_snapshot(db_engine, resume_id="res_l1", document_id="doc_l1", created_at=t1)

    insert_doc(db_engine, doc_id="doc_l2", content_hash="hash_l2")
    insert_snapshot(db_engine, resume_id="res_l2", document_id="doc_l2", created_at=t2)

    insert_doc(db_engine, doc_id="doc_l3", content_hash="hash_l3")
    insert_snapshot(db_engine, resume_id="res_l3", document_id="doc_l3", created_at=t3)

    # Range covering t2 and t3
    after_str = "2026-09-15T10:00:00Z"
    before_str = "2026-09-20T10:00:00Z"
    resp = client.get(f"/api/v1/resumes?created_after={after_str}&created_before={before_str}").json()
    assert resp["total"] == 2
    matched = {item["resume_id"] for item in resp["items"]}
    assert matched == {"res_l2", "res_l3"}


def test_m_candidate_specific_listing(db_engine, client):
    """M. Candidate-specific listing: returns candidate snapshots and 404 on missing/deleted candidate."""
    insert_candidate(db_engine, candidate_id="cand_m1", name="Mary Jane")
    insert_doc(db_engine, doc_id="doc_m1", content_hash="hash_m1", candidate_id="cand_m1")
    insert_snapshot(db_engine, resume_id="res_m1", document_id="doc_m1", candidate_id="cand_m1", is_latest=True)

    insert_doc(db_engine, doc_id="doc_m2", content_hash="hash_m2", candidate_id="cand_m1")
    insert_snapshot(db_engine, resume_id="res_m2", document_id="doc_m2", candidate_id="cand_m1", is_latest=False)

    # Active listing
    resp = client.get("/api/v1/candidates/cand_m1/resumes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["candidate_id"] == "cand_m1"
    assert data["total"] == 2

    # Filter is_latest=true
    resp_latest = client.get("/api/v1/candidates/cand_m1/resumes?is_latest=true").json()
    assert resp_latest["total"] == 1
    assert resp_latest["items"][0]["resume_id"] == "res_m1"

    # Nonexistent candidate -> 404
    resp_missing = client.get("/api/v1/candidates/cand_nonexistent/resumes")
    assert resp_missing.status_code == 404
    assert resp_missing.json()["detail"]["error"] == "CANDIDATE_NOT_FOUND"


def test_n_include_provenance_behavior(db_engine, client):
    """N. include_provenance behavior: omitted by default, populated when include_provenance=true."""
    insert_doc(db_engine, doc_id="doc_n", content_hash="hash_n")
    insert_snapshot(db_engine, resume_id="res_n", document_id="doc_n")
    with db_engine.begin() as conn:
        ResumeProvenanceRepository.insert(
            conn,
            "res_n",
            [
                {
                    "canonicalField": "personal.name",
                    "extractedValue": "Alice Smith",
                    "sourceBlockIds": ["b1"],
                }
            ],
        )

    # 1. Default (omitted)
    resp_def = client.get("/api/v1/resumes/res_n").json()
    assert resp_def["provenance"] is None

    # 2. include_provenance=false
    resp_false = client.get("/api/v1/resumes/res_n?include_provenance=false").json()
    assert resp_false["provenance"] is None

    # 3. include_provenance=true
    resp_true = client.get("/api/v1/resumes/res_n?include_provenance=true").json()
    assert resp_true["provenance"] is not None
    assert len(resp_true["provenance"]) == 1
    assert resp_true["provenance"][0]["canonicalField"] == "personal.name"


def test_o_empty_result_behavior(client):
    """O. Empty result behavior: returns 200 with empty list and total=0."""
    resp = client.get("/api/v1/resumes?query=NonExistentSearchTermXYZ123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["has_more"] is False


def test_p_pagination_bounds(client):
    """P. Pagination bounds: rejects invalid limits (<1 or >100) or negative offset."""
    # limit < 1
    assert client.get("/api/v1/resumes?limit=0").status_code == 422
    # limit > 100
    assert client.get("/api/v1/resumes?limit=101").status_code == 422
    # offset < 0
    assert client.get("/api/v1/resumes?offset=-1").status_code == 422

    # candidate endpoint pagination bounds
    assert client.get("/api/v1/candidates/cand_01/resumes?limit=0").status_code == 422
    assert client.get("/api/v1/candidates/cand_01/resumes?limit=101").status_code == 422
    assert client.get("/api/v1/candidates/cand_01/resumes?offset=-1").status_code == 422
