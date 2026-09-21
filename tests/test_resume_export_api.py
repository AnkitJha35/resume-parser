"""Focused tests for Product v1 Resume Export APIs (Phase 11E-2).

Tests:
A. JSON export
B. Text export
C. CSV export
D. Default format=json
E. Invalid format
F. Missing resume -> 404
G. Soft-deleted resume -> 404
H. Soft-deleted document visibility
I. Nested experience/education/projects/certifications
J. Empty optional collections
K. Unicode text
L. Deterministic repeated export
M. Correct Content-Type
N. Correct Content-Disposition
O. Export performs ZERO LLM/parser executions
"""

from __future__ import annotations

import csv
import io
import json
import os
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.api.v1.endpoints import set_extractor_override
from app.infrastructure.database.connection import reset_engine
from app.infrastructure.database.repositories import (
    CandidateRepository,
    DocumentRepository,
    ResumeProvenanceRepository,
    ResumeSnapshotRepository,
)
from app.main import app
from app.pipeline.parser import ResumeParser

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


def insert_doc(
    engine,
    doc_id: str = "doc_exp_01",
    content_hash: str = "hash_exp_01",
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
    resume_id: str = "res_exp_01",
    document_id: str = "doc_exp_01",
    candidate_id: str | None = None,
    parse_status: str = "SUCCESS",
    success: bool = True,
    candidate_name: str | None = "Alice Wonder",
    candidate_email: str | None = "alice@example.com",
    candidate_phone: str | None = "+15551234567",
    candidate_location: str | None = "San Francisco, CA",
    skills: list[str] | None = None,
    resume_data: dict | None = None,
) -> dict:
    if resume_data is None:
        resume_data = {
            "schemaVersion": "1.0",
            "parserVersion": "1.0.0",
            "personal": {
                "name": candidate_name,
                "email": candidate_email,
                "phone": candidate_phone,
                "location": candidate_location,
                "linkedin": "https://linkedin.com/in/alicewonder",
                "github": "https://github.com/alicewonder",
            },
            "summary": "Accomplished AI and backend software architect.",
            "skills": skills or ["Python", "FastAPI", "PostgreSQL", "Docker"],
            "experience": [
                {
                    "company": "Apex Technologies",
                    "designation": "Principal Engineer",
                    "location": "San Francisco, CA",
                    "startDate": "2021-03",
                    "endDate": None,
                    "current": True,
                    "description": "Architected low-latency distributed parser pipelines.",
                    "technologies": ["Python", "AsyncIO", "PostgreSQL"],
                },
                {
                    "company": "Core Systems Corp",
                    "designation": "Senior Engineer",
                    "location": "Seattle, WA",
                    "startDate": "2018-06",
                    "endDate": "2021-02",
                    "current": False,
                    "description": "Maintained high-throughput data pipelines.",
                    "technologies": ["Go", "Kafka"],
                },
            ],
            "education": [
                {
                    "institution": "Stanford University",
                    "degree": "M.S.",
                    "fieldOfStudy": "Computer Science",
                    "startDate": "2016-09",
                    "endDate": "2018-06",
                    "grade": "3.9 GPA",
                }
            ],
            "projects": [
                {
                    "name": "AutoParser",
                    "description": "High performance layout-aware parser engine.",
                    "technologies": ["PyMuPDF", "FastAPI"],
                    "startDate": "2022-01",
                    "endDate": None,
                    "current": True,
                    "url": "https://autoparser.dev",
                }
            ],
            "certifications": [
                {
                    "name": "AWS Solutions Architect Professional",
                    "issuingOrganization": "Amazon Web Services",
                    "issueDate": "2022-05",
                    "expiryDate": "2025-05",
                    "credentialId": "AWS-12345",
                    "credentialUrl": "https://aws.cert/12345",
                }
            ],
            "achievements": [
                "Keynote speaker at Global Python Summit 2023",
                "1st place winner at ACM Regional Hackathon",
            ],
            "languages": ["English", "French", "German"],
            "metadata": {"archetype": "standard_cv"},
        }

    with engine.begin() as conn:
        return ResumeSnapshotRepository.insert(
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
            parse_config_hash="cfg_hash_exp",
            resume_data=resume_data,
            violations=[],
            metadata={"archetype": "standard_cv"},
            latency_ms=1000.0,
            request_count=1,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            candidate_phone=candidate_phone,
            candidate_location=candidate_location,
            skills=skills or ["Python", "FastAPI", "PostgreSQL", "Docker"],
            is_latest=True,
        )


# =====================================================================
# Tests A through O
# =====================================================================


def test_a_json_export(db_engine, client):
    """A. JSON export: returns canonical persisted Resume JSON, 200, Content-Type, Content-Disposition."""
    insert_doc(db_engine, doc_id="doc_a", content_hash="hash_a")
    snap = insert_snapshot(db_engine, resume_id="res_a", document_id="doc_a")

    resp = client.get("/api/v1/resumes/res_a/export?format=json")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    assert 'attachment; filename="resume_res_a.json"' in resp.headers["content-disposition"]

    data = resp.json()
    assert data["schemaVersion"] == "1.0"
    assert data["parserVersion"] == "1.0.0"
    assert data["personal"]["name"] == "Alice Wonder"
    assert len(data["experience"]) == 2
    assert data["experience"][0]["company"] == "Apex Technologies"


def test_b_text_export(db_engine, client):
    """B. Text export: human-readable representation following prescribed order, omitting provenance."""
    insert_doc(db_engine, doc_id="doc_b", content_hash="hash_b")
    insert_snapshot(db_engine, resume_id="res_b", document_id="doc_b")
    with db_engine.begin() as conn:
        ResumeProvenanceRepository.insert(
            conn,
            "res_b",
            [{"canonicalField": "personal.name", "extractedValue": "Alice Wonder"}],
        )

    resp = client.get("/api/v1/resumes/res_b/export?format=text")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert 'attachment; filename="resume_res_b.txt"' in resp.headers["content-disposition"]

    text_body = resp.text
    # Verify prescribed section order
    assert "Alice Wonder" in text_body
    pos_summary = text_body.find("SUMMARY")
    pos_skills = text_body.find("SKILLS")
    pos_exp = text_body.find("EXPERIENCE")
    pos_edu = text_body.find("EDUCATION")
    pos_prj = text_body.find("PROJECTS")
    pos_cert = text_body.find("CERTIFICATIONS")
    pos_ach = text_body.find("ACHIEVEMENTS")
    pos_lang = text_body.find("LANGUAGES")

    assert pos_summary < pos_skills < pos_exp < pos_edu < pos_prj < pos_cert < pos_ach < pos_lang
    # Verify provenance is NOT included in export text
    assert "canonicalField" not in text_body
    assert "extractedValue" not in text_body


def test_c_csv_export(db_engine, client):
    """C. CSV export: stable 5-column EAV model, preserves order and nested arrays."""
    insert_doc(db_engine, doc_id="doc_c", content_hash="hash_c")
    insert_snapshot(db_engine, resume_id="res_c", document_id="doc_c")

    resp = client.get("/api/v1/resumes/res_c/export?format=csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert 'attachment; filename="resume_res_c.csv"' in resp.headers["content-disposition"]

    reader = csv.reader(io.StringIO(resp.text))
    rows = list(reader)

    # 1. Check header
    assert rows[0] == ["resume_id", "section", "item_index", "field", "value"]

    # 2. Check personal name
    name_row = next(r for r in rows if r[1] == "personal" and r[3] == "name")
    assert name_row == ["res_c", "personal", "0", "name", "Alice Wonder"]

    # 3. Check experience collection ordering and nested technology
    exp0_company = next(r for r in rows if r[1] == "experience" and r[2] == "0" and r[3] == "company")
    assert exp0_company[4] == "Apex Technologies"

    exp0_tech0 = next(r for r in rows if r[1] == "experience" and r[2] == "0" and r[3] == "technology[0]")
    assert exp0_tech0[4] == "Python"

    exp1_company = next(r for r in rows if r[1] == "experience" and r[2] == "1" and r[3] == "company")
    assert exp1_company[4] == "Core Systems Corp"


def test_d_default_format_is_json(db_engine, client):
    """D. Default format=json: omitting format query parameter produces JSON."""
    insert_doc(db_engine, doc_id="doc_d", content_hash="hash_d")
    insert_snapshot(db_engine, resume_id="res_d", document_id="doc_d")

    resp = client.get("/api/v1/resumes/res_d/export")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    assert 'attachment; filename="resume_res_d.json"' in resp.headers["content-disposition"]
    data = resp.json()
    assert data["personal"]["name"] == "Alice Wonder"


def test_e_invalid_format_422(db_engine, client):
    """E. Invalid format: returns 422 on unsupported format."""
    insert_doc(db_engine, doc_id="doc_e", content_hash="hash_e")
    insert_snapshot(db_engine, resume_id="res_e", document_id="doc_e")

    resp = client.get("/api/v1/resumes/res_e/export?format=xml")
    assert resp.status_code == 422
    err = resp.json()
    assert err["detail"]["error"] == "UNSUPPORTED_FORMAT"


def test_f_missing_resume_404(client):
    """F. Missing resume -> 404."""
    resp = client.get("/api/v1/resumes/res_nonexistent/export?format=json")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "RESUME_NOT_FOUND"


def test_g_soft_deleted_resume_404(db_engine, client):
    """G. Soft-deleted resume -> 404."""
    insert_doc(db_engine, doc_id="doc_g", content_hash="hash_g")
    insert_snapshot(db_engine, resume_id="res_g", document_id="doc_g")
    with db_engine.begin() as conn:
        ResumeSnapshotRepository.soft_delete(conn, "res_g")

    resp = client.get("/api/v1/resumes/res_g/export?format=json")
    assert resp.status_code == 404


def test_h_soft_deleted_document_visibility(db_engine, client):
    """H. Soft-deleted document visibility: returns 404 when document is soft-deleted."""
    insert_doc(db_engine, doc_id="doc_h", content_hash="hash_h")
    insert_snapshot(db_engine, resume_id="res_h", document_id="doc_h")
    with db_engine.begin() as conn:
        DocumentRepository.soft_delete(conn, "doc_h")

    resp = client.get("/api/v1/resumes/res_h/export?format=json")
    assert resp.status_code == 404


def test_i_nested_collections(db_engine, client):
    """I. Nested experience/education/projects/certifications preserved across formats."""
    insert_doc(db_engine, doc_id="doc_i", content_hash="hash_i")
    insert_snapshot(db_engine, resume_id="res_i", document_id="doc_i")

    # JSON preserves all nested structures
    rj = client.get("/api/v1/resumes/res_i/export?format=json").json()
    assert len(rj["experience"]) == 2
    assert len(rj["education"]) == 1
    assert len(rj["projects"]) == 1
    assert len(rj["certifications"]) == 1

    # Text preserves details
    rt = client.get("/api/v1/resumes/res_i/export?format=text").text
    assert "AWS Solutions Architect Professional" in rt
    assert "Stanford University" in rt
    assert "AutoParser" in rt

    # CSV preserves explicit item indices
    rc = client.get("/api/v1/resumes/res_i/export?format=csv").text
    assert "projects,0,name,AutoParser" in rc
    assert "certifications,0,name,AWS Solutions Architect Professional" in rc


def test_j_empty_optional_collections(db_engine, client):
    """J. Empty optional collections: exports cleanly without crashing."""
    empty_resume = {
        "schemaVersion": "1.0",
        "parserVersion": "1.0.0",
        "personal": {"name": "Solo Worker"},
        "summary": None,
        "skills": [],
        "experience": [],
        "education": [],
        "projects": [],
        "certifications": [],
        "achievements": [],
        "languages": [],
    }
    insert_doc(db_engine, doc_id="doc_j", content_hash="hash_j")
    insert_snapshot(db_engine, resume_id="res_j", document_id="doc_j", resume_data=empty_resume)

    # JSON
    rj = client.get("/api/v1/resumes/res_j/export?format=json")
    assert rj.status_code == 200

    # Text
    rt = client.get("/api/v1/resumes/res_j/export?format=text")
    assert rt.status_code == 200
    assert "Solo Worker" in rt.text

    # CSV
    rc = client.get("/api/v1/resumes/res_j/export?format=csv")
    assert rc.status_code == 200
    assert "personal,0,name,Solo Worker" in rc.text


def test_k_unicode_text(db_engine, client):
    """K. Unicode text: handles accents, foreign characters, and emojis cleanly."""
    unicode_resume = {
        "schemaVersion": "1.0",
        "parserVersion": "1.0.0",
        "personal": {
            "name": "Renée François 🚀",
            "email": "renee@example.fr",
            "location": "Montréal, Québec, Canada",
        },
        "summary": "Développeuse logicielle spécialisée en IA et systèmes distribués 🌟",
        "skills": ["Python 🐍", "C++", "Intelligence Artificielle"],
        "experience": [
            {
                "company": "Société Générale",
                "designation": "Ingénieure Principale",
                "location": "Paris, France",
                "startDate": "2020-01",
                "current": True,
                "description": "Conception d'architectures résilientes à haute performance.",
            }
        ],
        "education": [],
        "projects": [],
        "certifications": [],
        "achievements": [],
        "languages": ["Français", "English", "Español"],
    }
    insert_doc(db_engine, doc_id="doc_k", content_hash="hash_k")
    insert_snapshot(db_engine, resume_id="res_k", document_id="doc_k", resume_data=unicode_resume)

    # JSON UTF-8
    rj = client.get("/api/v1/resumes/res_k/export?format=json")
    assert rj.status_code == 200
    assert "Renée François 🚀" in rj.text

    # Text UTF-8
    rt = client.get("/api/v1/resumes/res_k/export?format=text")
    assert rt.status_code == 200
    assert "Renée François 🚀" in rt.text
    assert "Développeuse logicielle" in rt.text

    # CSV UTF-8
    rc = client.get("/api/v1/resumes/res_k/export?format=csv")
    assert rc.status_code == 200
    assert "Renée François 🚀" in rc.text


def test_l_deterministic_repeated_export(db_engine, client):
    """L. Deterministic repeated export: repeated requests produce identical byte outputs."""
    insert_doc(db_engine, doc_id="doc_l", content_hash="hash_l")
    insert_snapshot(db_engine, resume_id="res_l", document_id="doc_l")

    for fmt in ("json", "text", "csv"):
        resp1 = client.get(f"/api/v1/resumes/res_l/export?format={fmt}")
        resp2 = client.get(f"/api/v1/resumes/res_l/export?format={fmt}")
        assert resp1.content == resp2.content, f"Repeated export for {fmt} must be bit-identical"


def test_m_correct_content_type(db_engine, client):
    """M. Correct Content-Type for all export formats."""
    insert_doc(db_engine, doc_id="doc_m", content_hash="hash_m")
    insert_snapshot(db_engine, resume_id="res_m", document_id="doc_m")

    # JSON
    rj = client.get("/api/v1/resumes/res_m/export?format=json")
    assert rj.headers["content-type"] == "application/json"

    # Text
    rt = client.get("/api/v1/resumes/res_m/export?format=text")
    assert "text/plain; charset=utf-8" in rt.headers["content-type"]

    # CSV
    rc = client.get("/api/v1/resumes/res_m/export?format=csv")
    assert "text/csv; charset=utf-8" in rc.headers["content-type"]


def test_n_correct_content_disposition(db_engine, client):
    """N. Correct Content-Disposition with sanitized filename."""
    import urllib.parse

    # Special characters in resume_id must be sanitized in filename
    noisy_id = "res_abc-123_xyz!@#$%^&*"
    insert_doc(db_engine, doc_id="doc_n", content_hash="hash_n")
    insert_snapshot(db_engine, resume_id=noisy_id, document_id="doc_n")

    quoted_id = urllib.parse.quote(noisy_id, safe="")
    rj = client.get(f"/api/v1/resumes/{quoted_id}/export?format=json")
    assert rj.status_code == 200
    assert rj.headers["content-disposition"] == 'attachment; filename="resume_res_abc-123_xyz.json"'

    rt = client.get(f"/api/v1/resumes/{quoted_id}/export?format=text")
    assert rt.status_code == 200
    assert rt.headers["content-disposition"] == 'attachment; filename="resume_res_abc-123_xyz.txt"'

    rc = client.get(f"/api/v1/resumes/{quoted_id}/export?format=csv")
    assert rc.status_code == 200
    assert rc.headers["content-disposition"] == 'attachment; filename="resume_res_abc-123_xyz.csv"'


def test_o_export_performs_zero_llm_or_parser_executions(db_engine, client, monkeypatch):
    """O. Export performs ZERO LLM or parser executions."""
    insert_doc(db_engine, doc_id="doc_o", content_hash="hash_o")
    insert_snapshot(db_engine, resume_id="res_o", document_id="doc_o")

    # Mock parser pipeline and extractor to assert they are never called
    mock_parser = MagicMock(side_effect=RuntimeError("PARSER_CALLED_ERROR"))
    monkeypatch.setattr(ResumeParser, "parse_with_semantic_pipeline", mock_parser)

    # Calling all formats
    for fmt in ("json", "text", "csv"):
        resp = client.get(f"/api/v1/resumes/res_o/export?format={fmt}")
        assert resp.status_code == 200

    # Ensure parser was never touched
    mock_parser.assert_not_called()
