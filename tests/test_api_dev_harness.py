"""Focused unit and integration tests for the local developer API harness."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.dev_server import app
from app.api.v1.endpoints import set_extractor_override
from app.extractors.semantic_extractor import MockSemanticExtractor


@pytest.fixture(autouse=True)
def use_mock_extractor():
    """Ensure all test requests use MockSemanticExtractor without making real LLM calls."""
    set_extractor_override(lambda: MockSemanticExtractor())
    yield
    set_extractor_override(None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """A minimal valid 1-page PDF for testing."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Jane Doe")
    page.insert_text((50, 70), "jane.doe@example.com")
    page.insert_text((50, 90), "+1 555-0199")
    page.insert_text((50, 120), "Senior Software Engineer")
    page.insert_text((50, 140), "Acme Corporation")
    page.insert_text((50, 160), "Python, TypeScript, React, Docker")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "parser_version" in data
    assert "application_version" in data


def test_config_endpoint_does_not_leak_secrets(client: TestClient) -> None:
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    assert "model" in data
    assert "representation" in data
    assert "parser_version" in data
    assert "ocr_available" in data

    # Verify strictly no secrets or API keys are present anywhere in the response
    json_str = response.text.lower()
    for forbidden in ["api_key", "secret", "password", "token", "minio_secret"]:
        assert forbidden not in json_str, f"Found forbidden secret key '{forbidden}' in config response!"


def test_parse_rejects_non_pdf_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/parse",
        files={"file": ("resume.txt", b"This is plain text", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "INVALID_FILE_TYPE" in str(data)


def test_parse_rejects_invalid_pdf_magic_bytes(client: TestClient) -> None:
    response = client.post(
        "/api/v1/parse",
        files={"file": ("fake_resume.pdf", b"NOT A VALID PDF FILE", "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "INVALID_PDF_HEADER" in str(data)


def test_parse_valid_pdf_with_mock_extractor(client: TestClient, sample_pdf_bytes: bytes) -> None:
    response = client.post(
        "/api/v1/parse",
        files={"file": ("test_resume.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] in ("SUCCESS", "success")
    assert data["violations"] == []
    assert "resume" in data
    assert "metadata" in data
    assert data["metadata"]["filename"] == "test_resume.pdf"
    assert data["metadata"]["requestCount"] == 1
    assert data["metadata"]["pageCount"] >= 1
    assert "diagnostics" in data
    assert "provenance" in data["diagnostics"]


def test_parse_returns_validation_failed_with_usable_resume(client: TestClient, sample_pdf_bytes: bytes) -> None:
    from app.domain.semantic_contract import GroundedString

    class ValidationFailingMockExtractor:
        def extract(self, input_data):
            normal_output = MockSemanticExtractor().extract(input_data)
            normal_output.personal.location = GroundedString(
                value="Nowhere Land",
                raw_value="Nowhere Land",
                source_block_ids=[],  # missing provenance triggers validation failure
            )
            return normal_output

    set_extractor_override(lambda: ValidationFailingMockExtractor())

    response = client.post(
        "/api/v1/parse",
        files={"file": ("test_resume.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] in ("VALIDATION_FAILED", "validation_failed")
    assert "resume" in data and data["resume"] is not None
    assert data["resume"]["personal"]["name"] is not None
    assert "violations" in data and len(data["violations"]) > 0
    assert any("MISSING_PROVENANCE" in v for v in data["violations"])
    assert "metadata" in data
    assert "diagnostics" in data
    assert "provenance" in data["diagnostics"]


def test_diagnostic_endpoint_bounds(client: TestClient, sample_pdf_bytes: bytes) -> None:
    response = client.post(
        "/api/v1/parse/diagnostic?max_blocks=10&max_text_length=50",
        files={"file": ("test_resume.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["page_count"] >= 1
    assert data["block_count"] >= 1
    assert "archetype" in data
    assert "blocks" in data
    assert "regions" in data
    assert "pages" in data
    assert len(data["blocks"]) <= 10
    for b in data["blocks"]:
        assert "block_id" in b
        assert "reading_order" in b
        assert "bbox" in b
        assert len(b["text"]) <= 53  # max 50 chars + optional "..."


def test_fixtures_listing(client: TestClient) -> None:
    response = client.get("/api/v1/fixtures")
    assert response.status_code == 200
    fixtures = response.json()
    assert isinstance(fixtures, list)
    assert len(fixtures) >= 10

    filenames = [f["filename"] for f in fixtures]
    assert "AditCV_SOL.pdf" in filenames
    assert "fresher_hr_resume.pdf" in filenames
    assert "swe_experienced_resume.pdf" in filenames

    # Check fixture attributes
    item = next(f for f in fixtures if f["filename"] == "AditCV_SOL.pdf")
    assert item["suite"] == "regression_12"
    assert item["archetype"] == "standard_cv"
    assert item["candidate_name"] == "Aditi Anand"


def test_fixture_parse_not_found(client: TestClient) -> None:
    response = client.post("/api/v1/fixtures/non_existent_fixture.pdf/parse")
    assert response.status_code == 404
    data = response.json()
    assert "FIXTURE_NOT_FOUND" in str(data)


def test_fixture_parse_existing(client: TestClient) -> None:
    response = client.post("/api/v1/fixtures/AditCV_SOL.pdf/parse")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["metadata"]["filename"] == "AditCV_SOL.pdf"
    assert "resume" in data


def test_parse_partial_resume_status(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """A valid parse missing skills or experience should yield PARTIAL status with success=True."""
    class PartialMockExtractor:
        def extract(self, input_data):
            normal_output = MockSemanticExtractor().extract(input_data)
            normal_output.experience = []  # Empty history -> incomplete but usable
            return normal_output

    set_extractor_override(lambda: PartialMockExtractor())

    response = client.post(
        "/api/v1/parse",
        files={"file": ("test_resume.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "PARTIAL"
    assert data["violations"] == []
    assert data["resume"] is not None


def test_parse_provider_failures(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Ensure provider/network/transport errors return appropriate HTTP 5xx/4xx codes, not SUCCESS."""
    from app.extractors.semantic_extractor import (
        SemanticConfigurationError,
        SemanticRateLimitError,
        SemanticServerError,
        SemanticTimeoutError,
        SemanticTransportError,
    )

    # 1. Transport error -> 502
    class TransportFailExtractor:
        def extract(self, input_data):
            raise SemanticTransportError("Connection reset by peer")

    set_extractor_override(lambda: TransportFailExtractor())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 502
    assert res.json()["detail"]["status"] == "EXTRACTION_FAILED"

    # 2. Timeout error -> 504
    class TimeoutFailExtractor:
        def extract(self, input_data):
            raise SemanticTimeoutError("Provider timed out after 30s")

    set_extractor_override(lambda: TimeoutFailExtractor())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 504
    assert res.json()["detail"]["status"] == "ERROR"

    # 3. Rate limit error -> 429
    class RateLimitFailExtractor:
        def extract(self, input_data):
            raise SemanticRateLimitError("Quota exceeded", retry_after=10.0)

    set_extractor_override(lambda: RateLimitFailExtractor())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 429
    assert res.json()["detail"]["status"] == "ERROR"

    # 4. Configuration error -> 500
    class ConfigFailExtractor:
        def extract(self, input_data):
            raise SemanticConfigurationError("Missing GEMINI_API_KEY")

    set_extractor_override(lambda: ConfigFailExtractor())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 500
    assert res.json()["detail"]["status"] == "ERROR"


def test_parse_oversized_file_rejected(client: TestClient) -> None:
    """Files exceeding MAX_UPLOAD_SIZE (10MB) must be rejected with 413."""
    from app.api.v1.endpoints import MAX_UPLOAD_SIZE
    large_bytes = b"%PDF-1.4" + b"0" * (MAX_UPLOAD_SIZE + 100)
    response = client.post(
        "/api/v1/parse",
        files={"file": ("huge.pdf", large_bytes, "application/pdf")},
    )
    assert response.status_code == 413
    data = response.json()
    assert data["detail"]["error"] == "PAYLOAD_TOO_LARGE"
    assert data["detail"]["status"] == "ERROR"


def test_parse_exactly_one_llm_request_enforced(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Verify that exactly one LLM extract request is performed per parse."""
    call_count = 0

    class TrackingExtractor:
        def extract(self, input_data):
            nonlocal call_count
            call_count += 1
            return MockSemanticExtractor().extract(input_data)

    set_extractor_override(lambda: TrackingExtractor())

    response = client.post(
        "/api/v1/parse",
        files={"file": ("test_resume.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert call_count == 1
    assert data["metadata"]["requestCount"] == 1


def test_parse_preserves_provenance_structure(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Verify provenance records retain canonicalField, sourceBlockIds, sourceTexts, and pageNumbers."""
    response = client.post(
        "/api/v1/parse",
        files={"file": ("test_resume.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    prov_list = data["diagnostics"]["provenance"]
    assert len(prov_list) > 0

    # Inspect a record
    rec = prov_list[0]
    assert "canonicalField" in rec
    assert "sourceBlockIds" in rec
    assert "sourceTexts" in rec
    assert "pageNumbers" in rec
    assert isinstance(rec["sourceBlockIds"], list)
    assert isinstance(rec["sourceTexts"], list)

