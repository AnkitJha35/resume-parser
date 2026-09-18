"""Production API contract hardening tests covering all 9 requirement scenarios."""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.api.dev_server import app
from app.api.models import ParseStatus
from app.api.v1.endpoints import MAX_UPLOAD_SIZE, set_extractor_override
from app.domain.semantic_contract import GroundedString
from app.extractors.semantic_extractor import (
    MockSemanticExtractor,
    SemanticConfigurationError,
    SemanticRateLimitError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
)


@pytest.fixture(autouse=True)
def cleanup_extractor():
    """Ensure mock extractor overrides are cleanly reset between tests."""
    yield
    set_extractor_override(None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a minimal valid 1-page PDF for test requests."""
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


# 1. valid resume -> SUCCESS
def test_api_contract_valid_resume_success(client: TestClient, sample_pdf_bytes: bytes) -> None:
    set_extractor_override(lambda: MockSemanticExtractor())
    response = client.post(
        "/api/v1/parse",
        files={"file": ("jane_doe.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == ParseStatus.SUCCESS.value
    assert data["violations"] == []
    assert data["resume"]["personal"]["name"] == "Jane Doe"
    assert data["metadata"]["requestCount"] == 1


# 2. partial resume -> PARTIAL
def test_api_contract_partial_resume_partial(client: TestClient, sample_pdf_bytes: bytes) -> None:
    class PartialExtractor:
        def extract(self, input_data):
            output = MockSemanticExtractor().extract(input_data)
            output.experience = []  # Missing work/education history
            return output

    set_extractor_override(lambda: PartialExtractor())
    response = client.post(
        "/api/v1/parse",
        files={"file": ("partial_resume.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == ParseStatus.PARTIAL.value
    assert data["violations"] == []
    assert data["resume"] is not None


# 3. semantic validation failure -> VALIDATION_FAILED
def test_api_contract_semantic_validation_failure_validation_failed(client: TestClient, sample_pdf_bytes: bytes) -> None:
    class ValidationFailingExtractor:
        def extract(self, input_data):
            output = MockSemanticExtractor().extract(input_data)
            output.personal.location = GroundedString(
                value="Hallucinated City",
                raw_value="Hallucinated City",
                source_block_ids=[],  # Violates grounding invariant
            )
            return output

    set_extractor_override(lambda: ValidationFailingExtractor())
    response = client.post(
        "/api/v1/parse",
        files={"file": ("unsupported_resume.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] == ParseStatus.VALIDATION_FAILED.value
    assert len(data["violations"]) > 0
    assert any("MISSING_PROVENANCE" in v for v in data["violations"])
    assert data["resume"] is not None


# 4. malformed input -> appropriate 4xx
def test_api_contract_malformed_input_400(client: TestClient) -> None:
    response = client.post(
        "/api/v1/parse",
        files={"file": ("corrupt.pdf", b"NOT_A_VALID_PDF_HEADER", "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "INVALID_PDF_HEADER"
    assert data["detail"]["status"] == ParseStatus.ERROR.value


# 5. unsupported file -> appropriate 4xx
def test_api_contract_unsupported_file_400(client: TestClient) -> None:
    # Non-pdf extension
    response = client.post(
        "/api/v1/parse",
        files={"file": ("resume.docx", b"PK...", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "INVALID_FILE_TYPE"
    assert data["detail"]["status"] == ParseStatus.ERROR.value

    # Oversized file (> 10MB)
    huge_bytes = b"%PDF-1.5\n" + b"A" * (MAX_UPLOAD_SIZE + 100)
    response_large = client.post(
        "/api/v1/parse",
        files={"file": ("huge.pdf", huge_bytes, "application/pdf")},
    )
    assert response_large.status_code == 413
    data_large = response_large.json()
    assert data_large["detail"]["error"] == "PAYLOAD_TOO_LARGE"
    assert data_large["detail"]["status"] == ParseStatus.ERROR.value


# 6. provider failure -> ERROR / appropriate server response
def test_api_contract_provider_failure_responses(client: TestClient, sample_pdf_bytes: bytes) -> None:
    # 6a. Network / transport failure -> 502 with EXTRACTION_FAILED
    class TransportFail:
        def extract(self, _):
            raise SemanticTransportError("Network peer reset")

    set_extractor_override(lambda: TransportFail())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 502
    assert res.json()["detail"]["status"] == ParseStatus.EXTRACTION_FAILED.value

    # 6b. Timeout -> 504 with ERROR
    class TimeoutFail:
        def extract(self, _):
            raise SemanticTimeoutError("LLM call timed out")

    set_extractor_override(lambda: TimeoutFail())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 504
    assert res.json()["detail"]["status"] == ParseStatus.ERROR.value

    # 6c. Rate limit -> 429 with ERROR
    class RateLimitFail:
        def extract(self, _):
            raise SemanticRateLimitError("Rate limit exceeded", retry_after=5.0)

    set_extractor_override(lambda: RateLimitFail())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 429
    assert res.json()["detail"]["status"] == ParseStatus.ERROR.value
    assert res.json()["detail"]["retryAfter"] == 5.0

    # 6d. Config error -> 500 with ERROR
    class ConfigFail:
        def extract(self, _):
            raise SemanticConfigurationError("API credentials unset")

    set_extractor_override(lambda: ConfigFail())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 500
    assert res.json()["detail"]["status"] == ParseStatus.ERROR.value


# 7. response contains violations when applicable
def test_api_contract_response_contains_violations(client: TestClient, sample_pdf_bytes: bytes) -> None:
    class MultiViolationExtractor:
        def extract(self, input_data):
            output = MockSemanticExtractor().extract(input_data)
            output.personal.location = GroundedString(value="Tokyo", raw_value="Tokyo", source_block_ids=[])
            return output

    set_extractor_override(lambda: MultiViolationExtractor())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data["violations"], list)
    assert len(data["violations"]) > 0
    assert any("personal.location" in v for v in data["violations"])


# 8. response preserves provenance
def test_api_contract_response_preserves_provenance(client: TestClient, sample_pdf_bytes: bytes) -> None:
    set_extractor_override(lambda: MockSemanticExtractor())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 200
    data = res.json()
    provenance = data["diagnostics"]["provenance"]
    assert len(provenance) >= 2
    for record in provenance:
        assert "canonicalField" in record
        assert "sourceBlockIds" in record
        assert "sourceTexts" in record
        assert "pageNumbers" in record
        assert isinstance(record["sourceBlockIds"], list)


# 9. exactly one LLM request remains unchanged
def test_api_contract_exactly_one_llm_request_unchanged(client: TestClient, sample_pdf_bytes: bytes) -> None:
    call_count = 0

    class TrackingExtractor:
        def extract(self, input_data):
            nonlocal call_count
            call_count += 1
            return MockSemanticExtractor().extract(input_data)

    set_extractor_override(lambda: TrackingExtractor())
    res = client.post("/api/v1/parse", files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")})
    assert res.status_code == 200
    assert call_count == 1
    assert res.json()["metadata"]["requestCount"] == 1
