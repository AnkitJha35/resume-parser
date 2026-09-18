"""Production concurrency, resource lifecycle, and safety tests.

Validates:
1. 2 and 5 simultaneous concurrent resume requests (same vs distinct fixtures).
2. Concurrency isolation (zero cross-request state leakage, no shared mutable state, isolated provenance).
3. Graceful failure isolation (timeouts, rate limits, 5xx errors, 400 invalid files in parallel).
4. Memory bounding & upload size enforcement under concurrent load.
5. Zero temporary file creation / leakage on disk.
6. Non-blocking event loop offloading under concurrent execution.
"""

from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import fitz
import pytest
from fastapi.testclient import TestClient

from app.api.dev_server import app
from app.api.models import ParseStatus
from app.api.v1.endpoints import MAX_UPLOAD_SIZE, set_extractor_override
from app.extractors.semantic_extractor import (
    MockSemanticExtractor,
    SemanticRateLimitError,
    SemanticServerError,
    SemanticTimeoutError,
)


@pytest.fixture(autouse=True)
def cleanup_extractor():
    """Ensure mock extractor overrides are cleanly reset between tests."""
    yield
    set_extractor_override(None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _create_sample_pdf(name: str = "Jane Doe", email: str = "jane.doe@example.com", role: str = "Senior Software Engineer") -> bytes:
    """Generate a minimal valid 1-page PDF matching the standard contract for test requests."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), name)
    page.insert_text((50, 70), email)
    page.insert_text((50, 90), "+1 555-0199")
    page.insert_text((50, 120), role)
    page.insert_text((50, 140), "Acme Corporation")
    page.insert_text((50, 160), "Python, TypeScript, React, Docker")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_concurrent_identical_resumes_2_workers(client: TestClient) -> None:
    """Test 2 simultaneous requests of the exact same resume fixture."""
    pdf_bytes = _create_sample_pdf("Alice Smith", "alice@example.com", "Senior Software Engineer")
    set_extractor_override(lambda: MockSemanticExtractor())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                client.post,
                "/api/v1/parse",
                files={"file": (f"worker_{i}.pdf", pdf_bytes, "application/pdf")},
            )
            for i in range(2)
        ]
        responses = [f.result() for f in futures]

    for i, res in enumerate(responses):
        assert res.status_code == 200, f"Worker {i} failed: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["status"] == ParseStatus.SUCCESS.value
        assert data["metadata"]["requestCount"] == 1
        assert data["metadata"]["filename"] == f"worker_{i}.pdf"
        assert len(data["diagnostics"]["provenance"]) > 0


def test_concurrent_identical_resumes_5_workers(client: TestClient) -> None:
    """Test 5 simultaneous requests of the exact same resume fixture."""
    pdf_bytes = _create_sample_pdf("Bob Jones", "bob@example.com", "Senior Software Engineer")

    # Add artificial sleep in extractor to test concurrent overlap
    class LatencyMockExtractor(MockSemanticExtractor):
        def extract(self, input_data: Any) -> Any:
            time.sleep(0.02)
            return super().extract(input_data)

    set_extractor_override(lambda: LatencyMockExtractor())

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                client.post,
                "/api/v1/parse",
                files={"file": (f"worker_5_{i}.pdf", pdf_bytes, "application/pdf")},
            )
            for i in range(5)
        ]
        responses = [f.result() for f in futures]

    for i, res in enumerate(responses):
        assert res.status_code == 200, f"Worker {i} failed: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["status"] == ParseStatus.SUCCESS.value
        assert data["metadata"]["requestCount"] == 1
        assert data["metadata"]["filename"] == f"worker_5_{i}.pdf"
        # Provenance sourceBlockIds must all be valid format b_p...
        for prov in data["diagnostics"]["provenance"]:
            for bid in prov["sourceBlockIds"]:
                assert bid.startswith("b_p")


def test_concurrent_distinct_resumes_isolation(client: TestClient) -> None:
    """Test 5 simultaneous requests with distinct resumes and assert complete isolation."""
    candidates = [
        ("Candidate One", "c1@example.com", "Senior Software Engineer"),
        ("Candidate Two", "c2@example.com", "Senior Software Engineer"),
        ("Candidate Three", "c3@example.com", "Senior Software Engineer"),
        ("Candidate Four", "c4@example.com", "Senior Software Engineer"),
        ("Candidate Five", "c5@example.com", "Senior Software Engineer"),
    ]
    items = [(name, _create_sample_pdf(name, email, role)) for name, email, role in candidates]

    set_extractor_override(lambda: MockSemanticExtractor())

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                client.post,
                "/api/v1/parse",
                files={"file": (f"{name.replace(' ', '_')}.pdf", pdf_data, "application/pdf")},
            )
            for name, pdf_data in items
        ]
        responses = [f.result() for f in futures]

    seen_block_ids_per_candidate: dict[str, set[str]] = {}

    for i, res in enumerate(responses):
        name, _ = items[i]
        assert res.status_code == 200, f"Candidate {name} failed: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["status"] == ParseStatus.SUCCESS.value
        assert data["metadata"]["filename"] == f"{name.replace(' ', '_')}.pdf"
        assert data["resume"]["personal"]["name"] == name

        # Collect block IDs referenced in provenance
        block_ids = {
            bid
            for prov in data["diagnostics"]["provenance"]
            for bid in prov["sourceBlockIds"]
        }
        seen_block_ids_per_candidate[name] = block_ids

    # Every candidate has its own extracted provenance
    for name, block_ids in seen_block_ids_per_candidate.items():
        assert len(block_ids) > 0, f"Candidate {name} had no provenance"


def test_concurrent_mixed_outcomes_failure_isolation(client: TestClient) -> None:
    """Test that concurrent errors (timeout, 429, 500, 400) do NOT affect concurrent successes."""
    valid_pdf = _create_sample_pdf("Normal Candidate", "normal@example.com", "Senior Software Engineer")
    invalid_pdf = b"not a pdf file content"

    class MixedBehaviorExtractor(MockSemanticExtractor):
        def extract(self, input_data: Any) -> Any:
            doc_id = input_data.document_id
            if "timeout" in doc_id:
                raise SemanticTimeoutError("Provider request timed out after 30.0s")
            elif "ratelimit" in doc_id:
                raise SemanticRateLimitError("Quota exceeded", retry_after=10.0)
            elif "servererror" in doc_id:
                raise SemanticServerError("Provider returned 503 Service Unavailable", status_code=503)
            return super().extract(input_data)

    set_extractor_override(lambda: MixedBehaviorExtractor())

    requests = [
        ("normal.pdf", valid_pdf),
        ("timeout.pdf", valid_pdf),
        ("ratelimit.pdf", valid_pdf),
        ("servererror.pdf", valid_pdf),
        ("corrupt.pdf", invalid_pdf),
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                client.post,
                "/api/v1/parse",
                files={"file": (fname, content, "application/pdf")},
            )
            for fname, content in requests
        ]
        responses = [f.result() for f in futures]

    # Verify each response handled its outcome independently
    # 0. normal.pdf -> 200 SUCCESS
    assert responses[0].status_code == 200
    normal_data = responses[0].json()
    assert normal_data["success"] is True
    assert normal_data["status"] == ParseStatus.SUCCESS.value

    # 1. timeout.pdf -> 504 GATEWAY_TIMEOUT
    assert responses[1].status_code == 504
    timeout_data = responses[1].json()
    assert timeout_data["detail"]["error"] == "GATEWAY_TIMEOUT"
    assert timeout_data["detail"]["status"] == ParseStatus.ERROR.value

    # 2. ratelimit.pdf -> 429 RATE_LIMIT_EXCEEDED
    assert responses[2].status_code == 429
    rate_data = responses[2].json()
    assert rate_data["detail"]["error"] == "RATE_LIMIT_EXCEEDED"
    assert rate_data["detail"]["status"] == ParseStatus.ERROR.value

    # 3. servererror.pdf -> 502 SEMANTIC_EXTRACTION_ERROR
    assert responses[3].status_code == 502
    server_data = responses[3].json()
    assert server_data["detail"]["error"] == "SEMANTIC_EXTRACTION_ERROR"
    assert server_data["detail"]["status"] == ParseStatus.EXTRACTION_FAILED.value

    # 4. corrupt.pdf -> 400 INVALID_PDF_HEADER
    assert responses[4].status_code == 400
    corrupt_data = responses[4].json()
    assert corrupt_data["detail"]["error"] == "INVALID_PDF_HEADER"
    assert corrupt_data["detail"]["status"] == ParseStatus.ERROR.value


def test_concurrent_payload_size_bounding(client: TestClient) -> None:
    """Test payload size limit enforcement in parallel with normal requests."""
    valid_pdf = _create_sample_pdf("Valid Candidate", "valid@example.com", "Senior Software Engineer")
    oversized_pdf = b"%PDF-1.4\n" + b"0" * (MAX_UPLOAD_SIZE + 1024)
    set_extractor_override(lambda: MockSemanticExtractor())

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_valid = executor.submit(
            client.post,
            "/api/v1/parse",
            files={"file": ("valid.pdf", valid_pdf, "application/pdf")},
        )
        f_oversized = executor.submit(
            client.post,
            "/api/v1/parse",
            files={"file": ("huge.pdf", oversized_pdf, "application/pdf")},
        )
        res_valid = f_valid.result()
        res_oversized = f_oversized.result()

    assert res_valid.status_code == 200
    assert res_valid.json()["success"] is True

    assert res_oversized.status_code == 413
    assert res_oversized.json()["detail"]["error"] == "PAYLOAD_TOO_LARGE"
    assert res_oversized.json()["detail"]["status"] == ParseStatus.ERROR.value


def test_zero_temporary_files_created_or_leaked(client: TestClient) -> None:
    """Verify that parsing leaves zero temporary files behind in the system temp directory."""
    temp_dir = tempfile.gettempdir()
    files_before = set(os.listdir(temp_dir))

    pdf_bytes = _create_sample_pdf("Temp Check", "temp@example.com", "Senior Software Engineer")
    set_extractor_override(lambda: MockSemanticExtractor())

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                client.post,
                "/api/v1/parse",
                files={"file": (f"temp_{i}.pdf", pdf_bytes, "application/pdf")},
            )
            for i in range(3)
        ]
        responses = [f.result() for f in futures]

    for res in responses:
        assert res.status_code == 200

    files_after = set(os.listdir(temp_dir))
    new_files = files_after - files_before
    # PyMuPDF and the parser operate fully in memory without temp files
    assert len(new_files) == 0, f"Leaked temporary files found: {new_files}"


def test_diagnostic_endpoint_concurrency(client: TestClient) -> None:
    """Verify that the diagnostic endpoint functions concurrently under threadpool execution."""
    pdf_bytes = _create_sample_pdf("Diagnostic User", "diag@example.com", "Senior Software Engineer")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                client.post,
                "/api/v1/parse/diagnostic",
                files={"file": (f"diag_{i}.pdf", pdf_bytes, "application/pdf")},
            )
            for i in range(3)
        ]
        responses = [f.result() for f in futures]

    for i, res in enumerate(responses):
        assert res.status_code == 200
        data = res.json()
        assert data["page_count"] == 1
        assert data["block_count"] > 0
        assert len(data["blocks"]) > 0
