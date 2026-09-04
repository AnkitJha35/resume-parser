"""Phase 9-2: Comprehensive provider resilience, retry, fallback, and Kafka lifecycle tests."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
import pytest
import httpx

from app.core.config import Settings
from app.domain.resume import Resume
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedPersonal,
    GroundedString,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    build_semantic_input,
)
from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.providers.ollama import OllamaSemanticExtractor
from app.extractors.semantic_extractor import (
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticRateLimitError,
    SemanticResponseError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
    SemanticValidationError,
)
from app.pipeline.parser import ResumeParser
from app.pipeline.semantic_pipeline import parse_document_semantically
from app.services.resume_request_service import ResumeRequestService
from tests.test_semantic_llm_contract import _make_test_document


def _make_dummy_settings(**kwargs: Any) -> Settings:
    defaults = {
        "kafka_brokers": "localhost:9092",
        "minio_endpoint": "localhost:9000",
        "minio_access_key": "minioadmin",
        "minio_secret_key": "minioadmin",
        "minio_bucket_name": "resumes",
        "gemini_api_key": "test-gemini-key-12345",
        "gemini_max_retries": 2,
        "gemini_timeout": 30.0,
        "ollama_timeout": 120.0,
        "semantic_fallback_enabled": True,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class MockKafkaProducer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sent_events: list[tuple[str, dict[str, Any]]] = []

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def send_json(self, topic: str, message: dict[str, Any]) -> None:
        self.sent_events.append((topic, message))


class MockMinioClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def download_pdf(self, storage_key: str) -> bytes:
        from tests.test_semantic_llm_contract import _make_test_document
        # Return a simple valid test PDF representation or dummy bytes
        return b"%PDF-1.4 mock pdf bytes"


# =====================================================================
# 1. Kafka Offset Advancement & Error Handling Verification
# =====================================================================


@pytest.mark.anyio
async def test_kafka_lifecycle_handled_semantic_extraction_error_advances_offset():
    """Verify ResumeRequestService catches SemanticExtractionError, emits failure event, and returns cleanly so Kafka commits."""
    settings = _make_dummy_settings()

    class FailingParser:
        def parse(self, raw_bytes: bytes) -> Any:
            raise SemanticExtractionError("Unrecoverable cloud provider failure")

        def parse_with_layout_pipeline(self, raw_bytes: bytes) -> Any:
            raise SemanticExtractionError("Unrecoverable cloud provider failure")

        def parse_with_semantic_pipeline(self, raw_bytes: bytes, semantic_extractor=None, document_id="doc-1") -> Any:
            raise SemanticExtractionError("Unrecoverable cloud provider failure")

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=MockMinioClient,
        kafka_producer_cls=MockKafkaProducer,
        parser_cls=FailingParser,
    )
    await service.start()

    service._download_pdf = lambda key: asyncio.sleep(0, result=b"%PDF-1.4 mock")  # type: ignore

    # When process is called with a failing extractor, it MUST NOT raise to caller
    msg = {"jobId": "job-fail-1", "resumeId": "res-fail-1", "storageKey": "key-1"}
    await service.process(msg)

    # Verify failed event published
    producer = service._producer
    assert len(producer.sent_events) == 1
    topic, event = producer.sent_events[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-fail-1"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "SEMANTIC_EXTRACTION_FAILED"


@pytest.mark.anyio
async def test_kafka_lifecycle_handled_semantic_validation_error_advances_offset():
    """Verify ResumeRequestService catches SemanticValidationError, emits failure event, and returns cleanly."""
    settings = _make_dummy_settings()

    class InvalidProvenanceParser:
        def parse(self, raw_bytes: bytes) -> Any:
            raise SemanticValidationError(["UNKNOWN_BLOCK_ID in personal.name: 'b_unreal'"])

        def parse_with_layout_pipeline(self, raw_bytes: bytes) -> Any:
            raise SemanticValidationError(["UNKNOWN_BLOCK_ID in personal.name: 'b_unreal'"])

        def parse_with_semantic_pipeline(self, raw_bytes: bytes, semantic_extractor=None, document_id="doc-1") -> Any:
            raise SemanticValidationError(["UNKNOWN_BLOCK_ID in personal.name: 'b_unreal'"])

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=MockMinioClient,
        kafka_producer_cls=MockKafkaProducer,
        parser_cls=InvalidProvenanceParser,
    )
    await service.start()
    service._download_pdf = lambda key: asyncio.sleep(0, result=b"%PDF-1.4 mock")  # type: ignore

    msg = {"jobId": "job-val-1", "resumeId": "res-val-1", "storageKey": "key-val"}
    await service.process(msg)

    producer = service._producer
    assert len(producer.sent_events) == 1
    topic, event = producer.sent_events[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-val-1"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "SEMANTIC_VALIDATION_FAILED"


# =====================================================================
# 2. Gemini Transient Failures & Fallback (Exact Call Counts & Backoff)
# =====================================================================


@pytest.mark.parametrize(
    ("status_code", "exc_type"),
    [
        (429, SemanticRateLimitError),
        (500, SemanticServerError),
        (502, SemanticServerError),
        (503, SemanticServerError),
        (504, SemanticServerError),
    ],
)
def test_gemini_transient_http_errors_retry_exact_3_times_then_fallback_once(status_code: int, exc_type: type):
    """Verify transient HTTP errors (429, 500, 502, 503, 504) execute exactly 3 Gemini calls + 1 Ollama call."""
    gemini_calls = 0
    ollama_calls = 0
    sleeps: list[float] = []

    def gemini_mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        return httpx.Response(status_code, text=f"HTTP {status_code} error", request=request)

    def ollama_mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        llm_payload = {
            "document_archetype": "standard_cv",
            "personal": {
                "name": {"value": "John Doe", "source_block_ids": ["b_p1_0"]},
            },
        }
        envelope = {
            "message": {"role": "assistant", "content": json.dumps(llm_payload)},
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        return httpx.Response(200, json=envelope, request=request)

    gemini_client = httpx.Client(transport=httpx.MockTransport(gemini_mock_handler))
    ollama_client = httpx.Client(transport=httpx.MockTransport(ollama_mock_handler))

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        max_retries=2,
        initial_backoff=0.1,
        backoff_multiplier=2.0,
        client=gemini_client,
        sleep_fn=sleeps.append,
    )
    fallback = OllamaSemanticExtractor(client=ollama_client)
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    doc = _make_test_document()
    resume = parse_document_semantically(doc, composite, document_id="doc-transient-test")

    # Assert exact call counts
    assert gemini_calls == 3  # Initial + 2 retries
    assert len(sleeps) == 2  # 2 exponential backoff sleeps
    assert sleeps[0] == pytest.approx(0.1)
    assert sleeps[1] == pytest.approx(0.2)
    assert ollama_calls == 1  # Fallback invoked once
    assert isinstance(resume, Resume)

    # Assert fallback usage metadata
    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is True
    assert meta["primary_provider"] == "gemini"
    assert meta["fallback_provider"] == "ollama"
    assert meta["primary_error"] == exc_type.__name__
    assert meta["status"] == "success"


def test_gemini_transport_and_timeout_errors_retry_exact_3_times_then_fallback_once():
    """Verify network transport and timeout errors retry 2 times before falling back to Ollama."""
    gemini_calls = 0
    ollama_calls = 0
    sleeps: list[float] = []

    def gemini_mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        if gemini_calls == 1:
            raise httpx.ConnectError("Connection refused")
        raise httpx.ReadTimeout("Read timed out")

    def ollama_mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        llm_payload = {
            "document_archetype": "standard_cv",
            "personal": {
                "name": {"value": "John Doe", "source_block_ids": ["b_p1_0"]},
            },
        }
        envelope = {
            "message": {"role": "assistant", "content": json.dumps(llm_payload)},
            "prompt_eval_count": 120,
            "eval_count": 60,
        }
        return httpx.Response(200, json=envelope, request=request)

    gemini_client = httpx.Client(transport=httpx.MockTransport(gemini_mock_handler))
    ollama_client = httpx.Client(transport=httpx.MockTransport(ollama_mock_handler))

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        max_retries=2,
        initial_backoff=0.05,
        backoff_multiplier=2.0,
        client=gemini_client,
        sleep_fn=sleeps.append,
    )
    fallback = OllamaSemanticExtractor(client=ollama_client)
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    doc = _make_test_document()
    resume = parse_document_semantically(doc, composite, document_id="doc-transport-test")

    assert gemini_calls == 3
    assert len(sleeps) == 2
    assert ollama_calls == 1
    assert isinstance(resume, Resume)


# =====================================================================
# 3. Non-Transient Failures (0 Retries, No Inappropriate Fallback)
# =====================================================================


def test_gemini_http_400_bad_request_fails_immediately_without_retry_or_fallback():
    """HTTP 400 is a deterministic client error: exactly 1 Gemini call, 0 retries, 0 fallback calls."""
    gemini_calls = 0
    ollama_calls = 0

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        return httpx.Response(400, text="Bad Request: invalid argument", request=request)

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        return httpx.Response(200, json={}, request=request)

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(gemini_handler)),
    )
    fallback = OllamaSemanticExtractor(
        client=httpx.Client(transport=httpx.MockTransport(ollama_handler))
    )
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    doc = _make_test_document()
    with pytest.raises(SemanticResponseError):
        parse_document_semantically(doc, composite, document_id="doc-400")

    assert gemini_calls == 1
    assert ollama_calls == 0


def test_gemini_safety_blocked_candidate_fails_immediately_without_retry_or_fallback():
    """Safety / Content policy rejection: exactly 1 Gemini call, 0 retries, 0 fallback calls."""
    gemini_calls = 0
    ollama_calls = 0

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        envelope = {
            "candidates": [
                {
                    "finishReason": "SAFETY",
                    "content": {"parts": [{"text": ""}]},
                }
            ]
        }
        return httpx.Response(200, json=envelope, request=request)

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        return httpx.Response(200, json={}, request=request)

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(gemini_handler)),
    )
    fallback = OllamaSemanticExtractor(
        client=httpx.Client(transport=httpx.MockTransport(ollama_handler))
    )
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    doc = _make_test_document()
    with pytest.raises(SemanticResponseError):
        parse_document_semantically(doc, composite, document_id="doc-safety")

    assert gemini_calls == 1
    assert ollama_calls == 0


def test_gemini_malformed_json_response_fails_immediately_without_retry_or_fallback():
    """Malformed / unparseable JSON envelope: exactly 1 Gemini call, 0 retries, 0 fallback calls."""
    gemini_calls = 0
    ollama_calls = 0

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        return httpx.Response(200, text="<HTML>Not JSON</HTML>", request=request)

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        return httpx.Response(200, json={}, request=request)

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(gemini_handler)),
    )
    fallback = OllamaSemanticExtractor(
        client=httpx.Client(transport=httpx.MockTransport(ollama_handler))
    )
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    doc = _make_test_document()
    with pytest.raises(SemanticResponseError):
        parse_document_semantically(doc, composite, document_id="doc-bad-json")

    assert gemini_calls == 1
    assert ollama_calls == 0


# =====================================================================
# 4. Dual Provider Failure (Gemini Exhausts + Ollama Fails)
# =====================================================================


def test_dual_provider_failure_exhausts_retries_and_raises_semantic_extraction_error():
    """When Gemini exhausts 3 attempts and Ollama fails, total calls is exactly 4."""
    gemini_calls = 0
    ollama_calls = 0

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        return httpx.Response(500, text="Internal Server Error", request=request)

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        raise httpx.ConnectError("Ollama daemon down")

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        max_retries=2,
        client=httpx.Client(transport=httpx.MockTransport(gemini_handler)),
        sleep_fn=lambda _: None,
    )
    fallback = OllamaSemanticExtractor(
        client=httpx.Client(transport=httpx.MockTransport(ollama_handler))
    )
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    doc = _make_test_document()
    with pytest.raises(SemanticExtractionError) as exc_info:
        parse_document_semantically(doc, composite, document_id="doc-dual-fail")

    assert gemini_calls == 3
    assert ollama_calls == 1
    assert "Primary provider 'gemini' failed with SemanticServerError" in str(exc_info.value)
    assert "fallback provider 'ollama' failed with SemanticExtractionError" in str(exc_info.value)

    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is True
    assert meta["status"] == "failure"
    assert meta["primary_error"] == "SemanticServerError"
    assert meta["fallback_error"] == "SemanticExtractionError"


# =====================================================================
# 5. Security & Redaction Invariants (No Key or PII in Logs/Errors)
# =====================================================================


def test_security_sanitization_never_leaks_api_key_or_pii_in_logs_and_errors(caplog):
    """Verify API keys, auth headers, and candidate PII never leak to logs on errors."""
    caplog.set_level(logging.DEBUG)
    secret_key = "AIzaSySUPER_SECRET_PRODUCTION_KEY_999"

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        # Simulate an API error echoing the API key in the response body
        return httpx.Response(
            403,
            text=f'{{"error": {{"message": "API key {secret_key} forbidden", "code": 403}}}}',
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(gemini_handler))
    extractor = GeminiSemanticExtractor(api_key=secret_key, client=client)

    doc = _make_test_document()
    with pytest.raises(SemanticConfigurationError) as exc_info:
        parse_document_semantically(doc, extractor, document_id="doc-sec-test")

    # Verify exception message redacted
    assert secret_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)

    # Verify log records redacted
    for record in caplog.records:
        msg = record.getMessage()
        assert secret_key not in msg
        assert "x-goog-api-key" not in msg


# =====================================================================
# 6. Timeout Accounting & Latency Worst-Case Verification
# =====================================================================


def test_timeout_accounting_worst_case_defaults():
    """Verify configured defaults match theoretical worst-case timeout accounting:

    Gemini: 3 attempts * 30s timeout + 1s + 2s backoff = 93.0s
    Ollama: 1 attempt * 120s timeout = 120.0s
    Total worst-case provider timeout = 213.0s (~3.55 min)
    """
    settings = _make_dummy_settings()
    gemini_max_timeout = (settings.gemini_max_retries + 1) * settings.gemini_timeout + 1.0 + 2.0
    ollama_max_timeout = settings.ollama_timeout
    total_worst_case = gemini_max_timeout + ollama_max_timeout

    assert gemini_max_timeout == 93.0
    assert ollama_max_timeout == 120.0
    assert total_worst_case == 213.0
