"""Phase 9-3: End-to-end production failure-path and orchestration verification."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
import pytest
import httpx

from app.core.config import Settings
from app.core.exceptions import StorageClientError
from app.domain.document import Document
from app.domain.resume import Resume
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedPersonal,
    GroundedString,
    SemanticInput,
    SemanticOutput,
)
from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.providers.ollama import OllamaSemanticExtractor
from app.extractors.semantic_extractor import (
    SemanticExtractionError,
    SemanticServerError,
    SemanticValidationError,
)
from app.pipeline.parser import PipelineError, ResumeParser
from app.pipeline.semantic_pipeline import parse_document_semantically
from app.services.resume_request_service import ResumeRequestService
from tests.test_semantic_llm_contract import _make_test_document


def _make_e2e_settings(**kwargs: Any) -> Settings:
    defaults = {
        "kafka_brokers": "localhost:9092",
        "minio_endpoint": "localhost:9000",
        "minio_access_key": "minioadmin",
        "minio_secret_key": "minioadmin",
        "minio_bucket_name": "resumes",
        "gemini_api_key": "AIzaSyPROD_GEMINI_KEY_987654321",
        "gemini_max_retries": 2,
        "gemini_timeout": 30.0,
        "ollama_timeout": 120.0,
        "semantic_fallback_enabled": True,
        "parser_mode": "layout",
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class TrackingKafkaProducer:
    """Mock Kafka producer that tracks all published events and can simulate broker failures."""

    def __init__(self, settings: Settings, should_fail_topic: str | None = None) -> None:
        self.settings = settings
        self.should_fail_topic = should_fail_topic
        self.published_events: list[tuple[str, dict[str, Any]]] = []

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def send_json(self, topic: str, message: dict[str, Any]) -> None:
        if self.should_fail_topic == topic or self.should_fail_topic == "all":
            raise RuntimeError(f"Kafka broker unavailable on topic {topic}")
        self.published_events.append((topic, message))


class TrackingStorageClient:
    """Mock storage client that counts download attempts and can simulate failures."""

    def __init__(self, settings: Settings, max_failures: int = 0) -> None:
        self.settings = settings
        self.max_failures = max_failures
        self.download_attempts = 0

    def download_pdf(self, storage_key: str) -> bytes:
        self.download_attempts += 1
        if self.download_attempts <= self.max_failures:
            raise StorageClientError(f"MinIO read error attempt={self.download_attempts}")
        return b"%PDF-1.4 mock valid pdf bytes"


class SemanticLayoutParser:
    """Deterministic parser adapter connecting layout parsing with an injected SemanticExtractor."""

    def __init__(self, extractor: Any, document: Document | None = None) -> None:
        self.extractor = extractor
        self.document = document or _make_test_document()
        self.parse_calls = 0

    def parse_with_layout_pipeline(self, raw_bytes: bytes) -> Resume:
        self.parse_calls += 1
        return parse_document_semantically(
            self.document,
            self.extractor,
            document_id="doc-e2e-1",
        )

    def parse_with_semantic_pipeline(
        self,
        raw_bytes: bytes,
        semantic_extractor: Any = None,
        document_id: str = "doc-e2e-1",
    ) -> Resume:
        self.parse_calls += 1
        ext = semantic_extractor or self.extractor
        return parse_document_semantically(
            self.document,
            ext,
            document_id=document_id,
        )

    def parse(self, raw_bytes: bytes) -> Resume:
        return self.parse_with_layout_pipeline(raw_bytes)


# =====================================================================
# 1. Complete Successful Production Path
# =====================================================================


@pytest.mark.anyio
async def test_e2e_complete_successful_production_path():
    """1. Full success: Kafka payload -> storage -> parser -> semantic extraction -> validation -> completed Kafka event."""
    settings = _make_e2e_settings()
    producer = TrackingKafkaProducer(settings)
    storage = TrackingStorageClient(settings, max_failures=0)

    # Gemini mock returns valid grounded output
    gemini_calls = 0

    def gemini_mock(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        llm_payload = {
            "document_archetype": "standard_cv",
            "personal": {
                "name": {"value": "John Doe", "source_block_ids": ["b_p1_0"]},
                "email": {"value": "john.doe@example.com", "source_block_ids": ["b_p1_1"]},
            },
            "experience": [
                {
                    "title": {"value": "Senior Software Engineer", "source_block_ids": ["b_p1_5"]},
                    "company": {"value": "Acme Corporation", "source_block_ids": ["b_p1_4"]},
                }
            ],
        }
        envelope = {
            "candidates": [
                {
                    "content": {"parts": [{"text": json.dumps(llm_payload)}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 3500,
                "candidatesTokenCount": 250,
                "totalTokenCount": 3750,
            },
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(gemini_mock))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    parser = SemanticLayoutParser(extractor=extractor)

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=lambda: parser,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-success",
        "resumeId": "resume-e2e-success",
        "storageKey": "resumes/2026/09/sample.pdf",
    }
    await service.process(msg)
    await service.shutdown()

    # Exact call counts and events
    assert storage.download_attempts == 1
    assert parser.parse_calls == 1
    assert gemini_calls == 1
    assert len(producer.published_events) == 1

    topic, event = producer.published_events[0]
    assert topic == settings.kafka_topic_completed
    assert event["jobId"] == "job-e2e-success"
    assert event["resumeId"] == "resume-e2e-success"
    assert event["status"] == "COMPLETED"
    assert event["result"]["personal"]["name"] == "John Doe"
    assert event["result"]["personal"]["email"] == "john.doe@example.com"
    assert "processingTimeMs" in event["metadata"]


# =====================================================================
# 2. Gemini Transient Failure -> Ollama Fallback Success
# =====================================================================


@pytest.mark.anyio
async def test_e2e_gemini_transient_failure_falls_back_to_ollama_and_succeeds():
    """2. Gemini 429 retries 2 times (3 calls total) -> fallback to Ollama (1 call) -> completed event."""
    settings = _make_e2e_settings(semantic_fallback_enabled=True)
    producer = TrackingKafkaProducer(settings)
    storage = TrackingStorageClient(settings)

    gemini_calls = 0
    ollama_calls = 0

    def gemini_mock(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        return httpx.Response(429, text="Resource has been exhausted (e.g. check quota)", request=request)

    def ollama_mock(request: httpx.Request) -> httpx.Response:
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
            "prompt_eval_count": 210,
            "eval_count": 85,
        }
        return httpx.Response(200, json=envelope, request=request)

    gemini_client = httpx.Client(transport=httpx.MockTransport(gemini_mock))
    ollama_client = httpx.Client(transport=httpx.MockTransport(ollama_mock))

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        max_retries=2,
        initial_backoff=0.01,
        client=gemini_client,
        sleep_fn=lambda _: None,
    )
    fallback = OllamaSemanticExtractor(client=ollama_client)
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    parser = SemanticLayoutParser(extractor=composite)

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=lambda: parser,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-fallback",
        "resumeId": "resume-e2e-fallback",
        "storageKey": "resumes/sample.pdf",
    }
    await service.process(msg)
    await service.shutdown()

    # Exact call count assertions
    assert gemini_calls == 3  # Initial + 2 retries
    assert ollama_calls == 1  # Fallback invoked once
    assert len(producer.published_events) == 1

    topic, event = producer.published_events[0]
    assert topic == settings.kafka_topic_completed
    assert event["jobId"] == "job-e2e-fallback"
    assert event["status"] == "COMPLETED"

    # Telemetry assertions
    meta = composite.last_usage_metadata
    assert meta["fallback_invoked"] is True
    assert meta["primary_provider"] == "gemini"
    assert meta["fallback_provider"] == "ollama"
    assert meta["primary_error"] == "SemanticRateLimitError"
    assert meta["status"] == "success"


# =====================================================================
# 3. Dual Provider Failure
# =====================================================================


@pytest.mark.anyio
async def test_e2e_dual_provider_failure_publishes_semantic_extraction_failed():
    """3. Gemini exhausts 3 calls + Ollama fails (1 call) -> publishes exactly 1 SEMANTIC_EXTRACTION_FAILED event."""
    settings = _make_e2e_settings(semantic_fallback_enabled=True)
    producer = TrackingKafkaProducer(settings)
    storage = TrackingStorageClient(settings)

    gemini_calls = 0
    ollama_calls = 0

    def gemini_mock(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        return httpx.Response(500, text="Internal Server Error", request=request)

    def ollama_mock(request: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        raise httpx.ConnectError("Ollama connection refused")

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        max_retries=2,
        initial_backoff=0.01,
        client=httpx.Client(transport=httpx.MockTransport(gemini_mock)),
        sleep_fn=lambda _: None,
    )
    fallback = OllamaSemanticExtractor(
        client=httpx.Client(transport=httpx.MockTransport(ollama_mock))
    )
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    parser = SemanticLayoutParser(extractor=composite)

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=lambda: parser,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-dual-fail",
        "resumeId": "resume-e2e-dual-fail",
        "storageKey": "resumes/sample.pdf",
    }
    # Must return normally without escaping
    await service.process(msg)
    await service.shutdown()

    # Exact call count assertions
    assert gemini_calls == 3
    assert ollama_calls == 1
    assert len(producer.published_events) == 1

    topic, event = producer.published_events[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-e2e-dual-fail"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "SEMANTIC_EXTRACTION_FAILED"
    assert event["error"]["message"] == "Semantic extraction failed."


# =====================================================================
# 4. Semantic Validation Failure
# =====================================================================


@pytest.mark.anyio
async def test_e2e_semantic_validation_failure_publishes_semantic_validation_failed():
    """4. Syntactically valid but provenance-invalid SemanticOutput -> publishes SEMANTIC_VALIDATION_FAILED, no fallback."""
    settings = _make_e2e_settings(semantic_fallback_enabled=True)
    producer = TrackingKafkaProducer(settings)
    storage = TrackingStorageClient(settings)

    gemini_calls = 0
    ollama_calls = 0

    def gemini_mock(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        llm_payload = {
            "document_archetype": "standard_cv",
            "personal": {
                # Hallucinated block id
                "name": {"value": "John Doe", "source_block_ids": ["b_nonexistent_999"]},
            },
            "experience": [
                {
                    "title": {"value": "Senior Software Engineer", "source_block_ids": ["b_p1_5"]},
                    "company": {"value": "Acme Corporation", "source_block_ids": ["b_p1_4"]},
                }
            ],
        }
        envelope = {
            "candidates": [
                {
                    "content": {"parts": [{"text": json.dumps(llm_payload)}]},
                    "finishReason": "STOP",
                }
            ]
        }
        return httpx.Response(200, json=envelope, request=request)

    def ollama_mock(request: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        return httpx.Response(200, json={}, request=request)

    primary = GeminiSemanticExtractor(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(gemini_mock)),
    )
    fallback = OllamaSemanticExtractor(
        client=httpx.Client(transport=httpx.MockTransport(ollama_mock))
    )
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    parser = SemanticLayoutParser(extractor=composite)

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=lambda: parser,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-val-fail",
        "resumeId": "resume-e2e-val-fail",
        "storageKey": "resumes/sample.pdf",
    }
    await service.process(msg)
    await service.shutdown()

    # Exact call count: exactly 1 Gemini call, 0 Ollama calls
    assert gemini_calls == 1
    assert ollama_calls == 0
    assert len(producer.published_events) == 1

    topic, event = producer.published_events[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-e2e-val-fail"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "SEMANTIC_VALIDATION_FAILED"
    assert event["error"]["message"] == "Semantic output failed provenance validation."


# =====================================================================
# 5. Storage Failure (Exhausts Download Retries)
# =====================================================================


@pytest.mark.anyio
async def test_e2e_storage_failure_exhausts_retries_and_skips_semantic_extraction():
    """5. Storage client fails 3 times -> publishes STORAGE_ERROR, semantic extractor NEVER called."""
    settings = _make_e2e_settings()
    producer = TrackingKafkaProducer(settings)
    # Storage fails permanently (all 3 attempts)
    storage = TrackingStorageClient(settings, max_failures=10)

    extractor_calls = 0

    class TrackingExtractor:
        def extract(self, input_data: SemanticInput) -> SemanticOutput:
            nonlocal extractor_calls
            extractor_calls += 1
            return SemanticOutput()

    parser = SemanticLayoutParser(extractor=TrackingExtractor())

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=lambda: parser,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-storage-fail",
        "resumeId": "resume-e2e-storage-fail",
        "storageKey": "resumes/missing.pdf",
    }
    await service.process(msg)
    await service.shutdown()

    assert storage.download_attempts == 3  # Exactly 3 download attempts
    assert extractor_calls == 0  # Semantic extraction never reached
    assert len(producer.published_events) == 1

    topic, event = producer.published_events[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-e2e-storage-fail"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "STORAGE_ERROR"


# =====================================================================
# 6. Pipeline Failure (Invalid / Corrupted PDF)
# =====================================================================


@pytest.mark.anyio
async def test_e2e_pipeline_failure_publishes_pipeline_error_and_skips_semantic_extraction():
    """6. Layout/parser pipeline raises PipelineError -> publishes pipeline error code, semantic extractor NEVER called."""
    settings = _make_e2e_settings()
    producer = TrackingKafkaProducer(settings)
    storage = TrackingStorageClient(settings)

    extractor_calls = 0

    class CorruptedPDFParser:
        def parse_with_layout_pipeline(self, raw_bytes: bytes) -> Resume:
            raise PipelineError("INVALID_PDF", "Corrupted or non-PDF file detected.")

        def parse_with_semantic_pipeline(self, raw_bytes: bytes, semantic_extractor=None, document_id="doc-1") -> Resume:
            raise PipelineError("INVALID_PDF", "Corrupted or non-PDF file detected.")

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=CorruptedPDFParser,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-pipe-fail",
        "resumeId": "resume-e2e-pipe-fail",
        "storageKey": "resumes/corrupt.bin",
    }
    await service.process(msg)
    await service.shutdown()

    assert storage.download_attempts == 1
    assert extractor_calls == 0
    assert len(producer.published_events) == 1

    topic, event = producer.published_events[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-e2e-pipe-fail"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "INVALID_PDF"
    assert event["error"]["message"] == "Corrupted or non-PDF file detected."


# =====================================================================
# 7. Kafka Publication Failure Lifecycle
# =====================================================================


@pytest.mark.anyio
async def test_e2e_kafka_producer_failure_lifecycle(caplog):
    """7. If Kafka producer fails while publishing completed event, process() catches exception and tries publishing failed event."""
    caplog.set_level(logging.ERROR)
    settings = _make_e2e_settings()
    # Producer will fail on completed topic
    producer = TrackingKafkaProducer(settings, should_fail_topic=settings.kafka_topic_completed)
    storage = TrackingStorageClient(settings)

    class FastSuccessParser:
        def parse_with_layout_pipeline(self, raw_bytes: bytes) -> Any:
            class DummyResume:
                metadata = {"pageCount": 1}

                def model_dump(self):
                    return {"name": "Dummy"}

            return DummyResume()

        def parse_with_semantic_pipeline(self, raw_bytes: bytes, semantic_extractor=None, document_id="doc-1") -> Any:
            class DummyResume:
                metadata = {"pageCount": 1}

                def model_dump(self):
                    return {"name": "Dummy"}

            return DummyResume()

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=FastSuccessParser,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-kafka-fail",
        "resumeId": "resume-e2e-kafka-fail",
        "storageKey": "resumes/sample.pdf",
    }
    await service.process(msg)
    await service.shutdown()

    # The failure to publish COMPLETED was caught and published to FAILED topic
    assert len(producer.published_events) == 1
    topic, event = producer.published_events[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-e2e-kafka-fail"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "RESUME_PARSE_FAILED"


# =====================================================================
# 8. Security & Secret Redaction Invariants
# =====================================================================


@pytest.mark.anyio
async def test_e2e_security_invariants_no_leaks_in_kafka_events_or_logs(caplog):
    """8. Ensure zero API keys, auth tokens, candidate PII, or raw LLM prompts leak into Kafka events or logs."""
    caplog.set_level(logging.DEBUG)
    secret_key = "AIzaSyPROD_GEMINI_KEY_987654321"
    settings = _make_e2e_settings(gemini_api_key=secret_key)
    producer = TrackingKafkaProducer(settings)
    storage = TrackingStorageClient(settings)

    # Gemini throws error with sensitive candidate text and API key in body
    def gemini_mock(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            text=f'{{"error": "Internal error for key {secret_key} and prompt candidate John Doe"}}',
            request=request,
        )

    primary = GeminiSemanticExtractor(
        api_key=secret_key,
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(gemini_mock)),
    )
    parser = SemanticLayoutParser(extractor=primary)

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=lambda: parser,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-sec",
        "resumeId": "resume-e2e-sec",
        "storageKey": "resumes/sample.pdf",
    }
    await service.process(msg)
    await service.shutdown()

    # Check Kafka events
    assert len(producer.published_events) == 1
    topic, event = producer.published_events[0]
    event_str = json.dumps(event)
    assert secret_key not in event_str
    assert "AIzaSy" not in event_str
    assert "x-goog-api-key" not in event_str
    assert "John Doe" not in event_str

    # Check logs
    for record in caplog.records:
        msg_str = record.getMessage()
        assert secret_key not in msg_str
        assert "x-goog-api-key" not in msg_str


# =====================================================================
# 9. Handler Boundary Lifecycle (No Unhandled Exceptions)
# =====================================================================


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure_parser_cls",
    [
        lambda: SemanticLayoutParser(extractor=GeminiSemanticExtractor(api_key="bad-key", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, request=r))))),
        lambda: SemanticLayoutParser(extractor=GeminiSemanticExtractor(api_key="bad-key", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, request=r))))),
        lambda: SemanticLayoutParser(extractor=GeminiSemanticExtractor(api_key="bad-key", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"candidates": []}, request=r))))),
    ],
)
async def test_e2e_handler_boundary_all_failures_handled_without_unhandled_exceptions(failure_parser_cls):
    """9. Verify all semantic failure modes return normally across the Kafka handler boundary."""
    settings = _make_e2e_settings(semantic_fallback_enabled=False)
    producer = TrackingKafkaProducer(settings)
    storage = TrackingStorageClient(settings)

    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=lambda s: storage,
        kafka_producer_cls=lambda s: producer,
        parser_cls=failure_parser_cls,
    )
    await service.start()

    msg = {
        "jobId": "job-e2e-boundary",
        "resumeId": "resume-e2e-boundary",
        "storageKey": "resumes/sample.pdf",
    }
    # Crucial assertion: process() MUST NOT raise
    await service.process(msg)
    await service.shutdown()

    assert len(producer.published_events) == 1
    topic, event = producer.published_events[0]
    assert topic == settings.kafka_topic_failed
    assert event["status"] == "FAILED"
