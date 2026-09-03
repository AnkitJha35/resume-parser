import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import StorageClientError
from app.extractors.semantic_extractor import (
    SemanticExtractionError,
    SemanticServerError,
    SemanticValidationError,
)
from app.services.resume_request_service import ResumeRequestService


class DummyStorageClient:
    def __init__(self, settings: Settings) -> None:
        self.calls = 0

    def download_pdf(self, storage_key: str) -> bytes:
        self.calls += 1
        if self.calls < 3:
            raise StorageClientError("Temporary storage failure")
        return b"%PDF-1.4\n"


class DummyKafkaProducer:
    def __init__(self, settings: Settings) -> None:
        self.sent = []

    async def start(self) -> None:
        pass

    async def send_json(self, topic: str, message: dict) -> None:
        self.sent.append((topic, message))

    async def shutdown(self) -> None:
        pass


class DummyResumeParser:
    def __init__(self) -> None:
        self.parsed = False
        self.layout_parsed = False

    def parse(self, raw_pdf_bytes: bytes):
        self.parsed = True
        return self._resume()

    def parse_with_layout_pipeline(self, raw_pdf_bytes: bytes):
        self.layout_parsed = True
        return self._resume()

    def _resume(self):
        class DummyResume:
            def __init__(self):
                self.metadata = {"pageCount": 1, "ocrUsed": False}

            def model_dump(self):
                return {"dummy": True}

        return DummyResume()


def test_resume_request_service_retries_storage_and_publishes_completed(monkeypatch):
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")
    monkeypatch.setenv("PARSER_VERSION", "1.0.0")
    monkeypatch.setenv("PARSER_MODE", "legacy")

    settings = Settings()
    service = ResumeRequestService(
        settings,
        storage_client_cls=DummyStorageClient,
        kafka_producer_cls=DummyKafkaProducer,
        parser_cls=DummyResumeParser,
    )

    async def run_test() -> None:
        await service.start()
        try:
            await service.process({
                "jobId": "job-1",
                "resumeId": "resume-1",
                "storageKey": "resumes/user/resume.pdf",
            })
            assert len(service._producer.sent) == 1
            topic, message = service._producer.sent[0]
            assert topic == settings.kafka_topic_completed
            assert message["jobId"] == "job-1"
            assert message["resumeId"] == "resume-1"
            assert message["status"] == "COMPLETED"
            assert message["result"] == {"dummy": True}
            assert service._parser.parsed is True
            assert service._parser.layout_parsed is False
        finally:
            await service.shutdown()

    import asyncio

    asyncio.run(run_test())


def test_resume_request_service_uses_layout_parser_when_enabled(monkeypatch):
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")
    monkeypatch.setenv("PARSER_MODE", "layout")

    settings = Settings()
    service = ResumeRequestService(
        settings,
        storage_client_cls=DummyStorageClient,
        kafka_producer_cls=DummyKafkaProducer,
        parser_cls=DummyResumeParser,
    )

    async def run_test() -> None:
        await service.start()
        try:
            await service.process({
                "jobId": "job-1",
                "resumeId": "resume-1",
                "storageKey": "resumes/user/resume.pdf",
            })
            assert service._parser.parsed is False
            assert service._parser.layout_parsed is True
        finally:
            await service.shutdown()

    import asyncio

    asyncio.run(run_test())


def test_resume_request_service_uses_layout_parser_in_auto_mode(monkeypatch):
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")
    monkeypatch.setenv("PARSER_MODE", "auto")

    settings = Settings()
    service = ResumeRequestService(
        settings,
        storage_client_cls=DummyStorageClient,
        kafka_producer_cls=DummyKafkaProducer,
        parser_cls=DummyResumeParser,
    )

    async def run_test() -> None:
        await service.start()
        try:
            await service.process({
                "jobId": "job-1",
                "resumeId": "resume-1",
                "storageKey": "resumes/user/resume.pdf",
            })
            assert service._parser.parsed is False
            assert service._parser.layout_parsed is True
        finally:
            await service.shutdown()

    import asyncio

    asyncio.run(run_test())


def test_invalid_parser_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")
    monkeypatch.setenv("PARSER_MODE", "unsupported")

    with pytest.raises(ValidationError):
        Settings()


def test_resume_request_service_handles_semantic_validation_error(monkeypatch):
    """Test A: SemanticValidationError publishes SEMANTIC_VALIDATION_FAILED with sanitized message."""
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")

    class SemanticFailingParser:
        def parse_with_layout_pipeline(self, raw_pdf_bytes: bytes):
            # Simulate semantic validation rejection with potential PII
            raise SemanticValidationError([
                "UNKNOWN_BLOCK_ID in personal.name: 'secret_candidate_name_john_doe'",
                "UNSUPPORTED_CANONICAL_VALUE in personal.phone: '+1 555 123 4567'",
            ])

    settings = Settings()
    producer = DummyKafkaProducer(settings)
    service = ResumeRequestService(
        settings,
        storage_client_cls=DummyStorageClient,
        kafka_producer_cls=lambda s: producer,
        parser_cls=SemanticFailingParser,
    )

    import asyncio

    async def run_test():
        await service.start()
        try:
            await service.process({
                "jobId": "job-sem-val",
                "resumeId": "resume-sem-val",
                "storageKey": "resumes/user/resume.pdf",
            })
        finally:
            await service.shutdown()

    asyncio.run(run_test())

    assert len(producer.sent) == 1
    topic, event = producer.sent[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-sem-val"
    assert event["resumeId"] == "resume-sem-val"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "SEMANTIC_VALIDATION_FAILED"
    assert event["error"]["message"] == "Semantic output failed provenance validation."
    # Verify no candidate PII or raw violation details in event
    event_str = str(event)
    assert "secret_candidate_name" not in event_str
    assert "+1 555 123 4567" not in event_str


def test_resume_request_service_handles_semantic_extraction_error(monkeypatch):
    """Test B: SemanticExtractionError publishes SEMANTIC_EXTRACTION_FAILED with sanitized message."""
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")

    class SemanticExtractionFailingParser:
        def parse_with_layout_pipeline(self, raw_pdf_bytes: bytes):
            raise SemanticServerError("Gemini upstream 503 service unavailable", status_code=503)

    settings = Settings()
    producer = DummyKafkaProducer(settings)
    service = ResumeRequestService(
        settings,
        storage_client_cls=DummyStorageClient,
        kafka_producer_cls=lambda s: producer,
        parser_cls=SemanticExtractionFailingParser,
    )

    import asyncio

    async def run_test():
        await service.start()
        try:
            await service.process({
                "jobId": "job-sem-ext",
                "resumeId": "resume-sem-ext",
                "storageKey": "resumes/user/resume.pdf",
            })
        finally:
            await service.shutdown()

    asyncio.run(run_test())

    assert len(producer.sent) == 1
    topic, event = producer.sent[0]
    assert topic == settings.kafka_topic_failed
    assert event["jobId"] == "job-sem-ext"
    assert event["resumeId"] == "resume-sem-ext"
    assert event["status"] == "FAILED"
    assert event["error"]["code"] == "SEMANTIC_EXTRACTION_FAILED"
    assert event["error"]["message"] == "Semantic extraction failed."


def test_resume_request_service_handles_unrelated_generic_exception(monkeypatch):
    """Test C: Unrelated generic exceptions continue to publish RESUME_PARSE_FAILED."""
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")

    class GenericFailingParser:
        def parse_with_layout_pipeline(self, raw_pdf_bytes: bytes):
            raise RuntimeError("Unexpected memory allocation failure")

    settings = Settings()
    producer = DummyKafkaProducer(settings)
    service = ResumeRequestService(
        settings,
        storage_client_cls=DummyStorageClient,
        kafka_producer_cls=lambda s: producer,
        parser_cls=GenericFailingParser,
    )

    import asyncio

    async def run_test():
        await service.start()
        try:
            await service.process({
                "jobId": "job-generic-fail",
                "resumeId": "resume-generic-fail",
                "storageKey": "resumes/user/resume.pdf",
            })
        finally:
            await service.shutdown()

    asyncio.run(run_test())

    assert len(producer.sent) == 1
    topic, event = producer.sent[0]
    assert topic == settings.kafka_topic_failed
    assert event["error"]["code"] == "RESUME_PARSE_FAILED"
    assert event["error"]["message"] == "An unexpected error occurred."


def test_resume_request_service_no_duplicate_publication_on_failure(monkeypatch):
    """Test E: Failed requests emit exactly one failure event."""
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")

    class SemanticFailingParser:
        def parse_with_layout_pipeline(self, raw_pdf_bytes: bytes):
            raise SemanticExtractionError("Extraction failed")

    settings = Settings()
    producer = DummyKafkaProducer(settings)
    service = ResumeRequestService(
        settings,
        storage_client_cls=DummyStorageClient,
        kafka_producer_cls=lambda s: producer,
        parser_cls=SemanticFailingParser,
    )

    import asyncio

    async def run_test():
        await service.start()
        try:
            await service.process({
                "jobId": "job-single-pub",
                "resumeId": "resume-single-pub",
                "storageKey": "resumes/user/resume.pdf",
            })
        finally:
            await service.shutdown()

    asyncio.run(run_test())

    assert len(producer.sent) == 1
