import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import StorageClientError
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
