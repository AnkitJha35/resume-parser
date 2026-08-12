import pytest

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

    def parse(self, raw_pdf_bytes: bytes):
        self.parsed = True
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
        finally:
            await service.shutdown()

    import asyncio

    asyncio.run(run_test())
