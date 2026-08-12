from app.core.config import Settings
from app.infrastructure.kafka.consumer import KafkaConsumerClient


class DummyHandler:
    def __init__(self) -> None:
        self.called = False

    async def __call__(self, message: dict) -> None:
        self.called = True


def test_kafka_consumer_can_be_instantiated(monkeypatch):
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")

    settings = Settings()
    handler = DummyHandler()
    consumer = KafkaConsumerClient(settings, handler)

    assert consumer is not None
    assert hasattr(consumer, "start")
    assert hasattr(consumer, "shutdown")
    assert hasattr(consumer, "_message_handler")
    assert consumer._message_handler is handler
