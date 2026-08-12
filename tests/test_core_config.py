import os

from app.core.config import Settings, LOW_CONFIDENCE_THRESHOLD


def test_settings_loads_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAFKA_BROKERS", "broker1:9092,broker2:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "play.min.io")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")

    settings = Settings()

    assert settings.kafka_brokers == "broker1:9092,broker2:9092"
    assert settings.minio_endpoint == "play.min.io"
    assert settings.minio_access_key == "test-access-key"
    assert settings.minio_secret_key == "test-secret-key"
    assert settings.minio_bucket_name == "resumes"
    assert LOW_CONFIDENCE_THRESHOLD == 0.70
