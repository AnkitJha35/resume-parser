from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

LOW_CONFIDENCE_THRESHOLD = 0.70


class Settings(BaseSettings):
    # Required environment values (no default) — pydantic-settings will
    # map field names like `kafka_brokers` -> `KAFKA_BROKERS` automatically.
    kafka_brokers: str

    # Topics with reasonable defaults.
    kafka_topic_request: str = "resume.parse.requested"
    kafka_topic_completed: str = "resume.parse.completed"
    kafka_topic_failed: str = "resume.parse.failed"

    # MinIO configuration
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket_name: str
    minio_secure: bool = True

    # Parser version
    parser_version: str = "1.0.0"
    parser_mode: Literal["auto", "legacy", "layout"] = "auto"

    # pydantic-settings v2 configuration
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
