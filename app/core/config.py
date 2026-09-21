from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

LOW_CONFIDENCE_THRESHOLD = 0.70


class Settings(BaseSettings):
    # Kafka configuration (disabled by default in HTTP standalone mode)
    kafka_enabled: bool = False
    kafka_brokers: str = "localhost:9092"

    # Topics with reasonable defaults.
    kafka_topic_request: str = "resume.parse.requested"
    kafka_topic_completed: str = "resume.parse.completed"
    kafka_topic_failed: str = "resume.parse.failed"

    # MinIO configuration (optional in HTTP standalone mode)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "resumes"
    minio_secure: bool = False

    # Database configuration (PostgreSQL persistence foundation - optional for parser-only API)
    database_url: str | None = None
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout: float = 30.0

    # Upload & OCR configuration
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    ocr_enabled: bool = True
    ocr_engine: str = "pymupdf"

    # Parser version
    parser_version: str = "1.0.0"
    parser_mode: Literal["auto", "legacy", "layout"] = "auto"

    # Provider selection
    semantic_provider: Literal["gemini", "openrouter", "ollama", "nvidia"] = "gemini"

    # Semantic LLM configuration (optional)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_timeout: float = 30.0
    gemini_max_retries: int = 2
    gemini_two_pass: bool = False

    # OpenRouter LLM configuration
    openrouter_api_key: str | None = None
    openrouter_model: str = "google/gemini-3.5-flash-lite"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout: float = 60.0
    openrouter_max_retries: int = 2
    openrouter_max_tokens: int = 16384
    openrouter_two_pass: bool = False

    # NVIDIA NIM configuration
    nvidia_api_key: str | None = None
    nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_timeout: float = 60.0
    nvidia_max_retries: int = 2
    nvidia_max_tokens: int = 16384
    nvidia_two_pass: bool = False
    nvidia_response_format_type: Literal["json_object", "json_schema"] = "json_schema"
    nvidia_enable_thinking: bool = False

    # Ollama LLM configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_timeout: float = 120.0
    ollama_num_threads: int = 8
    ollama_think: bool = False

    # Fallback configuration
    semantic_fallback_enabled: bool = False

    # pydantic-settings v2 configuration
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
