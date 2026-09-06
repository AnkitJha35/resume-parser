"""Unit tests for semantic extractor factory and production wiring."""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.extractors.factory import get_semantic_extractor
from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.providers.ollama import OllamaSemanticExtractor
from app.extractors.semantic_extractor import SemanticExtractor
from app.services.resume_request_service import ResumeRequestService


def _make_test_settings(**kwargs) -> Settings:
    """Create a test Settings instance with minimum required fields."""
    defaults = {
        "kafka_brokers": "localhost:9092",
        "minio_endpoint": "localhost:9000",
        "minio_access_key": "minioadmin",
        "minio_secret_key": "minioadmin",
        "minio_bucket_name": "resumes",
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def test_factory_default_configuration_returns_gemini_only():
    """Test 1: Default configuration (fallback=False) returns GeminiSemanticExtractor directly."""
    settings = _make_test_settings(semantic_fallback_enabled=False)
    extractor = get_semantic_extractor(settings)

    assert isinstance(extractor, GeminiSemanticExtractor)
    assert not isinstance(extractor, FallbackSemanticExtractor)
    assert isinstance(extractor, SemanticExtractor)


def test_factory_fallback_enabled_returns_fallback_extractor():
    """Test 2: Fallback enabled returns FallbackSemanticExtractor wrapping Gemini and Ollama."""
    settings = _make_test_settings(semantic_fallback_enabled=True)
    extractor = get_semantic_extractor(settings)

    assert isinstance(extractor, FallbackSemanticExtractor)
    assert isinstance(extractor.primary, GeminiSemanticExtractor)
    assert isinstance(extractor.fallback, OllamaSemanticExtractor)
    assert isinstance(extractor, SemanticExtractor)


def test_factory_gemini_configuration_propagation():
    """Test 3: Gemini settings are accurately passed to GeminiSemanticExtractor."""
    settings = _make_test_settings(
        gemini_api_key="test-api-key-12345",
        gemini_model="gemini-2.5-pro",
        gemini_base_url="https://custom.gemini.endpoint",
        gemini_timeout=45.0,
        gemini_max_retries=4,
        semantic_fallback_enabled=False,
    )
    extractor = get_semantic_extractor(settings)

    assert isinstance(extractor, GeminiSemanticExtractor)
    assert extractor._explicit_api_key == "test-api-key-12345"
    assert extractor._explicit_model == "gemini-2.5-pro"
    assert extractor._explicit_base_url == "https://custom.gemini.endpoint"
    assert extractor._explicit_timeout == 45.0
    assert extractor._explicit_max_retries == 4


def test_factory_ollama_configuration_propagation():
    """Test 4: Ollama settings are accurately passed to OllamaSemanticExtractor in fallback mode."""
    settings = _make_test_settings(
        ollama_base_url="http://remote-ollama:11434",
        ollama_model="llama3.3:70b",
        ollama_timeout=180.0,
        ollama_num_threads=16,
        ollama_think=True,
        semantic_fallback_enabled=True,
    )
    extractor = get_semantic_extractor(settings)

    assert isinstance(extractor, FallbackSemanticExtractor)
    fallback = extractor.fallback
    assert isinstance(fallback, OllamaSemanticExtractor)
    assert fallback._explicit_base_url == "http://remote-ollama:11434"
    assert fallback._explicit_model == "llama3.3:70b"
    assert fallback._explicit_timeout == 180.0
    assert fallback._explicit_num_threads == 16
    assert fallback._explicit_think is True


def test_factory_uses_supplied_settings_object():
    """Test 5: Factory respects supplied Settings object without creating a second configuration source."""
    settings = _make_test_settings(
        gemini_model="custom-gemini-test-model",
        semantic_fallback_enabled=False,
    )
    extractor = get_semantic_extractor(settings)
    assert isinstance(extractor, GeminiSemanticExtractor)
    assert extractor._explicit_model == "custom-gemini-test-model"


def test_factory_construction_makes_no_network_calls(monkeypatch):
    """Test 6: Factory creation does not execute any HTTP or network calls."""
    def _disallowed_request(*args, **kwargs):
        raise AssertionError("Network call attempted during factory construction!")

    monkeypatch.setattr(httpx.Client, "request", _disallowed_request)
    monkeypatch.setattr(httpx.Client, "send", _disallowed_request)

    settings = _make_test_settings(semantic_fallback_enabled=True)
    extractor = get_semantic_extractor(settings)

    assert isinstance(extractor, FallbackSemanticExtractor)


def test_production_service_wiring(monkeypatch):
    """Test 7: ResumeRequestService initializes and receives factory-created SemanticExtractor."""
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "resumes")

    class DummyStorage:
        def __init__(self, s):
            pass

    class DummyProducer:
        def __init__(self, s):
            pass

    class DummyParser:
        def __init__(self):
            pass

    settings = _make_test_settings(semantic_fallback_enabled=True)
    service = ResumeRequestService(
        settings=settings,
        storage_client_cls=DummyStorage,  # type: ignore[arg-type]
        kafka_producer_cls=DummyProducer,  # type: ignore[arg-type]
        parser_cls=DummyParser,  # type: ignore[arg-type]
    )

    assert hasattr(service, "_semantic_extractor")
    assert isinstance(service._semantic_extractor, FallbackSemanticExtractor)


def test_default_backward_compatibility():
    """Test 8: Default configuration produces Gemini-only extractor without fallback overhead."""
    settings = _make_test_settings()
    assert settings.semantic_fallback_enabled is False

    extractor = get_semantic_extractor(settings)
    assert isinstance(extractor, GeminiSemanticExtractor)
    assert not isinstance(extractor, FallbackSemanticExtractor)


def test_factory_returns_nvidia_when_specified():
    """Test 9: Factory returns NvidiaSemanticExtractor when provider='nvidia'."""
    from app.extractors.providers.nvidia import NvidiaSemanticExtractor

    settings = _make_test_settings(
        nvidia_api_key="nvapi-key",
        nvidia_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_max_tokens=8192,
    )
    extractor = get_semantic_extractor(settings, provider="nvidia")
    assert isinstance(extractor, NvidiaSemanticExtractor)
    assert extractor._explicit_api_key == "nvapi-key"
    assert extractor._explicit_model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert extractor._explicit_max_tokens == 8192


def test_factory_returns_nvidia_from_semantic_provider_setting():
    """Test 10: Factory returns NvidiaSemanticExtractor when settings.semantic_provider='nvidia'."""
    from app.extractors.providers.nvidia import NvidiaSemanticExtractor

    settings = _make_test_settings(
        semantic_provider="nvidia",
        nvidia_api_key="nvapi-key-2",
    )
    extractor = get_semantic_extractor(settings)
    assert isinstance(extractor, NvidiaSemanticExtractor)
    assert extractor._explicit_api_key == "nvapi-key-2"
