"""Unit tests for OpenRouter REST provider adapter."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedString,
    PersonalSemanticOutput,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
)
from app.extractors.factory import get_semantic_extractor
from app.extractors.providers.openrouter import (
    OpenRouterSemanticExtractor,
    _classify_http_error,
    _parse_retry_after,
    _sanitize_error_message,
)
from app.extractors.semantic_extractor import (
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticRateLimitError,
    SemanticResponseError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
)


def _sample_input(archetype: DocumentArchetype = DocumentArchetype.STANDARD_CV) -> SemanticInput:
    return SemanticInput(
        document_id="test_doc_1",
        page_count=1,
        archetype=archetype,
        blocks=[
            SemanticBlockInput(
                block_id="b0",
                text="Jane Doe",
                page=1,
                bbox=[72.0, 72.0, 200.0, 86.0],
                region_id="reg_0",
                region_kind="header",
                reading_order=0,
                suggested_role="HEADER",
            ),
            SemanticBlockInput(
                block_id="b1",
                text="jane@example.com",
                page=1,
                bbox=[72.0, 90.0, 200.0, 104.0],
                region_id="reg_0",
                region_kind="header",
                reading_order=1,
                suggested_role="CONTACT",
            ),
            SemanticBlockInput(
                block_id="b2",
                text="EXPERIENCE",
                page=1,
                bbox=[72.0, 120.0, 200.0, 134.0],
                region_id="reg_1",
                region_kind="physical_region",
                reading_order=2,
                suggested_role="SECTION_HEADING",
            ),
            SemanticBlockInput(
                block_id="b3",
                text="Acme Corp - Software Engineer (2020 - Present)",
                page=1,
                bbox=[72.0, 140.0, 400.0, 154.0],
                region_id="reg_1",
                region_kind="physical_region",
                reading_order=3,
                suggested_role="ENTRY_TITLE",
            ),
        ],
    )


def _make_mock_client(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


# =====================================================================
# 1. Success Tests: Single-Pass and Two-Pass
# =====================================================================

def test_openrouter_single_pass_success():
    """Single pass sends correct OpenAI-compatible payload and returns validated SemanticOutput."""
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "google/gemini-3.5-flash-lite"
        assert body["messages"][0]["role"] == "user"
        assert body["temperature"] == 0.0
        assert body["response_format"]["type"] == "json_schema"
        assert "schema" in body["response_format"]["json_schema"]

        resp_data = {
            "id": "gen-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "document_archetype": "standard_cv",
                            "personal": {
                                "name": {"value": "Jane Doe", "source_block_ids": ["b0"]},
                                "email": {"value": "jane@example.com", "source_block_ids": ["b1"]},
                            },
                            "experience": [
                                {
                                    "company": {"value": "Acme Corp", "source_block_ids": ["b3"]},
                                    "designation": {"value": "Software Engineer", "source_block_ids": ["b3"]},
                                    "current": {"value": True, "source_block_ids": ["b3"]},
                                    "source_block_ids": ["b3"],
                                }
                            ],
                        }),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 65,
                "total_tokens": 215,
            },
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(
        api_key="sk-or-v1-testkey123",
        model="google/gemini-3.5-flash-lite",
        client=client,
        two_pass=False,
    )

    result = extractor.extract(_sample_input())
    assert isinstance(result, SemanticOutput)
    assert result.personal.name.value == "Jane Doe"
    assert result.personal.email.value == "jane@example.com"
    assert len(result.experience) == 1
    assert result.experience[0].company.value == "Acme Corp"
    assert result.experience[0].current.value is True

    # Usage telemetry
    meta = extractor.last_run_meta
    assert meta["prompt_tokens"] == 150
    assert meta["output_tokens"] == 65
    assert meta["total_tokens"] == 215
    assert meta["two_pass"] is False
    assert meta["retry_count"] == 0

    # Authorization header
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.headers["Authorization"] == "Bearer sk-or-v1-testkey123"
    assert "chat/completions" in str(req.url)


def test_openrouter_two_pass_success():
    """Two pass runs personal and body passes concurrently, merges results, and aggregates tokens."""
    captured_passes = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompt = body["messages"][0]["content"]

        if "PersonalSemanticOutput" in prompt:
            captured_passes.append("personal")
            content = json.dumps({
                "personal": {
                    "name": {"value": "Jane Doe", "source_block_ids": ["b0"]},
                    "email": {"value": "jane@example.com", "source_block_ids": ["b1"]},
                }
            })
            tokens = {"prompt_tokens": 80, "completion_tokens": 25, "total_tokens": 105}
        else:
            captured_passes.append("body")
            content = json.dumps({
                "experience": [
                    {
                        "company": {"value": "Acme Corp", "source_block_ids": ["b3"]},
                        "designation": {"value": "Software Engineer", "source_block_ids": ["b3"]},
                        "current": {"value": True, "source_block_ids": ["b3"]},
                        "source_block_ids": ["b3"],
                    }
                ],
            })
            tokens = {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165}

        resp_data = {
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": tokens,
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(
        api_key="sk-or-testkey",
        client=client,
        two_pass=True,
    )

    result = extractor.extract(_sample_input())
    assert result.personal.name.value == "Jane Doe"
    assert len(result.experience) == 1
    assert result.experience[0].company.value == "Acme Corp"

    meta = extractor.last_run_meta
    assert meta["two_pass"] is True
    assert meta["prompt_tokens"] == 200  # 80 + 120
    assert meta["output_tokens"] == 70   # 25 + 45
    assert meta["total_tokens"] == 270   # 105 + 165
    assert "personal" in captured_passes
    assert "body" in captured_passes


# =====================================================================
# 2. Error Handling & Transient Retry Tests
# =====================================================================

def test_openrouter_malformed_json_in_envelope():
    """Non-JSON response envelope raises SemanticResponseError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway from cloudflare</html>")

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(api_key="test-key", client=client, two_pass=False)

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(_sample_input())
    assert "Failed to decode OpenRouter API response envelope as JSON" in str(exc_info.value)


def test_openrouter_malformed_json_in_content():
    """Malformed JSON string in choices content raises SemanticExtractionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "choices": [{"message": {"content": "{invalid json: true,"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(api_key="test-key", client=client, two_pass=False)

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(_sample_input())
    assert "Malformed JSON response" in str(exc_info.value)


def test_openrouter_schema_failure():
    """Content violating Pydantic schema raises SemanticExtractionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        # Array instead of object
        resp_data = {
            "choices": [{"message": {"content": "[1, 2, 3]"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(api_key="test-key", client=client, two_pass=False)

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(_sample_input())
    assert "Expected JSON object at root" in str(exc_info.value)


def test_openrouter_429_rate_limit_with_retry_after():
    """HTTP 429 respects retry-after header and retries until success."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "3"}, text=json.dumps({"error": {"message": "Rate limited"}}))
        resp_data = {
            "choices": [{"message": {"content": json.dumps({"document_archetype": "standard_cv"})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(
        api_key="test-key",
        client=client,
        two_pass=False,
        sleep_fn=sleeps.append,
        max_retries=2,
    )

    result = extractor.extract(_sample_input())
    assert attempts == 2
    assert sleeps == [3.0]
    assert extractor.last_run_meta["retry_count"] == 1


def test_openrouter_429_exhausted_raises_ratelimit_error():
    """Exhausting retries on 429 raises SemanticRateLimitError."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, headers={"retry-after": "1"}, text=json.dumps({"error": {"message": "Quota exceeded", "code": 429}}))

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(
        api_key="test-key",
        client=client,
        two_pass=False,
        sleep_fn=sleeps.append,
        max_retries=1,
    )

    with pytest.raises(SemanticRateLimitError) as exc_info:
        extractor.extract(_sample_input())
    assert attempts == 2  # initial + 1 retry
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value) or "Quota exceeded" in str(exc_info.value)


def test_openrouter_5xx_server_error_retries_and_raises():
    """5xx error retries up to max_retries and raises SemanticServerError."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text=json.dumps({"error": {"message": "Service unavailable"}}))

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(
        api_key="test-key",
        client=client,
        two_pass=False,
        sleep_fn=sleeps.append,
        max_retries=2,
        initial_backoff=0.5,
    )

    with pytest.raises(SemanticServerError) as exc_info:
        extractor.extract(_sample_input())
    assert attempts == 3  # 1 initial + 2 retries
    assert exc_info.value.status_code == 503
    assert len(sleeps) == 2


def test_openrouter_timeout_retries_and_raises():
    """httpx.TimeoutException retries and raises SemanticTimeoutError."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("Connection read timed out")

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(
        api_key="test-key",
        client=client,
        two_pass=False,
        sleep_fn=sleeps.append,
        max_retries=1,
    )

    with pytest.raises(SemanticTimeoutError) as exc_info:
        extractor.extract(_sample_input())
    assert attempts == 2
    assert "timed out" in str(exc_info.value)


def test_openrouter_transport_failure():
    """httpx.ConnectError retries and raises SemanticTransportError."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("Failed to resolve host openrouter.ai")

    client = _make_mock_client(handler)
    extractor = OpenRouterSemanticExtractor(
        api_key="test-key",
        client=client,
        two_pass=False,
        sleep_fn=sleeps.append,
        max_retries=1,
    )

    with pytest.raises(SemanticTransportError) as exc_info:
        extractor.extract(_sample_input())
    assert attempts == 2
    assert "transport failure" in str(exc_info.value)


def test_openrouter_missing_api_key_raises_configuration_error(monkeypatch):
    """Missing OPENROUTER_API_KEY raises SemanticConfigurationError."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    extractor = OpenRouterSemanticExtractor(two_pass=False)
    # Patch settings to have None key
    with pytest.raises(SemanticConfigurationError) as exc_info:
        extractor.extract(_sample_input())
    assert "OPENROUTER_API_KEY is not configured" in str(exc_info.value)


def test_openrouter_secret_redaction():
    """API key is never leaked in sanitized error messages or logs."""
    raw_key = "sk-or-v1-supersecretkey98765"
    err_text = f"Invalid auth key {raw_key} at endpoint"
    sanitized = _sanitize_error_message(err_text, raw_key)
    assert raw_key not in sanitized
    assert "[REDACTED]" in sanitized


# =====================================================================
# 3. Factory and Configuration Tests
# =====================================================================

def test_factory_returns_openrouter_when_selected():
    """get_semantic_extractor returns OpenRouterSemanticExtractor when provider='openrouter'."""
    settings = Settings(
        kafka_brokers="localhost:9092",
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket_name="resumes",
        openrouter_api_key="sk-or-factory-test",
        openrouter_model="anthropic/claude-3-haiku",
    )
    extractor = get_semantic_extractor(settings, provider="openrouter")
    assert isinstance(extractor, OpenRouterSemanticExtractor)
    api_key, model, base_url, timeout, max_retries = extractor._resolve_config()
    assert api_key == "sk-or-factory-test"
    assert model == "anthropic/claude-3-haiku"


def test_factory_returns_openrouter_from_settings():
    """get_semantic_extractor returns OpenRouterSemanticExtractor when semantic_provider='openrouter'."""
    settings = Settings(
        kafka_brokers="localhost:9092",
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket_name="resumes",
        semantic_provider="openrouter",
        openrouter_api_key="sk-or-settings-test",
    )
    extractor = get_semantic_extractor(settings)
    assert isinstance(extractor, OpenRouterSemanticExtractor)
