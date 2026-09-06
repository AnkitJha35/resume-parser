"""Unit tests for NVIDIA NIM REST provider adapter."""

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
from app.extractors.providers.nvidia import (
    NvidiaSemanticExtractor,
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
# 1. Output-Token Budget (max_tokens) & Payload Structure Tests
# =====================================================================

def test_nvidia_default_max_tokens_and_format():
    """Default max_tokens (16384) and json_object response format are used."""
    captured_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        resp_data = {
            "choices": [{"message": {"content": json.dumps({"document_archetype": "standard_cv"})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="nvapi-testkey",
        client=client,
        two_pass=False,
    )

    extractor.extract(_sample_input())
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["max_tokens"] == 16384
    assert captured_payloads[0]["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert captured_payloads[0]["response_format"] == {"type": "json_object"}
    assert captured_payloads[0]["temperature"] == 0.0


def test_nvidia_configured_max_tokens():
    """Explicitly configured max_tokens is sent in payload."""
    captured_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        resp_data = {
            "choices": [{"message": {"content": json.dumps({"document_archetype": "standard_cv"})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="nvapi-testkey",
        max_tokens=4096,
        client=client,
        two_pass=False,
    )

    extractor.extract(_sample_input())
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["max_tokens"] == 4096


def test_nvidia_env_max_tokens(monkeypatch):
    """NVIDIA_MAX_TOKENS environment variable is honored when unset explicitly."""
    monkeypatch.setenv("NVIDIA_MAX_TOKENS", "8192")
    captured_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        resp_data = {
            "choices": [{"message": {"content": json.dumps({"document_archetype": "standard_cv"})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="nvapi-testkey",
        client=client,
        two_pass=False,
    )

    extractor.extract(_sample_input())
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["max_tokens"] == 8192


def test_nvidia_json_schema_response_format_support():
    """When configured with response_format_type='json_schema', schema payload is included."""
    captured_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        resp_data = {
            "choices": [{"message": {"content": json.dumps({"document_archetype": "standard_cv"})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="nvapi-testkey",
        response_format_type="json_schema",
        client=client,
        two_pass=False,
    )

    extractor.extract(_sample_input())
    assert len(captured_payloads) == 1
    fmt = captured_payloads[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert "schema" in fmt["json_schema"]


# =====================================================================
# 2. HTTP 402 Insufficient Credit / Non-Transient Handling Tests
# =====================================================================

def test_nvidia_402_not_retried():
    """HTTP 402 is classified as non-transient SemanticConfigurationError and NOT retried."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(402, json={"error": {"message": "Credit limit exceeded", "code": 402}})

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="nvapi-testkey",
        client=client,
        two_pass=False,
        max_retries=3,
    )

    with pytest.raises(SemanticConfigurationError) as exc_info:
        extractor.extract(_sample_input())

    assert attempts == 1  # Strictly 1 attempt, NO retries
    err_str = str(exc_info.value)
    assert "HTTP 402" in err_str
    assert "Credit limit exceeded" in err_str or "credits" in err_str


# =====================================================================
# 3. Success Tests: Single-Pass and Two-Pass
# =====================================================================

def test_nvidia_single_pass_success():
    """Single pass sends correct OpenAI-compatible payload and returns validated SemanticOutput."""
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert body["messages"][0]["role"] == "user"
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 16384
        assert body["response_format"]["type"] == "json_object"

        resp_data = {
            "id": "nv-123",
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
                "prompt_tokens": 140,
                "completion_tokens": 60,
                "total_tokens": 200,
            },
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="nvapi-v1-secret123",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
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
    assert meta["prompt_tokens"] == 140
    assert meta["output_tokens"] == 60
    assert meta["total_tokens"] == 200
    assert meta["two_pass"] is False
    assert meta["retry_count"] == 0

    # Authorization header
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.headers["Authorization"] == "Bearer nvapi-v1-secret123"
    assert "chat/completions" in str(req.url)


def test_nvidia_two_pass_success():
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
            tokens = {"prompt_tokens": 70, "completion_tokens": 20, "total_tokens": 90}
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
            tokens = {"prompt_tokens": 110, "completion_tokens": 40, "total_tokens": 150}

        resp_data = {
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": tokens,
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="nvapi-testkey",
        client=client,
        two_pass=True,
    )

    result = extractor.extract(_sample_input())
    assert result.personal.name.value == "Jane Doe"
    assert len(result.experience) == 1
    assert result.experience[0].company.value == "Acme Corp"

    meta = extractor.last_run_meta
    assert meta["two_pass"] is True
    assert meta["prompt_tokens"] == 180  # 70 + 110
    assert meta["output_tokens"] == 60   # 20 + 40
    assert meta["total_tokens"] == 240   # 90 + 150
    assert "personal" in captured_passes
    assert "body" in captured_passes


# =====================================================================
# 4. Error Handling & Transient Retry Tests
# =====================================================================

def test_nvidia_malformed_json_in_envelope():
    """Non-JSON response envelope raises SemanticResponseError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway</html>")

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(api_key="test-key", client=client, two_pass=False)

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(_sample_input())
    assert "Failed to decode NVIDIA API response envelope as JSON" in str(exc_info.value)


def test_nvidia_malformed_json_in_content():
    """Malformed JSON string in choices content raises SemanticExtractionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "choices": [{"message": {"content": "{invalid json: true,"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(api_key="test-key", client=client, two_pass=False)

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(_sample_input())
    assert "Malformed JSON response" in str(exc_info.value)


def test_nvidia_schema_failure():
    """Content violating Pydantic schema raises SemanticExtractionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "choices": [{"message": {"content": "[1, 2, 3]"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(api_key="test-key", client=client, two_pass=False)

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(_sample_input())
    assert "Expected JSON object at root" in str(exc_info.value)


def test_nvidia_401_auth_failure_not_retried():
    """HTTP 401 raises SemanticConfigurationError with zero retries."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(api_key="bad-key", client=client, two_pass=False, max_retries=3)

    with pytest.raises(SemanticConfigurationError) as exc_info:
        extractor.extract(_sample_input())

    assert attempts == 1
    assert "HTTP 401" in str(exc_info.value)


def test_nvidia_429_rate_limit_with_retry_after():
    """HTTP 429 respects retry-after header and retries until success."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, text=json.dumps({"error": {"message": "Rate limited"}}))
        resp_data = {
            "choices": [{"message": {"content": json.dumps({"document_archetype": "standard_cv"})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=resp_data)

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="test-key",
        client=client,
        two_pass=False,
        sleep_fn=sleeps.append,
        max_retries=2,
    )

    result = extractor.extract(_sample_input())
    assert attempts == 2
    assert sleeps == [2.0]
    assert extractor.last_run_meta["retry_count"] == 1


def test_nvidia_429_exhausted_raises_ratelimit_error():
    """Exhausting retries on 429 raises SemanticRateLimitError."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, headers={"retry-after": "1"}, text=json.dumps({"error": {"message": "Quota exceeded", "code": 429}}))

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="test-key",
        client=client,
        two_pass=False,
        sleep_fn=sleeps.append,
        max_retries=1,
    )

    with pytest.raises(SemanticRateLimitError) as exc_info:
        extractor.extract(_sample_input())
    assert attempts == 2
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value) or "Quota exceeded" in str(exc_info.value)


def test_nvidia_5xx_server_error_retries_and_raises():
    """5xx error retries up to max_retries and raises SemanticServerError."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text=json.dumps({"error": {"message": "Service unavailable"}}))

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
        api_key="test-key",
        client=client,
        two_pass=False,
        sleep_fn=sleeps.append,
        max_retries=2,
        initial_backoff=0.5,
    )

    with pytest.raises(SemanticServerError) as exc_info:
        extractor.extract(_sample_input())
    assert attempts == 3
    assert exc_info.value.status_code == 503
    assert len(sleeps) == 2


def test_nvidia_timeout_retries_and_raises():
    """httpx.TimeoutException retries and raises SemanticTimeoutError."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("Connection read timed out")

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
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


def test_nvidia_transport_failure():
    """httpx.ConnectError retries and raises SemanticTransportError."""
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("Failed to resolve host integrate.api.nvidia.com")

    client = _make_mock_client(handler)
    extractor = NvidiaSemanticExtractor(
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


def test_nvidia_missing_api_key_raises_configuration_error(monkeypatch):
    """Missing NVIDIA_API_KEY raises SemanticConfigurationError."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    extractor = NvidiaSemanticExtractor(two_pass=False)
    with pytest.raises(SemanticConfigurationError) as exc_info:
        extractor.extract(_sample_input())
    assert "NVIDIA_API_KEY is not configured" in str(exc_info.value)


def test_nvidia_secret_redaction():
    """API key is never leaked in sanitized error messages or logs."""
    raw_key = "nvapi-v1-supersecretkey98765"
    err_text = f"Invalid auth key {raw_key} at endpoint"
    sanitized = _sanitize_error_message(err_text, raw_key)
    assert raw_key not in sanitized
    assert "[REDACTED]" in sanitized


# =====================================================================
# 5. Factory and Configuration Tests
# =====================================================================

def test_factory_returns_nvidia_when_selected():
    """get_semantic_extractor returns NvidiaSemanticExtractor when provider='nvidia'."""
    settings = Settings(
        kafka_brokers="localhost:9092",
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket_name="resumes",
        nvidia_api_key="nvapi-factory-test",
        nvidia_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_max_tokens=4096,
        nvidia_response_format_type="json_object",
    )
    extractor = get_semantic_extractor(settings, provider="nvidia")
    assert isinstance(extractor, NvidiaSemanticExtractor)
    api_key, model, base_url, timeout, max_retries, max_tokens, fmt_type = extractor._resolve_config()
    assert api_key == "nvapi-factory-test"
    assert model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert max_tokens == 4096
    assert fmt_type == "json_object"


def test_factory_returns_nvidia_from_settings():
    """get_semantic_extractor returns NvidiaSemanticExtractor when semantic_provider='nvidia'."""
    settings = Settings(
        kafka_brokers="localhost:9092",
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket_name="resumes",
        semantic_provider="nvidia",
        nvidia_api_key="nvapi-settings-test",
    )
    extractor = get_semantic_extractor(settings)
    assert isinstance(extractor, NvidiaSemanticExtractor)


# =====================================================================
# 6. Benchmark CLI Option Tests
# =====================================================================

def test_benchmark_runner_accepts_nvidia_provider(monkeypatch):
    """The benchmark CLI parser accepts --provider nvidia without error."""
    from tests.benchmark.run_semantic import main

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    # When NVIDIA_API_KEY is not set, main should exit cleanly with return code 1 and error message
    rc = main(["--provider", "nvidia"])
    assert rc == 1
