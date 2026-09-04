"""Phase 8-3: Tests for SemanticInput serialization, prompt contracts, and Gemini LLM adapter."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pytest
import httpx

from app.domain.resume import Resume
from app.domain.semantic_contract import (
    BlockClassification,
    DocumentArchetype,
    GroundedPersonal,
    GroundedString,
    SemanticBlockCategory,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    validate_semantic_output,
)
from app.extractors.providers.gemini import GeminiSemanticExtractor, pydantic_to_gemini_schema
from app.extractors.semantic_extractor import (
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticRateLimitError,
    SemanticResponseError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
    SemanticValidationError,
)
from app.extractors.semantic_prompt import (
    SEMANTIC_EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
    parse_semantic_output,
    serialize_semantic_input,
)
from app.pipeline.semantic_pipeline import parse_document_semantically
from tests.test_semantic_llm_contract import _make_test_document


def _sample_semantic_input() -> SemanticInput:
    blocks = [
        SemanticBlockInput(
            block_id="b_2",
            text="Second block in reading order",
            page=1,
            bbox=[50.0, 100.0, 200.0, 120.0],
            region_id="r1",
            region_kind="physical_region",
            reading_order=2,
            column_id=1,
            suggested_role="ORGANIZATION",
            table_id="tbl_1",
            row_index=1,
            column_index=0,
            cell_role="DATA",
        ),
        SemanticBlockInput(
            block_id="b_1",
            text="First block in reading order",
            page=1,
            bbox=[50.0, 50.0, 200.0, 70.0],
            region_id="r0",
            region_kind="header",
            reading_order=1,
            column_id=None,
            is_bold=True,
            font_size=16.0,
            suggested_role="HEADER",
            table_id=None,
            row_index=None,
            column_index=None,
            cell_role=None,
        ),
    ]
    return SemanticInput(
        document_id="doc_test_123",
        page_count=1,
        pages=[],
        blocks=blocks,
    )


# =====================================================================
# A. SemanticInput Serialization Tests
# =====================================================================


def test_serialization_deterministic_and_sorted():
    sem_input = _sample_semantic_input()
    raw_json = serialize_semantic_input(sem_input)
    parsed = json.loads(raw_json)

    assert parsed["document_id"] == "doc_test_123"
    assert parsed["page_count"] == 1
    assert len(parsed["blocks"]) == 2

    # Blocks must be sorted by page and reading order
    assert parsed["blocks"][0]["block_id"] == "b_1"
    assert parsed["blocks"][0]["reading_order"] == 1
    assert parsed["blocks"][0]["suggested_role"] == "HEADER"
    assert parsed["blocks"][0]["is_bold"] is True
    assert parsed["blocks"][0]["font_size"] == 16.0
    assert parsed["blocks"][0]["table_id"] is None

    assert parsed["blocks"][1]["block_id"] == "b_2"
    assert parsed["blocks"][1]["reading_order"] == 2
    assert parsed["blocks"][1]["table_id"] == "tbl_1"
    assert parsed["blocks"][1]["row_index"] == 1
    assert parsed["blocks"][1]["cell_role"] == "DATA"


def test_serialization_stability():
    sem_input = _sample_semantic_input()
    json1 = serialize_semantic_input(sem_input)
    json2 = serialize_semantic_input(sem_input)
    assert json1 == json2


# =====================================================================
# B. Prompt Contract Tests
# =====================================================================


def test_prompt_contains_grounding_and_no_fixture_rules():
    sem_input = _sample_semantic_input()
    prompt = build_extraction_prompt(sem_input)

    # Required instructions must be present
    assert "source_block_ids" in prompt
    assert "raw_value" in prompt
    assert "CRITICAL GROUNDING AND PROVENANCE RULES" in prompt
    assert "DO NOT treat document headers, form titles" in prompt
    assert "DO NOT treat table column headers" in prompt
    assert "DO NOT classify referee" in prompt
    assert "DOCUMENT BLOCKS (JSON):" in prompt

    # No benchmark-specific PDF names or fixture names
    disallowed_fixtures = [
        "Mayur",
        "AASHISH",
        "AKIBUL",
        "Rishabh",
        "JOSH",
        "MUKUND",
        "Rajeev",
        "Shubham",
        "Sendrick",
        "AditCV",
        "fresher_hr",
        "swe_experienced",
    ]
    for fix in disallowed_fixtures:
        assert fix not in SEMANTIC_EXTRACTION_SYSTEM_PROMPT


# =====================================================================
# C. Structured SemanticOutput Parsing Tests
# =====================================================================


def test_parse_valid_semantic_output():
    valid_payload = {
        "document_archetype": "standard_cv",
        "personal": {
            "name": {"value": "Jane Smith", "raw_value": "Jane Smith", "source_block_ids": ["b1"]},
            "email": {"value": "jane@example.com", "source_block_ids": ["b2"]},
        },
        "experience": [],
        "education": [],
        "projects": [],
        "skills": [{"value": "Python", "source_block_ids": ["b3"]}],
    }
    raw_str = f"```json\n{json.dumps(valid_payload)}\n```"
    output = parse_semantic_output(raw_str)

    assert isinstance(output, SemanticOutput)
    assert output.document_archetype == DocumentArchetype.STANDARD_CV
    assert output.personal.name is not None
    assert output.personal.name.value == "Jane Smith"
    assert output.skills[0].value == "Python"


def test_parse_malformed_json_rejected():
    with pytest.raises(SemanticExtractionError) as exc_info:
        parse_semantic_output("Not valid JSON {foo: bar}")
    assert "Malformed JSON response from LLM" in str(exc_info.value)


def test_prose_wrapped_json_rejected():
    # Defensive fence parsing must NOT accept conversational prose surrounding the JSON
    payload = {"document_archetype": "standard_cv"}
    prose_wrapped = f"Sure, here is your extracted resume:\n```json\n{json.dumps(payload)}\n```\nHope this helps!"
    with pytest.raises(SemanticExtractionError) as exc_info:
        parse_semantic_output(prose_wrapped)
    assert "Malformed JSON response from LLM" in str(exc_info.value)


def test_parse_invalid_structure_rejected():
    # Array instead of object
    with pytest.raises(SemanticExtractionError) as exc_info:
        parse_semantic_output("[1, 2, 3]")
    assert "Expected JSON object at root" in str(exc_info.value)


def test_structural_parsing_succeeds_but_semantic_validation_rejects_missing_provenance():
    # GroundedString without source_block_ids succeeds structural Pydantic parsing,
    # but fails deterministic validation invariant MISSING_PROVENANCE at the trust boundary.
    payload = {
        "personal": {
            "name": {"value": "Jane Smith", "source_block_ids": []},  # empty provenance
        }
    }
    # 1. Structural parsing succeeds
    output = parse_semantic_output(payload)
    assert isinstance(output, SemanticOutput)
    assert output.personal.name is not None

    # 2. Semantic validation catches missing provenance
    dummy_input = SemanticInput(document_id="d1", page_count=1, pages=[], blocks=[])
    violations = validate_semantic_output(output, dummy_input)
    assert any("MISSING_PROVENANCE in personal.name" in v for v in violations)


# =====================================================================
# D. Provider Isolation Tests
# =====================================================================


def test_provider_isolation():
    # Provider-neutral modules must not import httpx or provider SDKs
    for path in (
        Path("app/domain/semantic_contract.py"),
        Path("app/extractors/semantic_prompt.py"),
        Path("app/pipeline/semantic_pipeline.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        for disallowed in ("httpx", "google", "openai", "anthropic"):
            assert not any(m.startswith(disallowed) for m in imported_modules), (
                f"{path} should not import {disallowed}"
            )


# =====================================================================
# E, F, G. Provider Request Construction, Error Handling, and End-to-End
# =====================================================================


def test_pydantic_to_gemini_schema_derivation():
    # 1. Verify responseSchema is derived from SemanticOutput and has no unresolved $ref pointers
    schema = pydantic_to_gemini_schema(SemanticOutput)

    assert schema["type"] == "object"
    assert "properties" in schema
    assert "$ref" not in json.dumps(schema)

    # Check key domain properties are present
    props = schema["properties"]
    assert "personal" in props
    assert "experience" in props
    assert "education" in props
    assert "skills" in props
    assert "document_archetype" in props

    # Check nested personal properties (optional fields use anyOf with object and null)
    personal_props = props["personal"]["properties"]
    assert "name" in personal_props
    assert "email" in personal_props
    assert "phone" in personal_props

    name_schema = personal_props["name"]
    name_obj = name_schema["anyOf"][0] if "anyOf" in name_schema else name_schema
    assert "source_block_ids" in name_obj["properties"]


def test_gemini_endpoint_url():
    url = GeminiSemanticExtractor.get_endpoint_url("gemini-2.5-flash")
    expected_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert url == expected_url
    assert "[" not in url and "]" not in url and "(" not in url and ")" not in url


def test_gemini_config_defaults_and_override():
    # Test default configuration with explicit API key
    extractor = GeminiSemanticExtractor(api_key="my-test-key")
    k, m, b, t, r = extractor._resolve_config()
    assert k == "my-test-key"
    assert m == "gemini-3.5-flash-lite"
    assert b == "https://generativelanguage.googleapis.com"
    assert t == 30.0
    assert r == 2

    # Test explicit override of all settings
    extractor2 = GeminiSemanticExtractor(
        api_key="custom-key",
        model="gemini-custom",
        base_url="https://custom.endpoint.internal/",
        timeout=15.0,
        max_retries=4,
    )
    k2, m2, b2, t2, r2 = extractor2._resolve_config()
    assert k2 == "custom-key"
    assert m2 == "gemini-custom"
    assert b2 == "https://custom.endpoint.internal"
    assert t2 == 15.0
    assert r2 == 4


def test_gemini_config_precedence(monkeypatch):
    # 1. Environment variables override defaults
    monkeypatch.setenv("GEMINI_API_KEY", "env-api-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-model")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://env.endpoint.com")
    monkeypatch.setenv("GEMINI_TIMEOUT", "45.0")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "3")

    extractor = GeminiSemanticExtractor()
    k, m, b, t, r = extractor._resolve_config()
    assert k == "env-api-key"
    assert m == "gemini-env-model"
    assert b == "https://env.endpoint.com"
    assert t == 45.0
    assert r == 3

    # 2. Explicit constructor overrides environment variables
    extractor_explicit = GeminiSemanticExtractor(
        api_key="explicit-key",
        model="explicit-model",
        timeout=10.0,
        max_retries=1,
    )
    ke, me, be, te, re = extractor_explicit._resolve_config()
    assert ke == "explicit-key"
    assert me == "explicit-model"
    assert be == "https://env.endpoint.com"
    assert te == 10.0
    assert re == 1

    # 3. Invalid environment variables fall back safely
    monkeypatch.setenv("GEMINI_TIMEOUT", "invalid_timeout")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "-5")
    extractor_invalid = GeminiSemanticExtractor(api_key="valid-key")
    _, _, _, ti, ri = extractor_invalid._resolve_config()
    assert ti == 30.0
    assert ri == 2


def test_gemini_config_precedence_env_over_settings_regression(monkeypatch):
    # Mock Settings to return different values from env
    class MockSettings:
        gemini_api_key = "settings-key"
        gemini_model = "settings-model"
        gemini_base_url = "https://settings.endpoint.com"
        gemini_timeout = 60.0
        gemini_max_retries = 5

    monkeypatch.setattr("app.extractors.providers.gemini.Settings", MockSettings)

    # Regression Case 1: GEMINI_TIMEOUT=30 vs Settings timeout=60 -> result must be 30.0
    monkeypatch.setenv("GEMINI_TIMEOUT", "30")
    # Regression Case 2: GEMINI_MAX_RETRIES=2 vs Settings max_retries=5 -> result must be 2
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "2")
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")

    extractor = GeminiSemanticExtractor()
    k, m, b, t, r = extractor._resolve_config()
    assert k == "env-key"
    assert t == 30.0
    assert r == 2


def test_missing_api_key_raises_configuration_error(monkeypatch):
    from app.extractors.semantic_extractor import SemanticConfigurationError

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Ensure Settings does not supply a key
    monkeypatch.setattr("app.extractors.providers.gemini.Settings", lambda: type("S", (), {"gemini_api_key": None})())

    extractor = GeminiSemanticExtractor(api_key=None)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticConfigurationError) as exc_info:
        extractor.extract(sem_input)
    assert "Gemini API key is required but not configured" in str(exc_info.value)


def test_api_key_never_leaks_in_error_message():
    secret_key = "AIzaSySecretApiKey12345"

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        # Simulate server error returning request details including the secret key
        return httpx.Response(500, text=f"Error processing request with key {secret_key}", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(api_key=secret_key, client=client, max_retries=0)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(sem_input)

    err_str = str(exc_info.value)
    assert secret_key not in err_str
    assert "[REDACTED]" in err_str


def test_provider_http_rate_limit_error_classification():
    from app.extractors.semantic_extractor import SemanticRateLimitError

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "5"}, text="Quota exceeded", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client, max_retries=0)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticRateLimitError) as exc_info:
        extractor.extract(sem_input)
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 5.0
    assert "rate limit exceeded" in str(exc_info.value)


def test_provider_http_server_error_classification():
    from app.extractors.semantic_extractor import SemanticServerError

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client, max_retries=0)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticServerError) as exc_info:
        extractor.extract(sem_input)
    assert exc_info.value.status_code == 503
    assert "server error" in str(exc_info.value)


def test_provider_timeout_error_classification():
    from app.extractors.semantic_extractor import SemanticTimeoutError

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client, timeout=10.0, max_retries=0)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticTimeoutError) as exc_info:
        extractor.extract(sem_input)
    assert "timed out after 10.0s" in str(exc_info.value)


def test_transient_retry_success_with_exponential_backoff():
    call_count = 0
    sleep_calls: list[float] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx.Response(503, text="Temporary server error", request=request)
        # Attempt 3 succeeds
        payload = {
            "personal": {"name": {"value": "Jane Doe", "source_block_ids": ["b_1"]}},
        }
        envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=3,
        initial_backoff=1.0,
        backoff_multiplier=2.0,
        sleep_fn=sleep_calls.append,
    )
    sem_input = _sample_semantic_input()

    output = extractor.extract(sem_input)
    assert call_count == 3
    assert output.personal.name.value == "Jane Doe"
    # Attempt 1 failed -> slept 1.0s; Attempt 2 failed -> slept 2.0s
    assert sleep_calls == [1.0, 2.0]


def test_transient_retry_rate_limit_uses_retry_after():
    call_count = 0
    sleep_calls: list[float] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, text="Too Many Requests", request=request)
        payload = {"personal": {"name": {"value": "Jane", "source_block_ids": ["b1"]}}}
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=2,
        sleep_fn=sleep_calls.append,
    )
    sem_input = _sample_semantic_input()

    output = extractor.extract(sem_input)
    assert call_count == 2
    assert output.personal.name.value == "Jane"
    # Should use Retry-After header (7.0s) instead of default backoff (1.0s)
    assert sleep_calls == [7.0]


def test_transient_retry_exhaustion_no_sleep_after_final():
    from app.extractors.semantic_extractor import SemanticServerError

    call_count = 0
    sleep_calls: list[float] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(500, text="Persistent glitch", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=2,
        initial_backoff=1.0,
        backoff_multiplier=2.0,
        sleep_fn=sleep_calls.append,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticServerError):
        extractor.extract(sem_input)
    # Initial attempt + 2 retries = 3 calls
    assert call_count == 3
    # Slept after attempt 0 (1.0s) and attempt 1 (2.0s); NO sleep after attempt 2
    assert sleep_calls == [1.0, 2.0]


def test_non_transient_client_error_not_retried_no_sleep():
    call_count = 0
    sleep_calls: list[float] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, text="Bad Request", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=3,
        sleep_fn=sleep_calls.append,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    # Must fail on attempt 1 without retry or sleep
    assert call_count == 1
    assert sleep_calls == []
    assert "HTTP 400" in str(exc_info.value)


def test_deterministic_malformed_json_not_retried_no_sleep():
    call_count = 0
    sleep_calls: list[float] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        envelope = {"candidates": [{"content": {"parts": [{"text": "Not valid JSON output"}]}}]}
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=3,
        sleep_fn=sleep_calls.append,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(sem_input)
    assert call_count == 1
    assert sleep_calls == []
    assert "Malformed JSON response" in str(exc_info.value)


def test_provider_malformed_envelope_handling():
    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client, max_retries=0)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(sem_input)
    assert "returned no candidates" in str(exc_info.value)


def test_provider_request_and_end_to_end_pipeline_seam():
    # Verify request payload and successful parse through validation into Resume
    captured_requests: list[httpx.Request] = []

    def mock_transport_grounded(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        llm_response_json = {
            "document_archetype": "standard_cv",
            "personal": {
                "name": {"value": "John Doe", "raw_value": "John Doe", "source_block_ids": ["b_p1_0"]},
                "email": {"value": "john.doe@example.com", "source_block_ids": ["b_p1_1"]},
            },
            "experience": [
                {
                    "company": {"value": "Acme Corporation", "source_block_ids": ["b_p1_4"]},
                    "designation": {"value": "Senior Software Engineer", "source_block_ids": ["b_p1_5"]},
                    "source_block_ids": ["b_p1_4", "b_p1_5"],
                }
            ],
        }
        gemini_envelope = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(llm_response_json)}],
                    }
                }
            ]
        }
        return httpx.Response(200, json=gemini_envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_grounded))
    extractor = GeminiSemanticExtractor(api_key="fake-gemini-key", model="gemini-3.5-flash-lite", client=client)

    doc = _make_test_document()
    resume = parse_document_semantically(doc, extractor, document_id="gemini-e2e-1")

    # 1. Verify request was constructed properly
    assert len(captured_requests) > 0
    req = captured_requests[-1]
    # Exact URL assertion: normal URL string without Markdown links
    expected_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
    assert str(req.url) == expected_url
    assert req.headers["x-goog-api-key"] == "fake-gemini-key"

    # Verify request body contains responseMimeType and responseSchema
    req_body = json.loads(req.content.decode("utf-8"))
    gen_config = req_body["generationConfig"]
    assert gen_config["responseMimeType"] == "application/json"
    assert "responseSchema" in gen_config
    assert gen_config["responseSchema"]["type"] == "object"
    assert "personal" in gen_config["responseSchema"]["properties"]

    # Verify prompt contains serialized blocks
    assert "DOCUMENT BLOCKS (JSON):" in req_body["contents"][0]["parts"][0]["text"]

    # 2. Verify response passed validation and reached Resume projection
    assert isinstance(resume, Resume)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"
    assert len(resume.experience) == 1
    assert resume.experience[0].company == "Acme Corporation"


def test_gemini_observability_success_metadata():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = {"personal": {"name": {"value": "Alice", "source_block_ids": ["b1"]}}}
        envelope = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
            "usageMetadata": {
                "promptTokenCount": 2219,
                "candidatesTokenCount": 404,
                "totalTokenCount": 2623,
            },
        }
        return httpx.Response(200, json=envelope, request=request)

    clock_values = [10.0, 12.6155]  # 2615.5 ms
    clock_idx = 0

    def mock_clock() -> float:
        nonlocal clock_idx
        val = clock_values[min(clock_idx, len(clock_values) - 1)]
        clock_idx += 1
        return val

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-2.5-flash",
        client=client,
        time_fn=mock_clock,
    )
    sem_input = _sample_semantic_input()
    output = extractor.extract(sem_input)

    assert output.personal.name.value == "Alice"
    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["provider"] == "gemini"
    assert meta["model"] == "gemini-2.5-flash"
    assert meta["prompt_tokens"] == 2219
    assert meta["output_tokens"] == 404
    assert meta["total_tokens"] == 2623
    assert meta["latency_ms"] == 2615.5
    assert meta["retry_count"] == 0
    assert meta["status"] == "success"


def test_gemini_observability_missing_usage_metadata():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = {"personal": {"name": {"value": "Alice", "source_block_ids": ["b1"]}}}
        envelope = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
            # No usageMetadata provided
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
    )
    sem_input = _sample_semantic_input()
    extractor.extract(sem_input)

    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["prompt_tokens"] is None
    assert meta["output_tokens"] is None
    assert meta["total_tokens"] is None
    assert meta["latency_ms"] >= 0.0
    assert meta["status"] == "success"
    assert meta["retry_count"] == 0


def test_gemini_observability_retry_metadata_on_recovery():
    call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(503, text="Temporary error", request=request)
        payload = {"personal": {"name": {"value": "Bob", "source_block_ids": ["b1"]}}}
        envelope = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150},
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=2,
        sleep_fn=lambda _: None,
    )
    sem_input = _sample_semantic_input()
    extractor.extract(sem_input)

    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["retry_count"] == 1
    assert meta["status"] == "success"
    assert meta["prompt_tokens"] == 100


def test_gemini_observability_retry_exhaustion_failure_metadata():
    from app.extractors.semantic_extractor import SemanticServerError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Server Error", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-2.5-flash",
        client=client,
        max_retries=2,
        sleep_fn=lambda _: None,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticServerError):
        extractor.extract(sem_input)

    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["provider"] == "gemini"
    assert meta["model"] == "gemini-2.5-flash"
    assert meta["retry_count"] == 2
    assert meta["status"] == "failure"
    assert meta["error_type"] == "SemanticServerError"
    assert meta["latency_ms"] >= 0.0


def test_gemini_observability_deterministic_malformed_json_failure_metadata():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        envelope = {"candidates": [{"content": {"parts": [{"text": "INVALID_JSON_HERE"}]}}]}
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticExtractionError):
        extractor.extract(sem_input)

    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["retry_count"] == 0
    assert meta["status"] == "failure"
    assert meta["error_type"] == "SemanticExtractionError"


def test_gemini_observability_configuration_failure_metadata(monkeypatch):
    from app.extractors.semantic_extractor import SemanticConfigurationError

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("app.extractors.providers.gemini.Settings", lambda: type("S", (), {"gemini_api_key": None})())

    extractor = GeminiSemanticExtractor(api_key=None, model="gemini-custom")
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticConfigurationError):
        extractor.extract(sem_input)

    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["status"] == "failure"
    assert meta["error_type"] == "SemanticConfigurationError"
    assert meta["retry_count"] == 0


def test_gemini_observability_no_credentials_or_content_stored():
    secret_key = "AIzaSySuperSecretKey999"

    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = {"personal": {"name": {"value": "SecretName", "source_block_ids": ["b1"]}}}
        envelope = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20},
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key=secret_key,
        client=client,
    )
    sem_input = _sample_semantic_input()
    extractor.extract(sem_input)
    meta = extractor.last_usage_metadata
    meta_str = json.dumps(meta)

    # Assert API key, prompt text, and resume content are completely absent
    assert secret_key not in meta_str
    assert "SecretName" not in meta_str
    assert "DOCUMENT BLOCKS" not in meta_str


def test_gemini_observability_preserves_usage_metadata_on_post_response_failure():
    # 1. HTTP 200 with valid usageMetadata
    # 2. Malformed semantic JSON in content parts
    def mock_transport(request: httpx.Request) -> httpx.Response:
        envelope = {
            "candidates": [{"content": {"parts": [{"text": "THIS_IS_MALFORMED_JSON"}]}}],
            "usageMetadata": {
                "promptTokenCount": 2219,
                "candidatesTokenCount": 404,
                "totalTokenCount": 2623,
            },
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-2.5-flash",
        client=client,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticExtractionError):
        extractor.extract(sem_input)

    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["provider"] == "gemini"
    assert meta["model"] == "gemini-2.5-flash"
    assert meta["prompt_tokens"] == 2219
    assert meta["output_tokens"] == 404
    assert meta["total_tokens"] == 2623
    assert meta["status"] == "failure"
    assert meta["error_type"] == "SemanticExtractionError"
    assert meta["retry_count"] == 0
    assert meta["latency_ms"] >= 0.0


def test_gemini_observability_preserves_usage_metadata_on_empty_candidates():
    from app.extractors.semantic_extractor import SemanticResponseError

    # HTTP 200 with valid usageMetadata but empty candidates list
    def mock_transport(request: httpx.Request) -> httpx.Response:
        envelope = {
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 1500,
                "candidatesTokenCount": 0,
                "totalTokenCount": 1500,
            },
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-2.5-flash",
        client=client,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError):
        extractor.extract(sem_input)

    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["prompt_tokens"] == 1500
    assert meta["output_tokens"] == 0
    assert meta["total_tokens"] == 1500
    assert meta["status"] == "failure"
    assert meta["error_type"] == "SemanticResponseError"


# =====================================================================
# Checkpoint 3: Robust Response-Envelope Validation Tests
# =====================================================================

def test_gemini_envelope_missing_candidates():
    from app.extractors.semantic_extractor import SemanticResponseError

    call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client, max_retries=3)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "missing 'candidates' field" in str(exc_info.value)
    assert call_count == 1  # Fails fast, zero retries


def test_gemini_envelope_candidate_missing_content():
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "missing 'content'" in str(exc_info.value)


def test_gemini_envelope_content_missing_parts():
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {}}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "missing 'parts'" in str(exc_info.value)


def test_gemini_envelope_empty_parts():
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": []}}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "contains no parts" in str(exc_info.value)


def test_gemini_envelope_part_missing_text():
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{}]}}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "missing 'text'" in str(exc_info.value)


@pytest.mark.parametrize("empty_text", ["", "   ", "\n\t  "])
def test_gemini_envelope_empty_text(empty_text):
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": empty_text}]}}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "empty text" in str(exc_info.value)


def test_gemini_envelope_blocked_safety():
    from app.extractors.semantic_extractor import SemanticResponseError

    call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        envelope = {
            "candidates": [
                {
                    "finishReason": "SAFETY",
                    "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "HIGH"}],
                }
            ],
            "usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 0, "totalTokenCount": 200},
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client, max_retries=3)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "finishReason=SAFETY" in str(exc_info.value)
    assert call_count == 1  # Must not retry safety blocks
    assert extractor.last_usage_metadata["prompt_tokens"] == 200
    assert extractor.last_usage_metadata["status"] == "failure"


@pytest.mark.parametrize(
    "blocked_reason",
    [
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "NO_IMAGE",
        "UNEXPECTED_TOOL_CALL",
        "TOO_MANY_TOOL_CALLS",
        "MAX_TOKENS",
        "OTHER",
    ],
)
def test_gemini_envelope_blocked_other_finish_reasons(blocked_reason):
    from app.extractors.semantic_extractor import SemanticResponseError

    call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        envelope = {
            "candidates": [{"finishReason": blocked_reason}],
            "usageMetadata": {"promptTokenCount": 150, "candidatesTokenCount": 0, "totalTokenCount": 150},
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client, max_retries=2)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert f"finishReason={blocked_reason}" in str(exc_info.value)
    assert call_count == 1  # Fails fast, zero retries
    assert extractor.last_usage_metadata["retry_count"] == 0
    assert extractor.last_usage_metadata["status"] == "failure"
    assert extractor.last_usage_metadata["error_type"] == "SemanticResponseError"
    assert extractor.last_usage_metadata["prompt_tokens"] == 150


def test_gemini_envelope_prompt_feedback_blocked():
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        envelope = {
            "promptFeedback": {"blockReason": "SAFETY"},
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "blockReason=SAFETY" in str(exc_info.value)


@pytest.mark.parametrize("bad_candidates", [None, "invalid_string", 123, {}])
def test_gemini_envelope_malformed_candidates_type(bad_candidates):
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": bad_candidates}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError):
        extractor.extract(sem_input)


@pytest.mark.parametrize("bad_content", [None, "invalid", 123, []])
def test_gemini_envelope_malformed_content_type(bad_content):
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": bad_content}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError):
        extractor.extract(sem_input)


@pytest.mark.parametrize("bad_parts", [None, "invalid", 123, {}])
def test_gemini_envelope_malformed_parts_type(bad_parts):
    from app.extractors.semantic_extractor import SemanticResponseError

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": bad_parts}}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError):
        extractor.extract(sem_input)


# =====================================================================
# Checkpoint 4: HTTP Status Classification, Retry Correctness, and Config Safety
# =====================================================================

@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [
        (400, SemanticResponseError),
        (401, SemanticConfigurationError),
        (403, SemanticConfigurationError),
        (404, SemanticConfigurationError),
        (405, SemanticExtractionError),
        (422, SemanticExtractionError),
    ],
)
def test_gemini_http_status_classification_deterministic(status_code, expected_exception):
    call_count = 0
    sleep_calls = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, text=f"HTTP {status_code} Error", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=3,
        sleep_fn=sleep_calls.append,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(expected_exception):
        extractor.extract(sem_input)

    assert call_count == 1  # Deterministic failure, zero retries
    assert sleep_calls == []
    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["status"] == "failure"
    assert meta["error_type"] == expected_exception.__name__
    assert meta["retry_count"] == 0
    assert meta["prompt_tokens"] is None


@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [
        (429, SemanticRateLimitError),
        (500, SemanticServerError),
        (502, SemanticServerError),
        (503, SemanticServerError),
    ],
)
def test_gemini_http_status_classification_transient_retries(status_code, expected_exception):
    call_count = 0
    sleep_calls = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, text=f"Transient {status_code}", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=2,
        initial_backoff=1.0,
        backoff_multiplier=2.0,
        sleep_fn=sleep_calls.append,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(expected_exception):
        extractor.extract(sem_input)

    assert call_count == 3  # Initial attempt + 2 retries
    assert sleep_calls == [1.0, 2.0]  # No sleep after final attempt
    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["status"] == "failure"
    assert meta["error_type"] == expected_exception.__name__
    assert meta["retry_count"] == 2


@pytest.mark.parametrize(
    ("retry_header", "expected_sleep"),
    [
        ("10", [10.0]),         # Positive integer honored
        ("0", []),              # Zero -> no sleep performed
        ("-5", [1.0]),          # Negative -> ignored, fallback to initial backoff
        ("5.5", [1.0]),         # Decimal -> ignored, fallback to initial backoff
        ("invalid", [1.0]),     # Non-numeric -> ignored, fallback to initial backoff
        ("120", [30.0]),        # Excessive value -> capped at max_backoff (30.0s)
    ],
)
def test_gemini_retry_after_parsing_matrix(retry_header, expected_sleep):
    call_count = 0
    sleep_calls = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        headers = {"retry-after": retry_header}
        return httpx.Response(429, headers=headers, text="Rate limit hit", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        client=client,
        max_retries=1,
        initial_backoff=1.0,
        max_backoff=30.0,
        sleep_fn=sleep_calls.append,
    )
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticRateLimitError):
        extractor.extract(sem_input)

    assert call_count == 2
    assert sleep_calls == expected_sleep


def test_gemini_api_error_body_structured_json():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        error_body = {
            "error": {
                "code": 400,
                "message": "Invalid argument provided in generationConfig",
                "status": "INVALID_ARGUMENT",
            }
        }
        return httpx.Response(400, json=error_body, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticResponseError) as exc_info:
        extractor.extract(sem_input)
    assert "Invalid argument provided in generationConfig" in str(exc_info.value)
    assert "status=INVALID_ARGUMENT" in str(exc_info.value)


def test_gemini_api_error_body_redaction_and_malformed():
    secret_key = "AIzaSySuperSecretApiKey12345"

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=f"Forbidden request with key {secret_key}", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(api_key=secret_key, client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticConfigurationError) as exc_info:
        extractor.extract(sem_input)

    err_str = str(exc_info.value)
    assert secret_key not in err_str
    assert "[REDACTED]" in err_str


@pytest.mark.parametrize("empty_key", ["", "   ", "\t\n  "])
def test_gemini_configuration_rejects_empty_api_key(empty_key, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    extractor = GeminiSemanticExtractor(api_key=empty_key)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticConfigurationError) as exc_info:
        extractor.extract(sem_input)
    assert "Gemini API key is required" in str(exc_info.value)


def test_gemini_configuration_safely_defaults_invalid_numeric_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
    monkeypatch.setenv("GEMINI_TIMEOUT", "-10.0")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "-5")
    monkeypatch.setenv("GEMINI_BASE_URL", "   ")

    extractor = GeminiSemanticExtractor()
    api_key, model, base_url, timeout, max_retries = extractor._resolve_config()

    assert api_key == "valid-key"
    assert model == "gemini-3.5-flash-lite"
    assert base_url == "https://generativelanguage.googleapis.com"
    assert timeout == 30.0
    assert max_retries == 2
