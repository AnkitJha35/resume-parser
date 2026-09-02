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
from app.extractors.semantic_extractor import SemanticExtractionError
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


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    extractor = GeminiSemanticExtractor(api_key=None)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(sem_input)
    assert "Gemini API key is required but not configured" in str(exc_info.value)


def test_provider_http_error_handling():
    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(sem_input)
    assert "Gemini API HTTP error 500" in str(exc_info.value)


def test_provider_malformed_envelope_handling():
    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))
    extractor = GeminiSemanticExtractor(api_key="test-key", client=client)
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
    extractor = GeminiSemanticExtractor(api_key="fake-gemini-key", model="gemini-2.5-flash", client=client)

    doc = _make_test_document()
    resume = parse_document_semantically(doc, extractor, document_id="gemini-e2e-1")

    # 1. Verify request was constructed properly
    assert len(captured_requests) > 0
    req = captured_requests[-1]
    # Exact URL assertion: normal URL string without Markdown links
    expected_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
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
