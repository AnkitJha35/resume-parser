"""Comprehensive unit and integration tests for two-pass semantic extraction architecture."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from app.domain.document import Document, document_from_text_blocks
from app.domain.resume import Resume
from app.domain.semantic_contract import (
    BlockClassification,
    BodySemanticOutput,
    DocumentArchetype,
    GroundedBool,
    GroundedEducationItem,
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedProjectItem,
    GroundedString,
    PersonalSemanticOutput,
    SemanticBlockCategory,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    build_semantic_input,
    is_body_output_suspiciously_empty,
    merge_semantic_passes,
    semantic_output_to_resume,
    validate_semantic_output,
)
from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.semantic_extractor import (
    MockSemanticExtractor,
    SemanticExtractionError,
    SemanticRateLimitError,
    SemanticResponseError,
    SemanticServerError,
)
from app.extractors.semantic_prompt import (
    build_body_extraction_prompt,
    build_personal_extraction_prompt,
    get_body_schema,
    get_personal_schema,
    parse_body_output,
    parse_personal_output,
)
from app.pipeline.semantic_pipeline import parse_document_semantically
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor


def _sample_semantic_input() -> SemanticInput:
    """Construct a minimal valid SemanticInput for testing."""
    return SemanticInput(
        document_id="doc-test-1",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        blocks=[
            SemanticBlockInput(
                block_id="b1",
                text="Jane Doe",
                page=1,
                bbox=[50.0, 50.0, 200.0, 70.0],
                region_id="r1",
                region_kind="header",
                reading_order=1,
                suggested_role="HEADER",
            ),
            SemanticBlockInput(
                block_id="b2",
                text="Email: jane.doe@example.com",
                page=1,
                bbox=[50.0, 75.0, 250.0, 90.0],
                region_id="r1",
                region_kind="header",
                reading_order=2,
                suggested_role="FIELD",
            ),
            SemanticBlockInput(
                block_id="b3",
                text="Acme Corp - Senior Engineer (2020 - Present)",
                page=1,
                bbox=[50.0, 100.0, 400.0, 120.0],
                region_id="r2",
                region_kind="column",
                reading_order=3,
                suggested_role="ENTRY",
            ),
            SemanticBlockInput(
                block_id="b4",
                text="B.Sc. Computer Science - MIT (2016 - 2020)",
                page=1,
                bbox=[50.0, 130.0, 400.0, 150.0],
                region_id="r2",
                region_kind="column",
                reading_order=4,
                suggested_role="ENTRY",
            ),
            SemanticBlockInput(
                block_id="b5",
                text="AWS Certified Solutions Architect",
                page=1,
                bbox=[50.0, 160.0, 300.0, 180.0],
                region_id="r2",
                region_kind="column",
                reading_order=5,
                suggested_role="ENTRY",
            ),
        ],
    )


# =====================================================================
# 1. Partial Schema & Prompt Construction Tests
# =====================================================================


def test_personal_semantic_output_schema_and_defaults():
    """Verify PersonalSemanticOutput instantiation and defaults."""
    out = PersonalSemanticOutput()
    assert out.document_archetype == DocumentArchetype.UNKNOWN
    assert out.block_classifications == []
    assert out.personal.name is None
    assert out.personal.email is None

    populated = PersonalSemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="Jane Doe", source_block_ids=["b1"]),
            email=GroundedString(value="jane.doe@example.com", source_block_ids=["b2"]),
        ),
        block_classifications=[
            BlockClassification(block_id="b1", category=SemanticBlockCategory.PERSONAL),
            BlockClassification(block_id="b2", category=SemanticBlockCategory.PERSONAL),
        ],
    )
    assert populated.personal.name.value == "Jane Doe"
    assert populated.personal.email.value == "jane.doe@example.com"
    assert len(populated.block_classifications) == 2


def test_body_semantic_output_schema_and_defaults():
    """Verify BodySemanticOutput instantiation and defaults."""
    out = BodySemanticOutput()
    assert out.document_archetype == DocumentArchetype.UNKNOWN
    assert out.summary is None
    assert out.experience == []
    assert out.education == []
    assert out.certifications == []

    populated = BodySemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Acme Corp", source_block_ids=["b3"]),
                designation=GroundedString(value="Senior Engineer", source_block_ids=["b3"]),
                source_block_ids=["b3"],
            )
        ],
        education=[
            GroundedEducationItem(
                institution=GroundedString(value="MIT", source_block_ids=["b4"]),
                degree=GroundedString(value="B.Sc. Computer Science", source_block_ids=["b4"]),
                source_block_ids=["b4"],
            )
        ],
        certifications=[
            GroundedString(value="AWS Certified", source_block_ids=["b5"])
        ],
    )
    assert len(populated.experience) == 1
    assert len(populated.education) == 1
    assert len(populated.certifications) == 1


def test_personal_and_body_prompt_builders():
    """Verify focused prompt builders generate appropriate schema contracts and instructions."""
    sem_input = _sample_semantic_input()

    personal_prompt = build_personal_extraction_prompt(sem_input)
    assert "PersonalSemanticOutput" in personal_prompt
    assert "PERSONAL CONTACT & IDENTITY EXTRACTION GUIDELINES" in personal_prompt
    assert "DOCUMENT BLOCKS (JSON):" in personal_prompt
    assert "Jane Doe" in personal_prompt

    body_prompt = build_body_extraction_prompt(sem_input)
    assert "BodySemanticOutput" in body_prompt
    assert "OUTPUT COMPLETENESS & STRUCTURE REQUIREMENTS" in body_prompt
    assert "Education Tables:" in body_prompt
    assert "Maritime Sea-Service & Employment Tables:" in body_prompt
    assert "Certification, Course, & Endorsement Tables:" in body_prompt
    assert "DOCUMENT BLOCKS (JSON):" in body_prompt

    personal_schema = get_personal_schema()
    assert personal_schema["type"] == "object"
    assert "personal" in personal_schema["properties"]
    assert "experience" not in personal_schema["properties"]

    body_schema = get_body_schema()
    assert body_schema["type"] == "object"
    assert "experience" in body_schema["properties"]
    assert "education" in body_schema["properties"]
    assert "personal" not in body_schema["properties"]


# =====================================================================
# 2. Parsing Tests
# =====================================================================


def test_parse_personal_output_valid_and_fenced():
    """Verify parse_personal_output handles raw JSON and markdown code fences."""
    raw = json.dumps({
        "document_archetype": "standard_cv",
        "personal": {
            "name": {"value": "Alice", "source_block_ids": ["b1"]},
            "email": {"value": "alice@test.com", "source_block_ids": ["b2"]},
        },
        "block_classifications": [
            {"block_id": "b1", "category": "PERSONAL"},
            {"block_id": "b2", "category": "PERSONAL"},
        ],
    })
    parsed = parse_personal_output(raw)
    assert parsed.personal.name.value == "Alice"
    assert parsed.personal.email.value == "alice@test.com"

    fenced = f"```json\n{raw}\n```"
    parsed_fenced = parse_personal_output(fenced)
    assert parsed_fenced.personal.name.value == "Alice"


def test_parse_personal_output_errors():
    """Verify parse_personal_output raises on invalid JSON or bad types."""
    with pytest.raises(SemanticExtractionError, match="Malformed JSON"):
        parse_personal_output("not json")

    with pytest.raises(SemanticExtractionError, match="Expected JSON object"):
        parse_personal_output(json.dumps(["not a dict"]))

    with pytest.raises(SemanticExtractionError, match="Invalid PersonalSemanticOutput"):
        parse_personal_output(json.dumps({"personal": "invalid_type"}))


def test_parse_body_output_valid_and_fenced():
    """Verify parse_body_output handles raw JSON and markdown code fences."""
    raw = json.dumps({
        "document_archetype": "standard_cv",
        "experience": [
            {
                "company": {"value": "Google", "source_block_ids": ["b3"]},
                "designation": {"value": "Software Engineer", "source_block_ids": ["b3"]},
                "source_block_ids": ["b3"],
            }
        ],
        "education": [
            {
                "institution": {"value": "Stanford", "source_block_ids": ["b4"]},
                "degree": {"value": "BS", "source_block_ids": ["b4"]},
                "source_block_ids": ["b4"],
            }
        ],
        "certifications": [
            {"value": "GCP Architect", "source_block_ids": ["b5"]}
        ],
    })
    parsed = parse_body_output(raw)
    assert len(parsed.experience) == 1
    assert parsed.experience[0].company.value == "Google"
    assert len(parsed.education) == 1
    assert parsed.education[0].institution.value == "Stanford"
    assert len(parsed.certifications) == 1

    fenced = f"```json\n{raw}\n```"
    parsed_fenced = parse_body_output(fenced)
    assert parsed_fenced.experience[0].company.value == "Google"


def test_parse_body_output_errors():
    """Verify parse_body_output raises on invalid JSON or bad types."""
    with pytest.raises(SemanticExtractionError, match="Malformed JSON"):
        parse_body_output("not json")

    with pytest.raises(SemanticExtractionError, match="Expected JSON object"):
        parse_body_output(json.dumps(["not a dict"]))

    with pytest.raises(SemanticExtractionError, match="Invalid BodySemanticOutput"):
        parse_body_output(json.dumps({"experience": "invalid_type"}))


# =====================================================================
# 3. Deterministic Merge Tests
# =====================================================================


def test_merge_semantic_passes_clean_combination():
    """Verify merge_semantic_passes combines personal and body outputs into canonical SemanticOutput."""
    personal_out = PersonalSemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="Bob", source_block_ids=["b1"]),
            email=GroundedString(value="bob@example.com", source_block_ids=["b2"]),
        ),
        block_classifications=[
            BlockClassification(block_id="b1", category=SemanticBlockCategory.PERSONAL),
            BlockClassification(block_id="b2", category=SemanticBlockCategory.PERSONAL),
        ],
    )

    body_out = BodySemanticOutput(
        document_archetype=DocumentArchetype.STRUCTURED_FORM,
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Tech Corp", source_block_ids=["b3"]),
                source_block_ids=["b3"],
            )
        ],
        education=[
            GroundedEducationItem(
                institution=GroundedString(value="Oxford", source_block_ids=["b4"]),
                source_block_ids=["b4"],
            )
        ],
        certifications=[GroundedString(value="PMP", source_block_ids=["b5"])],
        block_classifications=[
            BlockClassification(block_id="b3", category=SemanticBlockCategory.EXPERIENCE),
            BlockClassification(block_id="b4", category=SemanticBlockCategory.EDUCATION),
        ],
    )

    merged = merge_semantic_passes(personal_out, body_out)

    assert isinstance(merged, SemanticOutput)
    # Archetype precedence: specific body archetype preferred over standard personal
    assert merged.document_archetype == DocumentArchetype.STRUCTURED_FORM
    assert merged.personal.name.value == "Bob"
    assert merged.personal.email.value == "bob@example.com"
    assert len(merged.experience) == 1
    assert merged.experience[0].company.value == "Tech Corp"
    assert len(merged.education) == 1
    assert merged.education[0].institution.value == "Oxford"
    assert len(merged.certifications) == 1
    assert merged.certifications[0].value == "PMP"

    # Deterministic sorted block classifications
    assert len(merged.block_classifications) == 4
    block_ids = [bc.block_id for bc in merged.block_classifications]
    assert block_ids == ["b1", "b2", "b3", "b4"]


def test_merge_semantic_passes_duplicate_classification_handling():
    """Verify duplicate block classification preserves PERSONAL for personal blocks and sorts deterministically."""
    personal_out = PersonalSemanticOutput(
        document_archetype=DocumentArchetype.UNKNOWN,
        personal=GroundedPersonal(name=GroundedString(value="Alice", source_block_ids=["b1"])),
        block_classifications=[
            BlockClassification(block_id="b1", category=SemanticBlockCategory.PERSONAL),
            BlockClassification(block_id="b2", category=SemanticBlockCategory.PERSONAL),
        ],
    )

    body_out = BodySemanticOutput(
        document_archetype=DocumentArchetype.MARITIME_CV,
        block_classifications=[
            BlockClassification(block_id="b1", category=SemanticBlockCategory.TABLE_HEADER),  # Conflict with PERSONAL
            BlockClassification(block_id="b3", category=SemanticBlockCategory.EXPERIENCE),
        ],
    )

    merged = merge_semantic_passes(personal_out, body_out)
    assert merged.document_archetype == DocumentArchetype.MARITIME_CV

    bc_map = {bc.block_id: bc.category for bc in merged.block_classifications}
    # b1 retained PERSONAL category despite body pass classifying as TABLE_HEADER
    assert bc_map["b1"] == SemanticBlockCategory.PERSONAL
    assert bc_map["b2"] == SemanticBlockCategory.PERSONAL
    assert bc_map["b3"] == SemanticBlockCategory.EXPERIENCE


# =====================================================================
# 4. Concurrent Execution & Usage Aggregation in Gemini Provider
# =====================================================================


def test_gemini_two_pass_concurrent_execution_and_usage_aggregation():
    """Verify GeminiSemanticExtractor with two_pass=True executes concurrently and aggregates usage."""
    captured_requests: list[httpx.Request] = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]

        if "PersonalSemanticOutput" in prompt_text:
            payload = {
                "document_archetype": "standard_cv",
                "personal": {
                    "name": {"value": "Jane Doe", "source_block_ids": ["b1"]},
                    "email": {"value": "jane.doe@example.com", "source_block_ids": ["b2"]},
                },
                "block_classifications": [
                    {"block_id": "b1", "category": "PERSONAL"},
                    {"block_id": "b2", "category": "PERSONAL"},
                ],
            }
            envelope = {
                "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "candidatesTokenCount": 100,
                    "totalTokenCount": 1100,
                },
            }
        else:
            payload = {
                "document_archetype": "standard_cv",
                "experience": [
                    {
                        "company": {"value": "Acme Corp", "source_block_ids": ["b3"]},
                        "designation": {"value": "Senior Engineer", "source_block_ids": ["b3"]},
                        "source_block_ids": ["b3"],
                    }
                ],
                "education": [
                    {
                        "institution": {"value": "MIT", "source_block_ids": ["b4"]},
                        "degree": {"value": "B.Sc. Computer Science", "source_block_ids": ["b4"]},
                        "source_block_ids": ["b4"],
                    }
                ],
                "certifications": [
                    {"value": "AWS Certified", "source_block_ids": ["b5"]}
                ],
                "block_classifications": [
                    {"block_id": "b3", "category": "EXPERIENCE"},
                    {"block_id": "b4", "category": "EDUCATION"},
                    {"block_id": "b5", "category": "CERTIFICATION"},
                ],
            }
            envelope = {
                "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
                "usageMetadata": {
                    "promptTokenCount": 1500,
                    "candidatesTokenCount": 300,
                    "totalTokenCount": 1800,
                },
            }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-api-key",
        model="gemini-3.5-flash-lite",
        client=client,
        two_pass=True,
    )

    sem_input = _sample_semantic_input()
    output = extractor.extract(sem_input)

    # Verify both requests were sent
    assert len(captured_requests) == 2
    prompts = [json.loads(r.content.decode("utf-8"))["contents"][0]["parts"][0]["text"] for r in captured_requests]
    assert any("PersonalSemanticOutput" in p for p in prompts)
    assert any("BodySemanticOutput" in p for p in prompts)

    # Verify merged output
    assert output.personal.name.value == "Jane Doe"
    assert output.personal.email.value == "jane.doe@example.com"
    assert len(output.experience) == 1
    assert output.experience[0].company.value == "Acme Corp"
    assert len(output.education) == 1
    assert output.education[0].institution.value == "MIT"
    assert len(output.certifications) == 1

    # Verify aggregated usage metadata
    meta = extractor.last_usage_metadata
    assert meta is not None
    assert meta["two_pass"] is True
    assert meta["prompt_tokens"] == 2500  # 1000 + 1500
    assert meta["output_tokens"] == 400   # 100 + 300
    assert meta["total_tokens"] == 2900   # 1100 + 1800
    assert meta["retry_count"] == 0
    assert "pass_metadata" in meta
    assert "personal" in meta["pass_metadata"]
    assert "body" in meta["pass_metadata"]


def test_gemini_two_pass_independent_retries():
    """Verify Pass 1 can retry independently while Pass 2 succeeds, with retries aggregated."""
    personal_attempts = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal personal_attempts
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]

        if "PersonalSemanticOutput" in prompt_text:
            personal_attempts += 1
            if personal_attempts == 1:
                # First attempt fails with 429
                return httpx.Response(429, headers={"retry-after": "0"}, text="Rate limit", request=request)
            payload = {
                "personal": {"name": {"value": "Jane Doe", "source_block_ids": ["b1"]}},
            }
            envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
            return httpx.Response(200, json=envelope, request=request)
        else:
            payload = {
                "certifications": [{"value": "AWS Certified", "source_block_ids": ["b5"]}],
            }
            envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
            return httpx.Response(200, json=envelope, request=request)

    sleep_calls = []
    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-api-key",
        client=client,
        two_pass=True,
        max_retries=2,
        sleep_fn=sleep_calls.append,
    )

    sem_input = _sample_semantic_input()
    output = extractor.extract(sem_input)

    assert output.personal.name.value == "Jane Doe"
    assert len(output.certifications) == 1
    assert personal_attempts == 2
    assert extractor.last_usage_metadata["retry_count"] == 1


# =====================================================================
# 5. Atomic Failure & Fallback Tests
# =====================================================================


def test_gemini_two_pass_atomic_failure_on_personal_pass_error():
    """Verify if Pass 1 fails fatally, the whole extraction fails atomically."""
    def mock_transport(request: httpx.Request) -> httpx.Response:
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]

        if "PersonalSemanticOutput" in prompt_text:
            return httpx.Response(500, text="Internal Server Error", request=request)
        else:
            payload = {"certifications": [{"value": "AWS", "source_block_ids": ["b5"]}]}
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-api-key",
        client=client,
        two_pass=True,
        max_retries=0,
    )

    sem_input = _sample_semantic_input()
    with pytest.raises(SemanticServerError):
        extractor.extract(sem_input)

    assert extractor.last_usage_metadata["status"] == "failure"


def test_gemini_two_pass_atomic_failure_on_body_pass_error():
    """Verify if Pass 2 fails fatally, the whole extraction fails atomically."""
    def mock_transport(request: httpx.Request) -> httpx.Response:
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]

        if "PersonalSemanticOutput" in prompt_text:
            payload = {"personal": {"name": {"value": "Jane", "source_block_ids": ["b1"]}}}
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}, request=request)
        else:
            return httpx.Response(503, text="Service Unavailable", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-api-key",
        client=client,
        two_pass=True,
        max_retries=0,
    )

    sem_input = _sample_semantic_input()
    with pytest.raises(SemanticServerError):
        extractor.extract(sem_input)

    assert extractor.last_usage_metadata["status"] == "failure"


def test_fallback_extractor_triggers_full_fallback_on_two_pass_failure():
    """Verify FallbackSemanticExtractor routes the whole document to fallback on two-pass failure."""
    def mock_gemini(request: httpx.Request) -> httpx.Response:
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]
        if "PersonalSemanticOutput" in prompt_text:
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps({"personal": {"name": {"value": "Jane", "source_block_ids": ["b1"]}}})}]}}]}, request=request)
        # Body pass fails with 500
        return httpx.Response(500, text="Gemini Body Error", request=request)

    gemini_client = httpx.Client(transport=httpx.MockTransport(mock_gemini))
    gemini_primary = GeminiSemanticExtractor(
        api_key="test-key",
        client=gemini_client,
        two_pass=True,
        max_retries=0,
    )
    mock_fallback = MockSemanticExtractor()

    composite = FallbackSemanticExtractor(primary=gemini_primary, fallback=mock_fallback)
    sem_input = _sample_semantic_input()

    result = composite.extract(sem_input)
    assert isinstance(result, SemanticOutput)
    # The entire document was handled by MockSemanticExtractor
    assert composite.last_usage_metadata["fallback_invoked"] is True


# =====================================================================
# 6. End-to-End AKIBUL Production-Style Pipeline Test
# =====================================================================


def test_akibul_two_pass_e2e_mocked_gemini_pipeline():
    """Verify complete AKIBUL fixture flow: layout -> Candidate B SemanticInput -> two-pass extraction -> merge -> validation -> Resume."""
    from pathlib import Path
    pdf_path = "tests/fixtures/AKIBUL ALAM CV(JO).pdf"
    pdf_bytes = Path(pdf_path).read_bytes()
    text_blocks = PDFExtractor.extract(pdf_bytes)
    doc = document_from_text_blocks(text_blocks)
    interpreted = interpret_layout(doc)
    reconstructed = reconstruct_document(interpreted)

    # 1. Build Candidate B SemanticInput
    sem_input = build_semantic_input(reconstructed, document_id="akibul-e2e-test")
    assert sem_input.archetype == DocumentArchetype.STRUCTURED_FORM

    # 2. Mock Gemini Responses for Pass 1 and Pass 2 based on verified diagnostic data
    captured_requests: list[httpx.Request] = []

    def mock_gemini_akibul(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]

        if "PersonalSemanticOutput" in prompt_text:
            personal_payload = {
                "document_archetype": "structured_form",
                "personal": {
                    "name": {"value": "Akibul Alam", "raw_value": "Akibul Alam", "source_block_ids": ["b_p1_88", "b_p1_2"]},
                    "email": {"value": "akibulalam3@gmail.com", "raw_value": "akibulalam3@gmail.com", "source_block_ids": ["b_p1_13"]},
                    "phone": {"value": "+8801878143235", "raw_value": "+8801878143235", "source_block_ids": ["b_p2_102", "b_p1_12"]},
                },
                "block_classifications": [
                    {"block_id": "b_p1_88", "category": "PERSONAL"},
                    {"block_id": "b_p1_2", "category": "PERSONAL"},
                    {"block_id": "b_p1_13", "category": "PERSONAL"},
                    {"block_id": "b_p1_12", "category": "PERSONAL"},
                    {"block_id": "b_p2_102", "category": "PERSONAL"},
                ],
            }
            envelope = {
                "candidates": [{"content": {"parts": [{"text": json.dumps(personal_payload)}]}}],
                "usageMetadata": {"promptTokenCount": 18425, "candidatesTokenCount": 318, "totalTokenCount": 18743},
            }
            return httpx.Response(200, json=envelope, request=request)
        else:
            body_payload = {
                "document_archetype": "structured_form",
                "education": [
                    {
                        "institution": {"value": "International Maritime Academy", "raw_value": "International Maritime Academy", "source_block_ids": ["b_p3_221"]},
                        "degree": {"value": "Pre-sea Nautical Science", "raw_value": "Pre-sea Nautical Science", "source_block_ids": ["b_p3_262"]},
                        "startDate": {"value": "2020-04-01", "raw_value": "01-04-2020", "source_block_ids": ["b_p3_233"]},
                        "endDate": {"value": "2022-05-01", "raw_value": "01-05-2022", "source_block_ids": ["b_p3_261"]},
                        "source_block_ids": ["b_p3_221", "b_p3_262", "b_p3_233", "b_p3_261"],
                    }
                ],
                "experience": [
                    {
                        "company": {"value": "Unix Line PTE LTD", "raw_value": "Unix Line PTE LTD", "source_block_ids": ["b_p4_268", "b_p4_270"]},
                        "designation": {"value": "Deck Cadet", "raw_value": "DeckCadet", "source_block_ids": ["b_p4_297"]},
                        "startDate": {"value": "2023-02-18", "raw_value": "18-02-2023", "source_block_ids": ["b_p4_298", "b_p4_303"]},
                        "endDate": {"value": "2023-08-14", "raw_value": "14-08-2023", "source_block_ids": ["b_p4_299", "b_p4_304"]},
                        "description": {"value": "MT Furano Galaxy", "raw_value": "MT Furano Galaxy", "source_block_ids": ["b_p4_269", "b_p4_271"]},
                        "source_block_ids": ["b_p4_268", "b_p4_269", "b_p4_270", "b_p4_271", "b_p4_297", "b_p4_298", "b_p4_299", "b_p4_303", "b_p4_304"],
                    },
                    {
                        "company": {"value": "Unix Line PTE LTD", "raw_value": "Unix Line PTE LTD", "source_block_ids": ["b_p4_273", "b_p4_275"]},
                        "designation": {"value": "Deck Cadet", "raw_value": "DeckCadet", "source_block_ids": ["b_p4_306"]},
                        "startDate": {"value": "2024-03-07", "raw_value": "07-03-2024", "source_block_ids": ["b_p4_307", "b_p4_312"]},
                        "endDate": {"value": "2024-10-09", "raw_value": "09-10-2024", "source_block_ids": ["b_p4_308", "b_p4_313"]},
                        "description": {"value": "MT ELM Galaxy", "raw_value": "MT ELM Galaxy", "source_block_ids": ["b_p4_274", "b_p4_276"]},
                        "source_block_ids": ["b_p4_273", "b_p4_274", "b_p4_275", "b_p4_276", "b_p4_306", "b_p4_307", "b_p4_308", "b_p4_312", "b_p4_313"],
                    },
                ],
                "certifications": [
                    {"value": "Advanced Fire Fighting", "raw_value": "Advanced Fire Fighting", "source_block_ids": ["b_p2_111"]},
                    {"value": "Medical First Aid", "raw_value": "Medical First Aid", "source_block_ids": ["b_p2_114"]},
                ],
                "block_classifications": [
                    {"block_id": "b_p3_221", "category": "EDUCATION"},
                    {"block_id": "b_p4_268", "category": "EXPERIENCE"},
                    {"block_id": "b_p2_111", "category": "CERTIFICATION"},
                ],
            }
            envelope = {
                "candidates": [{"content": {"parts": [{"text": json.dumps(body_payload)}]}}],
                "usageMetadata": {"promptTokenCount": 19022, "candidatesTokenCount": 1200, "totalTokenCount": 20222},
            }
            return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_gemini_akibul))
    extractor = GeminiSemanticExtractor(
        api_key="test-api-key",
        model="gemini-3.6-flash",
        client=client,
        two_pass=True,
    )

    # 3. Execute production parse_document_semantically()
    resume = parse_document_semantically(reconstructed, extractor, document_id="akibul-e2e-doc")

    # 4. Verify canonical Resume fields
    assert isinstance(resume, Resume)
    assert resume.personal.name == "Akibul Alam"
    assert resume.personal.email == "akibulalam3@gmail.com"
    assert resume.personal.phone == "+8801878143235"

    assert len(resume.education) == 1
    assert resume.education[0].institution == "International Maritime Academy"
    assert resume.education[0].degree == "Pre-sea Nautical Science"
    assert resume.education[0].startDate == "2020-04-01"
    assert resume.education[0].endDate == "2022-05-01"

    assert len(resume.experience) == 2
    assert resume.experience[0].company == "Unix Line PTE LTD"
    assert resume.experience[0].designation == "Deck Cadet"
    assert resume.experience[0].startDate == "2023-02-18"
    assert resume.experience[0].endDate == "2023-08-14"
    assert "Furano Galaxy" in resume.experience[0].description

    assert resume.experience[1].company == "Unix Line PTE LTD"
    assert resume.experience[1].designation == "Deck Cadet"
    assert resume.experience[1].startDate == "2024-03-07"
    assert resume.experience[1].endDate == "2024-10-09"
    assert "ELM Galaxy" in resume.experience[1].description

    assert len(resume.certifications) == 2
    assert resume.certifications[0].name == "Advanced Fire Fighting"
    assert resume.certifications[1].name == "Medical First Aid"


# =====================================================================
# 7. Phase 10D: Silent Body Omission Detection & Recovery Tests
# =====================================================================


def test_is_body_output_suspiciously_empty_predicates():
    """Verify deterministic is_body_output_suspiciously_empty classification."""
    rich_input = _sample_semantic_input()  # Has b3 ENTRY, b4 ENTRY, b5 ENTRY (explicit roles)

    # 1. Empty body on evidence-rich input => True
    empty_body = BodySemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        summary=GroundedString(value="Some summary text", source_block_ids=["b3"]),
        skills=[],
        experience=[],
        education=[],
        projects=[],
        certifications=[],
        languages=[],
        achievements=[],
    )
    assert is_body_output_suspiciously_empty(empty_body, rich_input) is True

    # 2. Genuinely sparse input (only header/contact info, no body evidence) => False
    sparse_input = SemanticInput(
        document_id="sparse-doc",
        page_count=1,
        blocks=[
            SemanticBlockInput(
                block_id="b1",
                text="John Doe",
                page=1,
                bbox=[0, 0, 100, 20],
                region_id="r1",
                region_kind="header",
                reading_order=1,
                suggested_role="HEADER",
            ),
            SemanticBlockInput(
                block_id="b2",
                text="john@example.com",
                page=1,
                bbox=[0, 20, 100, 40],
                region_id="r1",
                region_kind="header",
                reading_order=2,
                suggested_role="CONTACT",
            ),
        ],
    )
    assert is_body_output_suspiciously_empty(empty_body, sparse_input) is False

    # 3. Populated body on rich input => False
    populated_skills = BodySemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        skills=[GroundedString(value="Python", source_block_ids=["b3"])],
    )
    assert is_body_output_suspiciously_empty(populated_skills, rich_input) is False

    populated_exp = BodySemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Acme Corp", source_block_ids=["b3"]),
                source_block_ids=["b3"],
            )
        ],
    )
    assert is_body_output_suspiciously_empty(populated_exp, rich_input) is False

    populated_edu = BodySemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        education=[
            GroundedEducationItem(
                institution=GroundedString(value="MIT", source_block_ids=["b4"]),
                source_block_ids=["b4"],
            )
        ],
    )
    assert is_body_output_suspiciously_empty(populated_edu, rich_input) is False


def test_gemini_two_pass_body_recovery_on_silent_omission_success():
    """Verify Gemini provider triggers Pass 2 recovery when empty body is returned on evidence-rich input, and recovers successfully."""
    captured_requests: list[httpx.Request] = []
    body_call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal body_call_count
        captured_requests.append(request)
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]

        if "PersonalSemanticOutput" in prompt_text:
            personal_payload = {
                "document_archetype": "standard_cv",
                "personal": {
                    "name": {"value": "Jane Doe", "source_block_ids": ["b1"]},
                    "email": {"value": "jane.doe@example.com", "source_block_ids": ["b2"]},
                },
                "block_classifications": [
                    {"block_id": "b1", "category": "PERSONAL"},
                    {"block_id": "b2", "category": "PERSONAL"},
                ],
            }
            envelope = {
                "candidates": [{"content": {"parts": [{"text": json.dumps(personal_payload)}]}}],
                "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 50, "totalTokenCount": 550},
            }
            return httpx.Response(200, json=envelope, request=request)
        else:
            body_call_count += 1
            if body_call_count == 1:
                # First Body pass returns empty body (silent omission)
                empty_body_payload = {
                    "document_archetype": "standard_cv",
                    "summary": {"value": "Experienced engineer", "source_block_ids": ["b3"]},
                    "skills": [],
                    "experience": [],
                    "education": [],
                    "projects": [],
                    "certifications": [],
                    "languages": [],
                    "achievements": [],
                }
                envelope = {
                    "candidates": [{"content": {"parts": [{"text": json.dumps(empty_body_payload)}]}}],
                    "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 40, "totalTokenCount": 1040},
                }
                return httpx.Response(200, json=envelope, request=request)
            else:
                # Second Body pass (recovery attempt) returns populated body
                assert "IMPORTANT RECOVERY INSTRUCTION" in prompt_text
                populated_body_payload = {
                    "document_archetype": "standard_cv",
                    "experience": [
                        {
                            "company": {"value": "Acme Corp", "source_block_ids": ["b3"]},
                            "designation": {"value": "Senior Engineer", "source_block_ids": ["b3"]},
                            "source_block_ids": ["b3"],
                        }
                    ],
                    "education": [
                        {
                            "institution": {"value": "MIT", "source_block_ids": ["b4"]},
                            "degree": {"value": "B.Sc. Computer Science", "source_block_ids": ["b4"]},
                            "source_block_ids": ["b4"],
                        }
                    ],
                    "certifications": [
                        {"value": "AWS Certified", "source_block_ids": ["b5"]}
                    ],
                }
                envelope = {
                    "candidates": [{"content": {"parts": [{"text": json.dumps(populated_body_payload)}]}}],
                    "usageMetadata": {"promptTokenCount": 1100, "candidatesTokenCount": 200, "totalTokenCount": 1300},
                }
                return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-api-key",
        client=client,
        two_pass=True,
    )

    sem_input = _sample_semantic_input()
    output = extractor.extract(sem_input)

    # 1. Verify requests sent: 1 personal + 2 body (1 initial + 1 recovery)
    assert len(captured_requests) == 3
    personal_prompts = [
        r for r in captured_requests
        if "PersonalSemanticOutput" in json.loads(r.content.decode("utf-8"))["contents"][0]["parts"][0]["text"]
    ]
    assert len(personal_prompts) == 1  # Personal pass was NOT repeated

    body_prompts = [
        r for r in captured_requests
        if "BodySemanticOutput" in json.loads(r.content.decode("utf-8"))["contents"][0]["parts"][0]["text"]
    ]
    assert len(body_prompts) == 2  # Exactly 2 body requests

    # 2. Verify recovered output contents
    assert output.personal.name.value == "Jane Doe"
    assert len(output.experience) == 1
    assert output.experience[0].company.value == "Acme Corp"
    assert len(output.education) == 1
    assert output.education[0].institution.value == "MIT"
    assert len(output.certifications) == 1

    # 3. Verify telemetry metadata
    meta = extractor.last_usage_metadata
    assert meta["two_pass"] is True
    assert meta["body_recovery_invoked"] is True
    assert meta["body_recovery_reason"] == "suspicious_empty_body"
    assert meta["body_recovery_attempts"] == 1
    # Aggregated token counts: 500 (personal) + 1000 (body1) + 1100 (body2) = 2600 prompt tokens
    assert meta["prompt_tokens"] == 2600
    # 50 (personal) + 40 (body1) + 200 (body2) = 290 output tokens
    assert meta["output_tokens"] == 290
    assert meta["total_tokens"] == 2890


def test_gemini_two_pass_body_recovery_bounded_when_second_remains_empty():
    """Verify Pass 2 recovery is strictly bounded to 1 attempt when second attempt also returns empty collections."""
    captured_requests: list[httpx.Request] = []
    body_call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal body_call_count
        captured_requests.append(request)
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]

        if "PersonalSemanticOutput" in prompt_text:
            personal_payload = {
                "document_archetype": "standard_cv",
                "personal": {"name": {"value": "Jane Doe", "source_block_ids": ["b1"]}},
            }
            envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(personal_payload)}]}}]}
            return httpx.Response(200, json=envelope, request=request)
        else:
            body_call_count += 1
            empty_body_payload = {
                "document_archetype": "standard_cv",
                "skills": [],
                "experience": [],
                "education": [],
                "projects": [],
                "certifications": [],
                "languages": [],
                "achievements": [],
            }
            envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(empty_body_payload)}]}}]}
            return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-api-key",
        client=client,
        two_pass=True,
    )

    sem_input = _sample_semantic_input()
    output = extractor.extract(sem_input)

    # Verify exactly 1 personal + 2 body requests (no infinite/unbounded retries)
    assert len(captured_requests) == 3
    assert body_call_count == 2
    assert output.personal.name.value == "Jane Doe"
    assert len(output.experience) == 0
    assert len(output.education) == 0

    meta = extractor.last_usage_metadata
    assert meta["body_recovery_invoked"] is True
    assert meta["body_recovery_attempts"] == 1


def test_gemini_two_pass_body_recovery_independent_of_http_retries():
    """Verify HTTP retries (e.g. 429) on Pass 2 remain independent of semantic body recovery."""
    captured_requests: list[httpx.Request] = []
    body_call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal body_call_count
        captured_requests.append(request)
        req_body = json.loads(request.content.decode("utf-8"))
        prompt_text = req_body["contents"][0]["parts"][0]["text"]

        if "PersonalSemanticOutput" in prompt_text:
            personal_payload = {
                "document_archetype": "standard_cv",
                "personal": {"name": {"value": "Jane Doe", "source_block_ids": ["b1"]}},
            }
            envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(personal_payload)}]}}]}
            return httpx.Response(200, json=envelope, request=request)
        else:
            body_call_count += 1
            if body_call_count == 1:
                # Attempt 1 of Pass 2 fails with 429 rate limit
                return httpx.Response(429, headers={"retry-after": "0"}, text="Rate limit", request=request)
            elif body_call_count == 2:
                # Attempt 2 (HTTP retry) succeeds with empty body
                empty_body_payload = {
                    "document_archetype": "standard_cv",
                    "skills": [],
                    "experience": [],
                    "education": [],
                    "projects": [],
                    "certifications": [],
                    "languages": [],
                    "achievements": [],
                }
                envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(empty_body_payload)}]}}]}
                return httpx.Response(200, json=envelope, request=request)
            else:
                # Semantic recovery pass succeeds with populated body
                populated_body_payload = {
                    "document_archetype": "standard_cv",
                    "experience": [
                        {
                            "company": {"value": "Acme Corp", "source_block_ids": ["b3"]},
                            "source_block_ids": ["b3"],
                        }
                    ],
                }
                envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(populated_body_payload)}]}}]}
                return httpx.Response(200, json=envelope, request=request)

    sleep_calls = []
    client = httpx.Client(transport=httpx.MockTransport(mock_transport))
    extractor = GeminiSemanticExtractor(
        api_key="test-api-key",
        client=client,
        two_pass=True,
        max_retries=2,
        sleep_fn=sleep_calls.append,
    )

    sem_input = _sample_semantic_input()
    output = extractor.extract(sem_input)

    assert len(output.experience) == 1
    assert extractor.last_usage_metadata["retry_count"] == 1  # 1 HTTP retry on initial body request
    assert extractor.last_usage_metadata["body_recovery_invoked"] is True
    assert extractor.last_usage_metadata["body_recovery_attempts"] == 1
