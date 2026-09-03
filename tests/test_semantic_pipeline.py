"""Phase 8-2B: Tests for provider-independent semantic extraction interface, mock extractor, and service."""

from __future__ import annotations

import ast
from pathlib import Path
import pytest

from app.domain.resume import Resume
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedPersonal,
    GroundedString,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    build_semantic_input,
    semantic_output_to_resume,
    validate_semantic_output,
)
from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.semantic_extractor import (
    MockSemanticExtractor,
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticExtractor,
    SemanticRateLimitError,
    SemanticResponseError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
    SemanticValidationError,
)
from app.pipeline.parser import ResumeParser
from app.pipeline.semantic_pipeline import parse_document_semantically
from tests.test_semantic_llm_contract import _make_test_document


def test_mock_extractor_returns_semantic_output():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-mock")
    extractor = MockSemanticExtractor()

    output = extractor.extract(sem_input)
    assert isinstance(output, SemanticOutput)
    assert output.document_archetype == DocumentArchetype.STANDARD_CV
    assert output.personal.name is not None
    assert output.personal.name.value == "John Doe"
    assert output.personal.email is not None
    assert output.personal.email.value == "john.doe@example.com"
    assert len(output.experience) == 1
    assert output.experience[0].company is not None
    assert output.experience[0].company.value == "Acme Corporation"


def test_mock_extractor_provenance_and_validation():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-mock")
    extractor = MockSemanticExtractor()

    output = extractor.extract(sem_input)
    known_block_ids = {b.block_id for b in sem_input.blocks}

    # Verify that every returned grounded field has valid source_block_ids
    assert set(output.personal.name.source_block_ids).issubset(known_block_ids)
    assert set(output.personal.email.source_block_ids).issubset(known_block_ids)
    assert set(output.personal.phone.source_block_ids).issubset(known_block_ids)
    assert set(output.experience[0].source_block_ids).issubset(known_block_ids)

    # Values pass validate_semantic_output()
    violations = validate_semantic_output(output, sem_input)
    assert violations == []

    # Valid SemanticOutput projects through semantic_output_to_resume()
    resume = semantic_output_to_resume(output)
    assert isinstance(resume, Resume)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"
    assert len(resume.experience) == 1
    assert resume.experience[0].company == "Acme Corporation"


def test_deliberately_minimal_semantic_input():
    """Requirement 5: Test minimal SemanticInput with explicit structural evidence.

    Demonstrates that MockSemanticExtractor acts purely as an extraction seam test
    double that maps explicit structural evidence without heuristic parsing.
    """
    minimal_blocks = [
        SemanticBlockInput(
            block_id="b_name",
            text="Alice Smith",
            page=1,
            bbox=[50.0, 50.0, 200.0, 70.0],
            region_id="r_head",
            region_kind="header",
            reading_order=1,
            suggested_role="HEADER",
        ),
        SemanticBlockInput(
            block_id="b_email",
            text="alice@example.com",
            page=1,
            bbox=[50.0, 75.0, 200.0, 90.0],
            region_id="r_head",
            region_kind="header",
            reading_order=2,
            suggested_role="CONTACT",
        ),
        SemanticBlockInput(
            block_id="b_org",
            text="Widget Global Corp",
            page=1,
            bbox=[50.0, 150.0, 200.0, 170.0],
            region_id="r_body",
            region_kind="physical_region",
            reading_order=3,
            suggested_role="ORGANIZATION",
        ),
        SemanticBlockInput(
            block_id="b_role",
            text="Systems Architect",
            page=1,
            bbox=[50.0, 175.0, 200.0, 190.0],
            region_id="r_body",
            region_kind="physical_region",
            reading_order=4,
            suggested_role="ENTRY_TITLE",
        ),
    ]
    sem_input = SemanticInput(
        document_id="minimal-test-doc",
        page_count=1,
        pages=[],
        blocks=minimal_blocks,
    )

    extractor = MockSemanticExtractor()
    output = extractor.extract(sem_input)

    assert output.personal.name is not None
    assert output.personal.name.value == "Alice Smith"
    assert output.personal.name.source_block_ids == ["b_name"]

    assert output.personal.email is not None
    assert output.personal.email.value == "alice@example.com"
    assert output.personal.email.source_block_ids == ["b_email"]

    assert len(output.experience) == 1
    assert output.experience[0].company is not None
    assert output.experience[0].company.value == "Widget Global Corp"
    assert output.experience[0].designation is not None
    assert output.experience[0].designation.value == "Systems Architect"
    assert "b_org" in output.experience[0].source_block_ids
    assert "b_role" in output.experience[0].source_block_ids

    violations = validate_semantic_output(output, sem_input)
    assert violations == []

    resume = semantic_output_to_resume(output)
    assert resume.personal.name == "Alice Smith"
    assert resume.experience[0].company == "Widget Global Corp"


def test_invalid_output_rejected_before_projection():
    class BrokenExtractor:
        def extract(self, input_data: SemanticInput) -> SemanticOutput:
            return SemanticOutput(
                personal=GroundedPersonal(
                    name=GroundedString(value="APPLICATION FORM", source_block_ids=["b_p1_0"]),
                )
            )

    doc = _make_test_document()
    with pytest.raises(SemanticValidationError) as exc_info:
        parse_document_semantically(doc, BrokenExtractor())

    assert "DOCUMENT_TITLE_AS_NAME" in str(exc_info.value)


def test_extractor_interface_replaceable():
    class CustomDeterministicExtractor:
        def extract(self, input_data: SemanticInput) -> SemanticOutput:
            return SemanticOutput(
                document_archetype=DocumentArchetype.STANDARD_CV,
                personal=GroundedPersonal(
                    name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
                    email=GroundedString(value="john.doe@example.com", source_block_ids=["b_p1_1"]),
                ),
            )

    doc = _make_test_document()
    # Verify interface can be implemented by an alternative extractor without modifying service
    custom_extractor: SemanticExtractor = CustomDeterministicExtractor()
    resume = parse_document_semantically(doc, custom_extractor)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"


def test_no_provider_dependency():
    # Verify that neither MockSemanticExtractor nor parse_document_semantically import external LLM SDKs
    for path in (
        Path("app/extractors/semantic_extractor.py"),
        Path("app/pipeline/semantic_pipeline.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)

        disallowed = ("openai", "google.genai", "anthropic", "ollama", "requests", "httpx")
        for dis in disallowed:
            assert not any(name.startswith(dis) for name in imported_names), f"{path} imports {dis}"


def test_complete_end_to_end_pipeline_seam():
    # End-to-end test exercising Document -> SemanticInput -> MockSemanticExtractor -> validate -> Resume
    doc = _make_test_document()
    extractor = MockSemanticExtractor()

    resume = parse_document_semantically(doc, extractor, document_id="e2e-test-1")
    assert isinstance(resume, Resume)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"
    assert resume.experience[0].company == "Acme Corporation"
    assert resume.experience[0].designation == "Senior Software Engineer"


def test_existing_parser_remains_unchanged():
    # Existing ResumeParser behavior is untouched
    parser = ResumeParser()
    assert hasattr(parser, "parse")
    assert hasattr(parser, "parse_with_layout_pipeline")


# =====================================================================
# Checkpoint 6: Fallback Integration Through the Real Semantic Pipeline
# =====================================================================

class _PipelineFakeExtractor:
    """Controllable test double tracking call counts and received input."""

    def __init__(
        self,
        provider_name: str,
        result: SemanticOutput | None = None,
        exception: Exception | None = None,
        usage_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.result = result
        self.exception = exception
        self.call_count = 0
        self.received_input: SemanticInput | None = None
        self.last_usage_metadata = usage_metadata or {
            "provider": provider_name,
            "model": f"{provider_name}-model",
            "prompt_tokens": 120,
            "output_tokens": 60,
            "total_tokens": 180,
            "latency_ms": 300.0,
            "status": "success" if exception is None else "failure",
        }

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        self.call_count += 1
        self.received_input = input_data
        if self.exception:
            raise self.exception
        return self.result or SemanticOutput(
            document_archetype=DocumentArchetype.STANDARD_CV,
            personal=GroundedPersonal(
                name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
                email=GroundedString(value="john.doe@example.com", source_block_ids=["b_p1_1"]),
            ),
        )


def test_fallback_pipeline_test_a_primary_success():
    """Test A: Primary succeeds -> fallback not called, exact Resume projected, provenance validated."""
    doc = _make_test_document()
    primary_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
            email=GroundedString(value="john.doe@example.com", source_block_ids=["b_p1_1"]),
        ),
    )
    fallback_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="Fallback Name", source_block_ids=["b_p1_0"]),
        ),
    )
    primary = _PipelineFakeExtractor("gemini", result=primary_output)
    fallback = _PipelineFakeExtractor("ollama", result=fallback_output)

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    resume = parse_document_semantically(doc, composite, document_id="doc-test-a")

    assert primary.call_count == 1
    assert fallback.call_count == 0
    assert isinstance(resume, Resume)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"
    assert composite.last_usage_metadata["fallback_invoked"] is False
    assert composite.last_usage_metadata["primary_provider"] == "gemini"


@pytest.mark.parametrize(
    "eligible_exc",
    [
        SemanticRateLimitError("Rate limit 429", status_code=429),
        SemanticServerError("Server outage 503", status_code=503),
        SemanticTimeoutError("Gemini timed out after 30s"),
        SemanticTransportError("Connection reset"),
        SemanticConfigurationError("API key not configured"),
    ],
)
def test_fallback_pipeline_test_b_eligible_primary_failure_fallback_success(eligible_exc):
    """Test B: Transient primary failure -> fallback called and succeeds through provenance validation."""
    doc = _make_test_document()
    primary = _PipelineFakeExtractor("gemini", exception=eligible_exc)
    fallback_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
            email=GroundedString(value="john.doe@example.com", source_block_ids=["b_p1_1"]),
        ),
    )
    fallback = _PipelineFakeExtractor("ollama", result=fallback_output)

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    resume = parse_document_semantically(doc, composite, document_id="doc-test-b")

    assert primary.call_count == 1
    assert fallback.call_count == 1
    assert isinstance(resume, Resume)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"
    assert composite.last_usage_metadata["fallback_invoked"] is True
    assert composite.last_usage_metadata["primary_provider"] == "gemini"
    assert composite.last_usage_metadata["fallback_provider"] == "ollama"
    assert composite.last_usage_metadata["primary_error"] == type(eligible_exc).__name__


@pytest.mark.parametrize(
    "non_eligible_exc",
    [
        SemanticResponseError("Gemini response candidate blocked with finishReason=SAFETY"),
        SemanticExtractionError("Generic unclassified client error"),
    ],
)
def test_fallback_pipeline_test_c_deterministic_primary_failure_no_fallback(non_eligible_exc):
    """Test C: Deterministic primary failures (e.g. SAFETY block) must fail fast without fallback."""
    doc = _make_test_document()
    primary = _PipelineFakeExtractor("gemini", exception=non_eligible_exc)
    fallback = _PipelineFakeExtractor("ollama")

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    with pytest.raises(type(non_eligible_exc)):
        parse_document_semantically(doc, composite, document_id="doc-test-c")

    assert primary.call_count == 1
    assert fallback.call_count == 0


def test_fallback_pipeline_test_d_fallback_output_strictly_validated_against_trust_boundary():
    """Test D: Fallback output must pass provenance validation; hallucinations raise SemanticValidationError."""
    doc = _make_test_document()
    primary = _PipelineFakeExtractor("gemini", exception=SemanticServerError("HTTP 500", status_code=500))

    # Fallback proposes a hallucinated block ID not present in Document IR
    hallucinated_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["hallucinated_block_999"]),
        ),
    )
    fallback = _PipelineFakeExtractor("ollama", result=hallucinated_output)

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    with pytest.raises(SemanticValidationError) as exc_info:
        parse_document_semantically(doc, composite, document_id="doc-test-d")

    assert primary.call_count == 1
    assert fallback.call_count == 1
    assert "UNKNOWN_BLOCK_ID" in str(exc_info.value)


def test_fallback_pipeline_test_e_fallback_returns_full_multi_field_resume():
    """Test E: Fallback returns complete structured output projecting to Resume fields."""
    doc = _make_test_document()
    primary = _PipelineFakeExtractor("gemini", exception=SemanticTimeoutError("Timeout"))

    from app.domain.semantic_contract import GroundedExperienceItem

    fallback_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
            email=GroundedString(value="john.doe@example.com", source_block_ids=["b_p1_1"]),
        ),
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Acme Corporation", source_block_ids=["b_p1_4"]),
                designation=GroundedString(value="Senior Software Engineer", source_block_ids=["b_p1_5"]),
                source_block_ids=["b_p1_4", "b_p1_5"],
            )
        ],
    )
    fallback = _PipelineFakeExtractor("ollama", result=fallback_output)

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    resume = parse_document_semantically(doc, composite, document_id="doc-test-e")

    assert isinstance(resume, Resume)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"
    assert len(resume.experience) == 1
    assert resume.experience[0].company == "Acme Corporation"
    assert resume.experience[0].designation == "Senior Software Engineer"


def test_fallback_pipeline_test_f_both_providers_fail():
    """Test F: Both providers fail -> combined SemanticExtractionError with context and no Resume."""
    doc = _make_test_document()
    primary_exc = SemanticServerError("Primary server 500", status_code=500)
    fallback_exc = SemanticTimeoutError("Fallback timed out after 120s")

    primary = _PipelineFakeExtractor("gemini", exception=primary_exc)
    fallback = _PipelineFakeExtractor("ollama", exception=fallback_exc)

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)

    with pytest.raises(SemanticExtractionError) as exc_info:
        parse_document_semantically(doc, composite, document_id="doc-test-f")

    assert primary.call_count == 1
    assert fallback.call_count == 1
    assert "gemini" in str(exc_info.value)
    assert "ollama" in str(exc_info.value)
    assert exc_info.value.__cause__ is fallback_exc


def test_fallback_pipeline_test_g_semantic_input_survives_fallback_unaltered():
    """Test G: Fallback receives the exact unmodified SemanticInput without reconstruction."""
    doc = _make_test_document()
    primary = _PipelineFakeExtractor("gemini", exception=SemanticTransportError("Network error"))
    fallback = _PipelineFakeExtractor("ollama")

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    parse_document_semantically(doc, composite, document_id="doc-test-g")

    assert primary.received_input is not None
    assert fallback.received_input is not None
    # Verify exact same object identity and block structures passed to fallback
    assert primary.received_input is fallback.received_input
    assert fallback.received_input.document_id == "doc-test-g"
    assert len(fallback.received_input.blocks) == len(primary.received_input.blocks)


def test_fallback_pipeline_test_h_metadata_preserved_on_fallback_success():
    """Test H: Fallback usage metadata is accurately preserved in composite extractor."""
    doc = _make_test_document()
    primary_meta = {"provider": "gemini", "model": "gemini-2.5-flash", "status": "failure"}
    fallback_meta = {
        "provider": "ollama",
        "model": "qwen2.5-coder:7b",
        "prompt_tokens": 1500,
        "output_tokens": 300,
        "total_tokens": 1800,
        "latency_ms": 12400.0,
        "retry_count": 0,
        "status": "success",
    }
    primary = _PipelineFakeExtractor(
        "gemini",
        exception=SemanticServerError("500", status_code=500),
        usage_metadata=primary_meta,
    )
    fallback = _PipelineFakeExtractor("ollama", usage_metadata=fallback_meta)

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    parse_document_semantically(doc, composite, document_id="doc-test-h")

    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["provider"] == "ollama"
    assert meta["model"] == "qwen2.5-coder:7b"
    assert meta["prompt_tokens"] == 1500
    assert meta["output_tokens"] == 300
    assert meta["total_tokens"] == 1800
    assert meta["latency_ms"] == 12400.0
    assert meta["status"] == "success"
    assert meta["fallback_invoked"] is True
    assert meta["primary_provider"] == "gemini"
    assert meta["fallback_provider"] == "ollama"
    assert meta["primary_error"] == "SemanticServerError"
