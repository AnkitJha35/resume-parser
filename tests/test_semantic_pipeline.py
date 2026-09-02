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
from app.extractors.semantic_extractor import (
    MockSemanticExtractor,
    SemanticExtractor,
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
