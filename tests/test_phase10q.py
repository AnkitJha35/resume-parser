"""Phase 10Q: Tests for Generalized Section-Aware Extraction and Deterministic Current Sanitization.

Validates:
1. Part A: should_use_section_aware_body_extraction() routing matrix:
   - ACADEMIC_CV with >= 1 supported section -> True
   - ACADEMIC_CV with 0 supported sections -> False
   - STANDARD_CV with >= 2 supported sections -> True
   - STANDARD_CV with 1 supported section -> False
   - STANDARD_CV with 0 supported sections -> False
   - MARITIME_TABULAR, MARITIME_CV, STRUCTURED_FORM, UNKNOWN -> False
   - Real scanned_clinical_specialist.pdf -> True (4 supported sections)
   - Real long_academic_tenured_professor_cv.pdf -> True (3 supported sections)
   - Gemini extractor sectioned routing for STANDARD_CV with >= 2 supported sections
   - Gemini extractor monolithic routing for STANDARD_CV with 1 supported section

2. Part B: sanitize_grounded_current_status() deterministic invariants:
   - Preserves grounded current=True for all ACCEPTED_CURRENT_MARKERS ('present', 'current', 'ongoing', 'till date', 'now')
   - Normalizes ungrounded current=True (e.g., '2021 - 2026') to None
   - Preserves explicit current=False
   - Preserves explicit current=None
   - Multi-block source text support
   - Strict provenance isolation: un-cited blocks with 'Present' do not justify current=True
   - Non-current fields (dates, designations, companies, descriptions) are unmodified
   - Idempotence: sanitize(sanitize(x)) == sanitize(x)
   - Works for both SemanticOutput and BodySemanticOutput

3. Part C: End-to-end flow and validation trust boundary:
   - validate_semantic_output() still rejects ungrounded current=True directly (validator intact!)
   - sanitize_grounded_current_status() + validate_semantic_output() produces 0 violations
   - parse_document_semantically() successfully sanitizes ungrounded current=True from mock extractor
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    ACCEPTED_CURRENT_MARKERS,
    BlockClassification,
    BodySemanticOutput,
    DocumentArchetype,
    GroundedBool,
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
    partition_semantic_input_into_sections,
    sanitize_grounded_current_status,
    should_use_section_aware_body_extraction,
    validate_semantic_output,
)
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.semantic_extractor import MockSemanticExtractor
from app.pipeline.semantic_pipeline import parse_document_semantically
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor


# =====================================================================
# Helpers
# =====================================================================

def _block(
    block_id: str,
    text: str,
    suggested_role: str = "UNKNOWN",
    region_kind: str = "physical_region",
    page: int = 1,
    reading_order: int = 0,
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        text=text,
        page=page,
        bbox=[0.0, float(reading_order * 20), 500.0, float(reading_order * 20 + 15)],
        region_id=f"page-{page}-region-0",
        region_kind=region_kind,
        reading_order=reading_order,
        suggested_role=suggested_role,
    )


def _input_with_archetype(
    archetype: DocumentArchetype,
    blocks: list[SemanticBlockInput],
) -> SemanticInput:
    return SemanticInput(
        document_id="doc-test",
        page_count=1,
        archetype=archetype,
        blocks=blocks,
    )


# =====================================================================
# Part A: Routing Tests for should_use_section_aware_body_extraction
# =====================================================================

def test_academic_cv_with_supported_sections_uses_section_aware():
    """ACADEMIC_CV with >= 1 supported section routes to section-aware extraction."""
    blocks = [
        _block("h0", "Dr. Jane Doe", suggested_role="HEADER", region_kind="header"),
        _block("b1", "ACADEMIC APPOINTMENTS", suggested_role="SECTION_HEADING"),
        _block("b2", "Assistant Professor", suggested_role="ENTRY_TITLE"),
        _block("b3", "MIT", suggested_role="ORGANIZATION"),
        _block("b4", "2020 - Present", suggested_role="DATE"),
    ]
    inp = _input_with_archetype(DocumentArchetype.ACADEMIC_CV, blocks)
    assert should_use_section_aware_body_extraction(inp) is True


def test_academic_cv_with_only_unsupported_sections_falls_back():
    """ACADEMIC_CV where all sections are unsupported returns False."""
    blocks = [
        _block("h0", "Dr. Jane Doe", suggested_role="HEADER", region_kind="header"),
        _block("b1", "PEER-REVIEWED PUBLICATIONS", suggested_role="SECTION_HEADING"),
        _block("b2", "1. Doe, J. et al. Nature 2021.", suggested_role="DESCRIPTION"),
        _block("b3", "2. Doe, J. et al. Science 2023.", suggested_role="DESCRIPTION"),
    ]
    inp = _input_with_archetype(DocumentArchetype.ACADEMIC_CV, blocks)
    assert should_use_section_aware_body_extraction(inp) is False


def test_standard_cv_with_two_supported_sections_uses_section_aware():
    """STANDARD_CV with >= 2 supported sections routes to section-aware extraction."""
    blocks = [
        _block("h0", "John Smith", suggested_role="HEADER", region_kind="header"),
        _block("b1", "EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b2", "Software Engineer", suggested_role="ENTRY_TITLE"),
        _block("b3", "Google", suggested_role="ORGANIZATION"),
        _block("b4", "2019 - Present", suggested_role="DATE"),
        _block("b5", "EDUCATION", suggested_role="SECTION_HEADING"),
        _block("b6", "B.S. Computer Science", suggested_role="DEGREE"),
        _block("b7", "Stanford University", suggested_role="INSTITUTION"),
    ]
    inp = _input_with_archetype(DocumentArchetype.STANDARD_CV, blocks)
    assert should_use_section_aware_body_extraction(inp) is True


def test_standard_cv_with_one_supported_section_falls_back_to_monolithic():
    """STANDARD_CV with only 1 supported section returns False (monolithic)."""
    blocks = [
        _block("h0", "John Smith", suggested_role="HEADER", region_kind="header"),
        _block("b1", "EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b2", "Software Engineer", suggested_role="ENTRY_TITLE"),
        _block("b3", "Google", suggested_role="ORGANIZATION"),
        _block("b4", "2019 - Present", suggested_role="DATE"),
    ]
    inp = _input_with_archetype(DocumentArchetype.STANDARD_CV, blocks)
    assert should_use_section_aware_body_extraction(inp) is False


def test_standard_cv_with_one_supported_and_one_unsupported_section_falls_back():
    """STANDARD_CV with 1 supported section and 1 unsupported section returns False (< 2 supported)."""
    blocks = [
        _block("h0", "John Smith", suggested_role="HEADER", region_kind="header"),
        _block("b1", "EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b2", "Software Engineer", suggested_role="ENTRY_TITLE"),
        _block("b3", "Google", suggested_role="ORGANIZATION"),
        _block("b4", "2019 - Present", suggested_role="DATE"),
        _block("b5", "REFERENCES", suggested_role="SECTION_HEADING"),
        _block("b6", "Available upon request", suggested_role="DESCRIPTION"),
    ]
    inp = _input_with_archetype(DocumentArchetype.STANDARD_CV, blocks)
    sections = partition_semantic_input_into_sections(inp)
    supported = [s for s in sections if s.canonical_target != "unsupported"]
    assert len(supported) == 1
    assert should_use_section_aware_body_extraction(inp) is False


def test_other_archetypes_always_monolithic():
    """MARITIME_TABULAR, MARITIME_CV, STRUCTURED_FORM, and UNKNOWN never use section-aware extraction."""
    blocks = [
        _block("h0", "Capt. Hook", suggested_role="HEADER", region_kind="header"),
        _block("b1", "EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b2", "Master", suggested_role="ENTRY_TITLE"),
        _block("b3", "Maersk", suggested_role="ORGANIZATION"),
        _block("b4", "2015 - 2020", suggested_role="DATE"),
        _block("b5", "EDUCATION", suggested_role="SECTION_HEADING"),
        _block("b6", "Nautical Science", suggested_role="DEGREE"),
        _block("b7", "Maritime Academy", suggested_role="INSTITUTION"),
    ]
    for arch in (
        DocumentArchetype.MARITIME_TABULAR,
        DocumentArchetype.MARITIME_CV,
        DocumentArchetype.STRUCTURED_FORM,
        DocumentArchetype.UNKNOWN,
    ):
        inp = _input_with_archetype(arch, blocks)
        assert should_use_section_aware_body_extraction(inp) is False, f"Failed for {arch}"


def test_real_clinical_specialist_fixture_routes_to_section_aware():
    """Real scanned_clinical_specialist.pdf has 4 supported sections and routes to section-aware."""
    pdf_path = Path("tests/fixtures/generalization/scanned_clinical_specialist.pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(
        layout_doc,
        document_id=pdf_path.name,
        archetype=DocumentArchetype.STANDARD_CV,
    )
    assert sinput.archetype == DocumentArchetype.STANDARD_CV
    sections = partition_semantic_input_into_sections(sinput)
    supported = [s for s in sections if s.canonical_target != "unsupported"]
    assert len(supported) == 4
    assert should_use_section_aware_body_extraction(sinput) is True


def test_real_academic_cv_fixture_routes_to_section_aware():
    """Real long_academic_tenured_professor_cv.pdf has 3 supported sections and routes to section-aware."""
    pdf_path = Path("tests/fixtures/generalization/long_academic_tenured_professor_cv.pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(
        layout_doc,
        document_id=pdf_path.name,
        archetype=DocumentArchetype.ACADEMIC_CV,
    )
    assert sinput.archetype == DocumentArchetype.ACADEMIC_CV
    sections = partition_semantic_input_into_sections(sinput)
    supported = [s for s in sections if s.canonical_target != "unsupported"]
    assert len(supported) == 3
    assert should_use_section_aware_body_extraction(sinput) is True


def test_gemini_extractor_routes_standard_cv_to_section_aware(monkeypatch):
    """GeminiSemanticExtractor routes STANDARD_CV with >= 2 sections through section-aware extraction."""
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        two_pass=True,
    )

    blocks = [
        _block("h0", "Jane Doe", suggested_role="HEADER", region_kind="header"),
        _block("b1", "EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b2", "Staff Engineer", suggested_role="ENTRY_TITLE"),
        _block("b3", "Stripe", suggested_role="ORGANIZATION"),
        _block("b4", "2021 - Present", suggested_role="DATE"),
        _block("b5", "EDUCATION", suggested_role="SECTION_HEADING"),
        _block("b6", "B.S. in Computer Science", suggested_role="DEGREE"),
        _block("b7", "MIT", suggested_role="INSTITUTION"),
    ]
    inp = _input_with_archetype(DocumentArchetype.STANDARD_CV, blocks)

    executed_passes: list[str] = []

    def mock_execute(prompt, response_schema, api_key, model, base_url, timeout, max_retries, pass_name):
        executed_passes.append(pass_name)
        if pass_name == "personal":
            data = {"personal": {"name": {"value": "Jane Doe", "source_block_ids": ["h0"]}}}
        elif "experience" in pass_name:
            data = {
                "experience": [
                    {
                        "company": {"value": "Stripe", "source_block_ids": ["b3"]},
                        "designation": {"value": "Staff Engineer", "source_block_ids": ["b2"]},
                        "startDate": {"value": "2021", "source_block_ids": ["b4"]},
                        "current": {"value": True, "source_block_ids": ["b4"]},
                        "source_block_ids": ["b2", "b3", "b4"],
                    }
                ]
            }
        elif "education" in pass_name:
            data = {
                "education": [
                    {
                        "institution": {"value": "MIT", "source_block_ids": ["b7"]},
                        "degree": {"value": "B.S. in Computer Science", "source_block_ids": ["b6"]},
                        "source_block_ids": ["b6", "b7"],
                    }
                ]
            }
        else:
            data = {}
        usage = {"prompt_tokens": 50, "output_tokens": 50, "total_tokens": 100}
        return json.dumps(data), usage, 0

    monkeypatch.setattr(extractor, "_execute_prompt_request", mock_execute)

    result = extractor.extract(inp)

    # Verify sectioned passes were executed, NOT monolithic "body"
    assert "personal" in executed_passes
    assert any("body_sec_experience" in p for p in executed_passes)
    assert any("body_sec_education" in p for p in executed_passes)
    assert "body" not in executed_passes

    # Verify merged results
    assert len(result.experience) == 1
    assert result.experience[0].company.value == "Stripe"
    assert result.experience[0].current.value is True
    assert len(result.education) == 1
    assert result.education[0].institution.value == "MIT"


# =====================================================================
# Part B: Deterministic Boolean Current Sanitization Tests
# =====================================================================

def test_sanitize_preserves_valid_current_markers():
    """current=True is preserved when source evidence contains accepted markers."""
    for marker in ("Present", "Current", "Currently employed", "Ongoing", "Till Date", "Now"):
        sem_input = SemanticInput(
            document_id="doc-1",
            page_count=1,
            blocks=[
                _block("b1", f"2019 - {marker}"),
                _block("b2", f"2020 - {marker.lower()}"),
            ],
        )
        output = SemanticOutput(
            experience=[
                GroundedExperienceItem(
                    current=GroundedBool(value=True, source_block_ids=["b1"]),
                    source_block_ids=["b1"],
                )
            ],
            projects=[
                GroundedProjectItem(
                    current=GroundedBool(value=True, source_block_ids=["b2"]),
                    source_block_ids=["b2"],
                )
            ],
        )
        sanitized = sanitize_grounded_current_status(output, sem_input)
        assert sanitized.experience[0].current is not None, f"Failed for {marker}"
        assert sanitized.experience[0].current.value is True
        assert sanitized.projects[0].current is not None, f"Failed for {marker}"
        assert sanitized.projects[0].current.value is True


def test_sanitize_normalizes_ungrounded_current_to_none():
    """current=True on future or bare date ranges without accepted markers is normalized to None."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[
            _block("b1", "2021 - 2026"),
            _block("b2", "NIH Grant R01: Neural Decoding ($2.4M, PI, 2021 - 2026)"),
        ],
    )
    output = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Lead Investigator", source_block_ids=["b1"]),
                current=GroundedBool(value=True, source_block_ids=["b1"]),
                source_block_ids=["b1"],
            )
        ],
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="NIH Grant R01", source_block_ids=["b2"]),
                current=GroundedBool(value=True, source_block_ids=["b2"]),
                source_block_ids=["b2"],
            )
        ],
    )
    sanitized = sanitize_grounded_current_status(output, sem_input)

    # current should become None, NOT False
    assert sanitized.experience[0].current is None
    assert sanitized.projects[0].current is None

    # Other grounded fields must remain intact
    assert sanitized.experience[0].designation.value == "Lead Investigator"
    assert sanitized.projects[0].name.value == "NIH Grant R01"


def test_sanitize_preserves_explicit_current_false():
    """current=False is preserved and not altered."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[_block("b1", "2015 - 2020")],
    )
    output = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                current=GroundedBool(value=False, source_block_ids=["b1"]),
                source_block_ids=["b1"],
            )
        ],
        projects=[
            GroundedProjectItem(
                current=GroundedBool(value=False, source_block_ids=["b1"]),
                source_block_ids=["b1"],
            )
        ],
    )
    sanitized = sanitize_grounded_current_status(output, sem_input)
    assert sanitized.experience[0].current is not None
    assert sanitized.experience[0].current.value is False
    assert sanitized.projects[0].current is not None
    assert sanitized.projects[0].current.value is False


def test_sanitize_preserves_current_none():
    """current=None remains None without error."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[_block("b1", "2018 - 2022")],
    )
    output = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                current=None,
                source_block_ids=["b1"],
            )
        ],
        projects=[
            GroundedProjectItem(
                current=None,
                source_block_ids=["b1"],
            )
        ],
    )
    sanitized = sanitize_grounded_current_status(output, sem_input)
    assert sanitized.experience[0].current is None
    assert sanitized.projects[0].current is None


def test_sanitize_multi_block_evidence():
    """Sanitizer concatenates all cited source_block_ids to check for current marker."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[
            _block("b1", "2021 -"),
            _block("b2", "Present"),
            _block("b3", "2021 -"),
            _block("b4", "2026)"),
        ],
    )
    output = SemanticOutput(
        experience=[
            # Cites b1 and b2 -> contains 'Present' -> kept
            GroundedExperienceItem(
                current=GroundedBool(value=True, source_block_ids=["b1", "b2"]),
                source_block_ids=["b1", "b2"],
            )
        ],
        projects=[
            # Cites b3 and b4 -> '2021 - 2026)' -> no marker -> sanitized to None
            GroundedProjectItem(
                current=GroundedBool(value=True, source_block_ids=["b3", "b4"]),
                source_block_ids=["b3", "b4"],
            )
        ],
    )
    sanitized = sanitize_grounded_current_status(output, sem_input)
    assert sanitized.experience[0].current is not None
    assert sanitized.experience[0].current.value is True
    assert sanitized.projects[0].current is None


def test_sanitize_provenance_isolation():
    """An un-cited block containing 'Present' cannot validate current=True for another block."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[
            _block("b1", "2021 - 2026"),
            _block("b2", "2019 - Present"),  # has Present, but NOT cited
        ],
    )
    output = SemanticOutput(
        projects=[
            GroundedProjectItem(
                current=GroundedBool(value=True, source_block_ids=["b1"]),  # cites only b1
                source_block_ids=["b1"],
            )
        ]
    )
    sanitized = sanitize_grounded_current_status(output, sem_input)
    assert sanitized.projects[0].current is None


def test_sanitize_works_on_body_semantic_output():
    """Sanitizer operates identically on BodySemanticOutput."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[
            _block("b1", "2021 - 2026"),
            _block("b2", "2019 - Present"),
        ],
    )
    body_output = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                current=GroundedBool(value=True, source_block_ids=["b2"]),
                source_block_ids=["b2"],
            )
        ],
        projects=[
            GroundedProjectItem(
                current=GroundedBool(value=True, source_block_ids=["b1"]),
                source_block_ids=["b1"],
            )
        ],
    )
    sanitized = sanitize_grounded_current_status(body_output, sem_input)
    assert sanitized.experience[0].current is not None
    assert sanitized.experience[0].current.value is True
    assert sanitized.projects[0].current is None


def test_sanitize_idempotence():
    """Calling sanitize_grounded_current_status multiple times is idempotent."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[
            _block("b1", "2021 - 2026"),
            _block("b2", "2019 - Present"),
        ],
    )
    output = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                current=GroundedBool(value=True, source_block_ids=["b2"]),
                source_block_ids=["b2"],
            )
        ],
        projects=[
            GroundedProjectItem(
                current=GroundedBool(value=True, source_block_ids=["b1"]),
                source_block_ids=["b1"],
            )
        ],
    )
    first_pass = sanitize_grounded_current_status(output, sem_input)
    second_pass = sanitize_grounded_current_status(first_pass, sem_input)
    assert second_pass.experience[0].current.value is True
    assert second_pass.projects[0].current is None


# =====================================================================
# Part C: End-to-End Flow & Validation Invariant Preservation
# =====================================================================

def test_validation_invariant_still_rejects_unsanitized_current():
    """Deterministic validator boundary is NOT weakened: direct unsanitized input is still rejected."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[
            _block("b1", "Professor Benjamin Thorne", suggested_role="HEADER", region_kind="header"),
            _block("b2", "? NIH R01: 'Neural Decoding' (2021 - 2026)"),
        ],
    )
    unsanitized_output = SemanticOutput(
        personal=GroundedPersonal(name=GroundedString(value="Professor Benjamin Thorne", source_block_ids=["b1"])),
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="NIH R01", source_block_ids=["b2"]),
                current=GroundedBool(value=True, source_block_ids=["b2"]),
                source_block_ids=["b2"],
            )
        ],
    )
    # Direct validation without sanitization MUST fail
    violations = validate_semantic_output(unsanitized_output, sem_input)
    assert len(violations) == 1
    assert "UNSUPPORTED_CURRENT_STATUS in projects[0].current" in violations[0]


def test_sanitized_output_passes_validation():
    """Sanitized output eliminates UNSUPPORTED_CURRENT_STATUS violation cleanly."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[
            _block("b1", "Professor Benjamin Thorne", suggested_role="HEADER", region_kind="header"),
            _block("b2", "? NIH R01: 'Neural Decoding' (2021 - 2026)"),
        ],
    )
    output = SemanticOutput(
        personal=GroundedPersonal(name=GroundedString(value="Professor Benjamin Thorne", source_block_ids=["b1"])),
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="NIH R01", source_block_ids=["b2"]),
                current=GroundedBool(value=True, source_block_ids=["b2"]),
                source_block_ids=["b2"],
            )
        ],
    )
    sanitized = sanitize_grounded_current_status(output, sem_input)
    violations = validate_semantic_output(sanitized, sem_input)
    assert violations == []


def test_parse_document_semantically_end_to_end_with_sanitization():
    """parse_document_semantically applies sanitization transparently before validation."""
    from app.pipeline.stages.text_extraction import TextBlock

    doc_blocks = [
        TextBlock(
            text="Alice Smith",
            page_number=1,
            x0=0.0,
            y0=0.0,
            x1=200.0,
            y1=20.0,
        ),
        TextBlock(
            text="Research Grant (2022 - 2025)",
            page_number=1,
            x0=0.0,
            y0=30.0,
            x1=400.0,
            y1=50.0,
        ),
    ]
    doc = document_from_text_blocks(doc_blocks)
    reconstructed = reconstruct_document(doc)
    layout_doc = interpret_layout(reconstructed)

    # Mock extractor returns ungrounded current=True for the project
    class FlawedMockExtractor(MockSemanticExtractor):
        def extract(self, input_data: SemanticInput) -> SemanticOutput:
            b0_id = input_data.blocks[0].block_id
            b1_id = input_data.blocks[1].block_id
            return SemanticOutput(
                personal=GroundedPersonal(
                    name=GroundedString(value="Alice Smith", source_block_ids=[b0_id])
                ),
                projects=[
                    GroundedProjectItem(
                        name=GroundedString(value="Research Grant", source_block_ids=[b1_id]),
                        current=GroundedBool(value=True, source_block_ids=[b1_id]),  # ungrounded!
                        source_block_ids=[b1_id],
                    )
                ],
            )

    extractor = FlawedMockExtractor()
    resume = parse_document_semantically(layout_doc, extractor, document_id="doc-test-sanitization")

    # Should succeed without validation error, with current normalized to None
    assert resume.personal.name == "Alice Smith"
    assert len(resume.projects) == 1
    assert resume.projects[0].name == "Research Grant"
    assert resume.projects[0].current is None
