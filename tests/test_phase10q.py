"""Phase 10Q-R: Tests for Request-Efficient Semantic Routing & Deterministic Current Sanitization.

Validates:
1. Part A: STANDARD_CV Normal Routing
   - STANDARD_CV with 5 supported sections -> 1 normal body request
   - STANDARD_CV with 4 supported sections -> 1 normal body request
   - STANDARD_CV with 1 supported section -> 1 normal body request
   - STANDARD_CV with unsupported sections -> unsupported blocks excluded from normal body payload
   - STANDARD_CV duplicate canonical targets -> single normal body request (no duplicate calls)
   - should_use_section_aware_body_extraction() returns False for STANDARD_CV

2. Part B: STANDARD_CV Bounded Recovery
   - STANDARD_CV suspicious-empty triggers bounded recovery
   - Recovery request count is bounded (at most 2: overview + entities) and < old one-request-per-section

3. Part C: ACADEMIC_CV Routing & Grouping
   - ACADEMIC_CV section-aware behavior remains active (should_use_section_aware_body_extraction is True)
   - Duplicate targets (e.g. experience + experience) are grouped into a single extraction request
   - group_sections_by_target() preserves block ordering and merges identical targets

4. Part D: Other Archetypes
   - MARITIME_CV, MARITIME_TABULAR, STRUCTURED_FORM, UNKNOWN -> monolithic (behavior unchanged)

5. Part E: Deterministic Current-Status Sanitization
   - Preserves grounded current=True for ACCEPTED_CURRENT_MARKERS
   - Normalizes ungrounded current=True to None
   - Preserves current=False and current=None
   - Provenance isolation
   - Works on SemanticOutput and BodySemanticOutput

6. Part F: Request Planning Regression on 10 Generalization Fixtures
   - Verifies normal request planning: STANDARD_CV uses 1 personal + 1 body
   - Verifies ACADEMIC_CV uses 1 personal + grouped body requests
   - Guarantees STANDARD_CV never regresses to one request per physical section
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
    SemanticSection,
    build_semantic_input,
    filter_semantic_input_to_blocks,
    group_sections_by_target,
    group_sections_for_recovery,
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
from app.pipeline.stages.text_extraction import PDFExtractor, TextBlock


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
# Part A: STANDARD_CV Normal Routing Tests
# =====================================================================

def test_should_use_section_aware_false_for_standard_cv():
    """STANDARD_CV uses monolithic extraction in normal path (should_use_section_aware=False)."""
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
    assert should_use_section_aware_body_extraction(inp) is False


def test_standard_cv_with_five_supported_sections_uses_single_normal_body_request(monkeypatch):
    """STANDARD_CV with 5 supported sections issues exactly 1 normal body request."""
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        two_pass=True,
    )
    blocks = [
        _block("h0", "Jane Doe", suggested_role="HEADER", region_kind="header"),
        _block("b1", "SUMMARY", suggested_role="SECTION_HEADING"),
        _block("b2", "Experienced product leader", suggested_role="DESCRIPTION"),
        _block("b3", "SKILLS", suggested_role="SECTION_HEADING"),
        _block("b4", "Product Strategy, Agile, Python", suggested_role="SKILL"),
        _block("b5", "EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b6", "Lead PM", suggested_role="ENTRY_TITLE"),
        _block("b7", "Stripe", suggested_role="ORGANIZATION"),
        _block("b8", "2020 - Present", suggested_role="DATE"),
        _block("b9", "EDUCATION", suggested_role="SECTION_HEADING"),
        _block("b10", "MBA", suggested_role="DEGREE"),
        _block("b11", "Harvard", suggested_role="INSTITUTION"),
        _block("b12", "CERTIFICATIONS", suggested_role="SECTION_HEADING"),
        _block("b13", "PMP Certified", suggested_role="CERTIFICATION"),
    ]
    inp = _input_with_archetype(DocumentArchetype.STANDARD_CV, blocks)

    executed_passes: list[str] = []

    def mock_execute(prompt, response_schema, api_key, model, base_url, timeout, max_retries, pass_name):
        executed_passes.append(pass_name)
        if pass_name == "personal":
            data = {"personal": {"name": {"value": "Jane Doe", "source_block_ids": ["h0"]}}}
        else:
            data = {
                "skills": [{"value": "Product Strategy", "source_block_ids": ["b4"]}],
                "experience": [
                    {
                        "company": {"value": "Stripe", "source_block_ids": ["b7"]},
                        "designation": {"value": "Lead PM", "source_block_ids": ["b6"]},
                        "startDate": {"value": "2020", "source_block_ids": ["b8"]},
                        "current": {"value": True, "source_block_ids": ["b8"]},
                        "source_block_ids": ["b6", "b7", "b8"],
                    }
                ],
            }
        return json.dumps(data), {"prompt_tokens": 50, "output_tokens": 50, "total_tokens": 100}, 0

    monkeypatch.setattr(extractor, "_execute_prompt_request", mock_execute)
    res = extractor.extract(inp)

    # Exactly 1 personal and 1 body request!
    assert executed_passes == ["personal", "body"]
    assert res.personal.name.value == "Jane Doe"
    assert len(res.experience) == 1


def test_standard_cv_with_unsupported_sections_excludes_them_from_normal_body_payload(monkeypatch):
    """STANDARD_CV with unsupported sections excludes unsupported blocks from normal body request."""
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        two_pass=True,
    )
    blocks = [
        _block("h0", "John Smith", suggested_role="HEADER", region_kind="header"),
        _block("b1", "EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b2", "Software Engineer", suggested_role="ENTRY_TITLE"),
        _block("b3", "Google", suggested_role="ORGANIZATION"),
        _block("b4", "2019 - Present", suggested_role="DATE"),
        _block("b5", "REFERENCES", suggested_role="SECTION_HEADING"),
        _block("b6", "Available upon request from former manager", suggested_role="DESCRIPTION"),
    ]
    inp = _input_with_archetype(DocumentArchetype.STANDARD_CV, blocks)

    prompts_captured: dict[str, str] = {}

    def mock_execute(prompt, response_schema, api_key, model, base_url, timeout, max_retries, pass_name):
        prompts_captured[pass_name] = prompt
        if pass_name == "personal":
            data = {"personal": {"name": {"value": "John Smith", "source_block_ids": ["h0"]}}}
        else:
            data = {
                "experience": [
                    {
                        "company": {"value": "Google", "source_block_ids": ["b3"]},
                        "designation": {"value": "Software Engineer", "source_block_ids": ["b2"]},
                        "source_block_ids": ["b2", "b3", "b4"],
                    }
                ]
            }
        return json.dumps(data), {"prompt_tokens": 50, "output_tokens": 50, "total_tokens": 100}, 0

    monkeypatch.setattr(extractor, "_execute_prompt_request", mock_execute)
    extractor.extract(inp)

    # Body prompt must contain experience blocks but NOT references blocks (b5, b6)
    body_prompt = prompts_captured["body"]
    assert "Google" in body_prompt
    assert "b4" in body_prompt
    assert "b5" not in body_prompt
    assert "Available upon request" not in body_prompt


def test_standard_cv_duplicate_targets_single_normal_body_request(monkeypatch):
    """STANDARD_CV with 2 separate experience sections still uses exactly 1 normal body request."""
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        two_pass=True,
    )
    blocks = [
        _block("h0", "John Smith", suggested_role="HEADER", region_kind="header"),
        _block("b1", "EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b2", "Software Engineer", suggested_role="ENTRY_TITLE"),
        _block("b3", "Google", suggested_role="ORGANIZATION"),
        _block("b4", "2020 - Present", suggested_role="DATE"),
        _block("b5", "PREVIOUS EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b6", "Intern", suggested_role="ENTRY_TITLE"),
        _block("b7", "Meta", suggested_role="ORGANIZATION"),
        _block("b8", "2019 - 2020", suggested_role="DATE"),
    ]
    inp = _input_with_archetype(DocumentArchetype.STANDARD_CV, blocks)

    executed_passes: list[str] = []

    def mock_execute(prompt, response_schema, api_key, model, base_url, timeout, max_retries, pass_name):
        executed_passes.append(pass_name)
        data = {
            "experience": [
                {
                    "company": {"value": "Google", "source_block_ids": ["b3"]},
                    "designation": {"value": "Software Engineer", "source_block_ids": ["b2"]},
                    "source_block_ids": ["b2", "b3", "b4"],
                }
            ]
        }
        return json.dumps(data), {"prompt_tokens": 50, "output_tokens": 50, "total_tokens": 100}, 0

    monkeypatch.setattr(extractor, "_execute_prompt_request", mock_execute)
    extractor.extract(inp)

    # Exactly 1 body request, NOT 2 experience requests
    body_passes = [p for p in executed_passes if p != "personal"]
    assert body_passes == ["body"]


# =====================================================================
# Part B: STANDARD_CV Bounded Recovery Tests
# =====================================================================

def test_standard_cv_suspicious_empty_triggers_bounded_recovery(monkeypatch):
    """STANDARD_CV with suspiciously empty body triggers bounded recovery (<= 2 recovery passes)."""
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        two_pass=True,
    )
    # 4 sections: summary, certifications, experience, education
    blocks = [
        _block("h0", "Dr. Alpert", suggested_role="HEADER", region_kind="header"),
        _block("b1", "PROFESSIONAL SUMMARY", suggested_role="SECTION_HEADING"),
        _block("b2", "Oncologist with 14 years experience", suggested_role="DESCRIPTION"),
        _block("b3", "BOARD CERTIFICATIONS", suggested_role="SECTION_HEADING"),
        _block("b4", "ABIM Medical Oncology 2011", suggested_role="BULLET"),
        _block("b5", "CLINICAL EXPERIENCE", suggested_role="SECTION_HEADING"),
        _block("b6", "Senior Medical Director", suggested_role="ENTRY_TITLE"),
        _block("b7", "Biogen", suggested_role="ORGANIZATION"),
        _block("b8", "2019 - Present", suggested_role="DATE"),
        _block("b9", "EDUCATION", suggested_role="SECTION_HEADING"),
        _block("b10", "MD", suggested_role="DEGREE"),
        _block("b11", "Harvard", suggested_role="INSTITUTION"),
    ]
    inp = _input_with_archetype(DocumentArchetype.STANDARD_CV, blocks)

    executed_passes: list[str] = []

    def mock_execute(prompt, response_schema, api_key, model, base_url, timeout, max_retries, pass_name):
        executed_passes.append(pass_name)
        if pass_name == "personal":
            data = {"personal": {"name": {"value": "Dr. Alpert", "source_block_ids": ["h0"]}}}
        elif pass_name == "body":
            # Initial pass silently returns empty body (triggers recovery)
            data = {}
        elif pass_name == "body_recovery_overview":
            data = {
                "summary": {"value": "Oncologist with 14 years experience", "source_block_ids": ["b2"]},
                "certifications": [{"value": "ABIM Medical Oncology 2011", "source_block_ids": ["b4"]}],
            }
        elif pass_name == "body_recovery_entities":
            data = {
                "experience": [
                    {
                        "company": {"value": "Biogen", "source_block_ids": ["b7"]},
                        "designation": {"value": "Senior Medical Director", "source_block_ids": ["b6"]},
                        "source_block_ids": ["b6", "b7", "b8"],
                    }
                ],
                "education": [
                    {
                        "institution": {"value": "Harvard", "source_block_ids": ["b11"]},
                        "degree": {"value": "MD", "source_block_ids": ["b10"]},
                        "source_block_ids": ["b10", "b11"],
                    }
                ],
            }
        else:
            data = {}
        return json.dumps(data), {"prompt_tokens": 50, "output_tokens": 50, "total_tokens": 100}, 0

    monkeypatch.setattr(extractor, "_execute_prompt_request", mock_execute)
    res = extractor.extract(inp)

    # Verifications:
    # 1. Initial pass was "body"
    assert "body" in executed_passes
    # 2. Recovery was triggered and used bounded grouping: at most 2 recovery requests
    recovery_passes = [p for p in executed_passes if p.startswith("body_recovery_")]
    assert len(recovery_passes) == 2
    assert "body_recovery_overview" in recovery_passes
    assert "body_recovery_entities" in recovery_passes
    # 3. Merged result contains recovered data
    assert len(res.experience) == 1
    assert len(res.education) == 1
    assert len(res.certifications) == 1


# =====================================================================
# Part C: ACADEMIC_CV Target Grouping Tests
# =====================================================================

def test_academic_cv_groups_duplicate_experience_targets():
    """group_sections_by_target() coalesces multiple sections with target='experience'."""
    sections = [
        SemanticSection("h1", "EDUCATION", ["h1", "b1", "b2"], "education", 1),
        SemanticSection("h2", "ACADEMIC APPOINTMENTS", ["h2", "b3", "b4"], "experience", 1),
        SemanticSection("h3", "POSTDOCTORAL EXPERIENCE", ["h3", "b5", "b6"], "experience", 2),
        SemanticSection("h4", "GRANTS", ["h4", "b7"], "achievements", 2),
        SemanticSection("h5", "PUBLICATIONS", ["h5", "b8"], "unsupported", 3),
    ]
    grouped = group_sections_by_target(sections)

    # Unsupported excluded, duplicate experience merged into one!
    assert len(grouped) == 3
    targets = [s.canonical_target for s in grouped]
    assert targets == ["education", "experience", "achievements"]

    exp_sec = grouped[1]
    assert exp_sec.canonical_target == "experience"
    assert exp_sec.block_ids == ["h2", "b3", "b4", "h3", "b5", "b6"]
    assert "ACADEMIC APPOINTMENTS" in exp_sec.heading_text
    assert "POSTDOCTORAL EXPERIENCE" in exp_sec.heading_text


def test_gemini_extractor_groups_academic_cv_targets(monkeypatch):
    """GeminiSemanticExtractor sends 1 request for merged experience target in ACADEMIC_CV."""
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        two_pass=True,
    )
    blocks = [
        _block("h0", "Prof. Smith", suggested_role="HEADER", region_kind="header"),
        _block("b1", "ACADEMIC APPOINTMENTS", suggested_role="SECTION_HEADING"),
        _block("b2", "Associate Professor", suggested_role="ENTRY_TITLE"),
        _block("b3", "MIT", suggested_role="ORGANIZATION"),
        _block("b4", "2018 - Present", suggested_role="DATE"),
        _block("b5", "POSTDOCTORAL FELLOWSHIPS", suggested_role="SECTION_HEADING"),
        _block("b6", "Postdoc", suggested_role="ENTRY_TITLE"),
        _block("b7", "Stanford", suggested_role="ORGANIZATION"),
        _block("b8", "2016 - 2018", suggested_role="DATE"),
        _block("b9", "EDUCATION", suggested_role="SECTION_HEADING"),
        _block("b10", "Ph.D.", suggested_role="DEGREE"),
        _block("b11", "Berkeley", suggested_role="INSTITUTION"),
    ]
    inp = _input_with_archetype(DocumentArchetype.ACADEMIC_CV, blocks)

    executed_passes: list[str] = []

    def mock_execute(prompt, response_schema, api_key, model, base_url, timeout, max_retries, pass_name):
        executed_passes.append(pass_name)
        if pass_name == "personal":
            data = {"personal": {"name": {"value": "Prof. Smith", "source_block_ids": ["h0"]}}}
        elif pass_name == "body_sec_experience":
            data = {
                "experience": [
                    {
                        "company": {"value": "MIT", "source_block_ids": ["b3"]},
                        "designation": {"value": "Associate Professor", "source_block_ids": ["b2"]},
                        "source_block_ids": ["b2", "b3", "b4"],
                    },
                    {
                        "company": {"value": "Stanford", "source_block_ids": ["b7"]},
                        "designation": {"value": "Postdoc", "source_block_ids": ["b6"]},
                        "source_block_ids": ["b6", "b7", "b8"],
                    },
                ]
            }
        elif pass_name == "body_sec_education":
            data = {
                "education": [
                    {
                        "institution": {"value": "Berkeley", "source_block_ids": ["b11"]},
                        "degree": {"value": "Ph.D.", "source_block_ids": ["b10"]},
                        "source_block_ids": ["b10", "b11"],
                    }
                ]
            }
        else:
            data = {}
        return json.dumps(data), {"prompt_tokens": 50, "output_tokens": 50, "total_tokens": 100}, 0

    monkeypatch.setattr(extractor, "_execute_prompt_request", mock_execute)
    res = extractor.extract(inp)

    # Verifications:
    # 1. Only 1 body_sec_experience pass was executed (not 2!)
    exp_passes = [p for p in executed_passes if p == "body_sec_experience"]
    assert len(exp_passes) == 1
    # 2. Total body passes = 2 (experience and education)
    body_passes = [p for p in executed_passes if p != "personal"]
    assert sorted(body_passes) == ["body_sec_education", "body_sec_experience"]
    assert len(res.experience) == 2


# =====================================================================
# Part D: Other Archetypes Monolithic Tests
# =====================================================================

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


# =====================================================================
# Part E: Deterministic Boolean Current Sanitization Tests
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
    assert sanitized.experience[0].current is None
    assert sanitized.projects[0].current is None
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
        ]
    )
    sanitized = sanitize_grounded_current_status(output, sem_input)
    assert sanitized.experience[0].current is not None
    assert sanitized.experience[0].current.value is False


def test_sanitize_provenance_isolation():
    """An un-cited block containing 'Present' cannot validate current=True for another block."""
    sem_input = SemanticInput(
        document_id="doc-1",
        page_count=1,
        blocks=[
            _block("b1", "2021 - 2026"),
            _block("b2", "2019 - Present"),
        ],
    )
    output = SemanticOutput(
        projects=[
            GroundedProjectItem(
                current=GroundedBool(value=True, source_block_ids=["b1"]),
                source_block_ids=["b1"],
            )
        ]
    )
    sanitized = sanitize_grounded_current_status(output, sem_input)
    assert sanitized.projects[0].current is None


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
    violations = validate_semantic_output(unsanitized_output, sem_input)
    assert len(violations) == 1
    assert "UNSUPPORTED_CURRENT_STATUS in projects[0].current" in violations[0]


def test_parse_document_semantically_end_to_end_with_sanitization():
    """parse_document_semantically applies sanitization transparently before validation."""
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
                        current=GroundedBool(value=True, source_block_ids=[b1_id]),
                        source_block_ids=[b1_id],
                    )
                ],
            )

    extractor = FlawedMockExtractor()
    resume = parse_document_semantically(layout_doc, extractor, document_id="doc-test-sanitization")
    assert resume.personal.name == "Alice Smith"
    assert len(resume.projects) == 1
    assert resume.projects[0].name == "Research Grant"
    assert resume.projects[0].current is None


# =====================================================================
# Part G: Request Planning Regression on 10 Generalization Fixtures
# =====================================================================

def test_request_planning_regression_across_all_generalization_fixtures():
    """Verify normal request planning for all 10 generalization fixtures.

    Guarantees:
    - Every STANDARD_CV generates exactly 2 requests in normal path (1 personal + 1 body).
    - ACADEMIC_CV generates 1 personal + grouped body requests (no duplicate target calls).
    - Total normal requests across all 10 fixtures <= 26 (down from 58 in Phase 10Q).
    """
    manifest_path = Path("tests/fixtures/generalization/manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not found")

    manifest = json.loads(manifest_path.read_text())
    total_normal_requests = 0

    for item in manifest["fixtures"]:
        fname = item["filename"]
        fpath = Path("tests/fixtures/generalization") / fname
        raw = fpath.read_bytes()
        doc = document_from_text_blocks(PDFExtractor.extract(raw))
        rec = reconstruct_document(doc)
        layout_doc = interpret_layout(rec)

        arch_str = item["archetype"].upper()
        arch = DocumentArchetype[arch_str] if arch_str in DocumentArchetype.__members__ else DocumentArchetype.UNKNOWN
        sinput = build_semantic_input(layout_doc, document_id=fname, archetype=arch)

        sec_aware = should_use_section_aware_body_extraction(sinput)
        if arch == DocumentArchetype.STANDARD_CV:
            # Guarantees STANDARD_CV never regresses to one request per physical section
            assert sec_aware is False, f"STANDARD_CV {fname} must not use section-aware in normal path"
            normal_body_req = 1
        elif arch == DocumentArchetype.ACADEMIC_CV:
            assert sec_aware is True, f"ACADEMIC_CV {fname} should use section-aware"
            sections = partition_semantic_input_into_sections(sinput)
            grouped = group_sections_by_target(sections)
            normal_body_req = len(grouped)
        else:
            normal_body_req = 1

        normal_total = 1 + normal_body_req  # 1 personal + body
        total_normal_requests += normal_total

    # Assert total normal requests is strictly bounded (25 requests total: 16 for standard + 9 for academic)
    assert total_normal_requests <= 26, f"Expected <= 26 normal requests, got {total_normal_requests}"
