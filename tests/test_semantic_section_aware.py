"""Phase 10M: Focused offline tests for section-aware semantic extraction architecture.

Tests cover:
  1. Generic section boundary detection (no fixture-specific strings).
  2. Multiple sections across pages.
  3. Section without recognized canonical destination (unsupported).
  4. Academic appointments mapping to experience.
  5. Education section mapping to education.
  6. Grants/awards mapping to supported collections.
  7. Unsupported publication/teaching/service sections NOT suppressing supported extraction.
  8. Exact source_block_ids preserved after section splitting.
  9. No fixture-specific section-title matching used.
  10. filter_semantic_input_to_blocks preserves header blocks + selected body blocks.
  11. merge_body_outputs deterministic merge rules.
  12. Existing empty-body recovery/completeness behavior unchanged.
  13. GeminiSemanticExtractor section-aware _run_body for ACADEMIC_CV archetype (mock HTTP).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.semantic_contract import (
    BlockClassification,
    BodySemanticOutput,
    DocumentArchetype,
    GroundedEducationItem,
    GroundedExperienceItem,
    GroundedString,
    PersonalSemanticOutput,
    SemanticBlockCategory,
    SemanticBlockInput,
    SemanticInput,
    filter_semantic_input_to_blocks,
    is_body_output_suspiciously_empty,
    merge_body_outputs,
    partition_semantic_input_into_sections,
)
from app.extractors.semantic_extractor import SemanticCompletenessError


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


def _academic_input(blocks: list[SemanticBlockInput]) -> SemanticInput:
    return SemanticInput(
        document_id="doc-academic",
        page_count=2,
        archetype=DocumentArchetype.ACADEMIC_CV,
        blocks=blocks,
    )


def _standard_input(blocks: list[SemanticBlockInput]) -> SemanticInput:
    return SemanticInput(
        document_id="doc-standard",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        blocks=blocks,
    )


# =====================================================================
# 1. Generic Section Boundary Detection
# =====================================================================


def test_partition_single_entry_title_section():
    """All-caps heading creates a section boundary; following blocks belong to that section."""
    blocks = [
        _block("b1", "Some Person", suggested_role="HEADER", region_kind="header", reading_order=0),
        _block("b2", "SOFTWARE ENGINEER", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b3", "Acme Corp", suggested_role="ORGANIZATION", reading_order=2),
        _block("b4", "2020 - 2024", suggested_role="DATE", reading_order=3),
        _block("b5", "Built distributed storage engines", suggested_role="DESCRIPTION", reading_order=4),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)

    assert len(sections) >= 1
    all_section_block_ids = [bid for s in sections for bid in s.block_ids]
    for b in blocks:
        if b.region_kind != "header":
            assert b.block_id in all_section_block_ids


def test_partition_no_body_blocks_returns_empty():
    """If all blocks are header blocks, partition returns an empty list."""
    blocks = [
        _block("h1", "Name", suggested_role="HEADER", region_kind="header", reading_order=0),
        _block("h2", "email@test.com", suggested_role="CONTACT", region_kind="header", reading_order=1),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    assert sections == []


def test_partition_blocks_before_first_boundary_form_preamble():
    """Blocks appearing before the first boundary signal form an implicit preamble section."""
    blocks = [
        _block("h1", "John Doe", suggested_role="HEADER", region_kind="header", reading_order=0),
        _block("b1", "Introductory summary statement", suggested_role="DESCRIPTION", reading_order=1),
        _block("b2", "More introductory text", suggested_role="DESCRIPTION", reading_order=2),
        _block("b3", "WORK EXPERIENCE", suggested_role="SECTION_HEADING", reading_order=3),
        _block("b4", "2018 - 2021", suggested_role="DATE", reading_order=4),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    assert len(sections) >= 2
    preamble = sections[0]
    assert "b1" in preamble.block_ids
    assert "b2" in preamble.block_ids


def test_partition_preserves_reading_order():
    """Section block_ids are ordered by reading_order within each section."""
    blocks = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header", reading_order=0),
        _block("b1", "SECTION A", suggested_role="SECTION_HEADING", reading_order=1),
        _block("b2", "2020 - 2022", suggested_role="DATE", reading_order=2),
        _block("b3", "Description", suggested_role="DESCRIPTION", reading_order=3),
        _block("b4", "SECTION B", suggested_role="SECTION_HEADING", reading_order=4),
        _block("b5", "2015 - 2019", suggested_role="DATE", reading_order=5),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    b1_section_index = next(i for i, s in enumerate(sections) if "b1" in s.block_ids)
    b4_section_index = next(i for i, s in enumerate(sections) if "b4" in s.block_ids)
    assert b1_section_index < b4_section_index


# =====================================================================
# 2. Multiple Sections Across Pages
# =====================================================================


def test_partition_sections_span_pages():
    """Sections correctly span page boundaries in reading order."""
    blocks = [
        _block("h1", "Dr. Someone", suggested_role="HEADER", region_kind="header", page=1, reading_order=0),
        _block("b1", "APPOINTMENTS", suggested_role="ENTRY_TITLE", page=1, reading_order=1),
        _block("b2", "Professor with Tenure", suggested_role="ENTRY_TITLE", page=1, reading_order=2),
        _block("b3", "2019 - Present", suggested_role="DATE", page=1, reading_order=3),
        _block("h2", "Running Header", suggested_role="HEADER", region_kind="header", page=2, reading_order=0),
        _block("b4", "EDUCATION", suggested_role="SECTION_HEADING", page=2, reading_order=1),
        _block("b5", "Doctor of Philosophy (Ph.D.) in Bioengineering", suggested_role="DESCRIPTION", page=2, reading_order=2),
        _block("b6", "Harvard University", suggested_role="LOCATION", page=2, reading_order=3),
    ]
    si = SemanticInput(
        document_id="doc-multipage",
        page_count=2,
        archetype=DocumentArchetype.ACADEMIC_CV,
        blocks=blocks,
    )
    sections = partition_semantic_input_into_sections(si)
    assert len(sections) >= 2
    all_ids = {bid: s for s in sections for bid in s.block_ids}
    assert all_ids["b1"] is not all_ids["b4"]


def test_partition_page_numbers_recorded_correctly():
    """Each section's page_start matches the page of its first block."""
    blocks = [
        _block("h1", "Header", suggested_role="HEADER", region_kind="header", page=1, reading_order=0),
        _block("b1", "EXPERIENCE", suggested_role="SECTION_HEADING", page=1, reading_order=1),
        _block("b2", "2020 - 2023", suggested_role="DATE", page=1, reading_order=2),
        _block("h2", "Page 2 Header", suggested_role="HEADER", region_kind="header", page=2, reading_order=0),
        _block("b3", "CERTIFICATIONS", suggested_role="SECTION_HEADING", page=2, reading_order=1),
        _block("b4", "AWS Certified", suggested_role="DESCRIPTION", page=2, reading_order=2),
    ]
    si = SemanticInput(
        document_id="doc-pages",
        page_count=2,
        archetype=DocumentArchetype.ACADEMIC_CV,
        blocks=blocks,
    )
    sections = partition_semantic_input_into_sections(si)
    page1_sections = [s for s in sections if s.page_start == 1]
    page2_sections = [s for s in sections if s.page_start == 2]
    assert page1_sections
    assert page2_sections


# =====================================================================
# 3. Section without Recognized Canonical Destination (Unsupported)
# =====================================================================


def test_section_without_recognized_canonical_destination():
    """A section with no matching canonical alias and unstructured prose is unsupported."""
    blocks = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b1", "SOME CUSTOM MISCELLANEOUS SECTION", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "Some random narrative without dates or job titles.", suggested_role="DESCRIPTION", reading_order=2),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    custom_section = next((s for s in sections if "b1" in s.block_ids), None)
    assert custom_section is not None
    assert custom_section.canonical_target == "unsupported"


def test_citations_section_classified_unsupported():
    """Citation blocks without employer/institution/degree/grant patterns are unsupported."""
    blocks = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b1", "PEER-REVIEWED JOURNAL PAPERS", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "[1] Author, A. et al., Journal of Science, 2023.", suggested_role="DESCRIPTION", reading_order=2),
        _block("b3", "[2] Author, B. et al., Physical Review Letters, 2022.", suggested_role="DESCRIPTION", reading_order=3),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    pub_section = next((s for s in sections if "b1" in s.block_ids), None)
    assert pub_section is not None
    assert pub_section.canonical_target == "unsupported"


def test_teaching_mentorship_classified_unsupported():
    """Teaching course lists without degrees or employers are unsupported."""
    blocks = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b1", "TEACHING & GRADUATE MENTORSHIP", suggested_role="ORGANIZATION", reading_order=1),
        _block("b2", "· BME 580: Neural Engineering (Spring 2020)", suggested_role="BULLET", reading_order=2),
        _block("b3", "· BME 310: Computational Neuroscience (Fall 2021)", suggested_role="BULLET", reading_order=3),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    teaching_section = next((s for s in sections if "b1" in s.block_ids), None)
    assert teaching_section is not None
    assert teaching_section.canonical_target == "unsupported"


def test_service_board_classified_unsupported():
    """Editorial board / committee service lists without employers are unsupported."""
    blocks = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b1", "PROFESSIONAL SERVICE & EDITORIAL BOARDS", suggested_role="ORGANIZATION", reading_order=1),
        _block("b2", "· Associate Editor, IEEE Transactions (2018 - Present)", suggested_role="BULLET", reading_order=2),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    svc_section = next((s for s in sections if "b1" in s.block_ids), None)
    assert svc_section is not None
    assert svc_section.canonical_target == "unsupported"


# =====================================================================
# 4. Academic Appointments Mapping to Experience
# =====================================================================


def test_academic_appointments_mapping_to_experience():
    """Academic appointments section with titles, organizations, dates, and locations maps to experience."""
    blocks = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b1", "ACADEMIC APPOINTMENTS", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "Professor with Tenure", suggested_role="ENTRY_TITLE", reading_order=2),
        _block("b3", "Department of Biomedical Engineering, Johns Hopkins University | Baltimore, MD", suggested_role="LOCATION", reading_order=3),
        _block("b4", "2019 - Present", suggested_role="DATE", reading_order=4),
        _block("b5", "Associate Professor", suggested_role="DESCRIPTION", reading_order=5),
        _block("b6", "2014 - 2019", suggested_role="DATE", reading_order=6),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    appt_section = next((s for s in sections if "b1" in s.block_ids), None)
    assert appt_section is not None
    assert appt_section.canonical_target == "experience"
    assert "b2" in appt_section.block_ids
    assert "b3" in appt_section.block_ids
    assert "b4" in appt_section.block_ids


# =====================================================================
# 5. Education Mapping to Education
# =====================================================================


def test_education_mapping_to_education():
    """Education section maps to education canonical target."""
    blocks = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b1", "EDUCATION", suggested_role="SECTION_HEADING", reading_order=1),
        _block("b2", "Doctor of Philosophy (Ph.D.) in Bioengineering", suggested_role="DESCRIPTION", reading_order=2),
        _block("b3", "Harvard University | Cambridge, MA", suggested_role="LOCATION", reading_order=3),
        _block("b4", "2001 - 2005", suggested_role="DATE", reading_order=4),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    edu_section = next((s for s in sections if "b1" in s.block_ids), None)
    assert edu_section is not None
    assert edu_section.canonical_target == "education"
    assert "b2" in edu_section.block_ids
    assert "b3" in edu_section.block_ids
    assert "b4" in edu_section.block_ids


# =====================================================================
# 6. Grants/Awards Mapping to Supported Collections
# =====================================================================


def test_grants_awards_mapping_to_supported_collections():
    """Grants and awards section maps to achievements."""
    blocks = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b1", "FUNDED RESEARCH GRANTS & AWARDS (Total: $5.8M)", suggested_role="DESCRIPTION", reading_order=1),
        _block("b2", "NIH R01: Real-Time Neural Signal Decoding ($2.4M, Principal Investigator, 2021 - 2026)", suggested_role="DESCRIPTION", reading_order=2),
        _block("b3", "NSF CAREER Award ($1.1M, Principal Investigator, 2016 - 2021)", suggested_role="DESCRIPTION", reading_order=3),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)
    grant_section = next((s for s in sections if "b1" in s.block_ids), None)
    assert grant_section is not None
    assert grant_section.canonical_target == "achievements"
    assert "b2" in grant_section.block_ids
    assert "b3" in grant_section.block_ids


# =====================================================================
# 7. Unsupported Sections Do NOT Suppress Supported Extraction
# =====================================================================


def test_unsupported_publication_teaching_service_sections_not_suppressing_supported_extraction():
    """Presence of publication, teaching, and service sections does NOT eliminate supported ones."""
    blocks = [
        _block("h1", "Dr. Thorne", suggested_role="HEADER", region_kind="header"),
        # Supported: experience
        _block("b1", "ACADEMIC APPOINTMENTS", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "Professor with Tenure", suggested_role="ENTRY_TITLE", reading_order=2),
        _block("b3", "Johns Hopkins University | Baltimore, MD", suggested_role="LOCATION", reading_order=3),
        _block("b4", "2019 - Present", suggested_role="DATE", reading_order=4),
        # Supported: education
        _block("b5", "EDUCATION", suggested_role="SECTION_HEADING", reading_order=5),
        _block("b6", "Ph.D. in Bioengineering", suggested_role="DESCRIPTION", reading_order=6),
        _block("b7", "Harvard University", suggested_role="LOCATION", reading_order=7),
        # Supported: achievements (grants)
        _block("b8", "FUNDED RESEARCH GRANTS & AWARDS (Total: $5.8M)", suggested_role="DESCRIPTION", reading_order=8),
        _block("b9", "NIH R01 Grant ($2.4M, 2021 - 2026)", suggested_role="DESCRIPTION", reading_order=9),
        # Unsupported: publications
        _block("b10", "PEER-REVIEWED JOURNAL PUBLICATIONS", suggested_role="ENTRY_TITLE", reading_order=10),
        _block("b11", "[1] Thorne et al. Nature 2023.", suggested_role="DESCRIPTION", reading_order=11),
        # Unsupported: teaching
        _block("b12", "TEACHING & GRADUATE MENTORSHIP", suggested_role="ORGANIZATION", reading_order=12),
        _block("b13", "· BME 580: Neural Engineering", suggested_role="BULLET", reading_order=13),
        # Unsupported: service
        _block("b14", "PROFESSIONAL SERVICE & EDITORIAL BOARDS", suggested_role="ORGANIZATION", reading_order=14),
        _block("b15", "· Associate Editor, IEEE Trans Biomed Eng", suggested_role="BULLET", reading_order=15),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)

    supported = [s for s in sections if s.canonical_target != "unsupported"]
    unsupported = [s for s in sections if s.canonical_target == "unsupported"]

    assert len(supported) == 3, f"Expected 3 supported sections, got {[s.heading_text for s in supported]}"
    assert len(unsupported) == 3, f"Expected 3 unsupported sections, got {[s.heading_text for s in unsupported]}"

    targets = [s.canonical_target for s in supported]
    assert "experience" in targets
    assert "education" in targets
    assert "achievements" in targets


# =====================================================================
# 8. Source Block IDs Preserved After Section Splitting
# =====================================================================


def test_exact_source_block_ids_preserved_after_section_splitting():
    """Every non-header block ID is retained without alteration, deletion, or renaming."""
    blocks = [
        _block("h1", "Name", suggested_role="HEADER", region_kind="header"),
        _block("b1", "ACADEMIC APPOINTMENTS", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "Professor", suggested_role="ENTRY_TITLE", reading_order=2),
        _block("b3", "Univ | City", suggested_role="LOCATION", reading_order=3),
        _block("b4", "2019 - Present", suggested_role="DATE", reading_order=4),
        _block("b5", "EDUCATION", suggested_role="SECTION_HEADING", reading_order=5),
        _block("b6", "Ph.D. in Physics", suggested_role="DESCRIPTION", reading_order=6),
        _block("b7", "MIT", suggested_role="LOCATION", reading_order=7),
    ]
    si = _academic_input(blocks)
    sections = partition_semantic_input_into_sections(si)

    collected_ids: list[str] = []
    for s in sections:
        collected_ids.extend(s.block_ids)

    expected_body_ids = ["b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    assert collected_ids == expected_body_ids


def test_filter_semantic_input_preserves_header_and_source_block_ids():
    """filter_semantic_input_to_blocks preserves exact block_ids for header + selected blocks."""
    blocks = [
        _block("h1", "Name", suggested_role="HEADER", region_kind="header"),
        _block("h2", "email@test.com", suggested_role="CONTACT", region_kind="header"),
        _block("b1", "Appointments", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "Professor", suggested_role="ENTRY_TITLE", reading_order=2),
        _block("b3", "Pubs", suggested_role="DESCRIPTION", reading_order=3),
    ]
    si = _academic_input(blocks)
    filtered = filter_semantic_input_to_blocks(si, ["b1", "b2"])
    filtered_ids = [b.block_id for b in filtered.blocks]

    assert "h1" in filtered_ids
    assert "h2" in filtered_ids
    assert "b1" in filtered_ids
    assert "b2" in filtered_ids
    assert "b3" not in filtered_ids


# =====================================================================
# 9. No Fixture-Specific Section-Title Matching
# =====================================================================


def test_no_fixture_specific_section_title_matching():
    """Generic sections with arbitrary words are classified using structural patterns, not fixture names."""
    # An unusual heading with organization + dates is experience
    blocks1 = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b1", "CLINICAL POSTS & ROLES", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "St. Jude Hospital", suggested_role="ORGANIZATION", reading_order=2),
        _block("b3", "2015 - 2020", suggested_role="DATE", reading_order=3),
    ]
    si1 = _academic_input(blocks1)
    sections1 = partition_semantic_input_into_sections(si1)
    s1 = next((s for s in sections1 if "b1" in s.block_ids), None)
    assert s1 is not None
    assert s1.canonical_target == "experience"

    # An unusual heading with degrees is education
    blocks2 = [
        _block("h1", "Person", suggested_role="HEADER", region_kind="header"),
        _block("b2_1", "POSTGRADUATE TRAINING", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2_2", "Master of Science (M.S.)", suggested_role="DESCRIPTION", reading_order=2),
        _block("b2_3", "Oxford University", suggested_role="LOCATION", reading_order=3),
    ]
    si2 = _academic_input(blocks2)
    sections2 = partition_semantic_input_into_sections(si2)
    s2 = next((s for s in sections2 if "b2_1" in s.block_ids), None)
    assert s2 is not None
    assert s2.canonical_target == "education"


# =====================================================================
# 10. Existing Empty-Body Recovery & Completeness Tests Remain Passing
# =====================================================================


def test_is_body_output_suspiciously_empty_still_detects_evidence_rich():
    """is_body_output_suspiciously_empty continues to detect suspiciously empty bodies."""
    blocks = [
        _block("h1", "Name", suggested_role="HEADER", region_kind="header"),
        _block("b1", "ACADEMIC APPOINTMENTS", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "Professor with Tenure", suggested_role="ENTRY_TITLE", reading_order=2),
        _block("b3", "2019 - Present", suggested_role="DATE", reading_order=3),
    ]
    si = _academic_input(blocks)
    empty_out = BodySemanticOutput()
    assert is_body_output_suspiciously_empty(empty_out, si) is True


    populated_out = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Professor", source_block_ids=["b1"]),
                source_block_ids=["b1"],
            )
        ]
    )
    assert is_body_output_suspiciously_empty(populated_out, si) is False


def test_merge_body_outputs_combines_sections_deterministically():
    """merge_body_outputs correctly merges experience, education, achievements across sections."""
    out_exp = BodySemanticOutput(
        document_archetype=DocumentArchetype.ACADEMIC_CV,
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Professor with Tenure", source_block_ids=["b2"]),
                source_block_ids=["b2"],
            )
        ],
    )
    out_edu = BodySemanticOutput(
        education=[
            GroundedEducationItem(
                degree=GroundedString(value="Ph.D.", source_block_ids=["b6"]),
                source_block_ids=["b6"],
            )
        ],
    )
    out_ach = BodySemanticOutput(
        achievements=[GroundedString(value="NIH R01 Grant", source_block_ids=["b9"])],
    )
    merged = merge_body_outputs([out_exp, out_edu, out_ach])
    assert len(merged.experience) == 1
    assert len(merged.education) == 1
    assert len(merged.achievements) == 1
    assert merged.document_archetype == DocumentArchetype.ACADEMIC_CV


# =====================================================================
# 11. Gemini Mock Execution for Section-Aware ACADEMIC_CV
# =====================================================================


def _make_gemini_resp(content_dict: dict) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(content_dict)}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 300,
            "candidatesTokenCount": 150,
            "totalTokenCount": 450,
        },
    }


def test_gemini_academic_multi_section_extraction_merges_successfully():
    """GeminiSemanticExtractor runs section extractions for ACADEMIC_CV and merges them."""
    from app.extractors.providers.gemini import GeminiSemanticExtractor

    blocks = [
        _block("h1", "Dr. Thorne", suggested_role="HEADER", region_kind="header"),
        _block("h2", "thorne@jhu.edu", suggested_role="CONTACT", region_kind="header"),
        # Section 0: appointments
        _block("b1", "ACADEMIC APPOINTMENTS", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "Professor with Tenure", suggested_role="ENTRY_TITLE", reading_order=2),
        _block("b3", "2019 - Present", suggested_role="DATE", reading_order=3),
        _block("b4", "Johns Hopkins University | Baltimore, MD", suggested_role="LOCATION", reading_order=4),
        # Section 1: education
        _block("b5", "EDUCATION", suggested_role="SECTION_HEADING", reading_order=5),
        _block("b6", "Ph.D. in Bioengineering", suggested_role="DESCRIPTION", reading_order=6),
        _block("b7", "Harvard University", suggested_role="LOCATION", reading_order=7),
        # Section 2: unsupported publications
        _block("b8", "PEER-REVIEWED JOURNAL PUBLICATIONS", suggested_role="ENTRY_TITLE", reading_order=8),
        _block("b9", "[1] Thorne et al. Nature 2023.", suggested_role="DESCRIPTION", reading_order=9),
    ]
    si = _academic_input(blocks)

    personal_dict = {
        "document_archetype": "academic_cv",
        "block_classifications": [{"block_id": "h1", "category": "PERSONAL"}],
        "personal": {
            "name": {"value": "Dr. Thorne", "source_block_ids": ["h1"]},
            "email": {"value": "thorne@jhu.edu", "source_block_ids": ["h2"]},
        },
    }
    exp_dict = {
        "document_archetype": "academic_cv",
        "block_classifications": [],
        "summary": None,
        "skills": [],
        "experience": [
            {
                "designation": {"value": "Professor with Tenure", "source_block_ids": ["b2"]},
                "startDate": {"value": "2019", "source_block_ids": ["b3"]},
                "location": {"value": "Baltimore, MD", "source_block_ids": ["b4"]},
                "source_block_ids": ["b2", "b3", "b4"],
            }
        ],
        "education": [],
        "projects": [],
        "certifications": [],
        "languages": [],
        "achievements": [],
    }
    edu_dict = {
        "document_archetype": "academic_cv",
        "block_classifications": [],
        "summary": None,
        "skills": [],
        "experience": [],
        "education": [
            {
                "institution": {"value": "Harvard University", "source_block_ids": ["b7"]},
                "degree": {"value": "Ph.D. in Bioengineering", "source_block_ids": ["b6"]},
                "source_block_ids": ["b6", "b7"],
            }
        ],
        "projects": [],
        "certifications": [],
        "languages": [],
        "achievements": [],
    }

    call_count = 0
    def mock_post(url, headers=None, json=None, timeout=None):
        nonlocal call_count
        call_count += 1
        prompt_text = (json or {}).get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        if "PersonalSemanticOutput" in prompt_text or "personal" in prompt_text.lower()[:200]:
            mock_resp.json.return_value = _make_gemini_resp(personal_dict)
        elif "b2" in prompt_text:
            mock_resp.json.return_value = _make_gemini_resp(exp_dict)
        elif "b6" in prompt_text:
            mock_resp.json.return_value = _make_gemini_resp(edu_dict)
        else:
            mock_resp.json.return_value = _make_gemini_resp(exp_dict)
        return mock_resp

    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-test",
        two_pass=True,
    )
    mock_client = MagicMock()
    mock_client.post = mock_post
    extractor._client = mock_client

    result = extractor.extract(si)

    assert len(result.experience) == 1
    assert result.experience[0].designation.value == "Professor with Tenure"
    assert len(result.education) == 1
    assert result.education[0].institution.value == "Harvard University"
    assert result.personal.name.value == "Dr. Thorne"
    # Verify request count: 1 personal + 2 supported sections (appointments + education) = 3 calls
    # Unsupported publications section b8-b9 was skipped!
    assert call_count == 3
