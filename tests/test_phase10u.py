"""Phase 10U Tests: Deterministic Academic Appointment-Scoped Extraction.

Validates:
1. Multiple academic appointments segmentation on real generalization fixtures:
   - long_academic_tenured_professor_cv.pdf (4 appointments)
   - academic_research_postdoc_cv.pdf (2 research + 2 teaching appointments)
2. Ordering invariance: DATE before vs after ORGANIZATION.
3. Multiple description bullets grouped into the correct appointment.
4. Teaching assistant compound title block classification as ENTRY_TITLE and negative controls.
5. Organization-before-title teaching layouts.
6. Section isolation and preservation of block IDs, SemanticBlockInput objects, and section metadata.
7. Gemini appointment-scoped extraction, single-entity constraint, deterministic merge, and prevention of cross-entity provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    BlockClassification,
    BodySemanticOutput,
    DeterministicAppointmentGroup,
    DocumentArchetype,
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedString,
    PersonalSemanticOutput,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    SemanticSection,
    build_deterministic_appointment_groups,
    build_semantic_input,
    merge_body_outputs,
    partition_semantic_input_into_sections,
    validate_semantic_output,
)
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.structural_roles import (
    StructuralRole,
    classify_structural_role,
)
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


# =====================================================================
# 1. Structural Role: Delimited Teaching Assistant Title Classification
# =====================================================================

def test_compound_teaching_assistant_title_classification():
    """Teaching assistant titles with course numbers are classified as ENTRY_TITLE."""
    following_meta = ("Spring 2024", "Stanford University")
    role, score, reasons = classify_structural_role(
        "Teaching Assistant · CS 244B: Distributed Systems",
        following_texts=following_meta,
    )
    assert role == StructuralRole.ENTRY_TITLE
    assert "entry_title_structure" in reasons

    role2, _, _ = classify_structural_role(
        "Teaching Assistant · 6.824: Distributed Systems",
        following_texts=("Fall 2021, Fall 2022", "MIT Department of EECS"),
    )
    assert role2 == StructuralRole.ENTRY_TITLE

    role3, _, _ = classify_structural_role(
        "Graduate Teaching Assistant - CS 101",
        following_texts=("Fall 2020", "UC Berkeley"),
    )
    assert role3 == StructuralRole.ENTRY_TITLE


def test_teaching_title_negative_controls():
    """Organizations, dates, section headings, and bullets are never misclassified as ENTRY_TITLE."""
    # Organizations
    r_org, _, _ = classify_structural_role("Stanford University")
    assert r_org != StructuralRole.ENTRY_TITLE

    r_org2, _, _ = classify_structural_role("MIT Department of Electrical Engineering and Computer Science")
    assert r_org2 != StructuralRole.ENTRY_TITLE

    # Dates
    r_date, _, _ = classify_structural_role("Spring 2024")
    assert r_date != StructuralRole.ENTRY_TITLE

    r_date2, _, _ = classify_structural_role("Fall 2021, Fall 2022")
    assert r_date2 != StructuralRole.ENTRY_TITLE

    # Bullets
    r_bullet, _, _ = classify_structural_role("• Assisted instructor with weekly grading and office hours.")
    assert r_bullet == StructuralRole.BULLET

    # Section headings
    r_sec, _, _ = classify_structural_role("TEACHING EXPERIENCE")
    assert r_sec == StructuralRole.SECTION_HEADING


# =====================================================================
# 2. Deterministic Appointment Grouping on Real Fixtures
# =====================================================================

def test_multiple_academic_appointments_tenured_professor_cv():
    """long_academic_tenured_professor_cv.pdf ACADEMIC APPOINTMENTS segments into 4 appointments."""
    fpath = Path("tests/fixtures/generalization/long_academic_tenured_professor_cv.pdf")
    raw = fpath.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(layout_doc, document_id=fpath.name, archetype=DocumentArchetype.ACADEMIC_CV)

    sections = partition_semantic_input_into_sections(sinput)
    appt_section = next(s for s in sections if s.heading_text == "ACADEMIC APPOINTMENTS")
    assert appt_section.canonical_target == "experience"

    groups = build_deterministic_appointment_groups(appt_section, sinput)
    assert len(groups) == 4

    expected_titles = [
        "Professor with Tenure",
        "Associate Professor",
        "Assistant Professor",
        "Postdoctoral Research Fellow",
    ]
    for idx, (grp, exp_title) in enumerate(zip(groups, expected_titles)):
        title_block = next(b for b in grp.blocks if b.block_id == grp.title_block_id)
        assert title_block.suggested_role == "ENTRY_TITLE"
        assert title_block.text == exp_title
        assert len(grp.block_ids) == len(grp.blocks)
        assert grp.canonical_target == "experience"
        assert grp.section_heading_text == "ACADEMIC APPOINTMENTS"


def test_multiple_academic_appointments_postdoc_cv():
    """academic_research_postdoc_cv.pdf segments into 2 research and 2 teaching appointments."""
    fpath = Path("tests/fixtures/generalization/academic_research_postdoc_cv.pdf")
    raw = fpath.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(layout_doc, document_id=fpath.name, archetype=DocumentArchetype.ACADEMIC_CV)

    sections = partition_semantic_input_into_sections(sinput)

    # 1. RESEARCH EXPERIENCE (2 appointments)
    res_section = next(s for s in sections if s.heading_text == "RESEARCH EXPERIENCE")
    res_groups = build_deterministic_appointment_groups(res_section, sinput)
    assert len(res_groups) == 2
    assert res_groups[0].title_block_id == "b_p1_13"
    assert res_groups[0].blocks[0].text == "Postdoctoral Research Fellow"
    assert res_groups[1].title_block_id == "b_p1_18"
    assert res_groups[1].blocks[0].text == "Graduate Research Assistant"

    # 2. TEACHING EXPERIENCE (2 appointments with delimited title blocks)
    teach_section = next(s for s in sections if s.heading_text == "TEACHING EXPERIENCE")
    teach_groups = build_deterministic_appointment_groups(teach_section, sinput)
    assert len(teach_groups) == 2
    assert teach_groups[0].title_block_id == "b_p2_37"
    assert "Teaching Assistant · CS 244B" in teach_groups[0].blocks[0].text
    assert teach_groups[1].title_block_id == "b_p2_40"
    assert "Teaching Assistant · 6.824" in teach_groups[1].blocks[0].text


# =====================================================================
# 3. Ordering Invariance: DATE Before vs After ORGANIZATION
# =====================================================================

def test_date_before_and_after_organization_ordering():
    """Appointment segmentation handles DATE before OR after ORGANIZATION cleanly."""
    blocks = [
        _block("h1", "EXPERIENCE", suggested_role="SECTION_HEADING", reading_order=0),
        # Appointment 1: Title -> DATE -> ORGANIZATION -> DESCRIPTION
        _block("b1", "Postdoctoral Fellow", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "2023 - Present", suggested_role="DATE", reading_order=2),
        _block("b3", "Stanford University", suggested_role="ORGANIZATION", reading_order=3),
        _block("b4", "Conducted distributed systems research.", suggested_role="DESCRIPTION", reading_order=4),
        # Appointment 2: Title -> ORGANIZATION -> DATE -> DESCRIPTION
        _block("b5", "Graduate Research Assistant", suggested_role="ENTRY_TITLE", reading_order=5),
        _block("b6", "MIT", suggested_role="ORGANIZATION", reading_order=6),
        _block("b7", "2018 - 2023", suggested_role="DATE", reading_order=7),
        _block("b8", "Designed consensus protocols.", suggested_role="DESCRIPTION", reading_order=8),
    ]
    sinput = SemanticInput(document_id="test-doc", page_count=1, archetype=DocumentArchetype.ACADEMIC_CV, blocks=blocks)
    sec = SemanticSection("h1", "EXPERIENCE", [b.block_id for b in blocks], "experience", 1)

    groups = build_deterministic_appointment_groups(sec, sinput)
    assert len(groups) == 2

    # Group 1: DATE before ORG
    assert groups[0].title_block_id == "b1"
    assert groups[0].block_ids == ["b1", "b2", "b3", "b4"]

    # Group 2: ORG before DATE
    assert groups[1].title_block_id == "b5"
    assert groups[1].block_ids == ["b5", "b6", "b7", "b8"]


# =====================================================================
# 4. Multiple Descriptions Attached to Appointment
# =====================================================================

def test_multiple_descriptions_attached_to_appointment():
    """Multiple description and bullet blocks are cleanly retained in their appointment group."""
    blocks = [
        _block("h1", "RESEARCH", suggested_role="SECTION_HEADING", reading_order=0),
        _block("b1", "Research Scientist", suggested_role="ENTRY_TITLE", reading_order=1),
        _block("b2", "Bell Labs", suggested_role="ORGANIZATION", reading_order=2),
        _block("b3", "2020 - 2024", suggested_role="DATE", reading_order=3),
        _block("b4", "• Published 5 papers in top venues.", suggested_role="BULLET", reading_order=4),
        _block("b5", "• Secured $1.2M in competitive grant funding.", suggested_role="BULLET", reading_order=5),
        _block("b6", "• Led cross-functional team of 6 researchers.", suggested_role="BULLET", reading_order=6),
        _block("b7", "Mentored graduate interns throughout summer terms.", suggested_role="DESCRIPTION", reading_order=7),
        # Appointment 2
        _block("b8", "Postdoctoral Fellow", suggested_role="ENTRY_TITLE", reading_order=8),
        _block("b9", "Caltech", suggested_role="ORGANIZATION", reading_order=9),
    ]
    sinput = SemanticInput(document_id="test-doc", page_count=1, archetype=DocumentArchetype.ACADEMIC_CV, blocks=blocks)
    sec = SemanticSection("h1", "RESEARCH", [b.block_id for b in blocks], "experience", 1)

    groups = build_deterministic_appointment_groups(sec, sinput)
    assert len(groups) == 2

    # Verify all 4 description/bullet blocks remain with appointment 1
    assert groups[0].block_ids == ["b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    assert groups[1].block_ids == ["b8", "b9"]


# =====================================================================
# 5. Organization-Before-Title Teaching Layout
# =====================================================================

def test_organization_before_title_teaching_layout():
    """When ORGANIZATION precedes ENTRY_TITLE in teaching entries, entries are partitioned accurately."""
    blocks = [
        _block("h1", "TEACHING APPOINTMENTS", suggested_role="SECTION_HEADING", reading_order=0),
        # Entry 1: Org -> Title -> Date
        _block("b1", "Stanford University", suggested_role="ORGANIZATION", reading_order=1),
        _block("b2", "Teaching Assistant · CS 244B", suggested_role="ENTRY_TITLE", reading_order=2),
        _block("b3", "Spring 2024", suggested_role="DATE", reading_order=3),
        # Entry 2: Org -> Title -> Date
        _block("b4", "Massachusetts Institute of Technology", suggested_role="ORGANIZATION", reading_order=4),
        _block("b5", "Teaching Assistant · 6.824", suggested_role="ENTRY_TITLE", reading_order=5),
        _block("b6", "Fall 2021", suggested_role="DATE", reading_order=6),
    ]
    sinput = SemanticInput(document_id="test-doc", page_count=1, archetype=DocumentArchetype.ACADEMIC_CV, blocks=blocks)
    sec = SemanticSection("h1", "TEACHING APPOINTMENTS", [b.block_id for b in blocks], "experience", 1)

    groups = build_deterministic_appointment_groups(sec, sinput)
    assert len(groups) == 2

    # Group 1: Stanford + CS 244B + Spring 2024
    assert groups[0].block_ids == ["b1", "b2", "b3"]
    assert groups[0].title_block_id == "b2"

    # Group 2: MIT + 6.824 + Fall 2021
    assert groups[1].block_ids == ["b4", "b5", "b6"]
    assert groups[1].title_block_id == "b5"


# =====================================================================
# 6. Section Isolation and Metadata Preservation
# =====================================================================

def test_section_isolation_and_metadata_preservation():
    """Appointment groups preserve original SemanticBlockInput objects and section metadata."""
    b_title = _block("b1", "Assistant Professor", suggested_role="ENTRY_TITLE", reading_order=1)
    b_org = _block("b2", "Harvard University", suggested_role="ORGANIZATION", reading_order=2)
    b_date = _block("b3", "2021 - Present", suggested_role="DATE", reading_order=3)
    blocks = [_block("h1", "APPOINTMENTS", suggested_role="SECTION_HEADING", reading_order=0), b_title, b_org, b_date]

    sinput = SemanticInput(document_id="test-iso", page_count=1, archetype=DocumentArchetype.ACADEMIC_CV, blocks=blocks)
    sec = SemanticSection("h1", "APPOINTMENTS", ["h1", "b1", "b2", "b3"], "experience", 1)

    groups = build_deterministic_appointment_groups(sec, sinput)
    assert len(groups) == 1
    grp = groups[0]

    assert grp.section_heading_text == "APPOINTMENTS"
    assert grp.section_heading_block_id == "h1"
    assert grp.canonical_target == "experience"
    assert grp.title_block_id == "b1"
    assert grp.block_ids == ["b1", "b2", "b3"]
    # Check object identity
    assert grp.blocks[0] is b_title
    assert grp.blocks[1] is b_org
    assert grp.blocks[2] is b_date


# =====================================================================
# 7. Gemini Appointment-Scoped Extraction & Provenance Integrity
# =====================================================================

def test_gemini_appointment_scoped_extraction_enforces_single_entity_and_prevents_cross_entity_provenance(monkeypatch):
    """Gemini extracts each appointment group independently, enforcing exactly 1 entity per group."""
    extractor = GeminiSemanticExtractor(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        two_pass=True,
    )
    blocks = [
        _block("h0", "Dr. Jane Doe", suggested_role="HEADER", region_kind="header", reading_order=0),
        _block("b1", "ACADEMIC APPOINTMENTS", suggested_role="SECTION_HEADING", reading_order=1),
        _block("b2", "Assistant Professor", suggested_role="ENTRY_TITLE", reading_order=2),
        _block("b3", "Carnegie Mellon University", suggested_role="ORGANIZATION", reading_order=3),
        _block("b4", "2020 - Present", suggested_role="DATE", reading_order=4),
        _block("b5", "Postdoctoral Researcher", suggested_role="ENTRY_TITLE", reading_order=5),
        _block("b6", "Cornell University", suggested_role="ORGANIZATION", reading_order=6),
        _block("b7", "2018 - 2020", suggested_role="DATE", reading_order=7),
    ]
    inp = SemanticInput(document_id="academic-doc-1", page_count=1, archetype=DocumentArchetype.ACADEMIC_CV, blocks=blocks)

    executed_passes: list[str] = []

    def mock_execute(prompt, response_schema, api_key, model, base_url, timeout, max_retries, pass_name):
        executed_passes.append(pass_name)
        if pass_name == "personal":
            data = {"personal": {"name": {"value": "Dr. Jane Doe", "source_block_ids": ["h0"]}}}
        elif pass_name == "body_appt_experience":
            if "Assistant Professor" in prompt:
                # Deliberately return multiple entities to test enforcement of exactly 1 entity
                data = {
                    "experience": [
                        {
                            "company": {"value": "Carnegie Mellon University", "source_block_ids": ["b3"]},
                            "designation": {"value": "Assistant Professor", "source_block_ids": ["b2"]},
                            "source_block_ids": ["b2", "b3", "b4"],
                        },
                        {
                            "company": {"value": "Spurious Extra Entity", "source_block_ids": ["b3"]},
                            "designation": {"value": "Extraneous Role", "source_block_ids": ["b2"]},
                            "source_block_ids": ["b2"],
                        },
                    ]
                }
            else:
                data = {
                    "experience": [
                        {
                            "company": {"value": "Cornell University", "source_block_ids": ["b6"]},
                            "designation": {"value": "Postdoctoral Researcher", "source_block_ids": ["b5"]},
                            "source_block_ids": ["b5", "b6", "b7"],
                        }
                    ]
                }
        else:
            data = {}
        return json.dumps(data), {"prompt_tokens": 50, "output_tokens": 50, "total_tokens": 100}, 0

    monkeypatch.setattr(extractor, "_execute_prompt_request", mock_execute)
    res = extractor.extract(inp)

    # 1. Verify two independent appointment extractions were executed
    appt_passes = [p for p in executed_passes if p == "body_appt_experience"]
    assert len(appt_passes) == 2

    # 2. Verify exactly 2 experience entities exist (extra spurious entity in appt 1 was purged)
    assert len(res.experience) == 2
    assert res.experience[0].designation.value == "Assistant Professor"
    assert res.experience[0].company.value == "Carnegie Mellon University"
    assert res.experience[0].source_block_ids == ["b2", "b3", "b4"]

    assert res.experience[1].designation.value == "Postdoctoral Researcher"
    assert res.experience[1].company.value == "Cornell University"
    assert res.experience[1].source_block_ids == ["b5", "b6", "b7"]

    # 3. Deterministic validation succeeds without cross-entity provenance errors
    violations = validate_semantic_output(res, inp)
    assert violations == []


# =====================================================================
# 8. Section-Aware Extraction Planning Parity & Section Isolation
# =====================================================================

def test_plan_section_aware_body_passes_isolates_physical_experience_sections():
    """plan_section_aware_body_passes isolates physical experience sections and plans 4 appointment units for postdoc CV."""
    from app.domain.semantic_contract import plan_section_aware_body_passes

    fpath = Path("tests/fixtures/generalization/academic_research_postdoc_cv.pdf")
    raw = fpath.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(layout_doc, document_id=fpath.name, archetype=DocumentArchetype.ACADEMIC_CV)

    units = plan_section_aware_body_passes(sinput)

    # Filter to appointment units
    appt_units = [u for u in units if u.is_appointment]
    assert len(appt_units) == 4

    # Appts 1 & 2 belong to RESEARCH EXPERIENCE
    assert appt_units[0].section_heading == "RESEARCH EXPERIENCE"
    assert appt_units[0].section_input.blocks[0].block_id == "b_p1_13"
    assert [b.block_id for b in appt_units[0].section_input.blocks] == ["b_p1_13", "b_p1_14", "b_p1_15", "b_p1_16", "b_p1_17"]

    assert appt_units[1].section_heading == "RESEARCH EXPERIENCE"
    assert appt_units[1].section_input.blocks[0].block_id == "b_p1_18"
    assert [b.block_id for b in appt_units[1].section_input.blocks] == ["b_p1_18", "b_p1_19", "b_p1_20", "b_p1_21", "b_p1_22"]

    # Appts 3 & 4 belong to TEACHING EXPERIENCE
    assert appt_units[2].section_heading == "TEACHING EXPERIENCE"
    assert appt_units[2].section_input.blocks[0].block_id == "b_p2_37"
    assert [b.block_id for b in appt_units[2].section_input.blocks] == ["b_p2_37", "b_p2_38", "b_p2_39"]

    assert appt_units[3].section_heading == "TEACHING EXPERIENCE"
    assert appt_units[3].section_input.blocks[0].block_id == "b_p2_40"
    assert [b.block_id for b in appt_units[3].section_input.blocks] == ["b_p2_40", "b_p2_41", "b_p2_42"]

    # Section headings do not appear in any appointment unit
    for u in appt_units:
        bids = [b.block_id for b in u.section_input.blocks]
        assert "b_p1_12" not in bids
        assert "b_p2_36" not in bids
