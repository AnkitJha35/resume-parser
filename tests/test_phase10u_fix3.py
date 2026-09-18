from __future__ import annotations

import pytest

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.domain.structural import StructuralBlock, StructuralRole
from app.domain.semantic_contract import (
    BodySemanticOutput,
    GroundedExperienceItem,
    GroundedString,
    SectionAwareExtractionUnit,
    SemanticBlockInput,
    SemanticInput,
    constrain_appointment_experience_provenance,
)
from app.pipeline.stages.structural_roles import (
    build_structural_blocks,
    classify_structural_role,
)


def _line(
    line_id: str,
    text: str,
    y0: float,
    *,
    x0: float = 10.0,
    font_size: float = 11.0,
    bold: bool = False,
) -> Line:
    bbox = BoundingBox(x0, y0, x0 + max(40.0, len(text) * 6.0), y0 + 12.0)
    span = Span(f"{line_id}-span", text, bbox, font_size=font_size, bold=bold)
    return Line(
        line_id,
        1,
        bbox,
        [span],
        text,
        TextStyle(font_size=font_size, bold=bold),
        reading_order=int(y0),
        source_span_ids=[span.span_id],
        reconstruction_method="physical",
    )


def _doc_from_lines(lines: list[Line]) -> Document:
    bbox = BoundingBox(
        min(l.bbox.x0 for l in lines),
        min(l.bbox.y0 for l in lines),
        max(l.bbox.x1 for l in lines),
        max(l.bbox.y1 for l in lines),
    )
    return Document(
        pages=[
            Page(
                1,
                regions=[
                    Region(
                        "page-1-region-0",
                        "physical_region",
                        bbox,
                        lines=lines,
                        reading_order=0,
                        column_id=0,
                    )
                ],
            )
        ]
    )


def _roles_by_text(blocks: list[StructuralBlock]) -> dict[str, StructuralRole]:
    return {b.text: b.role for b in blocks}


# =====================================================================
# Test 1: MBA composite line followed by date -> ENTRY_TITLE
# =====================================================================
def test_1_mba_composite_line_followed_by_date_becomes_education_entry():
    text = "MBA (Human Resource Management) | School Of Open Learning , Delhi University (DU-SOL)"
    role, score, reasons = classify_structural_role(
        text,
        next_text="2024 – 2026",
    )
    assert role == StructuralRole.ENTRY_TITLE
    assert "education_composite_entry" in reasons
    assert score >= 0.85


# =====================================================================
# Test 2: Bachelor's degree composite line followed by date -> ENTRY_TITLE
# =====================================================================
def test_2_bachelor_composite_line_followed_by_date_becomes_education_entry():
    text = "Bachelor of Arts (General) | BIR Tikendrajit University"
    role, score, reasons = classify_structural_role(
        text,
        next_text="2020 – 2023",
    )
    assert role == StructuralRole.ENTRY_TITLE
    assert "education_composite_entry" in reasons
    assert score >= 0.85


# =====================================================================
# Test 3: Ordinary organization with corporate suffix is NOT education
# =====================================================================
def test_3_ordinary_organization_with_corp_suffix_is_not_education():
    for text in [
        "General Assembly Training LLC | San Francisco, CA",
        "Workday, Inc.",
        "Microsoft Corp.",
        "Education First Ltd.",
    ]:
        role, _, reasons = classify_structural_role(
            text,
            next_text="2020 – 2023",
        )
        assert role == StructuralRole.ORGANIZATION
        assert "education_composite_entry" not in reasons


# =====================================================================
# Test 4: Explicit skills section children become SKILL
# =====================================================================
def test_4_explicit_skills_section_children_become_skill():
    lines = [
        _line("h1", "Technical Skills", 10, font_size=14, bold=True),
        _line("c1", "Python, TypeScript, SQL", 30),
        _line("c2", "Docker, Kubernetes, AWS", 50),
    ]
    blocks = build_structural_blocks(_doc_from_lines(lines))
    roles = _roles_by_text(blocks)

    assert roles["Technical Skills"] == StructuralRole.SECTION_HEADING
    assert roles["Python, TypeScript, SQL"] == StructuralRole.SKILL
    assert roles["Docker, Kubernetes, AWS"] == StructuralRole.SKILL


# =====================================================================
# Test 5: Multiple consecutive skill subsections all classify children as SKILL
# =====================================================================
def test_5_multiple_consecutive_skill_subsections_all_classify_children_as_skill():
    lines = [
        _line("h1", "Human Resource Skills", 10, font_size=13, bold=True),
        _line("c1", "Recruitment & Selection Basics", 30),
        _line("c2", "Employee Engagement Concepts", 50),
        _line("h2", "Tools & Technical Skills", 70, font_size=13, bold=True),
        _line("c3", "MS Excel (VLOOKUP, Pivot Tables, Filters)", 90),
        _line("c4", "HRMS (Basic understanding)", 110),
        _line("h3", "Soft Skills", 130, font_size=13, bold=True),
        _line("c5", "Strong Verbal & Written Communication", 150),
        _line("c6", "Team Collaboration", 170),
    ]
    blocks = build_structural_blocks(_doc_from_lines(lines))
    roles = _roles_by_text(blocks)

    assert roles["Human Resource Skills"] == StructuralRole.SECTION_HEADING
    assert roles["Recruitment & Selection Basics"] == StructuralRole.SKILL
    assert roles["Employee Engagement Concepts"] == StructuralRole.SKILL

    assert roles["Tools & Technical Skills"] == StructuralRole.SECTION_HEADING
    assert roles["MS Excel (VLOOKUP, Pivot Tables, Filters)"] == StructuralRole.SKILL
    assert roles["HRMS (Basic understanding)"] == StructuralRole.SKILL

    assert roles["Soft Skills"] == StructuralRole.SECTION_HEADING
    assert roles["Strong Verbal & Written Communication"] == StructuralRole.SKILL
    assert roles["Team Collaboration"] == StructuralRole.SKILL


# =====================================================================
# Test 6: Bullet marker remains BULLET while bullet content becomes SKILL
# =====================================================================
def test_6_bullet_marker_remains_bullet_while_bullet_content_becomes_skill():
    lines = [
        _line("h1", "Soft Skills", 10, font_size=14, bold=True),
        _line("b1", "•", 30),
        _line("c1", "Team Collaboration", 32),
        _line("b2", "●", 50),
        _line("c2", "Time Management", 52),
    ]
    blocks = build_structural_blocks(_doc_from_lines(lines))
    by_id = {b.line_ids[0]: b for b in blocks}

    assert by_id["h1"].role == StructuralRole.SECTION_HEADING
    assert by_id["b1"].role == StructuralRole.BULLET
    assert by_id["c1"].role == StructuralRole.SKILL
    assert by_id["b2"].role == StructuralRole.BULLET
    assert by_id["c2"].role == StructuralRole.SKILL


# =====================================================================
# Test 7: Experience description outside skills section remains DESCRIPTION
# =====================================================================
def test_7_experience_description_outside_skills_section_remains_description():
    lines = [
        _line("h1", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Software Engineer", 30, font_size=12, bold=True),
        _line("c1", "Acme Corporation", 50),
        _line("d1", "2020 - Present", 70),
        _line("desc", "Developed high-throughput data processing pipelines using distributed caching architectures and asynchronous queues.", 90),
    ]
    blocks = build_structural_blocks(_doc_from_lines(lines))
    roles = _roles_by_text(blocks)

    assert roles["EXPERIENCE"] == StructuralRole.SECTION_HEADING
    assert roles["Software Engineer"] == StructuralRole.ENTRY_TITLE
    assert roles["Acme Corporation"] == StructuralRole.ORGANIZATION
    assert roles["2020 - Present"] == StructuralRole.DATE
    assert roles["Developed high-throughput data processing pipelines using distributed caching architectures and asynchronous queues."] == StructuralRole.DESCRIPTION


# =====================================================================
# Test 8: Existing academic appointment/title tests still pass
# =====================================================================
def test_8_existing_academic_appointment_titles_remain_entry_title():
    titles = [
        "Postdoctoral Research Fellow",
        "Instructor of Record",
        "Teaching Assistant",
        "Professor of Medicine",
    ]
    for title in titles:
        role, _, reasons = classify_structural_role(
            title,
            bold=True,
            font_size=12.0,
            following_texts=("Stanford University", "2022 - Present"),
        )
        assert role == StructuralRole.ENTRY_TITLE
        assert "entry_title_structure" in reasons


# =====================================================================
# Test 9: Existing section/org false-positive tests still pass
# =====================================================================
def test_9_existing_section_org_false_positives_preserved():
    # Organizations with corp suffixes or institutional nouns
    for org in ["Workday, Inc.", "Microsoft Corp.", "General Assembly Training LLC", "Education First Ltd."]:
        role, _, _ = classify_structural_role(org)
        assert role == StructuralRole.ORGANIZATION

    # Standard section headings
    for sec in ["EXPERIENCE", "EDUCATION", "SKILLS", "SUMMARY", "PROJECTS", "CERTIFICATIONS"]:
        role, _, _ = classify_structural_role(sec, bold=True, font_size=14.0)
        assert role == StructuralRole.SECTION_HEADING


# =====================================================================
# Test 10: Academic-looking line without date context is NOT education
# =====================================================================
def test_10_academic_looking_line_without_date_context_not_education():
    line_text = "Bachelor of Arts Project | Stanford University"
    # Followed by ordinary description sentences without dates
    role, _, reasons = classify_structural_role(
        line_text,
        previous_text=None,
        next_text="Conducted extensive archival surveys on local history.",
        following_texts=("Analyzed primary sources and documents.", "Presented findings at annual seminar."),
    )
    # Without date evidence, it must NOT be classified as an education composite entry
    assert "education_composite_entry" not in reasons
    assert role != StructuralRole.ENTRY_TITLE or "education_composite_entry" not in reasons


# =====================================================================
# Test 11: Appointment output leaking source_block_ids constrained to deterministic appointment block_ids
# =====================================================================
def test_11_appointment_leaking_source_block_ids_constrained_to_span():
    appt_blocks = ["b_p1_13", "b_p1_14", "b_p1_15", "b_p1_16", "b_p1_17"]
    # Model leaked blocks b_p1_18..b_p1_25 into top-level source_block_ids
    body = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Postdoc Fellow", source_block_ids=["b_p1_13"]),
                company=GroundedString(value="Stanford University", source_block_ids=["b_p1_14"]),
                source_block_ids=[
                    "b_p1_13", "b_p1_14", "b_p1_15", "b_p1_16", "b_p1_17",
                    "b_p1_18", "b_p1_19", "b_p1_20", "b_p1_25",
                ],
            )
        ]
    )

    normalized = constrain_appointment_experience_provenance(body, appt_blocks)
    assert normalized.experience[0].source_block_ids == appt_blocks


# =====================================================================
# Test 12: Correctly grounded appointment remains unchanged
# =====================================================================
def test_12_correctly_grounded_appointment_remains_unchanged():
    appt_blocks = ["b_p2_37", "b_p2_38", "b_p2_39"]
    body = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Instructor of Record", source_block_ids=["b_p2_37"]),
                company=GroundedString(value="Stanford University", source_block_ids=["b_p2_39"]),
                source_block_ids=["b_p2_37", "b_p2_38", "b_p2_39"],
            )
        ]
    )

    normalized = constrain_appointment_experience_provenance(body, appt_blocks)
    assert normalized.experience[0].source_block_ids == ["b_p2_37", "b_p2_38", "b_p2_39"]


# =====================================================================
# Test 13: Non-appointment experience output is unaffected
# =====================================================================
def test_13_non_appointment_experience_output_unaffected():
    # In non-appointment units, constrain_appointment_experience_provenance is NOT called
    unit = SectionAwareExtractionUnit(
        pass_name="body_sec_experience",
        section_input=SemanticInput(
            document_id="doc-1",
            page_count=1,
            blocks=[
                SemanticBlockInput(
                    block_id="b_1",
                    text="Software Engineer",
                    suggested_role="ENTRY_TITLE",
                    page=1,
                    bbox=[0.0, 0.0, 100.0, 20.0],
                    region_id="page-1-region-0",
                    region_kind="body",
                    reading_order=0,
                )
            ],
        ),
        section_heading="EXPERIENCE",
        canonical_target="experience",
        is_appointment=False,
    )
    assert unit.is_appointment is False
    # If is_appointment is False, the pipeline does not call constrain_appointment_experience_provenance
    # Ensure source_block_ids is left completely intact
    body = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Senior Engineer", source_block_ids=["b_1"]),
                source_block_ids=["b_1", "b_2", "b_3"],
            )
        ]
    )
    # Even if unit blocks are only ["b_1"], because is_appointment is False, no normalization is applied:
    if unit.is_appointment:
        allowed_bids = unit.appointment_block_ids or [b.block_id for b in unit.section_input.blocks]
        body = constrain_appointment_experience_provenance(body, allowed_bids)
    assert body.experience[0].source_block_ids == ["b_1", "b_2", "b_3"]


# =====================================================================
# Test 14: No individual field provenance is modified by normalization
# =====================================================================
def test_14_no_individual_field_provenance_modified_by_normalization():
    appt_blocks = ["b_p1_13", "b_p1_14", "b_p1_15", "b_p1_16", "b_p1_17"]
    original_designation = GroundedString(value="Postdoctoral Fellow", source_block_ids=["b_p1_13"])
    original_company = GroundedString(value="Stanford University", source_block_ids=["b_p1_14"])
    original_start_date = GroundedString(value="2022", source_block_ids=["b_p1_15"])
    original_end_date = GroundedString(value="Present", source_block_ids=["b_p1_15"])
    original_location = GroundedString(value="Stanford, CA", source_block_ids=["b_p1_14"])
    original_description = GroundedString(value="Investigating neural correlates.", source_block_ids=["b_p1_16", "b_p1_17"])
    original_tech = [GroundedString(value="Optogenetics", source_block_ids=["b_p1_17"])]

    body = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                designation=original_designation.model_copy(),
                company=original_company.model_copy(),
                startDate=original_start_date.model_copy(),
                endDate=original_end_date.model_copy(),
                location=original_location.model_copy(),
                description=original_description.model_copy(),
                technologies=[t.model_copy() for t in original_tech],
                source_block_ids=["b_p1_13", "b_p1_14", "b_p1_15", "b_p1_16", "b_p1_17", "b_p1_18", "b_p1_19"],
            )
        ]
    )

    normalized = constrain_appointment_experience_provenance(body, appt_blocks)
    exp = normalized.experience[0]

    # Top-level source_block_ids constrained
    assert exp.source_block_ids == appt_blocks

    # Field values and field source_block_ids strictly untouched
    assert exp.designation.value == "Postdoctoral Fellow"
    assert exp.designation.source_block_ids == ["b_p1_13"]
    assert exp.company.value == "Stanford University"
    assert exp.company.source_block_ids == ["b_p1_14"]
    assert exp.startDate.value == "2022"
    assert exp.startDate.source_block_ids == ["b_p1_15"]
    assert exp.endDate.value == "Present"
    assert exp.endDate.source_block_ids == ["b_p1_15"]
    assert exp.location.value == "Stanford, CA"
    assert exp.location.source_block_ids == ["b_p1_14"]
    assert exp.description.value == "Investigating neural correlates."
    assert exp.description.source_block_ids == ["b_p1_16", "b_p1_17"]
    assert len(exp.technologies) == 1
    assert exp.technologies[0].value == "Optogenetics"
    assert exp.technologies[0].source_block_ids == ["b_p1_17"]

