from __future__ import annotations

from pathlib import Path
import pytest

from app.domain.document import document_from_text_blocks
from app.domain.resume import Resume
from app.domain.semantic_contract import (
    GroundedEducationItem,
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedString,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    _is_qualifying_education_entry_title,
    build_semantic_input,
    has_explicit_skills_evidence,
    partition_semantic_input_into_sections,
    supplement_high_confidence_semantic_fields,
    validate_semantic_output,
)
from app.extractors.semantic_extractor import MockSemanticExtractor
from app.pipeline.semantic_pipeline import parse_document_semantically
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor


def _make_block(
    block_id: str,
    text: str,
    *,
    page: int = 1,
    reading_order: int = 0,
    region_kind: str = "physical_region",
    region_id: str = "r1",
    bbox: list[float] | None = None,
    suggested_role: str | None = None,
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        page=page,
        reading_order=reading_order,
        region_kind=region_kind,
        region_id=region_id,
        bbox=bbox or [10.0, float(reading_order * 15), 200.0, float(reading_order * 15 + 12)],
        text=text,
        suggested_role=suggested_role,
    )


def _make_input(blocks: list[SemanticBlockInput], page_count: int = 1) -> SemanticInput:
    return SemanticInput(document_id="doc1", blocks=blocks, page_count=page_count)


# --- Test A: Missing email + explicit header email => deterministic supplementation ---
def test_missing_email_with_header_email_is_supplemented():
    blocks = [
        _make_block("b1", "Jane Doe", page=1, reading_order=0, region_kind="header", suggested_role="HEADER"),
        _make_block("b2", "Email: jane.doe@example.com", page=1, reading_order=1, region_kind="header", suggested_role="CONTACT"),
    ]
    sem_input = _make_input(blocks)
    sem_output = SemanticOutput(personal=GroundedPersonal(email=None))

    output, diagnostics = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert output.personal.email is not None
    assert output.personal.email.value == "jane.doe@example.com"
    assert output.personal.email.source_block_ids == ["b2"]
    assert any(d.get("field") == "personal.email" for d in diagnostics)


# --- Test B: Existing semantic email => unchanged ---
def test_existing_semantic_email_remains_unchanged():
    blocks = [
        _make_block("b1", "Email: fallback@example.com", page=1, reading_order=0, region_kind="header", suggested_role="CONTACT"),
    ]
    sem_input = _make_input(blocks)
    existing = GroundedString(value="existing@company.com", source_block_ids=["b_orig"])
    sem_output = SemanticOutput(personal=GroundedPersonal(email=existing))

    output, diagnostics = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert output.personal.email.value == "existing@company.com"
    assert output.personal.email.source_block_ids == ["b_orig"]
    assert not any(d.get("field") == "personal.email" for d in diagnostics)


# --- Test C: Body email does not populate personal.email ---
def test_body_email_does_not_populate_personal_email():
    blocks = [
        _make_block("b1", "Jane Doe", page=1, reading_order=0, region_kind="header", suggested_role="HEADER"),
        _make_block("b2", "Software Engineer", page=1, reading_order=1, region_kind="physical_region", suggested_role="ENTRY_TITLE"),
        _make_block("b3", "Contact company hr at jobs@bigcorp.com", page=1, reading_order=2, region_kind="physical_region", suggested_role="DESCRIPTION"),
        _make_block("b4", "Page footer: support@univ.edu", page=1, reading_order=3, region_kind="footer", suggested_role="FOOTER"),
        _make_block("b5", "Email: page2@example.com", page=2, reading_order=0, region_kind="header", suggested_role="CONTACT"),
    ]
    sem_input = _make_input(blocks, page_count=2)
    sem_output = SemanticOutput(personal=GroundedPersonal(email=None))

    output, diagnostics = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert output.personal.email is None
    assert not any(d.get("field") == "personal.email" for d in diagnostics)


# --- Test D: Missing skills + explicit SKILL blocks => all explicit skills recovered ---
def test_missing_skills_with_explicit_skill_blocks_are_supplemented():
    blocks = [
        _make_block("b0", "Jane Doe", page=1, reading_order=0, region_kind="header", suggested_role="HEADER"),
        _make_block("b_h", "Technical Skills", page=1, reading_order=1, region_kind="physical_region", suggested_role="SECTION_HEADING"),
        _make_block("b_s1", "Python", page=1, reading_order=2, region_kind="physical_region", suggested_role="SKILL"),
        _make_block("b_s2", "TypeScript", page=1, reading_order=3, region_kind="physical_region", suggested_role="SKILL"),
        _make_block("b_s3", "Docker", page=1, reading_order=4, region_kind="physical_region", suggested_role="SKILL"),
    ]
    sem_input = _make_input(blocks)
    sem_output = SemanticOutput(skills=[])

    output, diagnostics = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert len(output.skills) == 3
    assert [s.value for s in output.skills] == ["Python", "TypeScript", "Docker"]
    assert [s.source_block_ids for s in output.skills] == [["b_s1"], ["b_s2"], ["b_s3"]]
    assert any(d.get("field") == "skills" for d in diagnostics)


# --- Test E: Existing semantic skills => unchanged ---
def test_existing_semantic_skills_remain_unchanged():
    blocks = [
        _make_block("b_h", "Skills", page=1, reading_order=0, region_kind="physical_region", suggested_role="SECTION_HEADING"),
        _make_block("b_s1", "Python", page=1, reading_order=1, region_kind="physical_region", suggested_role="SKILL"),
        _make_block("b_s2", "TypeScript", page=1, reading_order=2, region_kind="physical_region", suggested_role="SKILL"),
    ]
    sem_input = _make_input(blocks)
    existing_skills = [GroundedString(value="Existing Skill", source_block_ids=["b_orig"])]
    sem_output = SemanticOutput(skills=list(existing_skills))

    output, diagnostics = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert len(output.skills) == 1
    assert output.skills[0].value == "Existing Skill"
    assert output.skills[0].source_block_ids == ["b_orig"]
    assert not any(d.get("field") == "skills" for d in diagnostics)


# --- Test F: SKILL blocks outside skills evidence do not become skills ---
def test_skill_blocks_outside_skills_evidence_do_not_become_skills():
    blocks = [
        _make_block("b_h", "Client Engagements", page=1, reading_order=0, region_kind="physical_region", suggested_role="SECTION_HEADING"),
        _make_block("b_s1", "Python Development", page=1, reading_order=1, region_kind="physical_region", suggested_role="SKILL"),
    ]
    sem_input = _make_input(blocks)
    sem_output = SemanticOutput(skills=[])

    output, diagnostics = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert output.skills == []
    assert not any(d.get("field") == "skills" for d in diagnostics)


# --- Test G: Summary/experience/project text is never converted to skills ---
def test_summary_experience_text_never_converted_to_skills():
    blocks = [
        _make_block("b_h", "Professional Summary", page=1, reading_order=0, region_kind="physical_region", suggested_role="SECTION_HEADING"),
        _make_block("b1", "Expert in React, Python, and PostgreSQL.", page=1, reading_order=1, region_kind="physical_region", suggested_role="DESCRIPTION"),
        _make_block("b2", "●", page=1, reading_order=2, region_kind="physical_region", suggested_role="BULLET"),
        _make_block("b_exp", "Experience", page=1, reading_order=3, region_kind="physical_region", suggested_role="SECTION_HEADING"),
        _make_block("b3", "Software Engineer | Tech Corp", page=1, reading_order=4, region_kind="physical_region", suggested_role="ENTRY_TITLE"),
        _make_block("b4", "Built cloud microservices in Go and AWS.", page=1, reading_order=5, region_kind="physical_region", suggested_role="DESCRIPTION"),
    ]
    sem_input = _make_input(blocks)
    sem_output = SemanticOutput(skills=[])

    output, diagnostics = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert output.skills == []


# --- Test H: Exact source_block_ids are preserved ---
def test_exact_source_block_ids_are_preserved():
    blocks = [
        _make_block("b_email_10", "Email: test.user@domain.org", page=1, reading_order=0, region_kind="header", suggested_role="CONTACT"),
        _make_block("b_sec", "Skills", page=1, reading_order=1, region_kind="physical_region", suggested_role="SECTION_HEADING"),
        _make_block("b_skill_20", "Kubernetes", page=1, reading_order=2, region_kind="physical_region", suggested_role="SKILL"),
    ]
    sem_input = _make_input(blocks)
    sem_output = SemanticOutput(personal=GroundedPersonal(email=None), skills=[])

    output, _ = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert output.personal.email.source_block_ids == ["b_email_10"]
    assert output.skills[0].source_block_ids == ["b_skill_20"]


# --- Test I: Duplicate explicit skill text is deterministically deduplicated ---
def test_duplicate_explicit_skills_are_deduplicated():
    blocks = [
        _make_block("b_sec", "Key Skills", page=1, reading_order=0, region_kind="physical_region", suggested_role="SECTION_HEADING"),
        _make_block("b_s1", "Time Management", page=1, reading_order=1, region_kind="physical_region", suggested_role="SKILL"),
        _make_block("b_s2", "time management", page=1, reading_order=2, region_kind="physical_region", suggested_role="SKILL"),
        _make_block("b_s3", "  Time   Management  ", page=1, reading_order=3, region_kind="physical_region", suggested_role="SKILL"),
        _make_block("b_s4", "● Adaptability", page=1, reading_order=4, region_kind="physical_region", suggested_role="SKILL"),
        _make_block("b_s5", "Adaptability", page=1, reading_order=5, region_kind="physical_region", suggested_role="SKILL"),
    ]
    sem_input = _make_input(blocks)
    sem_output = SemanticOutput(skills=[])

    output, _ = supplement_high_confidence_semantic_fields(sem_output, sem_input)
    assert len(output.skills) == 2
    assert output.skills[0].value == "Time Management"
    assert output.skills[0].source_block_ids == ["b_s1"]
    assert output.skills[1].value == "Adaptability"
    assert output.skills[1].source_block_ids == ["b_s4"]


# --- Test J: Implicit education produces exactly two entries for fresher fixture ---
def test_fresher_fixture_implicit_education_entries():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        pytest.skip("Fixture fresher_hr_resume.pdf not found")

    doc = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes()))))
    sem_input = build_semantic_input(doc)
    blocks_by_id = {b.block_id: b for b in sem_input.blocks}

    # Verify qualifying education entry titles
    mba_block = blocks_by_id["b_p1_9"]
    ba_block = blocks_by_id["b_p1_21"]
    assert _is_qualifying_education_entry_title(mba_block) is True
    assert _is_qualifying_education_entry_title(ba_block) is True

    sections = partition_semantic_input_into_sections(sem_input)
    edu_sections = [s for s in sections if s.canonical_target == "education"]
    assert len(edu_sections) == 1
    edu_sec = edu_sections[0]
    assert "b_p1_9" in edu_sec.block_ids
    assert "b_p1_21" in edu_sec.block_ids


# --- Test K: Implicit education does not consume preceding summary prose as education evidence ---
def test_fresher_fixture_summary_prose_not_consumed_by_education():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        pytest.skip("Fixture fresher_hr_resume.pdf not found")

    doc = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes()))))
    sem_input = build_semantic_input(doc)
    sections = partition_semantic_input_into_sections(sem_input)

    # First section should be summary with blocks b_p1_5..b_p1_8
    assert sections[0].canonical_target == "summary"
    assert sections[0].block_ids == ["b_p1_5", "b_p1_6", "b_p1_7", "b_p1_8"]

    # Education section should start at b_p1_9
    assert sections[1].canonical_target == "education"
    assert sections[1].block_ids[0] == "b_p1_9"
    for summary_bid in ("b_p1_5", "b_p1_6", "b_p1_7", "b_p1_8"):
        assert summary_bid not in sections[1].block_ids


# --- Test L: Full offline end-to-end supplementation for fresher fixture ---
def test_fresher_fixture_end_to_end_supplementation():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        pytest.skip("Fixture fresher_hr_resume.pdf not found")

    doc = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes()))))

    # Mock extractor that omits email and skills
    class OmittedSemanticExtractor(MockSemanticExtractor):
        def extract(self, input_data: SemanticInput) -> SemanticOutput:
            return SemanticOutput(
                personal=GroundedPersonal(name=None, email=None),
                skills=[],
                experience=[],
                education=[],
            )

    resume = parse_document_semantically(
        doc, OmittedSemanticExtractor(), document_id="fresher-test", enable_supplementation=True
    )
    assert resume.personal.email == "aditianand136@gmail.com"
    assert len(resume.skills) == 14

    expected_skills = [
        "Recruitment & Selection Basics",
        "Employee Engagement Concepts",
        "Onboarding Process Understanding",
        "HR Policy Awareness",
        "Training & Development Support",
        "Basic Labor Law Knowledge",
        "MS Excel (VLOOKUP, Pivot Tables, Filters)",
        "Google Sheets & Docs",
        "HRMS (Basic understanding)",
        "Email & Calendar Management",
        "Strong Verbal & Written Communication",
        "Team Collaboration",
        "Time Management",
        "Adaptability & Willingness to Learn",
    ]
    assert resume.skills == expected_skills
