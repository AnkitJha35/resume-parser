"""Phase 10O: Focused regression tests for generic section heading detection,
structural-role precedence, and offline integration with generalization fixtures.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.domain.structural import StructuralRole
from app.pipeline.stages.layout import interpret_layout, _is_section_heading_line
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.structural_roles import (
    build_structural_blocks,
    classify_structural_role,
    _is_known_section_alias,
)
from app.pipeline.stages.text_extraction import PDFExtractor
from app.domain.semantic_contract import build_semantic_input, DocumentArchetype
from app.extractors.semantic_prompt import serialize_compact_semantic_input


def _make_line(
    line_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    font_size: float = 10.0,
    bold: bool = False,
    page: int = 1,
) -> Line:
    bbox = BoundingBox(x0, y0, x1, y1)
    span = Span(f"{line_id}-s0", text, bbox, font_size=font_size, bold=bold)
    return Line(
        line_id=line_id,
        page_number=page,
        bbox=bbox,
        spans=[span],
        text=text,
        style=TextStyle(font_size=font_size, bold=bold),
        source_span_ids=[span.span_id],
    )


def _make_doc(lines: list[Line]) -> Document:
    return Document(
        pages=[
            Page(
                page_number=1,
                regions=[
                    Region("r0", "physical_page", BoundingBox(0, 0, 600, 800), lines=lines)
                ],
            )
        ]
    )


def _header_lines(doc: Document) -> list[str]:
    return [
        l.text
        for p in doc.pages
        for r in p.regions
        if r.kind == "header"
        for l in r.lines
    ]


def _body_lines(doc: Document) -> list[str]:
    return [
        l.text
        for p in doc.pages
        for r in p.regions
        if r.kind != "header"
        for l in r.lines
    ]


# ===========================================================================
# LAYOUT TESTS (1 - 5)
# ===========================================================================

def test_1_one_word_section_heading_education_caps_body_start():
    """One-word 'EDUCATION' heading near top caps body_start and stays in body."""
    doc = _make_doc([
        _make_line("n", "John Doe", 50, 20, 200, 36, font_size=18),
        _make_line("c", "john@example.com | (555) 123-4567", 50, 42, 300, 54, font_size=10),
        # Single-word heading in normal/un-bolded 11pt font (same as fresher fixture)
        _make_line("h", "EDUCATION", 50, 70, 150, 82, font_size=11, bold=False),
        _make_line("u", "State University", 50, 90, 250, 102, font_size=10),
        _make_line("d", "2020 - 2024", 50, 108, 150, 120, font_size=10),
    ])
    layout = interpret_layout(doc)
    header = _header_lines(layout)
    body = _body_lines(layout)

    assert "John Doe" in header
    assert "john@example.com | (555) 123-4567" in header
    assert "EDUCATION" in body
    assert "EDUCATION" not in header
    assert "State University" in body
    assert "2020 - 2024" in body


def test_2_one_word_known_section_aliases_recognized():
    """Single-word section aliases are recognized as section headings by layout."""
    aliases = ["EDUCATION", "EXPERIENCE", "SKILLS", "PROJECTS", "SUMMARY"]
    median_size = 10.0
    for alias in aliases:
        line = _make_line("h", alias, 50, 100, 150, 112, font_size=11.0, bold=False)
        assert _is_section_heading_line(line, median_size), f"Expected {alias} to be recognized"


def test_3_one_word_uppercase_name_not_section_heading():
    """A single-word all-caps non-alias name/title must NOT become a section heading."""
    median_size = 10.0
    for non_heading in ["ALICE", "JOHN", "RESUME", "ENGINEERING", "REPORT"]:
        line = _make_line("w", non_heading, 50, 100, 150, 112, font_size=11.0, bold=False)
        assert not _is_section_heading_line(line, median_size), (
            f"Expected {non_heading} to NOT be recognized as section heading"
        )


def test_4_contact_and_header_lines_remain_header():
    """Contact lines and name lines are guarded and remain header."""
    doc = _make_doc([
        _make_line("n", "Ethan Zhao", 50, 20, 200, 36, font_size=20),
        _make_line("t", "Software Engineer | Graduate", 50, 42, 280, 54, font_size=10),
        _make_line("c", "Email: ethan@alumni.edu | Phone: +1 312 555 0143", 50, 60, 400, 72, font_size=8),
        _make_line("h", "SKILLS", 50, 95, 120, 107, font_size=11),
        _make_line("s", "Python, Go, SQL", 50, 115, 200, 127, font_size=10),
    ])
    layout = interpret_layout(doc)
    header = _header_lines(layout)
    body = _body_lines(layout)

    assert "Ethan Zhao" in header
    assert "Software Engineer | Graduate" in header
    assert "Email: ethan@alumni.edu | Phone: +1 312 555 0143" in header
    assert "SKILLS" in body
    assert "Python, Go, SQL" in body


def test_5_phase_10k_multi_word_headings_still_pass():
    """Phase 10K multi-word uppercase headings continue to cap body_start."""
    doc = _make_doc([
        _make_line("n", "Dr. Jane Doe", 50, 20, 200, 36, font_size=18),
        _make_line("c", "jane@harvard.edu", 50, 42, 200, 54, font_size=10),
        _make_line("h", "ACADEMIC APPOINTMENTS", 50, 70, 350, 84, font_size=13, bold=True),
        _make_line("p", "Professor with Tenure", 50, 92, 300, 106, font_size=10),
    ])
    layout = interpret_layout(doc)
    header = _header_lines(layout)
    body = _body_lines(layout)

    assert "Dr. Jane Doe" in header
    assert "jane@harvard.edu" in header
    assert "ACADEMIC APPOINTMENTS" in body
    assert "Professor with Tenure" in body


# ===========================================================================
# STRUCTURAL ROLE PRECEDENCE TESTS (6 - 13)
# ===========================================================================

def test_6_academic_open_source_projects_is_section_heading():
    role, _, _ = classify_structural_role("ACADEMIC & OPEN-SOURCE PROJECTS", font_size=11.0)
    assert role == StructuralRole.SECTION_HEADING


def test_7_honors_and_leadership_is_section_heading():
    role, _, _ = classify_structural_role("HONORS & LEADERSHIP", font_size=11.0)
    assert role == StructuralRole.SECTION_HEADING


def test_8_board_certifications_and_licensure_is_section_heading():
    role, _, _ = classify_structural_role("BOARD CERTIFICATIONS & LICENSURE", font_size=10.5, bold=False)
    assert role == StructuralRole.SECTION_HEADING


def test_9_clinical_and_pharmaceutical_experience_is_section_heading():
    role, _, _ = classify_structural_role("CLINICAL & PHARMACEUTICAL EXPERIENCE", font_size=10.5, bold=False)
    assert role == StructuralRole.SECTION_HEADING


def test_10_education_and_postdoctoral_training_is_section_heading():
    role, _, _ = classify_structural_role("EDUCATION & POSTDOCTORAL TRAINING", font_size=10.5, bold=False)
    assert role == StructuralRole.SECTION_HEADING


def test_11_attending_physician_and_clinical_investigator_is_not_section_heading():
    """Attending Physician & Clinical Investigator is an entry title, NOT a section heading."""
    following = ("2013 - 2019", "Dana-Farber Cancer Institute | Boston, MA")
    role, _, _ = classify_structural_role(
        "Attending Physician & Clinical Investigator",
        font_size=10.0,
        bold=False,
        following_texts=following,
    )
    assert role != StructuralRole.SECTION_HEADING
    assert role == StructuralRole.ENTRY_TITLE


def test_12_legitimate_organization_with_ampersand_is_organization():
    """A legitimate company/org containing '&' remains classified as ORGANIZATION."""
    for org in ["Johnson & Johnson", "Barnes & Noble", "Procter & Gamble", "Ernst & Young"]:
        role, _, _ = classify_structural_role(org, font_size=10.0)
        assert role == StructuralRole.ORGANIZATION, f"Expected {org} to be ORGANIZATION"


def test_13_wrapped_body_description_with_ampersand_remains_description():
    """A wrapped body description that contains '&' remains DESCRIPTION, not ORGANIZATION."""
    role, _, _ = classify_structural_role(
        "using Kafka & RabbitMQ for event streaming.",
        previous_text="Architected high-throughput telemetry ingestion service",
        font_size=8.5,
        bold=False,
    )
    assert role == StructuralRole.DESCRIPTION


# ===========================================================================
# OFFLINE INTEGRATION TESTS (REAL FIXTURES)
# ===========================================================================

def test_integration_entry_level_fresher_swe_offline():
    """Offline integration test for entry_level_fresher_swe.pdf proving:
    - EDUCATION is no longer in header region.
    - UIUC and its date remain body.
    - education evidence is visible to semantic input.
    - INTERNSHIP EXPERIENCE is SECTION_HEADING.
    - ACADEMIC & OPEN-SOURCE PROJECTS is SECTION_HEADING.
    - HONORS & LEADERSHIP is SECTION_HEADING.
    """
    pdf_path = Path("tests/fixtures/generalization/entry_level_fresher_swe.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Fixture {pdf_path} not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    blocks = build_structural_blocks(layout_doc)

    block_map = {b.text: b for b in blocks}

    # 1. EDUCATION is no longer in header region
    edu_block = block_map["EDUCATION"]
    assert edu_block.region_kind != "header"
    assert edu_block.role == StructuralRole.SECTION_HEADING

    # 2. UIUC and date remain body
    uiuc_block = block_map["University of Illinois Urbana-Champaign (UIUC)"]
    assert uiuc_block.region_kind != "header"
    date_block = block_map["2020 - 2024"]
    assert date_block.region_kind != "header"
    assert date_block.role == StructuralRole.DATE

    # 3. INTERNSHIP EXPERIENCE is SECTION_HEADING
    intern_block = block_map["INTERNSHIP EXPERIENCE"]
    assert intern_block.role == StructuralRole.SECTION_HEADING

    # 4. ACADEMIC & OPEN-SOURCE PROJECTS is SECTION_HEADING
    proj_block = block_map["ACADEMIC & OPEN-SOURCE PROJECTS"]
    assert proj_block.role == StructuralRole.SECTION_HEADING

    # 5. HONORS & LEADERSHIP is SECTION_HEADING
    honors_block = block_map["HONORS & LEADERSHIP"]
    assert honors_block.role == StructuralRole.SECTION_HEADING

    # 6. Education evidence is visible to semantic input Candidate B serialization
    sinput = build_semantic_input(layout_doc, document_id=pdf_path.name, archetype=DocumentArchetype.STANDARD_CV)
    payload = serialize_compact_semantic_input(sinput)
    assert '"text":"EDUCATION","role":"SECTION_HEADING"' in payload
    assert "University of Illinois Urbana-Champaign (UIUC)" in payload
    assert "2020 - 2024" in payload


def test_integration_scanned_clinical_specialist_offline():
    """Offline integration test for scanned_clinical_specialist.pdf proving:
    - BOARD CERTIFICATIONS & LICENSURE is SECTION_HEADING.
    - CLINICAL & PHARMACEUTICAL EXPERIENCE is SECTION_HEADING.
    - EDUCATION & POSTDOCTORAL TRAINING is SECTION_HEADING.
    - Attending Physician & Clinical Investigator is NOT SECTION_HEADING (is ENTRY_TITLE).
    """
    pdf_path = Path("tests/fixtures/generalization/scanned_clinical_specialist.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Fixture {pdf_path} not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    blocks = build_structural_blocks(layout_doc)

    block_map = {b.text: b for b in blocks}

    # 1. BOARD CERTIFICATIONS & LICENSURE is SECTION_HEADING
    cert_block = block_map["BOARD CERTIFICATIONS & LICENSURE"]
    assert cert_block.role == StructuralRole.SECTION_HEADING

    # 2. CLINICAL & PHARMACEUTICAL EXPERIENCE is SECTION_HEADING
    clin_block = block_map["CLINICAL & PHARMACEUTICAL EXPERIENCE"]
    assert clin_block.role == StructuralRole.SECTION_HEADING

    # 3. EDUCATION & POSTDOCTORAL TRAINING is SECTION_HEADING
    edu_block = block_map["EDUCATION & POSTDOCTORAL TRAINING"]
    assert edu_block.role == StructuralRole.SECTION_HEADING

    # 4. Attending Physician & Clinical Investigator is NOT SECTION_HEADING
    job_block = block_map["Attending Physician & Clinical Investigator"]
    assert job_block.role != StructuralRole.SECTION_HEADING
    assert job_block.role == StructuralRole.ENTRY_TITLE


# ===========================================================================
# PHASE 10O-FIX FALSE-POSITIVE PREVENTION REGRESSION TESTS (14 - 17)
# ===========================================================================

def test_14_section_headings_recognized():
    """Verify standard, compound, and qualified section headings are SECTION_HEADING."""
    headings = [
        "EDUCATION",
        "EXPERIENCE",
        "SKILLS",
        "PROJECTS",
        "PROFESSIONAL EXPERIENCE",
        "TECHNICAL SKILLS",
        "HONORS & LEADERSHIP",
        "EDUCATION & TRAINING",
        "CLINICAL & PHARMACEUTICAL EXPERIENCE",
        "ACADEMIC & OPEN-SOURCE PROJECTS",
    ]
    for h in headings:
        role, _, _ = classify_structural_role(h, font_size=11.0)
        assert role == StructuralRole.SECTION_HEADING, f"Expected {h} to be SECTION_HEADING, got {role}"


def test_15_organizations_with_section_keywords_or_suffixes_are_organization():
    """Verify organizations with section keywords or corporate suffixes are classified as ORGANIZATION."""
    orgs = [
        "APPLIED RESEARCH LABORATORIES",
        "RESEARCH TRIANGLE INSTITUTE",
        "GLOBAL LEADERSHIP FORUM",
        "AMERICAN CANCER RESEARCH FOUNDATION",
        "NATIONAL EDUCATION ALLIANCE",
        "Workday, Inc.",
        "Microsoft Corp.",
        "General Assembly Training LLC",
        "Education First Ltd",
    ]
    for o in orgs:
        role, _, _ = classify_structural_role(o, font_size=10.0)
        assert role == StructuralRole.ORGANIZATION, f"Expected {o} to be ORGANIZATION, got {role}"


def test_16_job_titles_with_section_keywords_are_not_section_heading():
    """Verify job titles containing section keywords are NOT classified as SECTION_HEADING."""
    jobs = [
        ("RESEARCH SCIENTIST", ()),
        ("DIRECTOR OF RESEARCH", ()),
        ("TRAINING SPECIALIST", ()),
        ("Attending Physician & Clinical Investigator", ("2013 - 2019", "Dana-Farber Cancer Institute | Boston, MA")),
    ]
    for j, following in jobs:
        role, _, _ = classify_structural_role(j, font_size=10.0, following_texts=following)
        assert role != StructuralRole.SECTION_HEADING, f"Expected {j} NOT to be SECTION_HEADING, got {role}"
        assert role == StructuralRole.ENTRY_TITLE, f"Expected {j} to be ENTRY_TITLE, got {role}"


def test_17_body_descriptions_are_not_section_heading():
    """Verify normal sentences containing section keywords and wrapped lines with '&' are NOT SECTION_HEADING."""
    descriptions = [
        ("Conducted academic research on scalable distributed architectures.", None),
        ("Completed graduate education in computational linguistics and AI.", None),
        ("Led weekly training workshops on continuous delivery and testing.", None),
        ("using Kafka & RabbitMQ for event streaming.", "Architected high-throughput telemetry ingestion service"),
    ]
    for desc, prev in descriptions:
        role, _, _ = classify_structural_role(desc, font_size=8.5, previous_text=prev)
        assert role != StructuralRole.SECTION_HEADING, f"Expected description to NOT be SECTION_HEADING: {desc}"

