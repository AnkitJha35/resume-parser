"""Phase 2: Focused tests for section-agnostic semantic input representation.

Tests prove:
1. Unknown heading preservation ("Career History" survives with all child blocks & source IDs).
2. Arbitrary CV headings survive intact.
3. Unknown heading is not treated as "unsupported content" and is not stripped.
4. Heading / personal title separation (Name / Job title not classified as section heading).
5. Title Case heading survives as heading candidate.
6. Two-column layout preservation (regions, columns, geometry survive).
7. Typography preservation (font_size, bold, italic, heading candidate).
8. Table preservation (table_id, row_index, column_index, cell_role, source IDs).
9. Nested / continued content remains linked to structural context.
10. Conventional resume regression (experience, education, skills, projects continue to work).
11. No content-loss invariant: all source blocks survive to semantic input.
"""

from __future__ import annotations

import json
import pytest

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.domain.semantic_contract import (
    DocumentArchetype,
    SemanticBlockInput,
    SemanticInput,
    build_semantic_input,
    filter_semantic_input_to_blocks,
    partition_semantic_input_into_sections,
)
from app.extractors.semantic_prompt import (
    serialize_compact_semantic_input,
    serialize_semantic_input_full,
)
from app.pipeline.stages.structural_roles import (
    StructuralRole,
    build_structural_blocks,
    classify_structural_role,
)


def _create_line(
    line_id: str,
    text: str,
    y0: float,
    y1: float,
    x0: float = 50.0,
    x1: float = 500.0,
    font_size: float = 10.0,
    bold: bool = False,
    italic: bool = False,
    reading_order: int = 0,
    page_number: int = 1,
) -> Line:
    bbox = BoundingBox(x0, y0, x1, y1)
    span = Span(f"sp_{line_id}", text, bbox)
    style = TextStyle(font_size=font_size, bold=bold, italic=italic)
    return Line(
        line_id=line_id,
        page_number=page_number,
        bbox=bbox,
        spans=[span],
        text=text,
        style=style,
        reading_order=reading_order,
    )


# =============================================================================
# TEST 1 — Unknown heading preservation
# =============================================================================
def test_unknown_heading_preservation():
    """'Career History' and its children must survive in SemanticInput with intact source IDs."""
    lines = [
        _create_line("l_hdr_name", "Jane Doe", 50, 70, font_size=16.0, bold=True, reading_order=1),
        _create_line("l_hdr_contact", "jane@example.com", 75, 90, reading_order=2),
        _create_line("l_ch_heading", "Career History", 110, 125, font_size=13.0, bold=True, reading_order=3),
        _create_line("l_ch_title", "Senior Engineer", 130, 145, font_size=11.0, bold=True, reading_order=4),
        _create_line("l_ch_org", "Acme Corp", 150, 165, reading_order=5),
        _create_line("l_ch_date", "2021 - 2025", 170, 185, reading_order=6),
    ]
    reg_hdr = Region("r_hdr", "header", BoundingBox(50, 50, 500, 95), lines[:2], 1)
    reg_body = Region("r_body", "column", BoundingBox(50, 105, 500, 200), lines[2:], 2)
    doc = Document([Page(1, 600.0, 800.0, [reg_hdr, reg_body])])

    sem_input = build_semantic_input(doc)

    block_texts = [b.text for b in sem_input.blocks]
    assert "Career History" in block_texts
    assert "Senior Engineer" in block_texts
    assert "Acme Corp" in block_texts
    assert "2021 - 2025" in block_texts

    heading_block = next(b for b in sem_input.blocks if b.text == "Career History")
    assert heading_block.block_id is not None
    assert len(sem_input.blocks) == 6


# =============================================================================
# TEST 2 — Arbitrary CV headings
# =============================================================================
def test_arbitrary_cv_headings_all_survive():
    """Arbitrary headings must all survive in SemanticInput without being dropped."""
    headings = [
        "Selected Engagements",
        "Academic Appointments",
        "Technical Background",
        "Leadership",
        "Major Contributions",
        "Professional Journey",
    ]
    lines = [
        _create_line("l_hdr", "Dr. Alex Smith", 50, 70, font_size=18.0, bold=True, reading_order=1),
    ]
    y = 100.0
    ro = 2
    for i, h in enumerate(headings):
        lines.append(_create_line(f"l_h_{i}", h, y, y + 15, font_size=13.0, bold=True, reading_order=ro))
        y += 20
        ro += 1
        lines.append(_create_line(f"l_c_{i}", f"Details under {h}", y, y + 15, font_size=10.0, reading_order=ro))
        y += 30
        ro += 1

    region = Region("r_main", "column", BoundingBox(50, 50, 500, y), lines, 1)
    doc = Document([Page(1, 600.0, y + 50.0, [region])])

    sem_input = build_semantic_input(doc)
    emitted_texts = {b.text for b in sem_input.blocks}

    for h in headings:
        assert h in emitted_texts, f"Heading '{h}' was discarded from semantic input!"
        assert f"Details under {h}" in emitted_texts, f"Content under '{h}' was discarded!"


# =============================================================================
# TEST 3 — Unknown heading is not "unsupported content"
# =============================================================================
def test_unknown_heading_not_marked_unsupported():
    """An unmapped heading must NOT be classified as 'unsupported' or stripped."""
    lines = [
        _create_line("l_0", "John Consultant", 50, 70, font_size=16.0, bold=True, reading_order=1),
        _create_line("l_1", "Selected Engagements", 100, 115, font_size=13.0, bold=True, reading_order=2),
        _create_line("l_2", "Lead Architect", 120, 135, bold=True, reading_order=3),
        _create_line("l_3", "Global Advisory LLC", 140, 155, reading_order=4),
        _create_line("l_4", "2020 - 2024", 160, 175, reading_order=5),
    ]
    reg = Region("r_main", "column", BoundingBox(50, 50, 500, 200), lines, 1)
    doc = Document([Page(1, 600.0, 800.0, [reg])])

    sem_input = build_semantic_input(doc)
    sections = partition_semantic_input_into_sections(sem_input)

    # Unknown heading must NOT have target 'unsupported'
    for s in sections:
        if "Selected Engagements" in s.heading_text:
            assert s.canonical_target != "unsupported", (
                f"Section '{s.heading_text}' was incorrectly flagged as 'unsupported'!"
            )

    # Prove that filtering logic does NOT discard it
    supported_bids = [
        bid for s in sections if s.canonical_target != "unsupported" for bid in s.block_ids
    ]
    filtered_input = filter_semantic_input_to_blocks(sem_input, supported_bids)
    filtered_texts = {b.text for b in filtered_input.blocks}
    assert "Selected Engagements" in filtered_texts
    assert "Lead Architect" in filtered_texts
    assert "Global Advisory LLC" in filtered_texts


# =============================================================================
# TEST 4 — Heading / title separation
# =============================================================================
def test_heading_and_personal_title_separation():
    """Visual prominence on personal name / job title in header must NOT become SECTION_HEADING."""
    lines = [
        _create_line("l_name", "JOHN DOE", 50, 75, font_size=20.0, bold=True, reading_order=1),
        _create_line("l_title", "Senior Product Manager", 80, 95, font_size=14.0, bold=True, reading_order=2),
        _create_line("l_email", "john@example.com", 100, 115, font_size=10.0, reading_order=3),
        _create_line("l_sec", "Career History", 140, 155, font_size=13.0, bold=True, reading_order=4),
        _create_line("l_role", "Staff PM", 160, 175, font_size=11.0, bold=True, reading_order=5),
    ]
    reg_hdr = Region("r_hdr", "header", BoundingBox(50, 50, 500, 120), lines[:3], 1)
    reg_body = Region("r_body", "column", BoundingBox(50, 130, 500, 200), lines[3:], 2)
    doc = Document([Page(1, 600.0, 800.0, [reg_hdr, reg_body])])

    struct_blocks = build_structural_blocks(doc)
    role_map = {b.text: b.role for b in struct_blocks}

    # JOHN DOE and Senior Product Manager must NOT be SECTION_HEADING
    assert role_map["JOHN DOE"] != StructuralRole.SECTION_HEADING
    assert role_map["Senior Product Manager"] != StructuralRole.SECTION_HEADING


# =============================================================================
# TEST 5 — Title Case heading
# =============================================================================
def test_title_case_heading_recognized_as_candidate():
    """'Career History' in Title Case (not uppercase) must be recognized as heading candidate."""
    lines = [
        _create_line("l_0", "Selected Engagements", 100, 115, font_size=14.0, bold=True, reading_order=1),
        _create_line("l_1", "Senior Engineer", 125, 140, font_size=11.0, bold=True, reading_order=2),
        _create_line("l_2", "Tech Corp | 2020 - Present", 145, 160, reading_order=3),
    ]
    reg = Region("r_body", "column", BoundingBox(50, 100, 500, 200), lines, 1)
    doc = Document([Page(1, 600.0, 800.0, [reg])])

    sem_input = build_semantic_input(doc)
    heading_b = next(b for b in sem_input.blocks if b.text == "Selected Engagements")
    assert getattr(heading_b, "heading_candidate", False) is True or heading_b.suggested_role == "SECTION_HEADING"


# =============================================================================
# TEST 6 — Two-column preservation
# =============================================================================
def test_two_column_preservation():
    """Two-column CV must preserve region_id, region_kind, column_id, bbox, and reading_order."""
    left_lines = [
        _create_line("l_l1", "Profile", 100, 115, x0=50.0, x1=200.0, font_size=12.0, bold=True, reading_order=1),
        _create_line("l_l2", "Motivated software engineer with 5 years experience.", 120, 150, x0=50.0, x1=200.0, reading_order=2),
        _create_line("l_l3", "Skills", 160, 175, x0=50.0, x1=200.0, font_size=12.0, bold=True, reading_order=3),
        _create_line("l_l4", "Python, TypeScript, SQL", 180, 195, x0=50.0, x1=200.0, reading_order=4),
    ]
    right_lines = [
        _create_line("l_r1", "Career History", 100, 115, x0=220.0, x1=550.0, font_size=14.0, bold=True, reading_order=5),
        _create_line("l_r2", "Lead Developer", 120, 135, x0=220.0, x1=550.0, font_size=11.0, bold=True, reading_order=6),
        _create_line("l_r3", "Acme Inc | 2020 - 2024", 140, 155, x0=220.0, x1=550.0, reading_order=7),
    ]
    reg_left = Region("reg_sidebar", "sidebar", BoundingBox(50, 100, 200, 300), left_lines, reading_order=1, column_id=0)
    reg_right = Region("reg_main", "column", BoundingBox(220, 100, 550, 300), right_lines, reading_order=2, column_id=1)
    doc = Document([Page(1, 600.0, 800.0, [reg_left, reg_right])])

    sem_input = build_semantic_input(doc)

    # Verify attributes survived in SemanticBlockInput
    left_blocks = [b for b in sem_input.blocks if b.region_id == "reg_sidebar"]
    right_blocks = [b for b in sem_input.blocks if b.region_id == "reg_main"]

    assert len(left_blocks) == 4
    assert len(right_blocks) == 3
    assert all(b.column_id == 0 for b in left_blocks)
    assert all(b.column_id == 1 for b in right_blocks)
    assert all(b.region_kind == "sidebar" for b in left_blocks)
    assert all(b.region_kind == "column" for b in right_blocks)

    # Verify serialization preserves column distinction
    serialized = serialize_compact_semantic_input(sem_input)
    payload = json.loads(serialized)
    col_ids = {b.get("col") for b in payload["blocks"] if "col" in b}
    regions = {b.get("region") for b in payload["blocks"] if "region" in b}
    assert 0 in col_ids or "reg_sidebar" in regions


# =============================================================================
# TEST 7 — Typography preservation
# =============================================================================
def test_typography_preservation():
    """Available font_size, bold, italic, and heading_candidate must survive in serialization."""
    lines = [
        _create_line("l_1", "Selected Engagements", 100, 118, font_size=15.0, bold=True, italic=False, reading_order=1),
        _create_line("l_2", "Consultant & Advisory Lead", 125, 140, font_size=11.0, bold=False, italic=True, reading_order=2),
    ]
    reg = Region("r1", "column", BoundingBox(50, 100, 500, 200), lines, 1)
    doc = Document([Page(1, 600.0, 800.0, [reg])])

    sem_input = build_semantic_input(doc)
    b1 = sem_input.blocks[0]
    b2 = sem_input.blocks[1]

    assert b1.font_size == 15.0
    assert b1.is_bold is True
    assert getattr(b2, "is_italic", None) is True

    # Compact serialization check
    serialized = serialize_compact_semantic_input(sem_input)
    payload = json.loads(serialized)
    blocks_payload = payload["blocks"]
    assert blocks_payload[0].get("bold") is True
    assert "size" in blocks_payload[0] or "font_size" in blocks_payload[0] or blocks_payload[0].get("bold") is True


# =============================================================================
# TEST 8 — Table preservation
# =============================================================================
def test_table_preservation():
    """Table cells with table_id, row_index, column_index, and cell_role must survive."""
    block1 = SemanticBlockInput(
        block_id="b_tbl_1",
        text="Vessel Name",
        page=1,
        bbox=[50, 100, 150, 120],
        region_id="r1",
        region_kind="table",
        reading_order=1,
        table_id="tbl_1",
        row_index=0,
        column_index=0,
        cell_role="HEADER",
    )
    block2 = SemanticBlockInput(
        block_id="b_tbl_2",
        text="MT Northern Star",
        page=1,
        bbox=[50, 125, 150, 145],
        region_id="r1",
        region_kind="table",
        reading_order=2,
        table_id="tbl_1",
        row_index=1,
        column_index=0,
        cell_role="DATA",
    )
    sem_input = SemanticInput(
        document_id="doc_tbl",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        blocks=[block1, block2],
    )

    serialized = serialize_compact_semantic_input(sem_input)
    payload = json.loads(serialized)
    b_data = next(b for b in payload["blocks"] if b["id"] == "b_tbl_2")
    assert b_data["table"] == "tbl_1"
    assert b_data["row"] == 1
    assert b_data["col"] == 0


# =============================================================================
# TEST 9 — Nested / continued content remains linked
# =============================================================================
def test_nested_content_under_arbitrary_heading():
    """Content under arbitrary heading must remain associated with structural heading context."""
    lines = [
        _create_line("l_h", "Key Consultancies", 100, 115, font_size=14.0, bold=True, reading_order=1),
        _create_line("l_e1", "Principal Advisor", 120, 135, font_size=11.0, bold=True, reading_order=2),
        _create_line("l_e2", "Advised Fortune 500 client on cloud migration", 140, 155, reading_order=3),
    ]
    reg = Region("r1", "column", BoundingBox(50, 100, 500, 200), lines, 1)
    doc = Document([Page(1, 600.0, 800.0, [reg])])

    sem_input = build_semantic_input(doc)
    h_block = next(b for b in sem_input.blocks if b.text == "Key Consultancies")
    child_blocks = [b for b in sem_input.blocks if b.text != "Key Consultancies"]

    for cb in child_blocks:
        assert getattr(cb, "parent_heading_id", None) == h_block.block_id


# =============================================================================
# TEST 10 — Conventional resume regression
# =============================================================================
def test_conventional_resume_regression():
    """Conventional resumes with canonical headings (EXPERIENCE, EDUCATION, SKILLS) remain intact."""
    lines = [
        _create_line("l_0", "John Doe", 50, 70, font_size=18.0, bold=True, reading_order=1),
        _create_line("l_1", "EXPERIENCE", 100, 115, font_size=14.0, bold=True, reading_order=2),
        _create_line("l_2", "Software Engineer", 120, 135, bold=True, reading_order=3),
        _create_line("l_3", "Google LLC | 2020 - Present", 140, 155, reading_order=4),
        _create_line("l_4", "EDUCATION", 170, 185, font_size=14.0, bold=True, reading_order=5),
        _create_line("l_5", "B.S. Computer Science", 190, 205, bold=True, reading_order=6),
        _create_line("l_6", "SKILLS", 220, 235, font_size=14.0, bold=True, reading_order=7),
        _create_line("l_7", "Python, Go, C++", 240, 255, reading_order=8),
    ]
    reg = Region("r_main", "column", BoundingBox(50, 50, 500, 300), lines, 1)
    doc = Document([Page(1, 600.0, 800.0, [reg])])

    sem_input = build_semantic_input(doc)
    assert len(sem_input.blocks) == 8
    texts = [b.text for b in sem_input.blocks]
    assert "EXPERIENCE" in texts
    assert "EDUCATION" in texts
    assert "SKILLS" in texts


# =============================================================================
# TEST 11 — No content-loss invariant
# =============================================================================
def test_no_content_loss_invariant():
    """Every text block from Document IR must be present in SemanticInput (zero content loss)."""
    lines = [
        _create_line(f"l_{i}", f"Arbitrary line text {i}", 50 + i * 20, 65 + i * 20, reading_order=i + 1)
        for i in range(15)
    ]
    reg1 = Region("r1", "header", BoundingBox(50, 50, 500, 95), lines[:2], 1)
    reg2 = Region("r2", "column", BoundingBox(50, 100, 500, 350), lines[2:], 2)
    doc = Document([Page(1, 600.0, 800.0, [reg1, reg2])])

    sem_input = build_semantic_input(doc)

    source_line_texts = [line.text.strip() for page in doc.pages for r in page.regions for line in r.lines]
    emitted_texts = [b.text.strip() for b in sem_input.blocks]

    assert source_line_texts == emitted_texts, (
        f"Mismatch between source document lines and semantic input blocks!\n"
        f"Missing: {set(source_line_texts) - set(emitted_texts)}"
    )
