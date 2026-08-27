"""Phase 5O: order-independent CandidateEntry segmentation (DATE-first stacks)."""

from app.domain.candidate_entry import EntryType
from app.domain.candidate_section import SectionOrigin
from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.domain.structural import StructuralRole
from app.pipeline.stages.candidate_entries import build_candidate_entries
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.structural_roles import build_structural_blocks


def _line(
    line_id: str,
    text: str,
    y0: float,
    *,
    page: int = 1,
    x0: float = 10.0,
    font_size: float = 11.0,
    bold: bool = False,
) -> Line:
    bbox = BoundingBox(x0, y0, x0 + max(40.0, len(text) * 6.0), y0 + 12.0)
    span = Span(f"{line_id}-span", text, bbox, font_size=font_size, bold=bold)
    return Line(
        line_id,
        page,
        bbox,
        [span],
        text,
        TextStyle(font_size=font_size, bold=bold),
        reading_order=int(y0) + page * 1000,
        source_span_ids=[span.span_id],
        reconstruction_method="physical",
    )


def _single_column(lines: list[Line], *, region_id: str = "page-1-region-0") -> Document:
    page = lines[0].page_number if lines else 1
    bbox = BoundingBox(
        min(line.bbox.x0 for line in lines),
        min(line.bbox.y0 for line in lines),
        max(line.bbox.x1 for line in lines),
        max(line.bbox.y1 for line in lines),
    )
    return Document(
        pages=[Page(page, regions=[Region(region_id, "column", bbox, lines=lines, reading_order=0, column_id=0)])]
    )


def _two_page(page1: list[Line], page2: list[Line]) -> Document:
    p1_bbox = BoundingBox(
        min(line.bbox.x0 for line in page1),
        min(line.bbox.y0 for line in page1),
        max(line.bbox.x1 for line in page1),
        max(line.bbox.y1 for line in page1),
    )
    p2_bbox = BoundingBox(
        min(line.bbox.x0 for line in page2),
        min(line.bbox.y0 for line in page2),
        max(line.bbox.x1 for line in page2),
        max(line.bbox.y1 for line in page2),
    )
    return Document(
        pages=[
            Page(1, regions=[Region("page-1-region-0", "column", p1_bbox, lines=page1, reading_order=0, column_id=0)]),
            Page(2, regions=[Region("page-2-region-0", "column", p2_bbox, lines=page2, reading_order=0, column_id=0)]),
        ]
    )


def _experience_entries(sections):
    return build_candidate_entries(sections)


def test_title_org_date_body_order():
    """A: TITLE → ORG → DATE → BODY."""
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Operations Lead", 30, font_size=12, bold=True),
        _line("c", "Northwind Partners LLC", 50),
        _line("d", "2020-2022", 70),
        _line("b", "• Coordinated regional delivery.", 90),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = _experience_entries(sections)
    assert len(entries) == 1
    assert entries[0].entry_type == EntryType.EXPERIENCE
    texts = [block.text for block in entries[0].blocks]
    assert texts[0] == "Operations Lead"
    assert "2020-2022" in texts


def test_date_title_org_body_order():
    """B: DATE → TITLE → ORG → BODY."""
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("d", "2020-2022", 30),
        _line("t", "Operations Lead", 50, font_size=12, bold=True),
        _line("c", "Northwind Partners LLC", 70),
        _line("b", "• Coordinated regional delivery.", 90),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = _experience_entries(sections)
    assert len(entries) == 1
    assert entries[0].entry_type == EntryType.EXPERIENCE
    texts = [block.text for block in entries[0].blocks]
    assert texts[0] == "2020-2022"
    assert "Operations Lead" in texts
    assert "Northwind Partners LLC" in texts


def test_date_description_title_org_body_order():
    """C: DATE → DESCRIPTION(title-like) → ORG → BODY."""
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("d", "2020-2022", 30),
        _line("role", "Managed correspondence for regional offices.", 50),
        _line("c", "Summit Holdings Inc – Denver, CO", 70),
        _line("b", "Owned scheduling and vendor coordination.", 90),
    ])
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    entries = _experience_entries(sections)
    assert len(entries) == 1
    title_block = next(block for block in entries[0].blocks if block.text.startswith("Managed correspondence"))
    assert title_block.role in {StructuralRole.DESCRIPTION, StructuralRole.UNKNOWN}
    assert StructuralRole.DATE in {block.role for block in entries[0].blocks}


def test_date_org_location_body_without_title():
    """D: DATE → ORG/LOCATION → BODY (no ENTRY_TITLE)."""
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("d", "2020-2022", 30),
        _line("c", "Summit Holdings Inc – Denver, CO", 50),
        _line("b", "Prepared weekly operational status reports for leadership.", 70),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = _experience_entries(sections)
    assert len(entries) == 1
    roles = {block.role for block in entries[0].blocks}
    assert StructuralRole.DATE in roles
    assert StructuralRole.ENTRY_TITLE not in roles


def test_isolated_date_does_not_open_entry():
    """E: isolated DATE with no corroboration → no entry."""
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Operations Lead", 30, font_size=12, bold=True),
        _line("d", "2022 - Present", 50),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    assert _experience_entries(sections) == []


def test_weak_date_and_unrelated_text_no_entry():
    """F: DATE followed only by unrelated prose → no entry."""
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("d", "2020", 30),
        _line("p1", "Independent research notes from the field assignment.", 50),
        _line("p2", "Observations recorded during travel without employment context.", 70),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    assert _experience_entries(sections) == []


def test_continued_experience_date_first_stack_segments():
    """G: CONTINUED EXPERIENCE with DATE-first page-2 stack."""
    document = _two_page(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Picker", 30, font_size=12, bold=True),
            _line("c1", "Harbor Logistics LLC", 50),
            _line("d1", "2020-2022", 70),
        ],
        [
            _line("d2", "2022-2024", 20, page=2),
            _line("t2", "Coordinator", 40, page=2, font_size=12, bold=True),
            _line("c2", "Summit Retail Inc", 60, page=2),
            _line("b", "Prepared shift handoff notes.", 80, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    continued = next(
        s for s in sections if s.origin == SectionOrigin.CONTINUED and s.semantic_label == "EXPERIENCE"
    )
    entries = _experience_entries(sections)
    page2 = [e for e in entries if 2 in e.page_numbers and e.section_id == continued.section_id]
    assert len(page2) == 1
    assert page2[0].entry_type == EntryType.EXPERIENCE
    assert StructuralRole.DATE in {block.role for block in page2[0].blocks}


def test_summary_date_first_job_like_stack_entry_type_unknown():
    """H: SUMMARY ownership + job-like DATE-first stack → UNKNOWN entry_type."""
    document = _single_column([
        _line("h", "SUMMARY", 10, font_size=14, bold=True),
        _line("d", "2020-2022", 30),
        _line("t", "Operations Lead", 50, font_size=12, bold=True),
        _line("c", "Northwind Partners LLC", 70),
        _line("b", "Built regional delivery programs.", 90),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = _experience_entries(sections)
    assert entries
    assert all(entry.entry_type == EntryType.UNKNOWN for entry in entries)
    assert all(entry.entry_type != EntryType.EXPERIENCE for entry in entries)


def test_multiline_title_date_first_order():
    """I: multiline ENTRY_TITLE with DATE-first ordering."""
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("d", "2020-2022", 30),
        _line("t1", "Senior Operations", 50, font_size=12, bold=True),
        _line("t2", "Lead", 65, font_size=12, bold=True),
        _line("c", "Northwind Partners LLC", 85),
        _line("b", "• Coordinated regional delivery.", 105),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = _experience_entries(sections)
    assert len(entries) == 1
    texts = [block.text for block in entries[0].blocks]
    assert "Senior Operations" in texts
    assert "Lead" in texts


def test_provenance_aggregated_across_date_first_members():
    """J: provenance survives on DATE-first entries."""
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("d", "2020-2022", 30),
        _line("t", "Operations Lead", 50, font_size=12, bold=True),
        _line("c", "Northwind Partners LLC", 70),
        _line("b", "• Coordinated regional delivery.", 90),
    ])
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    entry = _experience_entries(sections)[0]
    member_blocks = [b for b in blocks if b.page_number == 1 and b.text != "EXPERIENCE"]
    expected_line_ids = [lid for b in member_blocks for lid in b.line_ids]
    expected_span_ids = [sid for b in member_blocks for sid in b.source_span_ids]
    assert list(entry.line_ids) == expected_line_ids
    assert list(entry.source_span_ids) == expected_span_ids
    assert entry.page_numbers == (1,)
    assert entry.path_ids
    assert entry.region_ids
    assert entry.reconstruction_methods
    assert all(block.reconstruction_method for block in entry.blocks)
