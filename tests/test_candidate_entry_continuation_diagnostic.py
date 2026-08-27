"""Phase 5N/5O: continued EXPERIENCE DATE-first stacks vs CandidateEntry segmentation."""

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


def _continued_experience_section(sections):
    return next(
        section
        for section in sections
        if section.origin == SectionOrigin.CONTINUED and section.semantic_label == "EXPERIENCE"
    )


def test_date_description_org_body_stack_segments_after_phase_5o():
    document = _two_page(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Operations Lead", 30, font_size=12, bold=True),
            _line("c1", "Northwind Partners LLC", 50),
            _line("d1", "2020-2022", 70),
        ],
        [
            _line("d2", "2022-2024", 20, page=2),
            _line("role", "Managed correspondence for regional offices.", 40, page=2),
            _line("c2", "Summit Holdings Inc – Denver, CO", 60, page=2),
            _line("b1", "Owned scheduling and vendor coordination.", 80, page=2),
            _line("b2", "• Prepared weekly status reports.", 100, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    continued = _continued_experience_section(sections)
    page2_entries = [
        entry
        for entry in build_candidate_entries(sections)
        if entry.section_id == continued.section_id
    ]
    assert len(page2_entries) == 1
    assert page2_entries[0].entry_type == EntryType.EXPERIENCE
    assert page2_entries[0].blocks[0].role == StructuralRole.DATE


def test_date_entry_title_org_body_title_first_order_segments():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Operations Lead", 30, font_size=12, bold=True),
        _line("c", "Northwind Partners LLC", 50),
        _line("d", "2020-2022", 70),
        _line("b", "• Coordinated regional delivery.", 90),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = build_candidate_entries(sections)
    assert len(entries) == 1
    assert entries[0].entry_type == EntryType.EXPERIENCE
    assert entries[0].blocks[0].role == StructuralRole.ENTRY_TITLE


def test_date_entry_title_org_body_date_first_order_segments():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("d", "2020-2022", 30),
        _line("t", "Operations Lead", 50, font_size=12, bold=True),
        _line("c", "Northwind Partners LLC", 70),
        _line("b", "• Coordinated regional delivery.", 90),
    ])
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = build_candidate_entries(sections)
    assert len(entries) == 1
    assert entries[0].blocks[0].text == "2020-2022"


def test_continued_page_date_description_org_body_segments():
    document = _two_page(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Warehouse Associate", 30, font_size=12, bold=True),
            _line("c1", "Harbor Logistics LLC", 50),
            _line("d1", "2020-2022", 70),
        ],
        [
            _line("d2", "2022-2024", 20, page=2),
            _line("role", "Shift Supervisor", 40, page=2, font_size=12, bold=True),
            _line("c2", "Summit Retail Inc – Austin, TX", 60, page=2),
            _line("b", "Led nightly inventory reconciliation.", 80, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    continued = _continued_experience_section(sections)
    entries = [
        entry for entry in build_candidate_entries(sections) if entry.section_id == continued.section_id
    ]
    assert len(entries) == 1
    assert all(block.page_number == 2 for block in entries[0].blocks)


def test_continued_date_entry_title_org_segments_when_date_precedes_title():
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
    continued = _continued_experience_section(sections)
    page2_entries = [
        entry for entry in build_candidate_entries(sections) if entry.section_id == continued.section_id
    ]
    assert len(page2_entries) == 1
    assert page2_entries[0].entry_type == EntryType.EXPERIENCE


def test_continued_date_short_title_with_org_segments():
    document = _two_page(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Picker", 30, font_size=12, bold=True),
            _line("c1", "Harbor Logistics LLC", 50),
            _line("d1", "2020-2022", 70),
        ],
        [
            _line("d2", "2022-2024", 20, page=2),
            _line("role", "Coordinator", 40, page=2),
            _line("c2", "Summit Retail Inc", 60, page=2),
            _line("b", "Prepared shift handoff notes.", 80, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    continued = _continued_experience_section(sections)
    entries = build_candidate_entries(sections)
    assert any(entry.section_id == continued.section_id for entry in entries)
