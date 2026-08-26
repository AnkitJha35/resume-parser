from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor
from app.domain.document import document_from_text_blocks


def _line(line_id: str, text: str, x0: float, y0: float, x1: float, y1: float, page: int = 1) -> Line:
    bbox = BoundingBox(x0, y0, x1, y1)
    span = Span(f"{line_id}-span-0", text, bbox, font_size=10.0)
    return Line(line_id, page, bbox, [span], text, TextStyle(font_size=10.0), source_span_ids=[span.span_id])


def _document(lines: list[Line]) -> Document:
    pages: dict[int, list[Line]] = {}
    for line in lines:
        pages.setdefault(line.page_number, []).append(line)
    return Document(
        pages=[
            Page(
                page_number,
                regions=[Region(f"page-{page_number}-region-0", "physical_page", BoundingBox(0, 0, 700, 900), lines=page_lines)],
            )
            for page_number, page_lines in sorted(pages.items())
        ]
    )


def _all_lines(document: Document) -> list[Line]:
    return [line for page in document.pages for region in page.regions for line in region.lines]


def test_single_column_has_one_top_to_bottom_region():
    source = _document([_line("a", "First", 10, 10, 80, 20), _line("b", "Second", 12, 30, 90, 40)])

    result = interpret_layout(source)

    assert len(result.pages[0].regions) == 1
    assert [line.text for line in result.pages[0].regions[0].lines] == ["First", "Second"]


def test_unequal_two_columns_remain_separate():
    source = _document([
        _line("left-a", "Left one", 10, 20, 110, 30),
        _line("left-b", "Left two", 10, 40, 120, 50),
        _line("right-a", "Right one", 300, 20, 650, 30),
        _line("right-b", "Right two", 300, 40, 640, 50),
    ])

    result = interpret_layout(source)

    assert [region.kind for region in result.pages[0].regions] == ["column", "column"]
    assert [[line.text for line in region.lines] for region in result.pages[0].regions] == [
        ["Left one", "Left two"],
        ["Right one", "Right two"],
    ]


def test_full_width_header_and_different_column_starts_are_preserved():
    source = _document([
        _line("header", "Full width header", 100, 10, 600, 20),
        _line("left", "Sidebar", 10, 40, 120, 50),
        _line("right", "Main content", 300, 80, 650, 90),
    ])

    result = interpret_layout(source)

    assert result.pages[0].regions[0].kind == "header"
    assert result.pages[0].regions[0].lines[0].text == "Full width header"
    assert {region.column_id for region in result.pages[0].regions[1:]} == {0, 1}
    assert {line.text for region in result.pages[0].regions for line in region.lines} == {
        "Full width header", "Sidebar", "Main content"
    }


def test_wide_body_line_is_not_classified_as_header():
    source = _document([
        _line("left", "Sidebar", 10, 40, 120, 50),
        _line("right", "Main content that spans broadly", 300, 80, 650, 90),
        _line("left-body", "More sidebar", 10, 100, 120, 110),
    ])

    result = interpret_layout(source)

    assert all(line.text != "Main content that spans broadly" for line in result.pages[0].regions[0].lines)


def test_multipage_regions_have_deterministic_ids_and_no_duplicate_sources():
    source = _document([
        _line("p1-left", "Page one left", 10, 20, 100, 30, page=1),
        _line("p1-right", "Page one right", 300, 20, 400, 30, page=1),
        _line("p2-left", "Page two left", 10, 20, 100, 30, page=2),
        _line("p2-right", "Page two right", 300, 20, 400, 30, page=2),
    ])

    first = interpret_layout(source)
    second = interpret_layout(source)
    lines = _all_lines(first)
    source_ids = [source_id for line in lines for source_id in line.source_span_ids]

    assert first == second
    assert [page.page_number for page in first.pages] == [1, 2]
    assert first.pages[0].regions[0].region_id == "page-1-region-0"
    assert first.pages[1].regions[0].region_id == "page-2-region-0"
    assert len(source_ids) == len(set(source_ids)) == 4
    assert {line.line_id for line in lines} == {"p1-left", "p1-right", "p2-left", "p2-right"}


def test_real_resume_2_has_page_local_columns_without_text_loss():
    raw = PDFExtractor.extract(Path("tests/fixtures/resume_2.pdf").read_bytes())
    source = reconstruct_document(document_from_text_blocks(raw))
    result = interpret_layout(source)
    source_text = [span.text for line in _all_lines(source) for span in line.spans]
    layout_text = [span.text for line in _all_lines(result) for span in line.spans]

    assert len(result.pages[0].regions) >= 2
    assert len(result.pages[1].regions) >= 2
    assert sorted(source_text) == sorted(layout_text)
    assert len(_all_lines(result)) == len({line.line_id for line in _all_lines(result)})


def test_protected_fixtures_keep_layout_regions():
    for filename in ("resume_1.pdf", "resume_7.pdf", "fresher_hr_resume.pdf"):
        raw = PDFExtractor.extract(Path("tests/fixtures", filename).read_bytes())
        result = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
        source_lines = _all_lines(document_from_text_blocks(raw))
        layout_lines = _all_lines(result)
        assert len(layout_lines) == len(source_lines)
        assert {span.span_id for line in layout_lines for span in line.spans} == {
            span.span_id for line in source_lines for span in line.spans
        }
        assert all(
            region.bbox.x0 <= line.bbox.x0
            and region.bbox.y0 <= line.bbox.y0
            and region.bbox.x1 >= line.bbox.x1
            and region.bbox.y1 >= line.bbox.y1
            for page in result.pages
            for region in page.regions
            for line in region.lines
        )
