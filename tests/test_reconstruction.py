from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.reconstruction import reconstruct_document


def _line(line_id: str, text: str, x0: float, y0: float, x1: float, y1: float, page: int = 1, font_size: float = 10.0) -> Line:
    bbox = BoundingBox(x0, y0, x1, y1)
    span = Span(f"{line_id}-span-0", text, bbox, font_size=font_size)
    return Line(line_id, page, bbox, [span], text, TextStyle(font_size=font_size), source_span_ids=[span.span_id])


def _document(lines: list[Line], page: int = 1) -> Document:
    pages: dict[int, list[Line]] = {}
    for line in lines:
        pages.setdefault(line.page_number, []).append(line)
    return Document(
        pages=[
            Page(
                page_number,
                regions=[
                    Region(
                        f"page-{page_number}-region-0",
                        "physical_page",
                        BoundingBox(0, 0, 500, 500),
                        lines=page_lines,
                    )
                ],
            )
            for page_number, page_lines in sorted(pages.items())
        ]
    )


def test_reconstructs_adjacent_email_fragments_with_provenance():
    document = _document([
        _line("line-0", "david.perez@gmail.co", 10, 20, 140, 30),
        _line("line-1", "m", 142, 20, 148, 30),
    ])

    result = reconstruct_document(document)
    line = result.pages[0].regions[0].lines[0]

    assert line.text == "david.perez@gmail.com"
    assert line.source_span_ids == ["line-0-span-0", "line-1-span-0"]
    assert line.bbox.x0 == 10
    assert line.bbox.x1 == 148
    assert line.reconstruction_method == "adjacent_physical_fragments"


def test_reconstructs_adjacent_url_fragments_without_space():
    document = _document([
        _line("line-0", "linkedin.com/in/", 10, 20, 100, 30),
        _line("line-1", "davidperez", 103, 20, 160, 30),
    ])

    assert reconstruct_document(document).pages[0].regions[0].lines[0].text == "linkedin.com/in/davidperez"


def test_does_not_chain_a_full_width_line_after_wrapped_fragment():
    document = _document([
        _line("line-0", "david.perez@gmail.co", 10, 20, 140, 30),
        _line("line-1", "m", 142, 30, 148, 40),
        _line("line-2", "1938 W Augusta Blvd,", 10, 40, 140, 50),
    ])

    lines = reconstruct_document(document).pages[0].regions[0].lines
    assert [line.text for line in lines] == ["david.perez@gmail.com", "1938 W Augusta Blvd,"]


def test_reconstructs_ordinary_words_with_space():
    document = _document([
        _line("line-0", "Senior", 10, 20, 45, 30),
        _line("line-1", "Engineer", 48, 20, 100, 30),
    ])

    assert reconstruct_document(document).pages[0].regions[0].lines[0].text == "Senior Engineer"


def test_does_not_merge_different_lines_columns_pages_or_large_gaps():
    lines = [
        _line("line-0", "John Smith", 10, 20, 80, 30),
        _line("line-1", "Software Engineer", 10, 40, 120, 50),
        _line("line-2", "Left", 10, 60, 35, 70),
        _line("line-3", "Right", 300, 60, 335, 70),
        _line("line-4", "Page one", 10, 80, 60, 90, page=1),
        _line("line-5", "Page two", 10, 80, 60, 90, page=2),
        _line("line-6", "Far", 10, 100, 30, 110),
        _line("line-7", "Away", 100, 100, 140, 110),
    ]

    result = reconstruct_document(_document(lines))
    assert [line.text for line in result.pages[0].regions[0].lines] == [
        "John Smith", "Software Engineer", "Left", "Right", "Page one", "Far", "Away"
    ]


def test_reconstruction_is_deterministic_and_source_text_is_not_lost():
    document = _document([_line("line-0", "First", 10, 20, 40, 30), _line("line-1", "Second", 43, 20, 80, 30)])

    first = reconstruct_document(document)
    second = reconstruct_document(document)

    assert first == second
    assert {source_id for line in first.pages[0].regions[0].lines for source_id in line.source_span_ids} == {
        "line-0-span-0", "line-1-span-0"
    }