"""Tests for column-aware document reconstruction.

Verifies that:
1. Two independent table columns (with 7-13pt gap) remain separate lines/blocks.
2. Legitimate adjacent words and fragments on the same line still merge.
3. Date cells in distinct columns do not merge into adjacent columns.
4. Existing word spacing and punctuation attachment are preserved.
"""

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.reconstruction import reconstruct_document


def _line(
    line_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page: int = 1,
    font_size: float = 10.0,
) -> Line:
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
                        BoundingBox(0, 0, 600, 800),
                        lines=page_lines,
                    )
                ],
            )
            for page_number, page_lines in sorted(pages.items())
        ]
    )


def test_independent_table_columns_remain_separate():
    """Two table columns separated by a 10pt gutter (gap < 15pt default tolerance)

    must NOT merge when they represent recurring column positions across multiple rows.
    """
    lines = [
        # Row 1: Col A (x=36..100) and Col B (x=110..200) -> gap = 10pt
        _line("r1-c1", "3RD OFFICER", 36.0, 100.0, 100.0, 112.0),
        _line("r1-c2", "M.T. HARI PRIYA", 110.0, 100.0, 200.0, 112.0),
        # Row 2: Col A (x=36..100) and Col B (x=110..200) -> gap = 10pt
        _line("r2-c1", "4TH OFFICER", 36.0, 120.0, 100.0, 132.0),
        _line("r2-c2", "M.T. MAHARSHI", 110.0, 120.0, 200.0, 132.0),
    ]

    doc = _document(lines)
    result = reconstruct_document(doc)
    out_lines = result.pages[0].regions[0].lines

    # Ensure all 4 cells remain distinct lines, not merged horizontally
    texts = [line.text for line in out_lines]
    assert "3RD OFFICER" in texts
    assert "M.T. HARI PRIYA" in texts
    assert "4TH OFFICER" in texts
    assert "M.T. MAHARSHI" in texts
    assert len(out_lines) == 4


def test_date_cells_do_not_merge_into_adjacent_columns():
    """Date columns (From / To) separated by 8pt gutter must remain separate."""
    lines = [
        # Row 1: From date (x=150..210) and To date (x=218..278) -> gap = 8pt
        _line("r1-d1", "12/03/2021", 150.0, 50.0, 210.0, 62.0),
        _line("r1-d2", "15/09/2022", 218.0, 50.0, 278.0, 62.0),
        # Row 2: From date and To date
        _line("r2-d1", "01/01/2020", 150.0, 70.0, 210.0, 82.0),
        _line("r2-d2", "10/02/2021", 218.0, 70.0, 278.0, 82.0),
    ]

    doc = _document(lines)
    result = reconstruct_document(doc)
    out_lines = result.pages[0].regions[0].lines

    texts = [line.text for line in out_lines]
    assert texts == ["12/03/2021", "15/09/2022", "01/01/2020", "10/02/2021"]


def test_adjacent_fragments_on_same_line_still_merge():
    """Non-column word fragments on the same line still merge normally."""
    lines = [
        _line("f1", "Software", 36.0, 50.0, 90.0, 62.0),
        _line("f2", "Engineer", 94.0, 50.0, 150.0, 62.0),
    ]

    doc = _document(lines)
    result = reconstruct_document(doc)
    out_lines = result.pages[0].regions[0].lines

    assert len(out_lines) == 1
    assert out_lines[0].text == "Software Engineer"


def test_word_spacing_and_punctuation_attachment_preserved():
    """Punctuation and URL/email fragments merge without inappropriate spacing."""
    lines = [
        _line("l1", "david.perez@gmail.co", 10.0, 20.0, 140.0, 30.0),
        _line("l2", "m", 142.0, 20.0, 148.0, 30.0),
    ]

    doc = _document(lines)
    result = reconstruct_document(doc)
    out_lines = result.pages[0].regions[0].lines

    assert len(out_lines) == 1
    assert out_lines[0].text == "david.perez@gmail.com"
