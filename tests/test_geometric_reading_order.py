from __future__ import annotations

from app.domain.document import BoundingBox, Document, Line, Page, Region, TextStyle
from app.pipeline.stages.layout import interpret_layout


def _make_line(
    line_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    reading_order: int,
) -> Line:
    return Line(
        line_id=line_id,
        page_number=1,
        bbox=BoundingBox(x0, y0, x1, y1),
        spans=[],
        text=text,
        style=TextStyle(font_size=10.0, bold=False),
        reading_order=reading_order,
    )


def test_geometric_reading_order_fixes_out_of_order_stream():
    """Lines written in reversed or scrambled PDF stream order are re-indexed to visual order."""
    line_top = _make_line("l_top", "Assistant Second Engineer", x0=50.0, y0=100.0, x1=250.0, y1=115.0, reading_order=50)
    line_mid = _make_line("l_mid", "Bullet 1: Maintained propulsion machinery", x0=50.0, y0=130.0, x1=400.0, y1=145.0, reading_order=10)
    line_bot = _make_line("l_bot", "Bullet 2: Conducted boiler inspections", x0=50.0, y0=155.0, x1=400.0, y1=170.0, reading_order=20)

    page = Page(
        page_number=1,
        regions=[
            Region(
                region_id="r0",
                kind="physical_page",
                bbox=BoundingBox(50.0, 100.0, 400.0, 170.0),
                lines=[line_mid, line_bot, line_top],
                reading_order=0,
            )
        ],
    )
    doc = Document(pages=[page])
    interpreted = interpret_layout(doc)

    out_page = interpreted.pages[0]
    out_lines = [line for r in out_page.regions for line in r.lines]
    assert len(out_lines) == 3

    assert [line.line_id for line in out_lines] == ["l_top", "l_mid", "l_bot"]
    assert [line.reading_order for line in out_lines] == [0, 1, 2]


def test_geometric_reading_order_across_header_and_columns():
    """Header lines receive lower reading orders than column lines, and column 0 precedes column 1."""
    header_line = _make_line("l_hdr", "Full width header", x0=100.0, y0=10.0, x1=600.0, y1=20.0, reading_order=99)
    col0_l1 = _make_line("c0_1", "Left one", x0=10.0, y0=40.0, x1=120.0, y1=50.0, reading_order=3)
    col0_l2 = _make_line("c0_2", "Left two", x0=10.0, y0=60.0, x1=120.0, y1=70.0, reading_order=4)
    col1_l1 = _make_line("c1_1", "Right one", x0=300.0, y0=40.0, x1=650.0, y1=50.0, reading_order=1)
    col1_l2 = _make_line("c1_2", "Right two", x0=300.0, y0=60.0, x1=650.0, y1=70.0, reading_order=2)

    page = Page(
        page_number=1,
        regions=[
            Region(
                region_id="r0",
                kind="physical_page",
                bbox=BoundingBox(10.0, 10.0, 650.0, 70.0),
                lines=[col1_l1, col1_l2, col0_l1, col0_l2, header_line],
                reading_order=0,
            )
        ],
    )
    doc = Document(pages=[page])
    interpreted = interpret_layout(doc)

    out_page = interpreted.pages[0]
    out_lines = [line for r in out_page.regions for line in r.lines]

    # Header is first (0)
    assert out_lines[0].line_id == "l_hdr"
    assert out_lines[0].reading_order == 0

    # Col 0 is next (1, 2)
    assert out_lines[1].line_id == "c0_1"
    assert out_lines[1].reading_order == 1
    assert out_lines[2].line_id == "c0_2"
    assert out_lines[2].reading_order == 2

    # Col 1 is last (3, 4)
    assert out_lines[3].line_id == "c1_1"
    assert out_lines[3].reading_order == 3
    assert out_lines[4].line_id == "c1_2"
    assert out_lines[4].reading_order == 4
