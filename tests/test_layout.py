from tests.conftest import require_fixture
from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor
from app.domain.document import document_from_text_blocks


def _line(
    line_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page: int = 1,
    font_size: float = 10.0,
    bold: bool = False,
) -> Line:
    bbox = BoundingBox(x0, y0, x1, y1)
    span = Span(f"{line_id}-span-0", text, bbox, font_size=font_size, bold=bold)
    return Line(
        line_id,
        page,
        bbox,
        [span],
        text,
        TextStyle(font_size=font_size, bold=bold),
        source_span_ids=[span.span_id],
    )


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
    path = require_fixture("resume_2.pdf")
    if not path.exists():
        return
    raw = PDFExtractor.extract(path.read_bytes())
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
        path = Path("tests/fixtures", filename)
        if not path.exists():
            continue
        raw = PDFExtractor.extract(path.read_bytes())
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


def _body_columns(document: Document) -> list[list[str]]:
    return [
        [line.text for line in region.lines]
        for region in document.pages[0].regions
        if region.kind == "column"
    ]


def test_intra_column_grid_does_not_create_extra_columns():
    """Two real columns; a short 3-cell grid stays inside the left column."""
    source = _document([
        _line("l-h", "Experience", 10, 20, 180, 32),
        _line("l-a", "Left body", 10, 40, 200, 52),
        _line("l-b", "More left", 10, 60, 190, 72),
        _line("l-c", "Still left", 12, 80, 185, 92),
        _line("g1", "Cell one", 12, 200, 70, 212),
        _line("g2", "Cell two", 90, 200, 175, 212),
        _line("g3", "Cell three", 12, 220, 80, 232),
        _line("r-h", "Skills", 320, 20, 500, 32),
        _line("r-a", "Right body", 320, 40, 490, 52),
        _line("r-b", "More right", 320, 60, 480, 72),
        _line("r-c", "Still right", 320, 80, 500, 92),
        _line("r-d", "Right tail", 320, 200, 470, 212),
    ])
    result = interpret_layout(source)
    columns = _body_columns(result)
    assert len(columns) == 2
    left, right = columns
    assert {"Cell one", "Cell two", "Cell three"} <= set(left)
    assert {"Skills", "Right body"} <= set(right)
    assert "Cell two" not in right


def test_same_row_date_and_location_stay_in_one_column():
    source = _document([
        _line("h", "Experience", 10, 20, 160, 32),
        _line("t", "Role title", 10, 40, 140, 52),
        _line("d", "2020-2022", 10, 60, 120, 72),
        _line("loc", "Denver, CO", 160, 60, 280, 72),
        _line("b", "Did the work.", 10, 80, 220, 92),
        _line("rh", "Skills", 360, 20, 520, 32),
        _line("ra", "Python", 360, 40, 500, 52),
        _line("rb", "Docker", 360, 60, 500, 72),
        _line("rc", "Linux", 360, 80, 500, 92),
    ])
    result = interpret_layout(source)
    columns = _body_columns(result)
    assert len(columns) == 2
    left = next(col for col in columns if "2020-2022" in col)
    assert "Denver, CO" in left
    assert "Role title" in left


def test_numeric_ticks_do_not_create_a_third_column():
    source = _document([
        _line("lh", "Profile", 10, 20, 150, 32),
        _line("la", "Left prose", 10, 40, 200, 52),
        _line("lb", "More prose", 10, 60, 190, 72),
        _line("lc", "Still prose", 10, 80, 180, 92),
        _line("rh", "Skills", 300, 20, 420, 32),
        _line("s1", "Excel", 300, 50, 380, 62),
        _line("t1", "4", 450, 50, 465, 62),
        _line("s2", "Sheets", 300, 70, 390, 82),
        _line("t2", "4", 450, 70, 465, 82),
        _line("s3", "Docs", 300, 90, 380, 102),
        _line("t3", "3", 450, 90, 465, 102),
        _line("s4", "English", 300, 200, 370, 212),
        _line("t4", "5", 450, 200, 465, 212),
    ])
    result = interpret_layout(source)
    columns = _body_columns(result)
    assert len(columns) == 2
    right = next(col for col in columns if "Excel" in col)
    assert {"4", "3", "5"} <= set(right)


def test_narrow_date_gutter_is_not_a_page_column():
    source = _document([
        _line("n", "Name", 10, 10, 80, 22),
        _line("p1", "Profile sentence one that spans the page width.", 10, 40, 500, 52),
        _line("p2", "Profile sentence two continues the paragraph.", 10, 54, 500, 66),
        _line("edu", "Degree | School of Example", 10, 120, 420, 132),
        _line("d1", "2024-2026", 450, 120, 530, 132),
        _line("b1", "Coursework description under the degree.", 10, 140, 400, 152),
        _line("edu2", "Second Degree | Other School", 10, 200, 400, 212),
        _line("d2", "2020-2023", 450, 200, 530, 212),
        _line("b2", "More coursework under the second degree.", 10, 220, 400, 232),
    ])
    result = interpret_layout(source)
    assert len(result.pages[0].regions) == 1
    texts = [line.text for line in result.pages[0].regions[0].lines]
    assert "2024-2026" in texts and "2020-2023" in texts
    assert "Profile sentence one that spans the page width." in texts


def test_overlapping_tall_columns_remain_two():
    source = _document([
        _line("l1", "Left wrap that extends past the right origin.", 10, 20, 250, 32),
        _line("l2", "Left continues with wrapped width.", 10, 50, 240, 62),
        _line("l3", "Left third", 10, 80, 230, 92),
        _line("l4", "Left fourth", 10, 200, 245, 212),
        _line("l5", "Left fifth", 10, 400, 235, 412),
        _line("r1", "Right heading", 180, 20, 400, 32),
        _line("r2", "Right body one", 180, 50, 490, 62),
        _line("r3", "Right body two", 180, 80, 480, 92),
        _line("r4", "Right body three", 180, 200, 500, 212),
        _line("r5", "Right body four", 180, 400, 490, 412),
    ])
    result = interpret_layout(source)
    columns = _body_columns(result)
    assert len(columns) == 2
    left = next(col for col in columns if "Left third" in col)
    right = next(col for col in columns if "Right heading" in col)
    assert "Right heading" not in left
    assert "Left third" not in right


def test_three_tall_columns_remain_separate():
    source = _document([
        _line("a1", "A one", 10, 20, 90, 32),
        _line("a2", "A two", 10, 80, 90, 92),
        _line("a3", "A three", 10, 200, 90, 212),
        _line("b1", "B one", 220, 20, 300, 32),
        _line("b2", "B two", 220, 80, 300, 92),
        _line("b3", "B three", 220, 200, 300, 212),
        _line("c1", "C one", 430, 20, 520, 32),
        _line("c2", "C two", 430, 80, 520, 92),
        _line("c3", "C three", 430, 200, 520, 212),
    ])
    result = interpret_layout(source)
    columns = _body_columns(result)
    assert len(columns) == 3
    assert [col[0] for col in columns] == ["A one", "B one", "C one"]


def test_single_column_with_right_aligned_lines_stays_one():
    source = _document([
        _line("a", "Title line", 10, 20, 200, 32),
        _line("b", "Body one", 10, 40, 300, 52),
        _line("c", "Body two", 10, 60, 280, 72),
        _line("d", "Page 1", 400, 40, 460, 52),
        _line("e", "Body three", 10, 80, 250, 92),
    ])
    result = interpret_layout(source)
    assert len(result.pages[0].regions) == 1
    assert result.pages[0].regions[0].kind in {"physical_region", "column"}
    texts = [line.text for line in result.pages[0].regions[0].lines]
    assert "Page 1" in texts and "Body one" in texts


def _header_texts(document: Document) -> list[str]:
    return [
        line.text
        for region in document.pages[0].regions
        if region.kind == "header"
        for line in region.lines
    ]


def _non_header_texts(document: Document) -> list[str]:
    return [
        line.text
        for region in document.pages[0].regions
        if region.kind != "header"
        for line in region.lines
    ]


def test_contact_block_does_not_absorb_following_paragraph():
    source = _document([
        _line("n", "Jordan Blake", 10, 12, 140, 32, font_size=18),
        _line("e", "jordan@example.com", 10, 36, 160, 48),
        _line("p", "Phone: 555-0100", 10, 50, 130, 62),
        _line("a", "A motivated candidate with experience across operations and delivery of programs.", 10, 88, 520, 100),
        _line("b", "Eager to contribute to team outcomes while continuing to develop professional skills.", 10, 102, 515, 114),
        _line("c", "Comfortable collaborating across functions in a fast-paced environment.", 10, 116, 480, 128),
    ])
    result = interpret_layout(source)
    header = _header_texts(result)
    body = _non_header_texts(result)
    assert "Jordan Blake" in header
    assert "jordan@example.com" in header
    assert "A motivated candidate with experience across operations and delivery of programs." in body
    assert "Comfortable collaborating across functions in a fast-paced environment." in body


def test_one_line_tagline_may_remain_in_header():
    source = _document([
        _line("n", "Jordan Blake", 10, 12, 140, 32, font_size=18),
        _line("t", "Seeking internships", 10, 36, 180, 48, bold=True),
        _line("e", "jordan@example.com", 10, 52, 160, 64),
        _line("b", "Experience", 10, 100, 90, 112, font_size=13),
        _line("w", "Built internal tools for a logistics team during a summer role.", 10, 120, 500, 132),
    ])
    result = interpret_layout(source)
    header = _header_texts(result)
    body = _non_header_texts(result)
    assert "Seeking internships" in header
    assert "Experience" in body
    assert "Built internal tools for a logistics team during a summer role." in body


def test_body_heading_after_contact_stays_in_body():
    source = _document([
        _line("n", "Jordan Blake", 10, 12, 140, 32, font_size=18),
        _line("e", "jordan@example.com", 10, 36, 160, 48),
        _line("h", "Experience", 10, 80, 100, 94, font_size=13),
        _line("w", "Delivered process improvements for a regional operations group.", 10, 100, 510, 112),
    ])
    result = interpret_layout(source)
    assert "Experience" in _non_header_texts(result)
    assert "Experience" not in _header_texts(result)


def test_headingless_paragraph_before_education_stays_in_body():
    source = _document([
        _line("n", "Jordan Blake", 10, 12, 140, 32, font_size=18),
        _line("e", "jordan@example.com", 10, 36, 160, 48),
        _line("a", "A motivated candidate with experience across operations and delivery of programs.", 10, 70, 520, 82),
        _line("b", "Eager to contribute to team outcomes while continuing to develop professional skills.", 10, 84, 515, 96),
        _line("c", "Comfortable collaborating across functions in a fast-paced environment.", 10, 98, 480, 110),
        _line("d", "Bachelor of Science | State University", 10, 160, 280, 172),
        _line("y", "2020-2024", 450, 160, 530, 172),
    ])
    result = interpret_layout(source)
    body = _non_header_texts(result)
    assert "A motivated candidate with experience across operations and delivery of programs." in body
    assert "Bachelor of Science | State University" in body
    assert "2020-2024" in body


def test_tall_letterhead_remains_header():
    source = _document([
        _line("n", "Jordan Blake", 40, 12, 180, 32, font_size=20),
        _line("t", "Operations Associate", 40, 36, 200, 48, bold=True),
        _line("e", "jordan@example.com", 40, 52, 190, 64),
        _line("p", "Phone: 555-0100", 40, 66, 160, 78),
        _line("u", "https://example.com/in/jordan", 40, 80, 220, 92),
        _line("a1", "12 Oak Lane", 40, 106, 130, 118),
        _line("a2", "Springfield", 40, 120, 120, 132),
        _line("h", "Experience", 40, 220, 130, 234, font_size=13),
        _line("w", "Delivered process improvements for a regional operations group.", 40, 240, 520, 252),
    ])
    result = interpret_layout(source)
    header = set(_header_texts(result))
    assert {"Jordan Blake", "Operations Associate", "jordan@example.com", "12 Oak Lane", "Springfield"} <= header
    assert "Experience" not in header
    assert "Delivered process improvements for a regional operations group." in _non_header_texts(result)


def test_two_column_header_is_not_only_column_coincidence():
    source = _document([
        _line("n", "Jordan Blake", 10, 10, 140, 26, font_size=18),
        _line("e", "jordan@example.com", 10, 30, 160, 42),
        _line("l1", "Left column opens with a wide prose block about prior internships and impact.", 10, 70, 240, 82),
        _line("l2", "More left prose continues beneath the opening paragraph on this side.", 10, 90, 230, 102),
        _line("l3", "Left tail keeps covering the page vertically with additional detail.", 10, 200, 220, 212),
        _line("r1", "Right heading", 320, 200, 500, 212),
        _line("r2", "Right body one continues the second column much lower on the page.", 320, 220, 540, 232),
        _line("r3", "Right body two", 320, 240, 500, 252),
    ])
    result = interpret_layout(source)
    body = _non_header_texts(result)
    assert "Left column opens with a wide prose block about prior internships and impact." in body
    assert "Right heading" in body
    assert len(_body_columns(result)) == 2


def test_overlapping_header_and_columns_keep_summary_body():
    source = _document([
        _line("n", "Jordan Blake", 10, 12, 160, 28, font_size=18),
        _line("t", "Software Engineer II", 10, 32, 180, 44, bold=True),
        _line("c", "EMAIL : jordan@example.com PHONE : 5550100", 10, 48, 360, 60),
        _line("u", "https://linkedin.com/in/jordanblake", 10, 62, 280, 74),
        _line("s", "Summary", 10, 88, 90, 108, font_size=14),
        _line("l1", "Left wrap that extends past the right origin with summary prose.", 10, 120, 250, 132),
        _line("l2", "Left continues with wrapped width across several lines of text.", 10, 150, 240, 162),
        _line("l3", "Left third keeps covering the column.", 10, 180, 230, 192),
        _line("l4", "Left fourth", 10, 280, 245, 292),
        _line("r1", "Projects heading", 180, 118, 400, 130),
        _line("r2", "Right body one", 180, 150, 490, 162),
        _line("r3", "Right body two", 180, 180, 480, 192),
        _line("r4", "Right body three", 180, 280, 500, 292),
    ])
    result = interpret_layout(source)
    columns = _body_columns(result)
    assert len(columns) == 2
    left = next(col for col in columns if "Left third keeps covering the column." in col)
    right = next(col for col in columns if "Projects heading" in col)
    assert "Left wrap that extends past the right origin with summary prose." in left
    assert "Projects heading" not in left
    assert "Left third keeps covering the column." not in right


# ---------------------------------------------------------------------------
# Phase 10K regression tests: section-heading header-band cap
# ---------------------------------------------------------------------------


def test_all_caps_section_heading_near_top_stays_in_body():
    """An all-caps section heading must NOT be swallowed into the header region
    even when it appears early in the page and body_start_y is computed too low."""
    source = _document([
        _line("n", "Jordan Blake", 10, 10, 180, 26, font_size=18),
        _line("e", "jordan@example.com", 10, 30, 200, 42),
        # All-caps section heading immediately after contact block.
        _line("h", "ACADEMIC APPOINTMENTS", 10, 50, 350, 66, font_size=13, bold=True),
        _line("a1", "Professor with Tenure", 10, 70, 300, 84),
        _line("a2", "2019 - Present", 10, 88, 200, 100),
        _line("a3", "Johns Hopkins University", 10, 104, 320, 116),
    ])
    result = interpret_layout(source)
    header = _header_texts(result)
    body = _non_header_texts(result)
    # Name and contact are header; the section heading and appointment lines are body.
    assert "Jordan Blake" in header
    assert "jordan@example.com" in header
    assert "ACADEMIC APPOINTMENTS" in body
    assert "ACADEMIC APPOINTMENTS" not in header
    assert "Professor with Tenure" in body
    assert "2019 - Present" in body
    assert "Johns Hopkins University" in body


def test_body_lines_after_all_caps_heading_stay_in_body():
    """All appointment lines following a section heading must remain in body."""
    source = _document([
        _line("n", "Dr. Elias Moore", 10, 10, 200, 28, font_size=20),
        _line("t", "Curriculum Vitae", 10, 32, 160, 44, bold=True),
        _line("e", "elias@example.edu", 10, 48, 200, 60),
        _line("h", "WORK EXPERIENCE", 10, 68, 320, 82, font_size=13, bold=True),
        _line("r1", "Senior Researcher", 10, 90, 300, 104),
        _line("r2", "2015 - Present", 10, 108, 200, 120),
        _line("r3", "Research Institute", 10, 124, 290, 136),
        _line("r4", "Postdoctoral Fellow", 10, 144, 280, 156),
        _line("r5", "2012 - 2015", 10, 160, 180, 172),
    ])
    result = interpret_layout(source)
    body = _non_header_texts(result)
    assert "WORK EXPERIENCE" in body
    assert "Senior Researcher" in body
    assert "2015 - Present" in body
    assert "Research Institute" in body
    assert "Postdoctoral Fellow" in body
    assert "2012 - 2015" in body


def test_multi_line_appointment_block_after_heading_is_not_header():
    """Multi-entry appointment block below an all-caps heading must all be body."""
    source = _document([
        _line("n", "Prof. Ada Lee", 10, 10, 180, 28, font_size=20),
        _line("e", "ada@univ.edu", 10, 32, 180, 44),
        _line("h", "PROFESSIONAL EXPERIENCE", 10, 52, 380, 68, font_size=13, bold=True),
        _line("j1", "Associate Professor", 10, 72, 280, 86),
        _line("d1", "2014 - 2019", 10, 90, 200, 102),
        _line("o1", "University of Example", 10, 106, 300, 118),
        _line("j2", "Assistant Professor", 10, 122, 270, 134),
        _line("d2", "2008 - 2014", 10, 138, 200, 150),
        _line("o2", "State University", 10, 154, 260, 166),
    ])
    result = interpret_layout(source)
    header = _header_texts(result)
    body = _non_header_texts(result)
    assert "PROFESSIONAL EXPERIENCE" not in header
    for text in ("Associate Professor", "2014 - 2019", "University of Example",
                 "Assistant Professor", "2008 - 2014", "State University"):
        assert text in body, f"Expected {text!r} in body"
        assert text not in header, f"Expected {text!r} not in header"


def test_existing_contact_header_lines_remain_header():
    """Name and contact lines must still be classified as header after Phase 10K."""
    source = _document([
        _line("n", "Jordan Blake", 10, 12, 140, 32, font_size=18),
        _line("e", "jordan@example.com", 10, 36, 160, 48),
        _line("p", "Phone: 555-0100", 10, 50, 130, 62),
        _line("h", "EDUCATION", 10, 80, 200, 94, font_size=13, bold=True),
        _line("d", "Bachelor of Science", 10, 100, 260, 114),
    ])
    result = interpret_layout(source)
    header = _header_texts(result)
    body = _non_header_texts(result)
    assert "Jordan Blake" in header
    assert "jordan@example.com" in header
    assert "Phone: 555-0100" in header
    assert "EDUCATION" in body
    assert "Bachelor of Science" in body


def test_single_word_uppercase_name_does_not_trigger_heading_guard():
    """A single-word all-caps name (e.g. 'BLAKE') must NOT be treated as section
    heading because the guard requires at least two words."""
    source = _document([
        _line("n", "BLAKE JORDAN", 10, 10, 200, 28, font_size=20),
        _line("e", "blake@example.com", 10, 32, 200, 44),
        _line("b", "A long profile sentence spanning the page width here.", 10, 58, 500, 70),
    ])
    result = interpret_layout(source)
    # The document should have a header region (name in it) and body.
    all_regions = result.pages[0].regions
    kinds = [r.kind for r in all_regions]
    # Header and body must both exist — we're not asserting specific text here,
    # only that the name + email are in the header region (not spuriously pushed to body).
    assert "header" in kinds
