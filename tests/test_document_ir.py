from app.domain.document import document_from_text_blocks
from app.pipeline.stages.text_extraction import TextBlock


def test_text_block_converts_to_deterministic_span_and_line():
    block = TextBlock(
        text="Senior Engineer",
        page_number=2,
        x0=12.5,
        y0=30.0,
        x1=140.25,
        y1=44.5,
        font_size=11.0,
        bold=True,
    )

    document = document_from_text_blocks([block])
    line = document.pages[0].regions[0].lines[0]
    span = line.spans[0]

    assert document.pages[0].page_number == 2
    assert line.line_id == "page-2-region-0-line-0"
    assert span.span_id == "page-2-region-0-line-0-span-0"
    assert line.text == block.text
    assert span.text == block.text
    assert line.bbox == span.bbox
    assert (line.bbox.x0, line.bbox.y0, line.bbox.x1, line.bbox.y1) == (
        block.x0,
        block.y0,
        block.x1,
        block.y1,
    )
    assert line.style.font_size == block.font_size
    assert line.style.bold is True
    assert span.font_size == block.font_size
    assert span.bold is True


def test_multiple_pages_preserve_text_and_physical_page_regions():
    blocks = [
        TextBlock("Page two", 2, 10, 20, 80, 30),
        TextBlock("Page one", 1, 5, 10, 70, 20),
        TextBlock("Second line", 1, 5, 25, 90, 35),
    ]

    document = document_from_text_blocks(blocks)

    assert [page.page_number for page in document.pages] == [1, 2]
    assert len(document.pages[0].regions) == 1
    assert document.pages[0].regions[0].kind == "physical_page"
    assert [line.text for line in document.pages[0].regions[0].lines] == [
        "Page one",
        "Second line",
    ]
    assert [line.text for line in document.pages[1].regions[0].lines] == ["Page two"]
    assert document.pages[0].regions[0].bbox.x0 == 5
    assert document.pages[0].regions[0].bbox.y0 == 10
    assert document.pages[0].regions[0].bbox.x1 == 90
    assert document.pages[0].regions[0].bbox.y1 == 35


def test_conversion_is_deterministic_and_does_not_drop_text():
    blocks = [
        TextBlock("First", 1, 1, 2, 3, 4, font_size=10.0),
        TextBlock("Second", 1, 5, 6, 7, 8, bold=False),
    ]

    first = document_from_text_blocks(blocks)
    second = document_from_text_blocks(blocks)

    first_lines = first.pages[0].regions[0].lines
    second_lines = second.pages[0].regions[0].lines
    assert [line.text for line in first_lines] == ["First", "Second"]
    assert first == second
    assert [span.text for line in first_lines for span in line.spans] == [
        block.text for block in blocks
    ]
