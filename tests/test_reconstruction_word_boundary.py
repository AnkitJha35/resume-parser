"""Phase 6K: reconstruction word-boundary / compact separator decisions."""

from __future__ import annotations

from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.reconstruction import (
    _is_compact_adjacent_fragment,
    _merge_lines,
    reconstruct_document,
)


def _line(
    line_id: str,
    text: str,
    x0: float,
    y0: float = 20.0,
    *,
    font_size: float = 10.0,
    char_width: float | None = None,
) -> Line:
    width = (char_width if char_width is not None else font_size * 0.5) * max(len(text), 1)
    x1 = x0 + max(width, 1.0)
    y1 = y0 + font_size
    bbox = BoundingBox(x0, y0, x1, y1)
    span = Span(f"{line_id}-span-0", text, bbox, font_size=font_size)
    return Line(
        line_id,
        1,
        bbox,
        [span],
        text,
        TextStyle(font_size=font_size),
        source_span_ids=[span.span_id],
        reconstruction_method="physical",
    )


def _place_sequence(parts: list[str], *, gap: float = 9.75, font_size: float = 9.0, y0: float = 20.0) -> list[Line]:
    """Place fragments left-to-right with a fixed inter-fragment gap."""
    lines: list[Line] = []
    x0 = 10.0
    for index, text in enumerate(parts):
        line = _line(f"w{index}", text, x0, y0, font_size=font_size)
        lines.append(line)
        x0 = line.bbox.x1 + gap
    return lines


def _document(lines: list[Line]) -> Document:
    return Document(
        pages=[
            Page(
                1,
                regions=[
                    Region(
                        "page-1-region-0",
                        "physical_page",
                        BoundingBox(0, 0, 800, 400),
                        lines=list(lines),
                    )
                ],
            )
        ]
    )


def _merged_text(parts: list[str], *, gap: float = 9.75) -> str:
    return reconstruct_document(_document(_place_sequence(parts, gap=gap))).pages[0].regions[0].lines[0].text


def test_a_pairwise_alphabetic_words_get_space():
    assert _merged_text(["hello", "world"]) == "hello world"
    assert _merged_text(["Google", "Docs"]) == "Google Docs"


def test_b_long_accumulated_line_then_and_keeps_spaces():
    parts = ["JMS", "to", "manage", "high-volume", "synchronous", "and"]
    assert _merged_text(parts) == "JMS to manage high-volume synchronous and"


def test_c_short_function_words_not_glued_after_growth():
    assert "toa" not in _merged_text(["Ready", "to", "ship", "to"])
    assert _merged_text(["Ready", "to", "ship", "to"]) == "Ready to ship to"
    assert _merged_text(["one", "of", "many", "of"]) == "one of many of"
    assert _merged_text(["big", "and", "tall", "and"]) == "big and tall and"


def test_d_punctuation_attachment():
    assert _merged_text(["text", ","], gap=1.0) == "text,"
    assert _merged_text(["text", ")"], gap=1.0) == "text)"
    assert _merged_text(["(", "text"], gap=1.0) == "(text"


def test_e_hyphenated_word_fragments():
    assert _merged_text(["high", "-", "volume"], gap=1.0) == "high-volume"


def test_f_date_hyphen_fragments():
    assert _merged_text(["2025", "-", "2026"], gap=1.0) == "2025-2026"


def test_g_url_like_fragments_remain_unspaced():
    assert _merged_text(["linkedin.com/in/", "davidperez"], gap=2.0) == "linkedin.com/in/davidperez"
    assert _merged_text(["https://", "example.com/", "path"], gap=2.0) == "https://example.com/path"


def test_h_email_like_fragments_remain_unspaced():
    document = _document(
        [
            _line("e0", "david.perez@gmail.co", 10, 20, char_width=6.0),
            _line("e1", "m", 142, 20, char_width=6.0),
        ]
    )
    # Force adjacency like the existing reconstruction email test.
    left = document.pages[0].regions[0].lines[0]
    right = _line("e1", "m", left.bbox.x1 + 2.0, 20, char_width=6.0)
    assert reconstruct_document(_document([left, right])).pages[0].regions[0].lines[0].text == (
        "david.perez@gmail.com"
    )
    assert _merged_text(["name", "@", "example.com"], gap=1.0) == "name@example.com"


def test_i_long_word_sequence_remains_spaced():
    parts = ["Built", "a", "real", "time", "backend", "for", "solar", "performance"]
    assert _merged_text(parts) == "Built a real time backend for solar performance"


def test_j_compact_uses_atomic_preceding_fragment_not_merged_bbox():
    parts = ["JMS", "to", "manage", "high-volume", "synchronous"]
    lines = _place_sequence(parts, gap=9.75)
    acc = lines[0]
    for nxt in lines[1:]:
        acc = _merge_lines(acc, nxt)
    and_line = _place_sequence(["and"], gap=9.75)[0]
    # Place "and" just after the grown line, with a normal word gap.
    and_line = _line("and", "and", acc.bbox.x1 + 9.75, font_size=9.0)
    assert acc.bbox.x1 - acc.bbox.x0 > 150  # accumulated width is large
    assert _is_compact_adjacent_fragment(acc, and_line) is False
    assert _merge_lines(acc, and_line).text.endswith("synchronous and")


def test_resume_c_project_has_synchronous_and_not_glued():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    from app.domain.document import document_from_text_blocks
    from app.pipeline.stages.text_extraction import PDFExtractor

    document = reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes())))
    texts = [line.text for page in document.pages for region in page.regions for line in region.lines]
    joined = "\n".join(texts)
    assert "synchronousand" not in joined
    assert "synchronous and" in joined
