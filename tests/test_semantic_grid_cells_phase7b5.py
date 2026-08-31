"""Phase 7B-5: semantic grid-cell / unknown-heading fix regression tests."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.semantic_paths import (
    _is_same_visual_row,
    _is_unknown_heading_line,
    detect_region_aware_sections,
)
from app.pipeline.stages.text_extraction import PDFExtractor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _shubham_fixture() -> Path:
    matches = sorted(glob.glob(str(FIXTURES_DIR / "*Shubham*.pdf")))
    if not matches:
        pytest.skip("missing fixture: *Shubham*.pdf")
    return Path(matches[0])


def _make_line(
    line_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    font_size: float = 10.0,
    bold: bool = False,
) -> Line:
    bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    span = Span(span_id=f"{line_id}-s0", text=text, bbox=bbox, font_size=font_size, bold=bold)
    return Line(
        line_id=line_id,
        page_number=1,
        bbox=bbox,
        spans=[span],
        text=text,
        style=TextStyle(font_size=font_size, bold=bold),
        source_span_ids=[span.span_id],
        reading_order=int(y0),
    )


def test_is_same_visual_row_tolerance():
    l1 = _make_line("l1", "Android Development", 10.0, 185.6, 120.0, 195.6)
    l2 = _make_line("l2", "MVC", 130.0, 185.6, 160.0, 195.6)
    assert _is_same_visual_row(l1, l2) is True

    l3 = _make_line("l3", "Slight Offset", 130.0, 186.2, 160.0, 196.2)
    assert _is_same_visual_row(l1, l3) is True

    l4 = _make_line("l4", "Different Row", 10.0, 200.0, 120.0, 210.0)
    assert _is_same_visual_row(l1, l4) is False


def test_same_row_grid_cell_not_unknown_heading():
    detector = SectionDetector()
    heading_line = _make_line("h1", "TECHNICAL SKILLS", 10.0, 50.0, 150.0, 62.0, font_size=12.0, bold=True)
    c1 = _make_line("c1", "Android Development", 10.0, 70.0, 120.0, 80.0)
    c2 = _make_line("c2", "Problem Solving", 130.0, 70.0, 220.0, 80.0)
    c3 = _make_line("c3", "MVC", 230.0, 70.0, 260.0, 80.0)  # looks like unknown heading if isolated
    following = _make_line("f1", "SQL / MYSQL", 10.0, 90.0, 80.0, 100.0)

    lines = [heading_line, c1, c2, c3, following]

    # Directly verify _is_unknown_heading_line rejects c3 due to same visual row as c2
    assert _is_unknown_heading_line(c3, lines, 3, detector, current_section="SKILLS") is False

    doc = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region(
                        region_id="reg-1",
                        kind="column",
                        bbox=BoundingBox(10.0, 50.0, 300.0, 120.0),
                        lines=lines,
                    )
                ],
            )
        ]
    )

    sem_doc = detect_region_aware_sections(doc, detector)
    skills_lines = [l.text for sec in sem_doc.sections.get("SKILLS", []) for l in sec.lines]

    assert "MVC" in skills_lines
    assert not any(l.text == "MVC" for l in sem_doc.unassigned_lines)
    assert not any(cand.heading.text == "MVC" for cand in sem_doc.unknown_candidates)


def test_genuine_heading_on_new_visual_row_remains_eligible():
    detector = SectionDetector()
    heading_line = _make_line("h1", "EXPERIENCE", 10.0, 50.0, 100.0, 62.0, font_size=12.0, bold=True)
    prev_line = _make_line("p1", "Software Engineer at Acme", 10.0, 70.0, 200.0, 80.0)
    unknown_hdr = _make_line("u1", "CORE COMPETENCIES", 10.0, 100.0, 150.0, 112.0, font_size=12.0, bold=True)
    item1 = _make_line("i1", "Python", 10.0, 120.0, 60.0, 130.0)
    item2 = _make_line("i2", "Django", 10.0, 135.0, 60.0, 145.0)

    lines = [heading_line, prev_line, unknown_hdr, item1, item2]

    # Unknown heading on new visual row must remain eligible
    assert _is_unknown_heading_line(unknown_hdr, lines, 2, detector, current_section="EXPERIENCE") is True


def test_shubham_mvc_ownership_and_no_regression():
    fixture_path = _shubham_fixture()
    raw = fixture_path.read_bytes()
    layout_doc = interpret_layout(
        reconstruct_document(document_from_text_blocks(PDFExtractor.extract(raw)))
    )
    sem_doc = detect_region_aware_sections(layout_doc, SectionDetector())

    skills_texts = [line.text for sec in sem_doc.sections.get("SKILLS", []) for line in sec.lines]
    experience_texts = [line.text for sec in sem_doc.sections.get("EXPERIENCE", []) for line in sec.lines]
    unassigned_texts = [line.text for line in sem_doc.unassigned_lines]

    # 1. MVC is owned by SKILLS
    assert "MVC" in skills_texts, f"'MVC' must be in SKILLS, got: {skills_texts}"

    # 2. MVC is not UNASSIGNED
    assert "MVC" not in unassigned_texts, f"'MVC' must not be unassigned"

    # 3. MVC is not owned by EXPERIENCE
    assert "MVC" not in experience_texts, f"'MVC' must not be in EXPERIENCE"

    # 4. React JS, AWS must not regress
    for chip in ("React JS", "AWS", "Android Development", "Problem Solving", "MongoDB", "Git"):
        assert chip in skills_texts, f"{chip!r} must be in SKILLS"
        assert chip not in experience_texts, f"{chip!r} must not be in EXPERIENCE"

    # SQL / MYSQL remains unassigned pending separate heading retention fix (Phase 7B-6/7)
    # and must not regress into EXPERIENCE
    assert "SQL / MYSQL" not in experience_texts

