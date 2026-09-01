"""Phase 7B-14: multi-page section continuation regression tests."""

from __future__ import annotations

from pathlib import Path
import pytest

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.semantic_paths import (
    _path_opening_heading,
    detect_region_aware_sections,
)
from app.pipeline.parser import ResumeParser

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _make_line(
    line_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_number: int = 1,
    *,
    font_size: float = 10.0,
    bold: bool = False,
) -> Line:
    bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    span = Span(span_id=f"{line_id}-s0", text=text, bbox=bbox, font_size=font_size, bold=bold)
    return Line(
        line_id=line_id,
        page_number=page_number,
        bbox=bbox,
        spans=[span],
        text=text,
        style=TextStyle(font_size=font_size, bold=bold),
        source_span_ids=[span.span_id],
        reading_order=int(y0),
    )


def test_synthetic_multipage_continuation_preserves_section_before_later_heading():
    """Test A: Page 1 ends in PROJECTS; Page 2 continues with body lines before EDUCATION."""
    # Page 1: Single column with PROJECTS heading and Project 1
    p1_lines = [
        _make_line("p1-l0", "PROJECTS", 50.0, 100.0, 150.0, 115.0, page_number=1),
        _make_line("p1-l1", "Project Alpha", 50.0, 130.0, 200.0, 145.0, page_number=1),
        _make_line("p1-l2", "Built backend services using Python.", 50.0, 150.0, 400.0, 165.0, page_number=1),
    ]
    r1 = Region(
        region_id="p1-r0",
        kind="physical_region",
        bbox=BoundingBox(x0=50.0, y0=100.0, x1=500.0, y1=700.0),
        lines=p1_lines,
    )
    page1 = Page(page_number=1, width=612.0, height=792.0, regions=[r1])

    # Page 2: Starts with continuation of projects, then EDUCATION heading appears lower
    p2_lines = [
        _make_line("p2-l0", "Project Beta", 50.0, 50.0, 200.0, 65.0, page_number=2),
        _make_line("p2-l1", "Built full-stack application using React.", 50.0, 70.0, 400.0, 85.0, page_number=2),
        _make_line("p2-l2", "EDUCATION", 50.0, 300.0, 150.0, 315.0, page_number=2),
        _make_line("p2-l3", "Master of Science in Computer Science", 50.0, 330.0, 350.0, 345.0, page_number=2),
    ]
    r2 = Region(
        region_id="p2-r0",
        kind="physical_region",
        bbox=BoundingBox(x0=50.0, y0=50.0, x1=500.0, y1=700.0),
        lines=p2_lines,
    )
    page2 = Page(page_number=2, width=612.0, height=792.0, regions=[r2])

    doc = Document(pages=[page1, page2])
    detector = SectionDetector()
    sem_doc = detect_region_aware_sections(doc, detector)

    project_texts = [
        line.text
        for section in sem_doc.sections.get("PROJECTS", [])
        for line in section.lines
    ]
    education_texts = [
        line.text
        for section in sem_doc.sections.get("EDUCATION", [])
        for line in section.lines
    ]

    # Page 2 content before EDUCATION must remain in PROJECTS
    assert "Project Beta" in project_texts
    assert "Built full-stack application using React." in project_texts

    # Page 2 content after EDUCATION must belong to EDUCATION
    assert "Master of Science in Computer Science" in education_texts
    assert "Master of Science in Computer Science" not in project_texts


def test_opening_heading_prevents_incorrect_continuation():
    """Test B: Page 2 begins directly with EDUCATION heading; continuation is not inherited."""
    p1_lines = [
        _make_line("p1-l0", "PROJECTS", 50.0, 100.0, 150.0, 115.0, page_number=1),
        _make_line("p1-l1", "Project Alpha", 50.0, 130.0, 200.0, 145.0, page_number=1),
    ]
    r1 = Region(
        region_id="p1-r0",
        kind="physical_region",
        bbox=BoundingBox(x0=50.0, y0=100.0, x1=500.0, y1=700.0),
        lines=p1_lines,
    )
    page1 = Page(page_number=1, width=612.0, height=792.0, regions=[r1])

    # Page 2 opens with EDUCATION
    p2_lines = [
        _make_line("p2-l0", "EDUCATION", 50.0, 50.0, 150.0, 65.0, page_number=2),
        _make_line("p2-l1", "Master of Science", 50.0, 80.0, 300.0, 95.0, page_number=2),
    ]
    r2 = Region(
        region_id="p2-r0",
        kind="physical_region",
        bbox=BoundingBox(x0=50.0, y0=50.0, x1=500.0, y1=700.0),
        lines=p2_lines,
    )
    page2 = Page(page_number=2, width=612.0, height=792.0, regions=[r2])

    doc = Document(pages=[page1, page2])
    detector = SectionDetector()
    sem_doc = detect_region_aware_sections(doc, detector)

    project_texts = [
        line.text
        for section in sem_doc.sections.get("PROJECTS", [])
        for line in section.lines
    ]
    education_texts = [
        line.text
        for section in sem_doc.sections.get("EDUCATION", [])
        for line in section.lines
    ]

    assert "Master of Science" in education_texts
    assert "Master of Science" not in project_texts


def test_rajeev_multipage_continuation_preserves_projects_and_fixes_location():
    """Test C: Rajeev multi-page parse retains all projects and stops location leakage."""
    fixture = FIXTURES_DIR / "Rajeev_Ranjan_Prajapati_FullStack_Engineer.pdf"
    if not fixture.exists():
        pytest.skip("missing fixture: Rajeev_Ranjan_Prajapati_FullStack_Engineer.pdf")

    raw = fixture.read_bytes()
    resume = ResumeParser().parse_with_layout_pipeline(raw)

    # 1. Assert leaked project name is no longer used as personal location
    assert resume.personal.location != "TradesMan"

    # 2. Assert all project names are retained across pages in PROJECTS section
    from app.domain.document import document_from_text_blocks
    from app.pipeline.stages.text_extraction import PDFExtractor
    from app.pipeline.stages.reconstruction import reconstruct_document
    from app.pipeline.stages.layout import interpret_layout
    layout_doc = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(raw))))
    sem_doc = detect_region_aware_sections(layout_doc, SectionDetector())
    project_lines = [
        l.text
        for s in sem_doc.sections.get("PROJECTS", [])
        for l in s.lines
    ]

    assert any("Client Management Tool" in l for l in project_lines)
    assert any("TradesMan" in l for l in project_lines)
    assert any("Marsonik" in l for l in project_lines)
    assert any("Farmo" in l for l in project_lines)
    assert any("FleetAnalytix" in l for l in project_lines)
