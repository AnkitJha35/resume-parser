"""Tests for archetype-agnostic table detection.

Verifies that:
1. Standard CV with genuine tables (table_heavy_consulting_projects.pdf) has tables bound
   despite archetype == STANDARD_CV (no archetype gating).
2. Maritime tables (Mayur, Mukund) and structured forms (Josh) continue to have tables bound.
3. Multi-column CV layouts (modern_two_column_product_manager.pdf, swe_experienced_resume.pdf)
   are NOT falsely converted to tables.
4. Sparse non-table text is never converted to tables.
"""

from pathlib import Path
import pytest

from app.domain.document import (
    BoundingBox,
    Document,
    Line,
    Page,
    Region,
    Span,
    TextStyle,
    document_from_text_blocks,
)
from app.domain.semantic_contract import DocumentArchetype, build_semantic_input
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor


def _process_pdf_to_semantic_input(pdf_path: Path):
    if not pdf_path.exists():
        pytest.skip(f"Fixture {pdf_path} not found")
    raw_bytes = pdf_path.read_bytes()
    extracted = PDFExtractor.extract(raw_bytes)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)
    return build_semantic_input(layout, document_id=pdf_path.name)


def test_standard_cv_project_tables_are_detected_without_archetype_gate():
    """table_heavy_consulting_projects.pdf has genuine project tables that must be bound

    even though it is classified as STANDARD_CV.
    """
    pdf_path = Path("tests/fixtures/generalization/table_heavy_consulting_projects.pdf")
    sem_input = _process_pdf_to_semantic_input(pdf_path)

    assert sem_input.archetype == DocumentArchetype.STANDARD_CV
    table_blocks = [b for b in sem_input.blocks if b.table_id is not None]
    assert len(table_blocks) > 0, "Expected genuine tables to be bound in consulting CV"
    assert sem_input.tables is not None and len(sem_input.tables) > 0


def test_maritime_and_structured_forms_have_tables_bound():
    """Maritime resumes (Mayur, Mukund) and forms (Josh) must have tables bound."""
    for rel_path in [
        "tests/fixtures/2nd Officer Mayur Agarwal_062029.pdf",
        "tests/fixtures/MUKUND 3RD OFF CV 2026.pdf",
        "tests/fixtures/JOSH PARASHAR MASTER CV2.pdf",
    ]:
        pdf_path = Path(rel_path)
        sem_input = _process_pdf_to_semantic_input(pdf_path)
        table_blocks = [b for b in sem_input.blocks if b.table_id is not None]
        assert len(table_blocks) > 0, f"Expected tables to be bound in {rel_path}"


def test_two_column_cvs_are_not_converted_to_tables():
    """Standard 2-column resumes must NOT be mistakenly converted to tables."""
    for rel_path in [
        "tests/fixtures/generalization/modern_two_column_product_manager.pdf",
        "tests/fixtures/swe_experienced_resume.pdf",
    ]:
        pdf_path = Path(rel_path)
        sem_input = _process_pdf_to_semantic_input(pdf_path)
        table_blocks = [b for b in sem_input.blocks if b.table_id is not None]
        assert len(table_blocks) == 0, f"2-column resume {rel_path} was falsely detected as a table"


def test_sparse_non_table_text_is_not_converted_to_table():
    """Sparse text lines on a single page must not be converted to a table."""
    def _line(line_id: str, text: str, y: float) -> Line:
        bbox = BoundingBox(50, y, 200, y + 12)
        span = Span(f"{line_id}-s", text, bbox, font_size=10.0)
        return Line(line_id, 1, bbox, [span], text, TextStyle(font_size=10.0))

    lines = [
        _line("l1", "John Doe", 50),
        _line("l2", "Software Engineer", 70),
        _line("l3", "john@example.com", 90),
        _line("l4", "Summary of qualifications", 130),
        _line("l5", "Over ten years of experience in backend development.", 150),
    ]
    doc = Document(pages=[Page(1, regions=[Region("p1-r0", "physical_page", BoundingBox(0, 0, 600, 800), lines=lines)])])
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)
    sem_input = build_semantic_input(layout, document_id="sparse-test")

    table_blocks = [b for b in sem_input.blocks if b.table_id is not None]
    assert len(table_blocks) == 0
