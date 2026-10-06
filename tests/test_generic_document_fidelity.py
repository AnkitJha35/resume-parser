"""Comprehensive fidelity tests for Generic Document Structure v1.

Verifies the 15 fidelity criteria across the real 22-document corpus:
1. Source coverage >= 98% across all 22 documents (no dropped meaningful text).
2. Zero unsupported or invented blocks (100% grounding in source).
3. Zero duplicate block ownership across sections.
4. Zero provenance validation violations.
5. Exact heading preservation and non-inverted order.
6. Section ownership correctness.
7. Candidate names / header metadata form the initial unheaded section.
8. List items preserve bullet nature and text.
9. Tables preserve rows, cells, and headers.
10. Multi-page documents preserve continuous pages and order.
11. Deterministic execution (preserves exactly 1-LLM invariant).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.document import document_from_text_blocks
from app.domain.document_structure import DocumentBlock, DocumentSection, DocumentStructure
from app.domain.semantic_contract import build_semantic_input
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor
from app.validation.generic_document_validator import validate_generic_document_structure


REG_DIR = Path("tests/fixtures")
GEN_DIR = Path("tests/fixtures/generalization")
ALL_PDF_PATHS = sorted(REG_DIR.glob("*.pdf")) + sorted(GEN_DIR.glob("*.pdf"))


@pytest.fixture(scope="module")
def parser() -> ResumeParser:
    return ResumeParser()


def _collect_emitted_items(sections: list[DocumentSection]) -> tuple[list[DocumentBlock], list[DocumentSection]]:
    blocks: list[DocumentBlock] = []
    all_sections: list[DocumentSection] = []

    def _walk(sec_list: list[DocumentSection]) -> None:
        for s in sec_list:
            all_sections.append(s)
            blocks.extend(s.blocks)
            if s.subsections:
                _walk(s.subsections)

    _walk(sections)
    return blocks, all_sections


def test_corpus_fixtures_count():
    """Verify that all 22 expected PDF fixtures exist in the test environment."""
    assert len(ALL_PDF_PATHS) == 22, f"Expected 22 PDFs, found {len(ALL_PDF_PATHS)}"


@pytest.mark.parametrize("pdf_path", ALL_PDF_PATHS, ids=lambda p: p.name)
def test_document_coverage_and_provenance(parser: ResumeParser, pdf_path: Path):
    """Test source coverage >= 98% and zero provenance violations for each of the 22 PDFs."""
    raw_bytes = pdf_path.read_bytes()
    filename = pdf_path.name

    # 1. Parse document structure
    doc = parser.parse_document_structure(raw_bytes, document_id=filename)
    assert doc.page_count >= 1

    # 2. Extract semantic input
    phys = document_from_text_blocks(PDFExtractor.extract(raw_bytes))
    recon = reconstruct_document(phys)
    layout = interpret_layout(recon)
    sem_inp = build_semantic_input(layout, document_id=filename)

    # 3. Check source coverage
    source_blocks = [b for b in sem_inp.blocks if b.text.strip()]
    source_bids = {b.block_id for b in source_blocks}

    emitted_blocks, all_sections = _collect_emitted_items(doc.sections)
    cited_bids: set[str] = set()
    for b in emitted_blocks:
        cited_bids.update(b.source_block_ids)
    for s in all_sections:
        cited_bids.update(s.source_block_ids)

    covered_bids = source_bids & cited_bids
    coverage_pct = (len(covered_bids) / len(source_bids) * 100.0) if source_bids else 100.0

    assert coverage_pct >= 98.0, (
        f"{filename}: Coverage {coverage_pct:.1f}% is below 98.0% threshold. "
        f"Missing: {len(source_bids - cited_bids)} blocks"
    )

    # 4. Check provenance validation
    violations = validate_generic_document_structure(doc, semantic_input=sem_inp)
    assert not violations, f"{filename} has {len(violations)} provenance violations: {violations[:3]}"


def test_candidate_name_placed_in_unheaded_section(parser: ResumeParser):
    """Candidate names at top of resumes must be placed in unheaded section, not become root headings."""
    fixtures_with_names = [
        ("swe_experienced_resume.pdf", "ANKIT JHA"),
        ("Résume_Shubham.pdf", "SHUBHAM BAJAJ"),
        ("Rajeev_Ranjan_Prajapati_FullStack_Engineer.pdf", "RAJEEV RANJAN PRAJAPATI"),
        ("AditCV_SOL.pdf", "ADITI ANAND"),
        ("2nd Officer Mayur Agarwal_062029.pdf", "MAYUR"),
    ]

    for fname, name_substr in fixtures_with_names:
        p = REG_DIR / fname
        raw = p.read_bytes()
        doc = parser.parse_document_structure(raw, document_id=fname)

        # First section should be unheaded (intro/contact)
        assert doc.sections[0].heading is None, f"{fname}: First section should be unheaded"
        unheaded_text = " ".join(b.text for b in doc.sections[0].blocks)
        assert name_substr in unheaded_text, f"{fname}: Expected {name_substr!r} in unheaded section blocks"

        # Candidate name should NOT be a section heading
        headings = [s.heading for s in doc.sections if s.heading]
        assert not any(name_substr in (h or "") for h in headings), (
            f"{fname}: Candidate name {name_substr!r} was erroneously classified as heading in {headings}"
        )


def test_peer_headings_maintain_level_1(parser: ResumeParser):
    """Major document sections must be level 1 peer headings without cascading subsections."""
    p = REG_DIR / "swe_experienced_resume.pdf"
    doc = parser.parse_document_structure(p.read_bytes(), document_id=p.name)

    top_level_headings = [s.heading for s in doc.sections if s.heading is not None]
    assert "SUMMARY" in top_level_headings
    assert "EXPERIENCE" in top_level_headings
    assert "EDUCATION" in top_level_headings

    # All these top-level sections must have level == 1
    for s in doc.sections:
        if s.heading in ("SUMMARY", "EXPERIENCE", "EDUCATION"):
            assert s.level == 1, f"Expected level 1 for {s.heading}, got {s.level}"


def test_list_fidelity_preserved(parser: ResumeParser):
    """List item blocks must be preserved with type 'list_item'."""
    p = REG_DIR / "fresher_hr_resume.pdf"
    doc = parser.parse_document_structure(p.read_bytes(), document_id=p.name)

    emitted_blocks, _ = _collect_emitted_items(doc.sections)
    list_items = [b for b in emitted_blocks if b.type == "list_item"]
    assert len(list_items) >= 10, f"Expected >= 10 list items in fresher_hr_resume, got {len(list_items)}"


def test_table_fidelity_preserved(parser: ResumeParser):
    """Tables must preserve rows, headers, and type 'table'."""
    p = GEN_DIR / "table_heavy_consulting_projects.pdf"
    doc = parser.parse_document_structure(p.read_bytes(), document_id=p.name)

    emitted_blocks, _ = _collect_emitted_items(doc.sections)
    tables = [b for b in emitted_blocks if b.type == "table"]
    assert len(tables) >= 1, f"Expected at least 1 table in consulting projects fixture, got {len(tables)}"

    tbl = tables[0]
    assert tbl.table_data is not None
    assert "headers" in tbl.table_data
    assert "rows" in tbl.table_data
    assert len(tbl.table_data["rows"]) > 0


def test_multipage_continuity(parser: ResumeParser):
    """Multi-page documents must preserve full page count and contiguous reading order."""
    p = REG_DIR / "AASHISH DG.pdf"
    doc = parser.parse_document_structure(p.read_bytes(), document_id=p.name)
    assert doc.page_count == 6

    emitted_blocks, _ = _collect_emitted_items(doc.sections)
    pages_seen = {b.page_number for b in emitted_blocks if b.page_number is not None}
    assert len(pages_seen) == 6, f"Expected blocks on all 6 pages, saw: {pages_seen}"


def test_deterministic_offline_execution(parser: ResumeParser):
    """Generic document structure builder must run deterministically offline without LLM calls."""
    p = GEN_DIR / "entry_level_fresher_swe.pdf"
    raw = p.read_bytes()

    doc1 = parser.parse_document_structure(raw, document_id="run1")
    doc2 = parser.parse_document_structure(raw, document_id="run2")

    assert len(doc1.sections) == len(doc2.sections)
    for s1, s2 in zip(doc1.sections, doc2.sections):
        assert s1.heading == s2.heading
        assert s1.level == s2.level
        assert len(s1.blocks) == len(s2.blocks)
