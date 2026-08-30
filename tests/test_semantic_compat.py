from tests.conftest import require_fixture
from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.semantic_compat import (
    semantic_header_to_text_blocks,
    semantic_sections_to_section_views,
    semantic_sections_to_text_blocks,
)
from app.pipeline.stages.semantic_paths import detect_region_aware_sections
from app.pipeline.stages.text_extraction import PDFExtractor, TextBlock


def _line(line_id: str, text: str, x0: float, y0: float, x1: float, y1: float, page: int = 1) -> Line:
    bbox = BoundingBox(x0, y0, x1, y1)
    span = Span(f"{line_id}-span-0", text, bbox, font_size=10.0, bold=True)
    return Line(line_id, page, bbox, [span], text, TextStyle(font_size=10.0, bold=True), source_span_ids=[span.span_id])


def _semantic_document():
    lines = [_line("line-0", "Summary text", 10, 10, 100, 20), _line("line-1", "Experience text", 200, 10, 300, 20)]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region("page-1-region-0", "column", BoundingBox(0, 0, 150, 100), [lines[0]], reading_order=0, column_id=0),
                    Region("page-1-region-1", "column", BoundingBox(150, 0, 350, 100), [lines[1]], reading_order=1, column_id=1),
                ],
            )
        ]
    )
    from app.pipeline.stages.semantic_paths import SemanticDocument, SemanticSection, SemanticPath

    paths = [
        SemanticPath("page-1-path-0", 1, "page-1-region-0", 0, [lines[0]]),
        SemanticPath("page-1-path-1", 1, "page-1-region-1", 1, [lines[1]]),
    ]
    return SemanticDocument(
        paths=paths,
        sections={
            "SUMMARY": [SemanticSection("SUMMARY", 1, "page-1-region-0", "page-1-path-0", [lines[0]])],
            "EXPERIENCE": [SemanticSection("EXPERIENCE", 1, "page-1-region-1", "page-1-path-1", [lines[1]])],
        },
    )


def test_semantic_sections_convert_to_textblocks_with_geometry_and_provenance():
    semantic = _semantic_document()
    sections = semantic_sections_to_text_blocks(semantic)

    block = sections["SUMMARY"][0]
    assert isinstance(block, TextBlock)
    assert block.text == "Summary text"
    assert (block.page_number, block.x0, block.y0, block.x1, block.y1) == (1, 10, 10, 100, 20)
    assert block.font_size == 10.0
    assert block.bold is True
    assert block.source_line_id == "line-0"
    assert block.source_span_ids == ["line-0-span-0"]


def test_section_views_preserve_path_and_region_boundaries():
    views = semantic_sections_to_section_views(_semantic_document())

    assert [(view.path_id, view.region_id, [block.text for block in view.blocks]) for view in views["SUMMARY"]] == [
        ("page-1-path-0", "page-1-region-0", ["Summary text"])
    ]
    assert [(view.path_id, view.region_id) for view in views["EXPERIENCE"]] == [("page-1-path-1", "page-1-region-1")]


def test_header_view_preserves_unassigned_contact_candidates():
    from app.pipeline.stages.semantic_paths import SemanticDocument

    line = _line("header", "david.perez@gmail.com", 10, 10, 150, 20)
    blocks = semantic_header_to_text_blocks(SemanticDocument(unassigned_lines=[line]))

    assert [block.text for block in blocks] == ["david.perez@gmail.com"]
    assert blocks[0].source_line_id == "header"
    assert blocks[0].source_span_ids == ["header-span-0"]


def test_conversion_is_deterministic_and_preserves_all_source_lines():
    semantic = _semantic_document()
    first = semantic_sections_to_text_blocks(semantic)
    second = semantic_sections_to_text_blocks(semantic)

    assert [(section, block.text, block.source_line_id) for section, blocks in first.items() for block in blocks] == [
        (section, block.text, block.source_line_id)
        for section, blocks in second.items()
        for block in blocks
    ]
    assert {block.source_line_id for blocks in first.values() for block in blocks} == {"line-0", "line-1"}


def test_resume_2_compatibility_representation_keeps_region_sections():
    raw = PDFExtractor.extract(require_fixture("resume_2.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    semantic = detect_region_aware_sections(document)
    sections = semantic_sections_to_text_blocks(semantic)
    views = semantic_sections_to_section_views(semantic)

    assert any("Administrative Assistant with 6+ years" in block.text for block in sections["SUMMARY"])
    assert any("September 2019" in block.text for block in sections["EXPERIENCE"])
    assert any("Bachelor Of Arts" in block.text for block in sections["EDUCATION"])
    assert any("Microsoft Office" in block.text for block in sections["SKILLS"])
    assert any("AWARD TITLE / Brand" in block.text for block in sections["ACHIEVEMENTS"])
    assert any("June 2015" in block.text and block.page_number == 2 for block in sections["EXPERIENCE"])
    all_blocks = [block for blocks in sections.values() for block in blocks]
    assert any(block.text == "david.perez@gmail.com" for block in all_blocks)
    assert any(block.text == "linkedin.com/in/davidperez" for block in all_blocks)
    assert all(view.path_id and view.region_id for section_views in views.values() for view in section_views)


def test_fresher_hr_compatibility_representation_keeps_education_and_skills():
    raw = PDFExtractor.extract(Path("tests/fixtures/fresher_hr_resume.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    semantic = detect_region_aware_sections(document)
    sections = semantic_sections_to_text_blocks(semantic)

    education_texts = {block.text for block in sections["EDUCATION"]}
    skills_texts = {block.text for block in sections["SKILLS"]}
    assert {"MBA (Human Resource Management) | School Of Open Learning , Delhi University (DU-SOL)", "2024 – 2026", "Bachelor of Arts (General) | BIR Tikendrajit University", "2020 – 2023"} <= education_texts
    assert "MS Excel (VLOOKUP, Pivot Tables, Filters)" in skills_texts
    assert "Human Resource Skills" not in education_texts
    assert "Tools & Technical Skills" not in education_texts
