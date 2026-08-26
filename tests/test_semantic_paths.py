from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.semantic_paths import (
    build_semantic_paths,
    detect_region_aware_sections,
)
from app.pipeline.stages.text_extraction import PDFExtractor


def _document_from_lines(lines):
    pages = {}
    for line_id, page_number, text, x0, y0, x1, y1 in lines:
        bbox = BoundingBox(x0, y0, x1, y1)
        span = Span(f"{line_id}-span-0", text, bbox)
        line = Line(line_id, page_number, bbox, [span], text, TextStyle(), source_span_ids=[span.span_id])
        pages.setdefault(page_number, []).append(line)
    return Document(
        pages=[
            Page(number, regions=[Region(f"page-{number}-region-0", "physical_page", BoundingBox(0, 0, 800, 1000), lines=page_lines)])
            for number, page_lines in sorted(pages.items())
        ]
    )


def test_paths_keep_regions_and_do_not_globally_sort_lines():
    document = _document_from_lines([
        ("left-0", 1, "Skills", 10, 100, 60, 110),
        ("left-1", 1, "Python", 10, 200, 60, 210),
        ("right-0", 1, "Experience", 300, 20, 380, 30),
        ("right-1", 1, "Company", 300, 40, 380, 50),
    ])
    document.pages[0].regions = [
        Region("page-1-region-0", "column", BoundingBox(0, 0, 100, 300), document.pages[0].regions[0].lines[:2], reading_order=1, column_id=0),
        Region("page-1-region-1", "column", BoundingBox(290, 0, 400, 300), document.pages[0].regions[0].lines[2:], reading_order=0, column_id=1),
    ]

    paths = build_semantic_paths(document)

    assert [path.region_id for path in paths] == ["page-1-region-1", "page-1-region-0"]
    assert [line.text for line in paths[0].lines] == ["Experience", "Company"]
    assert {line.line_id for path in paths for line in path.lines} == {"left-0", "left-1", "right-0", "right-1"}


def test_region_aware_sections_keep_section_state_local():
    document = _document_from_lines([
        ("left-0", 1, "Skills", 10, 10, 60, 20),
        ("left-1", 1, "Python", 10, 30, 60, 40),
        ("right-0", 1, "Experience", 300, 10, 380, 20),
        ("right-1", 1, "Company", 300, 30, 380, 40),
    ])
    document.pages[0].regions = [
        Region("page-1-region-0", "column", BoundingBox(0, 0, 100, 100), document.pages[0].regions[0].lines[:2], reading_order=0, column_id=0),
        Region("page-1-region-1", "column", BoundingBox(290, 0, 400, 100), document.pages[0].regions[0].lines[2:], reading_order=1, column_id=1),
    ]

    sections = detect_region_aware_sections(document).sections

    assert [line.text for section in sections["SKILLS"] for line in section.lines] == ["Python"]
    assert [line.text for section in sections["EXPERIENCE"] for line in section.lines] == ["Company"]
    assert sections["SKILLS"][0].region_id != sections["EXPERIENCE"][0].region_id


def test_adjacent_content_uses_last_heading_from_multi_section_region():
    document = _document_from_lines([
        ("summary", 1, "Summary", 10, 10, 80, 20),
        ("summary-text", 1, "Profile", 10, 30, 80, 40),
        ("skills", 1, "Skill Highlights", 10, 60, 120, 70),
        ("skill-content", 1, "Innovative", 300, 80, 380, 90),
    ])
    document.pages[0].regions = [
        Region(
            "page-1-region-0",
            "header",
            BoundingBox(0, 0, 150, 100),
            document.pages[0].regions[0].lines[:3],
            reading_order=0,
        ),
        Region(
            "page-1-region-1",
            "column",
            BoundingBox(290, 0, 400, 100),
            document.pages[0].regions[0].lines[3:],
            reading_order=1,
        ),
    ]

    sections = detect_region_aware_sections(document).sections

    assert [line.text for item in sections["SKILLS"] for line in item.lines] == ["Innovative"]


def test_existing_alias_resource_recognizes_common_heading_variants():
    document = _document_from_lines([
        ("summary", 1, "Professional Profile", 10, 10, 100, 20),
        ("profile-text", 1, "Profile text", 10, 30, 100, 40),
        ("employment", 1, "Employment", 10, 60, 100, 70),
        ("employment-text", 1, "Employment text", 10, 80, 100, 90),
        ("certification", 1, "Certification", 10, 110, 100, 120),
        ("certification-text", 1, "Certification text", 10, 130, 100, 140),
    ])

    sections = detect_region_aware_sections(document).sections

    assert [line.text for item in sections["SUMMARY"] for line in item.lines] == ["Profile text"]
    assert [line.text for item in sections["EXPERIENCE"] for line in item.lines] == ["Employment text"]
    assert [line.text for item in sections["CERTIFICATIONS"] for line in item.lines] == ["Certification text"]


def test_real_resume_2_preserves_region_specific_section_paths():
    raw = PDFExtractor.extract(Path("tests/fixtures/resume_2.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    result = detect_region_aware_sections(document)

    def texts(section):
        return [line.text for item in result.sections[section] for line in item.lines]

    assert any("Administrative Assistant with 6+ years" in text for text in texts("SUMMARY"))
    assert any("(September 2019" in text for text in texts("EXPERIENCE"))
    assert any("Bachelor Of Arts" in text for text in texts("EDUCATION"))
    assert any("Microsoft Office" in text for text in texts("SKILLS"))
    assert any("AWARD TITLE / Brand" in text for text in texts("ACHIEVEMENTS"))
    assert not any("SUNTRUST FINANCIAL" in text for text in texts("ACHIEVEMENTS"))


def test_real_fixtures_have_lossless_unique_semantic_paths():
    for filename in ("resume_1.pdf", "resume_2.pdf", "resume_7.pdf", "fresher_hr_resume.pdf"):
        raw = PDFExtractor.extract(Path("tests/fixtures", filename).read_bytes())
        source = reconstruct_document(document_from_text_blocks(raw))
        result = detect_region_aware_sections(interpret_layout(source))
        source_ids = [span.span_id for page in source.pages for region in page.regions for line in region.lines for span in line.spans]
        path_ids = [span.span_id for path in result.paths for line in path.lines for span in line.spans]
        assert sorted(source_ids) == sorted(path_ids)
        assert len(path_ids) == len(set(path_ids))