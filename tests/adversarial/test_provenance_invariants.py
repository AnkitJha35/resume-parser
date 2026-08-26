from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.semantic_paths import detect_region_aware_sections
from app.pipeline.stages.sections import SectionDetector


def line(line_id, text, y):
    bbox = BoundingBox(10, y, 200, y + 10)
    span = Span(f"{line_id}-span", text, bbox)
    return Line(line_id, 1, bbox, [span], text, TextStyle(font_size=11), source_span_ids=[span.span_id])


def test_unknown_section_keeps_line_provenance_and_unknown_fallback():
    source = [line("h", "RANDOM INFORMATION", 10), line("c", "generic text", 30)]
    document = Document(pages=[Page(1, regions=[Region("r", "column", BoundingBox(0, 0, 300, 100), source)])])
    result = detect_region_aware_sections(document)
    candidate = result.unknown_candidates[0]
    assert candidate.inference.section == "UNKNOWN"
    assert candidate.heading.line_id == "h"
    assert candidate.content[0].source_span_ids == ["c-span"]
    assert candidate.content[0].bbox == source[1].bbox
    assert candidate.content[0].text == "generic text"
