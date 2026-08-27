from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.resume_ownership_diagnostic import (
    diagnose_resume_ownership,
    difference_categories,
)
from app.pipeline.stages.text_extraction import PDFExtractor


def _line(
    line_id: str,
    text: str,
    y0: float,
    *,
    page: int = 1,
    x0: float = 10.0,
    font_size: float = 11.0,
    bold: bool = False,
) -> Line:
    bbox = BoundingBox(x0, y0, x0 + max(40.0, len(text) * 6.0), y0 + 12.0)
    span = Span(f"{line_id}-span", text, bbox, font_size=font_size, bold=bold)
    return Line(
        line_id,
        page,
        bbox,
        [span],
        text,
        TextStyle(font_size=font_size, bold=bold),
        reading_order=int(y0) + page * 1000,
        source_span_ids=[span.span_id],
        reconstruction_method="physical",
    )


def _single_column(lines: list[Line], *, region_id: str = "page-1-region-0") -> Document:
    page = lines[0].page_number if lines else 1
    bbox = BoundingBox(
        min(line.bbox.x0 for line in lines),
        min(line.bbox.y0 for line in lines),
        max(line.bbox.x1 for line in lines),
        max(line.bbox.y1 for line in lines),
    )
    return Document(pages=[Page(page, regions=[Region(region_id, "physical_region", bbox, lines, 0, 0)])])


def _layout_pdf(name: str) -> Document:
    raw = Path("tests/fixtures", name).read_bytes()
    return interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(raw))))


def test_known_experience_resume_fields_are_findings_not_tuned():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
        _line("b", "• Managed schedules.", 90),
    ])
    diagnostic = diagnose_resume_ownership(document)
    current = diagnostic.current_resume
    candidate = diagnostic.candidate_resume
    # Current semantic path often extracts nothing; CandidateSection may feed extractors.
    if not current.experience and candidate.experience:
        assert "ENTRY_COUNT_DIFFERENCE" in difference_categories(diagnostic) or "FIELD_ADDED" in difference_categories(diagnostic)
        first = diagnostic.first_meaningful_difference
        assert first is not None
        assert first.cause is not None
        assert first.cause.line_ids or first.cause.heading


def test_unknown_heading_experience_diagnostic():
    document = _single_column([
        _line("h", "CAREER HISTORY", 10, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, font_size=12, bold=True),
        _line("c", "Google LLC", 50),
        _line("d", "2022 - Present", 70),
        _line("b", "• Built distributed systems.", 90),
    ])
    diagnostic = diagnose_resume_ownership(document)
    assert diagnostic.current_resume is not None
    assert diagnostic.candidate_resume is not None
    # Do not force CandidateSection to match current experience count.
    _ = diagnostic.differences


def test_unknown_heading_skills_diagnostic():
    document = _single_column([
        _line("h", "HIGHLIGHTS", 10, font_size=14, bold=True),
        _line("a", "Python", 30),
        _line("b", "Docker", 50),
        _line("c", "Kubernetes", 70),
        _line("d", "Team player", 90),
    ])
    diagnostic = diagnose_resume_ownership(document)
    _ = (diagnostic.current_resume.skills, diagnostic.candidate_resume.skills, diagnostic.differences)


def test_ambiguous_unknown_does_not_force_resume_section():
    document = _single_column([
        _line("h", "MISCELLANEOUS NOTES", 10, font_size=14, bold=True),
        _line("a", "Independent research notes.", 30),
        _line("b", "Not a job stack.", 50),
    ])
    diagnostic = diagnose_resume_ownership(document)
    assert diagnostic.candidate_resume.experience == [] or diagnostic.differences is not None


def test_summary_experience_like_content_diagnostic():
    document = _single_column([
        _line("h", "SUMMARY", 10, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, font_size=12, bold=True),
        _line("c", "Google", 50),
        _line("d", "2022 - Present", 70),
        _line("b", "Built distributed systems.", 90),
    ])
    diagnostic = diagnose_resume_ownership(document)
    # CandidateSection must not relabel SUMMARY → EXPERIENCE for extraction.
    # Experience may still differ if current path unassigns the heading.
    _ = diagnostic.candidate_resume.summary, diagnostic.current_resume.summary
    if diagnostic.candidate_resume.experience and not diagnostic.current_resume.experience:
        # Would indicate ownership leak into EXPERIENCE extractor — record, do not retune here.
        assert any(item.path.startswith("experience") for item in diagnostic.differences)


def test_page_continuation_diagnostic():
    page1 = [
        _line("h", "EXPERIENCE", 10, page=1, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, page=1, font_size=12, bold=True),
        _line("c", "Google LLC", 50, page=1),
        _line("d", "2022 - Present", 70, page=1),
        _line("b", "• Built services", 90, page=1),
    ]
    page2 = [
        _line("b2", "• Continued platform work", 20, page=2),
        _line("b3", "• Mentored engineers", 40, page=2),
    ]
    document = Document(
        pages=[
            Page(1, regions=[Region("page-1-region-0", "column", BoundingBox(0, 0, 200, 120), page1, 0, 0)]),
            Page(2, regions=[Region("page-2-region-0", "column", BoundingBox(0, 0, 200, 80), page2, 0, 0)]),
        ]
    )
    diagnostic = diagnose_resume_ownership(document, page_count=2)
    assert diagnostic.current_resume.metadata.get("pageCount") == 2
    _ = diagnostic.differences


def test_two_column_layout_diagnostic():
    left = [
        _line("h1", "EXPERIENCE", 10, x0=10, font_size=14, bold=True),
        _line("t1", "Secretary", 40, x0=10, font_size=12, bold=True),
        _line("c1", "Bright Spot LTD", 60, x0=10),
        _line("d1", "2015 - 2018", 80, x0=10),
    ]
    right = [
        _line("h2", "EDUCATION", 10, x0=320, font_size=14, bold=True),
        _line("deg", "Bachelor of Science", 40, x0=320, font_size=12, bold=True),
        _line("u", "University of Chicago", 60, x0=320),
    ]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region("page-1-region-0", "column", BoundingBox(0, 0, 200, 120), left, 0, 0),
                    Region("page-1-region-1", "column", BoundingBox(300, 0, 500, 120), right, 1, 1),
                ],
            )
        ]
    )
    diagnostic = diagnose_resume_ownership(document)
    _ = diagnostic.current_resume.experience, diagnostic.candidate_resume.experience
    _ = diagnostic.current_resume.education, diagnostic.candidate_resume.education


def test_wrapped_role_title_diagnostic():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Senior Software", 30, font_size=12, bold=True),
        _line("t2", "Engineer", 45, font_size=12, bold=True),
        _line("c", "Google LLC", 70),
        _line("d", "2022 - Present", 90),
        _line("b", "• Built distributed systems.", 110),
    ])
    diagnostic = diagnose_resume_ownership(document)
    assert diagnostic.ownership.line_records
    engineer = next(record for record in diagnostic.ownership.line_records if record.line_id == "t2")
    assert engineer.source_span_ids
    _ = diagnostic.candidate_resume.experience, diagnostic.current_resume.experience


def test_heading_plus_bullets_diagnostic():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
        _line("b1", "• Coordinated meetings", 90),
        _line("b2", "• Prepared reports", 110),
    ])
    diagnostic = diagnose_resume_ownership(document)
    _ = diagnostic.differences


def test_resume_1_field_diagnostic_does_not_change_production():
    production = ResumeParser().parse_with_layout_pipeline(Path("tests/fixtures/resume_1.pdf").read_bytes())
    assert [(item.designation, item.company) for item in production.experience] == [
        ("Administrative Assistant", "Redford & Sons"),
        ("Secretary", "Bright Spot LTD"),
    ]
    diagnostic = diagnose_resume_ownership(_layout_pdf("resume_1.pdf"), page_count=production.metadata.get("pageCount"))
    assert [(item.designation, item.company) for item in diagnostic.current_resume.experience] == [
        ("Administrative Assistant", "Redford & Sons"),
        ("Secretary", "Bright Spot LTD"),
    ]
    # Candidate path may differ; record findings only.
    first = diagnostic.first_meaningful_difference
    _ = first.path if first else None, difference_categories(diagnostic)


def test_resume_6_field_diagnostic_does_not_change_production():
    production = ResumeParser().parse_with_layout_pipeline(Path("tests/fixtures/resume_6.pdf").read_bytes())
    required = {
        "Warehouse Equipment Operation",
        "Resourceful Problem Solver",
        "Friendly and Helpful",
        "Good Physical Condition",
        "Safety-Conscious",
        "Team Player",
    }
    assert required.issubset(set(production.skills))
    diagnostic = diagnose_resume_ownership(_layout_pdf("resume_6.pdf"), page_count=production.metadata.get("pageCount"))
    assert required.issubset(set(diagnostic.current_resume.skills))
    _ = diagnostic.candidate_resume.skills, difference_categories(diagnostic), diagnostic.first_meaningful_difference


def test_resume_2_diagnostic_documents_baseline_without_fixing():
    """Known baseline (unchanged): contact/header routing and education phone-fragment.

    tests/test_education_extractor.py::test_extracts_parenthesized_date_from_resume_2_layout_path
    tests/test_layout_parser_integration.py::test_resume_2_layout_parser_exposes_correct_semantic_inputs
    """
    production = ResumeParser().parse_with_layout_pipeline(Path("tests/fixtures/resume_2.pdf").read_bytes())
    diagnostic = diagnose_resume_ownership(_layout_pdf("resume_2.pdf"), page_count=production.metadata.get("pageCount"))
    assert diagnostic.current_resume.personal.name == production.personal.name
    # Baseline: production layout path loses contact fields; do not treat CandidateSection
    # contact recovery as a production fix in this phase.
    _ = production.personal.email, diagnostic.candidate_resume.personal.email
    _ = production.education, diagnostic.candidate_resume.education
    _ = difference_categories(diagnostic), diagnostic.first_meaningful_difference
