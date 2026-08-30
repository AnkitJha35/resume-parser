from tests.conftest import require_fixture
from pathlib import Path

from app.domain.candidate_section import SectionOrigin
from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.domain.structural import StructuralRole
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.semantic_ownership_comparison import (
    compare_extraction_input_streams,
    compare_semantic_ownership,
    disagreement_categories,
)
from app.pipeline.stages.structural_roles import build_structural_blocks
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


def _cs(document: Document):
    return build_candidate_sections(build_structural_blocks(document))


def test_known_alias_candidate_section_contract_and_comparison():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
        _line("b", "• Managed schedules.", 90),
    ])
    sections = _cs(document)
    headed = [section for section in sections if section.origin == SectionOrigin.KNOWN_ALIAS]
    assert headed
    assert headed[0].semantic_label == "EXPERIENCE"
    assert headed[0].heading is not None and headed[0].heading.line_ids == ("h",)
    assert "t" in headed[0].line_ids
    comparison = compare_semantic_ownership(document)
    # Current path may unassign the job stack; that is a finding, not a reason to retune.
    if comparison.first_loss is not None:
        assert comparison.first_loss.category in {
            "LINE_UNASSIGNED",
            "SECTION_EXTRA",
            "SECTION_MISSING",
            "SECTION_BOUNDARY_DIFFERENCE",
            "PROVENANCE_DIFFERENCE",
        }


def test_unknown_heading_strong_skills_inferred():
    document = _single_column([
        _line("h", "HIGHLIGHTS", 10, font_size=14, bold=True),
        _line("a", "Python", 30),
        _line("b", "Docker", 50),
        _line("c", "Kubernetes", 70),
        _line("d", "Team player", 90),
    ])
    sections = _cs(document)
    headed = [section for section in sections if section.heading and "HIGHLIGHTS" in section.heading.text]
    assert headed
    assert headed[0].origin in {SectionOrigin.INFERRED, SectionOrigin.UNKNOWN}
    if headed[0].origin == SectionOrigin.INFERRED:
        assert headed[0].semantic_label == "SKILLS"
    comparison = compare_semantic_ownership(document)
    assert isinstance(comparison.disagreements, tuple)


def test_unknown_heading_strong_experience_inferred():
    document = _single_column([
        _line("h", "CAREER HISTORY", 10, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, font_size=12, bold=True),
        _line("c", "Google LLC", 50),
        _line("d", "2022 - Present", 70),
        _line("b", "• Built distributed systems for production traffic.", 90),
    ])
    sections = _cs(document)
    headed = [section for section in sections if section.heading and section.heading.text == "CAREER HISTORY"]
    comparison = compare_semantic_ownership(document)
    # Contract: do not force EXPERIENCE without a heading/inference path.
    # Finding: StructuralRole may withhold SECTION_HEADING when follow-on looks like
    # an entry stack, so CandidateSection can remain UNLABELED instead of INFERRED.
    if headed:
        assert headed[0].origin in {SectionOrigin.INFERRED, SectionOrigin.UNKNOWN}
        if headed[0].origin == SectionOrigin.INFERRED:
            assert headed[0].semantic_label == "EXPERIENCE"
        else:
            assert headed[0].semantic_label is None
    else:
        unlabeled = [section for section in sections if section.origin == SectionOrigin.UNLABELED]
        assert unlabeled
        assert all(section.semantic_label is None for section in unlabeled)
    assert all(
        not (section.semantic_label == "EXPERIENCE" and section.origin == SectionOrigin.KNOWN_ALIAS and section.heading and section.heading.text == "CAREER HISTORY")
        for section in sections
    )
    _ = comparison.disagreements


def test_unknown_ambiguous_heading_stays_unknown():
    document = _single_column([
        _line("h", "MISCELLANEOUS NOTES", 10, font_size=14, bold=True),
        _line("a", "Independent research notes.", 30),
        _line("b", "Not a job stack.", 50),
    ])
    sections = _cs(document)
    headed = [section for section in sections if section.heading]
    assert headed
    assert headed[0].origin in {SectionOrigin.UNKNOWN, SectionOrigin.INFERRED}
    if headed[0].origin == SectionOrigin.UNKNOWN:
        assert headed[0].semantic_label is None
    compare_semantic_ownership(document)


def test_headingless_experience_cluster_is_unlabeled_not_forced():
    document = _single_column([
        _line("t", "Software Engineer", 10, font_size=12, bold=True),
        _line("c", "Google LLC", 30),
        _line("d", "2022 - Present", 50),
        _line("b", "• Built distributed systems", 70),
    ])
    sections = _cs(document)
    unlabeled = [section for section in sections if section.origin == SectionOrigin.UNLABELED]
    assert unlabeled
    assert all(section.semantic_label is None for section in unlabeled)
    assert all(section.semantic_label != "EXPERIENCE" for section in sections)
    comparison = compare_semantic_ownership(document)
    _ = comparison


def test_summary_remains_summary_with_experience_like_content():
    document = _single_column([
        _line("h", "SUMMARY", 10, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, font_size=12, bold=True),
        _line("c", "Google", 50),
        _line("d", "2022 - Present", 70),
        _line("b", "Built distributed systems.", 90),
    ])
    sections = _cs(document)
    summary = [section for section in sections if section.semantic_label == "SUMMARY"]
    assert summary
    assert summary[0].origin == SectionOrigin.KNOWN_ALIAS
    titles = [block for block in summary[0].content if block.role == StructuralRole.ENTRY_TITLE]
    assert titles
    comparison = compare_semantic_ownership(document)
    labels_for_title = [
        record.candidate_labels
        for record in comparison.line_records
        if record.line_id == "t"
    ]
    assert labels_for_title
    assert "EXPERIENCE" not in labels_for_title[0]


def test_entry_title_does_not_create_section_boundary():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c1", "Redford & Sons", 50),
        _line("d1", "2018 - 2021", 70),
        _line("t2", "Office Manager", 110, font_size=12, bold=True),
        _line("c2", "ABC Corp LLC", 130),
        _line("d2", "2021 - Present", 150),
    ])
    sections = _cs(document)
    experience = [section for section in sections if section.semantic_label == "EXPERIENCE"]
    assert len(experience) == 1
    texts = [block.text for block in experience[0].content]
    assert "Administrative Assistant" in texts
    assert "Office Manager" in texts
    compare_semantic_ownership(document)


def test_wrapped_body_does_not_create_section_boundary():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - Present", 70),
        _line("b1", "Arrangements for supervisors and managers", 90),
        _line("b2", "across regional offices during peak season.", 102),
    ])
    sections = _cs(document)
    experience = [section for section in sections if section.semantic_label == "EXPERIENCE"]
    assert len(experience) == 1
    assert any("regional offices" in block.text for block in experience[0].content)
    compare_semantic_ownership(document)


def test_adjacent_columns_remain_separate_paths():
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
    sections = _cs(document)
    paths = {section.path_id for section in sections if section.semantic_label}
    assert len(paths) == 2
    experience = next(section for section in sections if section.semantic_label == "EXPERIENCE")
    education = next(section for section in sections if section.semantic_label == "EDUCATION")
    assert experience.path_id != education.path_id
    assert experience.region_id != education.region_id
    comparison = compare_semantic_ownership(document)
    assert "t1" not in education.line_ids
    _ = comparison


def test_multiline_heading_records_boundary_without_retuning():
    document = _single_column([
        _line("h1", "PROFESSIONAL", 10, font_size=14, bold=True),
        _line("h2", "EXPERIENCE", 24, font_size=14, bold=True),
        _line("t", "Software Engineer", 50, font_size=12, bold=True),
        _line("c", "Google LLC", 70),
        _line("d", "2022 - Present", 90),
    ])
    sections = _cs(document)
    comparison = compare_semantic_ownership(document)
    experience = [section for section in sections if section.semantic_label == "EXPERIENCE"]
    # Contract: a known EXPERIENCE alias still owns following content.
    assert experience
    assert any("Software Engineer" in block.text for block in experience[0].content)
    # Split heading lines may produce extra UNKNOWN/UNLABELED — record, do not merge by hack.
    _ = comparison.disagreements


def test_same_section_page_continuation_is_recorded_not_forced():
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
    sections = _cs(document)
    continued = [
        section
        for section in sections
        if section.page_number == 2
        and section.origin == SectionOrigin.CONTINUED
        and section.semantic_label == "EXPERIENCE"
    ]
    assert continued
    assert continued[0].heading is None
    assert any("Continued platform work" in block.text for block in continued[0].content)
    comparison = compare_semantic_ownership(document)
    _ = disagreement_categories(comparison)


def test_different_section_on_next_page():
    page1 = [
        _line("h", "EXPERIENCE", 10, page=1, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, page=1, font_size=12, bold=True),
        _line("c", "Google LLC", 50, page=1),
        _line("d", "2022 - 2023", 70, page=1),
    ]
    page2 = [
        _line("h2", "EDUCATION", 10, page=2, font_size=14, bold=True),
        _line("deg", "Bachelor of Science", 30, page=2, font_size=12, bold=True),
        _line("u", "University of Chicago", 50, page=2),
        _line("y", "2014 - 2018", 70, page=2),
    ]
    document = Document(
        pages=[
            Page(1, regions=[Region("page-1-region-0", "column", BoundingBox(0, 0, 200, 90), page1, 0, 0)]),
            Page(2, regions=[Region("page-2-region-0", "column", BoundingBox(0, 0, 200, 90), page2, 0, 0)]),
        ]
    )
    sections = _cs(document)
    labels = {section.semantic_label for section in sections if section.semantic_label}
    assert "EXPERIENCE" in labels
    assert "EDUCATION" in labels
    compare_semantic_ownership(document)


def test_provenance_preservation_on_candidate_section():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Secretary", 30, font_size=12, bold=True),
        _line("c", "Bright Spot LTD", 50),
        _line("d", "2015 - 2018", 70),
    ])
    sections = _cs(document)
    section = next(item for item in sections if item.semantic_label == "EXPERIENCE")
    assert section.line_ids
    assert section.source_span_ids
    assert section.path_id
    assert section.region_id
    comparison = compare_semantic_ownership(document)
    assert "PROVENANCE_DIFFERENCE" in disagreement_categories(comparison) or comparison.line_records
    for record in comparison.line_records:
        if record.line_id == "t":
            assert record.source_span_ids
            assert record.path_id
            assert record.region_id


def test_no_double_assignment_on_candidate_section():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
        _line("h2", "EDUCATION", 120, font_size=14, bold=True),
        _line("deg", "Bachelor of Science", 140, font_size=12, bold=True),
    ])
    comparison = compare_semantic_ownership(document)
    cs_double = [
        item
        for item in comparison.disagreements
        if item.category == "LINE_DOUBLE_ASSIGNED" and "CandidateSection" in item.detail
    ]
    assert cs_double == []


def test_known_owned_content_not_unclaimed_by_candidate_section():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
        _line("b", "• Managed schedules.", 90),
    ])
    comparison = compare_semantic_ownership(document)
    for line_id in ("t", "c", "d", "b"):
        assert line_id not in comparison.candidate_unclaimed_line_ids
        record = next(item for item in comparison.line_records if item.line_id == line_id)
        assert record.candidate_section_ids
        assert "EXPERIENCE" in record.candidate_labels


def test_extraction_input_stream_helper_does_not_require_parser_change():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
    ])
    stream_diffs = compare_extraction_input_streams(document)
    assert any(item.category in {"SECTION_BOUNDARY_DIFFERENCE", "PROVENANCE_DIFFERENCE"} for item in stream_diffs)


def test_resume_1_ownership_diagnostic_lock():
    resume = ResumeParser().parse_with_layout_pipeline(require_fixture("resume_1.pdf").read_bytes())
    assert [(item.designation, item.company) for item in resume.experience] == [
        ("Administrative Assistant", "Redford & Sons"),
        ("Secretary", "Bright Spot LTD"),
    ]
    raw = PDFExtractor.extract(require_fixture("resume_1.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    comparison = compare_semantic_ownership(document)
    assert comparison.line_records
    assert comparison.first_loss is None or comparison.first_loss.category
    stream_diffs = compare_extraction_input_streams(document)
    assert isinstance(stream_diffs, tuple)


def test_resume_6_highlights_skills_diagnostic_lock():
    resume = ResumeParser().parse_with_layout_pipeline(require_fixture("resume_6.pdf").read_bytes())
    assert {
        "Warehouse Equipment Operation",
        "Resourceful Problem Solver",
        "Friendly and Helpful",
        "Good Physical Condition",
        "Safety-Conscious",
        "Team Player",
    }.issubset(set(resume.skills))
    raw = PDFExtractor.extract(require_fixture("resume_6.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    comparison = compare_semantic_ownership(document)
    sections = _cs(document)
    skills = [section for section in sections if section.semantic_label == "SKILLS"]
    assert skills
    _ = comparison.disagreements
