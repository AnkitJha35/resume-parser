from tests.conftest import require_fixture
from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.semantic_paths import (
    _is_unknown_heading_line,
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
    raw = PDFExtractor.extract(require_fixture("resume_2.pdf").read_bytes())
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
        raw = PDFExtractor.extract(require_fixture(filename).read_bytes())
        source = reconstruct_document(document_from_text_blocks(raw))
        result = detect_region_aware_sections(interpret_layout(source))
        source_ids = [span.span_id for page in source.pages for region in page.regions for line in region.lines for span in line.spans]
        path_ids = [span.span_id for path in result.paths for line in path.lines for span in line.spans]
        assert sorted(source_ids) == sorted(path_ids)
        assert len(path_ids) == len(set(path_ids))


def _styled_line(line_id, text, y0, *, font_size=11.0, bold=False, x0=10.0):
    bbox = BoundingBox(x0, y0, x0 + max(40.0, len(text) * 6.0), y0 + 12.0)
    span = Span(f"{line_id}-span", text, bbox, font_size=font_size, bold=bold)
    return Line(
        line_id,
        1,
        bbox,
        [span],
        text,
        TextStyle(font_size=font_size, bold=bold),
        reading_order=int(y0),
        source_span_ids=[span.span_id],
        reconstruction_method="physical",
    )


def _single_column(lines):
    bbox = BoundingBox(
        min(line.bbox.x0 for line in lines),
        min(line.bbox.y0 for line in lines),
        max(line.bbox.x1 for line in lines),
        max(line.bbox.y1 for line in lines),
    )
    return Document(
        pages=[Page(1, regions=[Region("page-1-region-0", "physical_region", bbox, lines, 0, 0)])]
    )


def _texts(result, section):
    return [line.text for item in result.sections[section] for line in item.lines]


def _ids(result, section):
    return {line.line_id for item in result.sections[section] for line in item.lines}


def _unassigned_ids(result):
    return {line.line_id for line in result.unassigned_lines}


def test_active_achievements_keeps_title_org_year_stack():
    document = _single_column([
        _styled_line("h", "AWARDS", 10, font_size=14, bold=True),
        _styled_line("t", "Honor Citation", 30, font_size=12, bold=True),
        _styled_line("o", "Civic League", 50),
        _styled_line("y", "2015", 70),
    ])
    result = detect_region_aware_sections(document)
    owned = _ids(result, "ACHIEVEMENTS")
    assert owned >= {"t", "o", "y"}
    assert "t" not in _unassigned_ids(result)
    assert "o" not in _ids(result, "EDUCATION")
    assert "y" not in _ids(result, "EDUCATION")
    assert _texts(result, "EDUCATION") == []


def test_active_achievements_keeps_multiple_award_entries():
    document = _single_column([
        _styled_line("h", "AWARDS", 10, font_size=14, bold=True),
        _styled_line("t1", "Honor Citation", 30, font_size=12, bold=True),
        _styled_line("o1", "Civic League", 50),
        _styled_line("y1", "2019", 70),
        _styled_line("t2", "Service Medal", 90, font_size=12, bold=True),
        _styled_line("o2", "County Board", 110),
        _styled_line("y2", "2021", 130),
    ])
    result = detect_region_aware_sections(document)
    owned = _ids(result, "ACHIEVEMENTS")
    assert owned >= {"t1", "o1", "y1", "t2", "o2", "y2"}
    assert not ({"t1", "t2", "y1", "y2"} & _ids(result, "EDUCATION"))
    assert _texts(result, "EDUCATION") == []


def test_education_degree_stack_stays_education():
    document = _single_column([
        _styled_line("h", "EDUCATION", 10, font_size=14, bold=True),
        _styled_line("d", "Bachelor of Science", 30, font_size=12, bold=True),
        _styled_line("u", "State University", 50),
        _styled_line("y", "2018", 70),
    ])
    result = detect_region_aware_sections(document)
    owned = _ids(result, "EDUCATION")
    assert owned >= {"d", "u", "y"}
    assert _texts(result, "ACHIEVEMENTS") == []


def test_highlights_infer_skills_when_no_active_section():
    document = _single_column([
        _styled_line("h", "HIGHLIGHTS", 10, font_size=14, bold=True),
        _styled_line("a", "Team player", 30),
        _styled_line("b", "Safety-conscious", 50),
        _styled_line("c", "Problem solving", 70),
    ])
    result = detect_region_aware_sections(document)
    skills = _texts(result, "SKILLS")
    assert "Team player" in skills
    assert "Safety-conscious" in skills
    assert result.unknown_candidates
    assert result.unknown_candidates[0].inference.section == "SKILLS"


def test_active_experience_job_stack_not_education():
    document = _single_column([
        _styled_line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _styled_line("t", "Software Engineer", 30, font_size=12, bold=True),
        _styled_line("c", "Example Corp LLC", 50),
        _styled_line("d", "2020 - 2023", 70),
    ])
    result = detect_region_aware_sections(document)
    owned = _ids(result, "EXPERIENCE")
    assert owned >= {"t", "c", "d"}
    assert _texts(result, "EDUCATION") == []


def test_summary_year_in_prose_does_not_become_education():
    document = _single_column([
        _styled_line("h", "SUMMARY", 10, font_size=14, bold=True),
        _styled_line("p", "Platform engineer with production work since 2018.", 30),
        _styled_line("q", "Seeking a senior role delivering reliable services.", 50),
    ])
    result = detect_region_aware_sections(document)
    assert any("2018" in text for text in _texts(result, "SUMMARY"))
    assert _texts(result, "EDUCATION") == []


def test_active_experience_keeps_split_job_title_and_unsuffixed_org():
    """Job title + unsuffixed org + separate date stay in EXPERIENCE (not unassigned)."""
    document = _single_column([
        _styled_line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _styled_line("t", "Software Engineer", 30, font_size=12, bold=True),
        _styled_line("c", "Northwind Partners", 50),
        _styled_line("d", "2018 - 2021", 70),
        _styled_line("b", "• Built internal scheduling tools.", 90),
    ])
    result = detect_region_aware_sections(document)
    experience = _texts(result, "EXPERIENCE")
    assert "Software Engineer" in experience
    assert "Northwind Partners" in experience
    assert "2018 - 2021" in experience
    assert any("scheduling tools" in text for text in experience)
    assigned = {line.line_id for item in result.sections["EXPERIENCE"] for line in item.lines}
    assert "t" in assigned
    assert "t" not in {line.line_id for line in result.unassigned_lines}


def test_ampersand_organization_is_entry_metadata_not_a_section_heading():
    document = _single_column([
        _styled_line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _styled_line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _styled_line("c", "Redford & Sons", 50),
        _styled_line("d", "2018 - 2021", 70),
        _styled_line("b", "• Managed schedules.", 90),
    ])
    result = detect_region_aware_sections(document)
    experience = _texts(result, "EXPERIENCE")
    assert "Administrative Assistant" in experience
    assert "Redford & Sons" in experience
    assert {line.line_id for line in result.unassigned_lines} == {"h"}


def test_combined_company_date_line_still_protects_item_heading():
    lines = [
        _styled_line("role", "ADMINISTRATIVE ASSISTANT", 10, font_size=12, bold=True),
        _styled_line("company", "Example Company, Boston, MA / September 2018 - Present", 30),
        _styled_line("description", "Schedule and coordinate meetings", 50),
    ]
    assert _is_unknown_heading_line(lines[0], lines, 0, SectionDetector(), "EXPERIENCE") is False


def test_career_history_infers_experience_when_no_active_section():
    document = _single_column([
        _styled_line("h", "CAREER HISTORY", 10, font_size=14, bold=True),
        _styled_line("t", "Software Engineer", 30, font_size=12, bold=True),
        _styled_line("c", "Example Corp LLC", 50),
        _styled_line("d", "2020 - 2023", 70),
        _styled_line("b", "Delivered production services for internal teams.", 90),
    ])
    result = detect_region_aware_sections(document)
    assert result.unknown_candidates
    assert result.unknown_candidates[0].heading.text == "CAREER HISTORY"
    experience = _texts(result, "EXPERIENCE")
    assert "Example Corp LLC" in experience or "Software Engineer" in experience or experience


def test_summary_with_experience_like_prose_stays_summary():
    document = _single_column([
        _styled_line("h", "SUMMARY", 10, font_size=14, bold=True),
        _styled_line("p", "Experienced software engineer seeking a platform role.", 30),
        _styled_line("q", "Led delivery of distributed systems from 2020 to 2023.", 50),
    ])
    result = detect_region_aware_sections(document)
    summary = _texts(result, "SUMMARY")
    assert any("platform role" in text for text in summary)
    assert _texts(result, "EXPERIENCE") == []


def test_wrapped_experience_body_stays_in_experience():
    document = _single_column([
        _styled_line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _styled_line("t", "Software Engineer", 30, font_size=12, bold=True),
        _styled_line("c", "Example Corp LLC", 50),
        _styled_line("d", "2018 - Present", 70),
        _styled_line("b1", "Arrangements for supervisors and managers", 90),
        _styled_line("b2", "across regional offices during peak season.", 102),
    ])
    result = detect_region_aware_sections(document)
    experience = _texts(result, "EXPERIENCE")
    assert any("regional offices" in text for text in experience)
    assert "b2" not in {line.line_id for line in result.unassigned_lines}


def test_ambiguous_unknown_heading_stays_unassigned_without_active_section():
    document = _single_column([
        _styled_line("h", "MISCELLANEOUS NOTES", 10, font_size=14, bold=True),
        _styled_line("a", "Independent research notes.", 30),
        _styled_line("b", "Not a job stack.", 50),
    ])
    result = detect_region_aware_sections(document)
    assert result.unknown_candidates
    assert result.unknown_candidates[0].inference.section == "UNKNOWN"
    for name in ("EXPERIENCE", "SKILLS", "SUMMARY", "EDUCATION"):
        assert _texts(result, name) == []
    unassigned = {line.line_id for line in result.unassigned_lines}
    assert {"h", "a", "b"} <= unassigned


def _prose(line_id: str, text: str, y0: float) -> Line:
    return _styled_line(line_id, text, y0, font_size=10.0)


def _header_then_body(header_lines: list[Line], body_lines: list[Line]) -> Document:
    header_bbox = BoundingBox(
        min(line.bbox.x0 for line in header_lines),
        min(line.bbox.y0 for line in header_lines),
        max(line.bbox.x1 for line in header_lines),
        max(line.bbox.y1 for line in header_lines),
    )
    body_bbox = BoundingBox(
        min(line.bbox.x0 for line in body_lines),
        min(line.bbox.y0 for line in body_lines),
        max(line.bbox.x1 for line in body_lines),
        max(line.bbox.y1 for line in body_lines),
    )
    regions = [
        Region("page-1-region-0", "header", header_bbox, header_lines, 0, None),
        Region("page-1-region-1", "physical_region", body_bbox, body_lines, 1, None),
    ]
    return Document(pages=[Page(1, regions=regions)])


def test_headingless_paragraph_before_education_may_be_summary():
    document = _header_then_body(
        [
            _styled_line("n", "Jordan Blake", 10, font_size=18),
            _styled_line("e", "jordan@example.com", 30),
        ],
        [
            _prose("p1", "A motivated candidate with experience across operations and delivery of programs for several teams.", 70),
            _prose("p2", "Eager to contribute to team outcomes while continuing to develop professional skills on the job.", 84),
            _prose("p3", "Comfortable collaborating across functions in a fast-paced environment with shifting priorities.", 98),
            _prose("p4", "Looking ahead to apply academic training while supporting day to day operational work.", 112),
            _styled_line("d", "Bachelor of Science | State University", 180),
            _styled_line("y", "2020-2024", 180, x0=450.0),
        ],
    )
    result = detect_region_aware_sections(document)
    summary = _texts(result, "SUMMARY")
    assert "p1" in _ids(result, "SUMMARY")
    assert any("motivated candidate" in text for text in summary)
    assert any("Bachelor of Science" in text for text in _texts(result, "EDUCATION"))


def test_short_headingless_prose_before_education_stays_unassigned():
    document = _header_then_body(
        [
            _styled_line("n", "Jordan Blake", 10, font_size=18),
            _styled_line("e", "jordan@example.com", 30),
        ],
        [
            _prose("p1", "Here is a brief note about recent independent study work.", 70),
            _prose("p2", "It continues onto a second line without more structure.", 84),
            _styled_line("d", "Bachelor of Science | State University", 140),
        ],
    )
    result = detect_region_aware_sections(document)
    assert "p1" not in _ids(result, "SUMMARY")
    assert "p2" not in _ids(result, "SUMMARY")
    assert any("Bachelor of Science" in text for text in _texts(result, "EDUCATION"))


def test_one_line_tagline_before_education_is_not_summary():
    document = _header_then_body(
        [
            _styled_line("n", "Jordan Blake", 10, font_size=18),
            _styled_line("e", "jordan@example.com", 30),
        ],
        [
            _styled_line("t", "Seeking internships", 70, bold=True),
            _styled_line("d", "Bachelor of Science | State University", 120),
        ],
    )
    result = detect_region_aware_sections(document)
    assert _texts(result, "SUMMARY") == []
    assert "t" not in _ids(result, "SUMMARY")


def test_skill_list_before_education_is_not_summary():
    document = _header_then_body(
        [
            _styled_line("n", "Jordan Blake", 10, font_size=18),
            _styled_line("e", "jordan@example.com", 30),
        ],
        [
            _styled_line("h", "Skills", 70, font_size=13, bold=True),
            _styled_line("b1", "•", 90),
            _styled_line("s1", "Python", 90, x0=30.0),
            _styled_line("b2", "•", 110),
            _styled_line("s2", "SQL", 110, x0=30.0),
            _styled_line("d", "Bachelor of Science | State University", 160),
        ],
    )
    result = detect_region_aware_sections(document)
    assert "Python" not in _texts(result, "SUMMARY")
    assert "Python" in _texts(result, "SKILLS")


def test_date_organization_stack_is_not_summary():
    document = _header_then_body(
        [
            _styled_line("n", "Jordan Blake", 10, font_size=18),
            _styled_line("e", "jordan@example.com", 30),
        ],
        [
            _styled_line("h", "Experience", 70, font_size=13, bold=True),
            _styled_line("d", "2019-2022", 90),
            _styled_line("c", "Example Corp LLC", 110),
            _styled_line("b", "• Delivered internal tools for operations teams.", 130),
        ],
    )
    result = detect_region_aware_sections(document)
    assert _texts(result, "SUMMARY") == []
    assert "Example Corp LLC" in _texts(result, "EXPERIENCE")


def test_degree_institution_at_body_start_is_education_not_summary():
    document = _header_then_body(
        [
            _styled_line("n", "Jordan Blake", 10, font_size=18),
            _styled_line("e", "jordan@example.com", 30),
        ],
        [
            _styled_line("d", "Bachelor of Science | State University", 70),
            _styled_line("y", "2020-2024", 70, x0=450.0),
            _styled_line("b", "Completed coursework in statistics and writing.", 90),
        ],
    )
    result = detect_region_aware_sections(document)
    assert _texts(result, "SUMMARY") == []
    assert "Bachelor of Science | State University" in _texts(result, "EDUCATION")


def test_explicit_summary_heading_behavior_unchanged():
    document = _single_column([
        _styled_line("h", "SUMMARY", 10, font_size=14, bold=True),
        _styled_line("p", "Platform engineer with production work since 2018.", 30),
        _styled_line("q", "Seeking a senior role delivering reliable services.", 50),
    ])
    result = detect_region_aware_sections(document)
    assert any("2018" in text for text in _texts(result, "SUMMARY"))
    assert "h" not in _ids(result, "SUMMARY")


def test_summary_heading_on_header_path_related_body_column_unchanged():
    header = [
        _styled_line("n", "Jordan Blake", 10, font_size=18),
        _styled_line("e", "jordan@example.com", 30),
        _styled_line("h", "SUMMARY", 50, font_size=14, bold=True),
    ]
    left = [
        _prose("l1", "I am a results-driven engineer with several years of backend delivery experience.", 80),
        _prose("l2", "I have a strong background in building services and deploying them to production.", 94),
        _prose("l3", "I am adept at application interfaces, messaging, and cloud based rollout work.", 108),
        _styled_line("sk", "SKILLS", 140, font_size=13, bold=True),
        _styled_line("sv", "Java, SQL", 160),
    ]
    right = [
        _styled_line("ph", "PROJECTS", 78, font_size=13, bold=True, x0=180.0),
        _styled_line("pt", "Head End System", 100, x0=180.0),
        _styled_line("pd", "Date : 02/2025 – present", 120, x0=180.0),
    ]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region("page-1-region-0", "header", BoundingBox(10, 10, 200, 64), header, 0, None),
                    Region("page-1-region-1", "column", BoundingBox(10, 80, 250, 180), left, 1, 0),
                    Region("page-1-region-2", "column", BoundingBox(180, 78, 500, 140), right, 2, 1),
                ],
            )
        ]
    )
    result = detect_region_aware_sections(document)
    summary = _texts(result, "SUMMARY")
    assert any("results-driven engineer" in text for text in summary)
    assert "Head End System" in _texts(result, "PROJECTS")
    assert "Java, SQL" in _texts(result, "SKILLS")


def test_explicit_profile_alias_on_resume_a_unchanged():
    path = Path("tests/fixtures/AditCV_SOL.pdf")
    if not path.exists():
        return
    document = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes()))))
    result = detect_region_aware_sections(document)
    summary = _texts(result, "SUMMARY")
    assert any("Results-oriented MBA graduate" in text for text in summary)
    assert any("Seeking an opportunity" in text for text in summary)


def test_headingless_profile_on_resume_b_is_summary_when_paragraph_evidence_holds():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        return
    document = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes()))))
    result = detect_region_aware_sections(document)
    summary = _texts(result, "SUMMARY")
    assert any("ambitious MBA student" in text for text in summary)
    education = _texts(result, "EDUCATION")
    assert any("School Of Open Learning" in text for text in education)
    assert any("BIR Tikendrajit" in text for text in education)
    assert "MS Excel (VLOOKUP, Pivot Tables, Filters)" in _texts(result, "SKILLS")