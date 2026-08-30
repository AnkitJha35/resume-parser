from tests.conftest import require_fixture
from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.domain.candidate_section import SectionOrigin
from app.domain.structural import StructuralRole
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.semantic_compat import candidate_sections_to_text_blocks
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
    reading_order: int | None = None,
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
        reading_order=int(y0) if reading_order is None else reading_order,
        source_span_ids=[span.span_id],
        reconstruction_method="physical",
    )


def _document(lines: list[Line], *, kind: str = "physical_region", region_id: str = "page-1-region-0") -> Document:
    page_number = lines[0].page_number if lines else 1
    bbox = BoundingBox(
        min(line.bbox.x0 for line in lines),
        min(line.bbox.y0 for line in lines),
        max(line.bbox.x1 for line in lines),
        max(line.bbox.y1 for line in lines),
    )
    return Document(
        pages=[Page(page_number, regions=[Region(region_id, kind, bbox, lines=lines, reading_order=0, column_id=0)])]
    )


def _two_page_document(
    page1: list[Line],
    page2: list[Line],
    *,
    page1_kind: str = "column",
    page2_kind: str = "column",
    extra_page2_regions: list[Region] | None = None,
) -> Document:
    page1_bbox = BoundingBox(
        min(line.bbox.x0 for line in page1),
        min(line.bbox.y0 for line in page1),
        max(line.bbox.x1 for line in page1),
        max(line.bbox.y1 for line in page1),
    )
    page2_bbox = BoundingBox(
        min(line.bbox.x0 for line in page2),
        min(line.bbox.y0 for line in page2),
        max(line.bbox.x1 for line in page2),
        max(line.bbox.y1 for line in page2),
    )
    page2_regions = [
        Region("page-2-region-0", page2_kind, page2_bbox, lines=page2, reading_order=0, column_id=0)
    ]
    if extra_page2_regions:
        page2_regions.extend(extra_page2_regions)
    return Document(
        pages=[
            Page(1, regions=[Region("page-1-region-0", page1_kind, page1_bbox, lines=page1, reading_order=0, column_id=0)]),
            Page(2, regions=page2_regions),
        ]
    )


def _sections_from_lines(lines: list[Line], **kwargs):
    document = _document(lines, **kwargs)
    blocks = build_structural_blocks(document)
    return build_candidate_sections(blocks), blocks


def test_path_order_follows_geometry_not_global_reading_order():
    """Heading A above B must win even if B has a lower extraction reading_order."""
    sections, _ = _sections_from_lines([
        _line("a", "SUMMARY", 10, font_size=14, bold=True, reading_order=50),
        _line("ba", "Experienced engineer seeking a platform role.", 30, reading_order=51),
        _line("b", "EXPERIENCE", 80, font_size=14, bold=True, reading_order=1),
        _line("t", "Software Engineer", 100, font_size=12, bold=True, reading_order=2),
        _line("c", "Example Corp LLC", 120, reading_order=3),
        _line("d", "2020 - 2023", 140, reading_order=4),
    ])
    headed = [section for section in sections if section.heading is not None]
    by_label = {section.semantic_label: section for section in headed}
    assert "SUMMARY" in by_label and "EXPERIENCE" in by_label
    summary_texts = [block.text for block in by_label["SUMMARY"].content]
    experience_texts = [block.text for block in by_label["EXPERIENCE"].content]
    assert "Experienced engineer seeking a platform role." in summary_texts
    assert "Software Engineer" not in summary_texts
    assert "Software Engineer" in experience_texts
    assert "Experienced engineer seeking a platform role." not in experience_texts
    assert headed.index(by_label["SUMMARY"]) < headed.index(by_label["EXPERIENCE"])


def test_known_experience_heading():
    sections, _ = _sections_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "Sept 2018 - Present", 70),
    ])
    headed = [section for section in sections if section.heading is not None]
    assert len(headed) == 1
    assert headed[0].origin == SectionOrigin.KNOWN_ALIAS
    assert headed[0].semantic_label == "EXPERIENCE"
    assert headed[0].heading.text == "EXPERIENCE"
    assert headed[0].content[0].text == "Administrative Assistant"
    assert headed[0].heading.block_id not in {block.block_id for block in headed[0].content}


def test_known_education_heading():
    sections, _ = _sections_from_lines([
        _line("h", "EDUCATION", 10, font_size=14, bold=True),
        _line("deg", "Bachelor of Arts", 30),
        _line("u", "River Brook University", 50),
    ])
    assert sections[0].origin == SectionOrigin.KNOWN_ALIAS
    assert sections[0].semantic_label == "EDUCATION"


def test_unknown_highlights_heading_inferred_skills():
    sections, _ = _sections_from_lines([
        _line("h", "HIGHLIGHTS", 10, font_size=14, bold=True),
        _line("a", "Team player", 30),
        _line("b", "Safety-conscious", 50),
        _line("c", "Resourceful problem solver", 70),
    ])
    headed = [section for section in sections if section.heading and section.heading.text == "HIGHLIGHTS"]
    assert headed
    assert headed[0].origin == SectionOrigin.INFERRED
    assert headed[0].semantic_label == "SKILLS"
    assert all(block.text != "HIGHLIGHTS" for block in headed[0].content)


def test_unknown_project_like_heading():
    sections, _ = _sections_from_lines([
        _line("h", "SELECTED WORK", 10, font_size=14, bold=True),
        _line("t", "Customer Portal", 30),
        _line("d", "Built a React and Spring Boot application for customer self-service.", 50),
    ])
    headed = [section for section in sections if section.heading and section.heading.text == "SELECTED WORK"]
    assert headed[0].origin == SectionOrigin.INFERRED
    assert headed[0].semantic_label == "PROJECTS"


def test_ambiguous_unknown_heading():
    sections, _ = _sections_from_lines([
        _line("h", "RANDOM INFORMATION", 10, font_size=14, bold=True),
        _line("c", "Managed systems and improved processes", 30),
    ])
    headed = [section for section in sections if section.heading and section.heading.text == "RANDOM INFORMATION"]
    # RANDOM INFORMATION may not qualify as SECTION_HEADING structurally (no list body).
    # If a headed unknown section exists, it must remain UNKNOWN — never forced.
    for section in headed:
        assert section.origin == SectionOrigin.UNKNOWN
        assert section.semantic_label is None


def test_entry_title_is_not_section_heading():
    sections, blocks = _sections_from_lines([
        _line("t", "Administrative Assistant", 10, font_size=14, bold=True),
        _line("c", "Redford & Sons", 30),
        _line("l", "Boston, MA", 50),
        _line("d", "Sept 2018 - Present", 70),
        _line("b", "• Schedule meetings", 90),
    ])
    assert all(
        section.heading is None or section.heading.text != "Administrative Assistant"
        for section in sections
    )
    title = next(block for block in blocks if block.text == "Administrative Assistant")
    assert title.role == StructuralRole.ENTRY_TITLE
    unlabeled = [section for section in sections if section.origin == SectionOrigin.UNLABELED]
    assert unlabeled
    assert unlabeled[0].heading is None
    assert any(block.text == "Administrative Assistant" for block in unlabeled[0].content)


def test_skill_phrases_are_not_independent_section_headings():
    sections, _ = _sections_from_lines([
        _line("a", "Resourceful problem solver", 10),
        _line("b", "Safety-conscious", 30),
        _line("c", "Team player", 50),
    ])
    assert all(section.heading is None for section in sections)
    unlabeled = [section for section in sections if section.origin == SectionOrigin.UNLABELED]
    assert unlabeled
    assert {block.text for block in unlabeled[0].content} >= {
        "Resourceful problem solver",
        "Safety-conscious",
        "Team player",
    }


def test_headingless_experience_like_cluster():
    sections, _ = _sections_from_lines([
        _line("t", "Software Engineer", 10, font_size=12, bold=True),
        _line("c", "Example Corp LLC", 30),
        _line("l", "New York, NY", 50),
        _line("d", "2020 - 2023", 70),
        _line("b", "• Built services", 90),
    ])
    unlabeled = [section for section in sections if section.origin == SectionOrigin.UNLABELED]
    assert unlabeled
    assert unlabeled[0].semantic_label is None
    assert unlabeled[0].heading is None
    assert "entry_title_present" in unlabeled[0].evidence


def test_headingless_education_like_cluster():
    sections, _ = _sections_from_lines([
        _line("deg", "Bachelor of Science", 10, font_size=12, bold=True),
        _line("u", "State University", 30),
        _line("d", "2014 - 2018", 50),
    ])
    unlabeled = [section for section in sections if section.origin == SectionOrigin.UNLABELED]
    assert unlabeled
    assert unlabeled[0].heading is None
    assert unlabeled[0].semantic_label is None


def test_headingless_skills_like_cluster():
    sections, _ = _sections_from_lines([
        _line("a", "Team player", 10),
        _line("b", "Safety-conscious", 30),
        _line("c", "Problem solving", 50),
        _line("d", "Friendly and helpful", 70),
    ])
    unlabeled = [section for section in sections if section.origin == SectionOrigin.UNLABELED]
    assert unlabeled
    assert unlabeled[0].origin == SectionOrigin.UNLABELED


def test_typography_transition_creates_boundary():
    sections, _ = _sections_from_lines([
        _line("t", "Software Engineer", 10, font_size=12, bold=True),
        _line("c", "Example Corp LLC", 30),
        _line("d", "2020 - 2021", 50),
        _line("h", "Tools", 120, font_size=16, bold=True),
        _line("s1", "Python", 140),
        _line("s2", "Docker", 160),
        _line("s3", "Kubernetes", 180),
    ])
    # Either a SECTION_HEADING for Tools if structural classifier says so, or a gap split.
    assert len(sections) >= 2
    assert sections[0].path_id == sections[1].path_id


def test_large_vertical_gap_creates_boundary():
    sections, _ = _sections_from_lines([
        _line("t", "Software Engineer", 10, font_size=12, bold=True),
        _line("c", "Example Corp LLC", 30),
        _line("d", "2020 - 2021", 50),
        _line("t2", "Data Analyst", 200, font_size=12, bold=True),
        _line("c2", "Other Company Inc", 220),
        _line("d2", "2018 - 2019", 240),
    ])
    unlabeled = [section for section in sections if section.origin == SectionOrigin.UNLABELED]
    assert len(unlabeled) >= 2


def test_adjacent_regions_do_not_merge():
    left = [
        _line("h1", "EXPERIENCE", 10, x0=10, font_size=14, bold=True),
        _line("t1", "Secretary", 40, x0=10, font_size=12, bold=True),
    ]
    right = [
        _line("h2", "SKILLS", 10, x0=320, font_size=14, bold=True),
        _line("s1", "Team player", 40, x0=320),
    ]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region("page-1-region-0", "column", BoundingBox(0, 0, 200, 100), left, 0, 0),
                    Region("page-1-region-1", "column", BoundingBox(300, 0, 500, 100), right, 1, 1),
                ],
            )
        ]
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    experience = next(section for section in sections if section.semantic_label == "EXPERIENCE")
    skills = next(section for section in sections if section.semantic_label == "SKILLS")
    assert experience.path_id != skills.path_id
    assert experience.region_id != skills.region_id


def test_provenance_survives_section_construction():
    sections, blocks = _sections_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Secretary", 30, font_size=12, bold=True),
    ])
    section = sections[0]
    assert "h" in section.line_ids[0] or any(line_id.endswith("h") or line_id == "h" or "line-h" in line_id or line_id == "h" for line_id in section.line_ids)
    assert section.line_ids
    assert section.source_span_ids
    assert section.heading is not None
    assert section.heading.line_ids == blocks[0].line_ids
    assert section.content[0].source_span_ids == blocks[1].source_span_ids


def test_heading_and_content_remain_separate():
    sections, _ = _sections_from_lines([
        _line("h", "SKILLS", 10, font_size=14, bold=True),
        _line("a", "Python", 30),
        _line("b", "Docker", 50),
    ])
    section = sections[0]
    assert section.heading is not None
    assert section.heading.text == "SKILLS"
    assert [block.text for block in section.content] == ["Python", "Docker"]


def test_multiline_heading_lines_retain_separate_provenance():
    # Multiline entry titles are not merged; both lines keep provenance for later entry work.
    sections, blocks = _sections_from_lines([
        _line("t1", "Senior Software", 10, font_size=12, bold=True),
        _line("t2", "Engineer", 25, font_size=12, bold=True),
        _line("c", "Example Corp LLC", 50),
        _line("d", "2020 - 2023", 70),
    ])
    unlabeled = next(section for section in sections if section.origin == SectionOrigin.UNLABELED)
    assert unlabeled.heading is None
    texts = [block.text for block in unlabeled.content]
    assert "Senior Software" in texts and "Engineer" in texts
    first = next(block for block in unlabeled.content if block.text == "Senior Software")
    second = next(block for block in unlabeled.content if block.text == "Engineer")
    assert first.line_ids != second.line_ids
    assert first.source_span_ids != second.source_span_ids


def test_compat_adapter_excludes_heading_from_section_content():
    sections, blocks = _sections_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Secretary", 30, font_size=12, bold=True),
    ])
    converted = candidate_sections_to_text_blocks(sections, blocks)
    assert all(block.text != "EXPERIENCE" for block in converted["EXPERIENCE"])
    assert any(block.text == "Secretary" for block in converted["EXPERIENCE"])


def test_resume_1_experience_invariant_unchanged():
    resume = ResumeParser().parse_with_layout_pipeline(require_fixture("resume_1.pdf").read_bytes())
    assert [(item.designation, item.company, item.location, item.startDate, item.endDate, item.current) for item in resume.experience] == [
        ("Administrative Assistant", "Redford & Sons", "Boston, MA", "2018-09", None, True),
        ("Secretary", "Bright Spot LTD", "Boston, MA", "2015-06", "2018-08", False),
    ]


def test_resume_6_highlights_skills_invariant_unchanged():
    resume = ResumeParser().parse_with_layout_pipeline(require_fixture("resume_6.pdf").read_bytes())
    assert {
        "Warehouse Equipment Operation",
        "Resourceful Problem Solver",
        "Friendly and Helpful",
        "Good Physical Condition",
        "Safety-Conscious",
        "Team Player",
    }.issubset(set(resume.skills))


def test_resume_1_candidate_sections_keep_experience_alias():
    raw = PDFExtractor.extract(require_fixture("resume_1.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    sections = build_candidate_sections(build_structural_blocks(document))
    assert any(section.semantic_label == "EXPERIENCE" and section.origin == SectionOrigin.KNOWN_ALIAS for section in sections)


def test_resume_6_candidate_sections_infer_highlights():
    raw = PDFExtractor.extract(require_fixture("resume_6.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    sections = build_candidate_sections(build_structural_blocks(document))
    highlights = [
        section
        for section in sections
        if section.heading is not None and section.heading.text.strip().upper() == "HIGHLIGHTS"
    ]
    assert highlights
    assert highlights[0].origin == SectionOrigin.INFERRED
    assert highlights[0].semantic_label == "SKILLS"


def test_experience_continues_headingless_next_page():
    """A. Page-2 DATE + role/org/body inherits EXPERIENCE as CONTINUED."""
    document = _two_page_document(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("c1", "Northwind Partners LLC", 30),
            _line("t1", "Operations Lead", 50, font_size=12, bold=True),
            _line("d1", "2020-2022", 70),
        ],
        [
            _line("d2", "2022-2024", 20, page=2),
            _line("t2", "Program Manager", 40, page=2, font_size=12, bold=True),
            _line("c2", "Contoso Holdings Inc", 60, page=2),
            _line("b2", "Owned delivery for regional operations.", 80, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    continued = [
        section
        for section in sections
        if section.origin == SectionOrigin.CONTINUED and section.semantic_label == "EXPERIENCE"
    ]
    assert continued
    assert continued[0].heading is None
    assert continued[0].continuation_of
    assert continued[0].page_number == 2
    texts = {block.text for block in continued[0].content}
    assert "2022-2024" in texts
    assert "Contoso Holdings Inc" in texts
    origin = next(section for section in sections if section.section_id == continued[0].continuation_of)
    assert origin.semantic_label == "EXPERIENCE"
    assert origin.page_number == 1
    assert all(block.page_number == 2 for block in continued[0].content)


def test_completed_previous_entry_does_not_block_experience_continuation():
    """B. A finished page-1 job still continues EXPERIENCE for a new page-2 job."""
    document = _two_page_document(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Warehouse Associate", 30, font_size=12, bold=True),
            _line("c1", "Harbor Logistics LLC", 50),
            _line("d1", "2020-2022", 70),
            _line("b1", "• Completed inbound audits.", 90),
        ],
        [
            _line("d2", "2022-2024", 20, page=2),
            _line("t2", "Shift Supervisor", 40, page=2, font_size=12, bold=True),
            _line("c2", "Summit Retail Inc", 60, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    continued = [
        section
        for section in sections
        if section.page_number == 2 and section.origin == SectionOrigin.CONTINUED
    ]
    assert continued
    assert continued[0].semantic_label == "EXPERIENCE"
    assert any(block.text == "Shift Supervisor" for block in continued[0].content)


def test_headingless_ambiguous_prose_stays_unlabeled():
    """C. Generic prose without structural evidence is UNLABELED, not inherited."""
    document = _two_page_document(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Analyst", 30, font_size=12, bold=True),
            _line("c1", "Example Corp LLC", 50),
            _line("d1", "2020-2022", 70),
        ],
        [
            _line("p1", "Independent research notes from the field assignment.", 20, page=2),
            _line("p2", "The following observations were recorded during travel.", 40, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    page2 = [section for section in sections if section.page_number == 2]
    assert page2
    assert all(section.semantic_label != "EXPERIENCE" for section in page2)
    assert all(section.origin != SectionOrigin.CONTINUED for section in page2)
    assert any(section.origin == SectionOrigin.UNLABELED for section in page2)


def test_competing_skills_content_does_not_inherit_experience():
    """D. Skill tokens on page 2 must not inherit EXPERIENCE from geometry."""
    document = _two_page_document(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Engineer", 30, font_size=12, bold=True),
            _line("c1", "Example Corp LLC", 50),
            _line("d1", "2020-2022", 70),
        ],
        [
            _line("s1", "Java", 20, page=2),
            _line("s2", "Python", 40, page=2),
            _line("s3", "Docker", 60, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    page2 = [section for section in sections if section.page_number == 2]
    assert all(section.semantic_label != "EXPERIENCE" for section in page2)
    assert all(section.origin != SectionOrigin.CONTINUED for section in page2)


def test_new_known_heading_is_not_continued_experience():
    """E. A known SKILLS heading starts SKILLS, not continued EXPERIENCE."""
    document = _two_page_document(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Engineer", 30, font_size=12, bold=True),
            _line("c1", "Example Corp LLC", 50),
            _line("d1", "2020-2022", 70),
        ],
        [
            _line("h2", "SKILLS", 10, page=2, font_size=14, bold=True),
            _line("s1", "Java", 30, page=2),
            _line("s2", "Python", 50, page=2),
        ],
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    page2 = [section for section in sections if section.page_number == 2]
    assert any(section.semantic_label == "SKILLS" and section.origin == SectionOrigin.KNOWN_ALIAS for section in page2)
    assert all(not (section.origin == SectionOrigin.CONTINUED and section.semantic_label == "EXPERIENCE") for section in page2)


def test_header_footer_do_not_continue_experience():
    """F. Repeated page-2 header/footer regions are not continued EXPERIENCE."""
    header_lines = [_line("hdr", "Candidate Name", 8, page=2, font_size=9)]
    footer_lines = [_line("ftr", "Page 2", 780, page=2, font_size=8)]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region(
                        "page-1-region-0",
                        "column",
                        BoundingBox(10, 10, 200, 90),
                        [
                            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
                            _line("t1", "Engineer", 30, font_size=12, bold=True),
                            _line("c1", "Example Corp LLC", 50),
                            _line("d1", "2020-2022", 70),
                        ],
                        0,
                        0,
                    )
                ],
            ),
            Page(
                2,
                regions=[
                    Region("page-2-region-h", "header", BoundingBox(10, 0, 200, 20), header_lines, 0, 0),
                    Region("page-2-region-f", "footer", BoundingBox(10, 770, 200, 800), footer_lines, 1, 0),
                ],
            ),
        ]
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    page2 = [section for section in sections if section.page_number == 2]
    assert all(section.origin != SectionOrigin.CONTINUED for section in page2)
    assert all(section.semantic_label != "EXPERIENCE" for section in page2)


def test_sidebar_contact_does_not_continue_experience():
    """G. Contact/sidebar paths do not inherit EXPERIENCE."""
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region(
                        "page-1-region-0",
                        "column",
                        BoundingBox(10, 10, 220, 90),
                        [
                            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
                            _line("t1", "Engineer", 30, font_size=12, bold=True),
                            _line("c1", "Example Corp LLC", 50),
                            _line("d1", "2020-2022", 70),
                        ],
                        0,
                        0,
                    )
                ],
            ),
            Page(
                2,
                regions=[
                    Region(
                        "page-2-region-0",
                        "sidebar",
                        BoundingBox(10, 10, 160, 80),
                        [
                            _line("e", "person@example.com", 20, page=2),
                            _line("p", "(555) 010-0199", 40, page=2),
                        ],
                        0,
                        0,
                    )
                ],
            ),
        ]
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    page2 = [section for section in sections if section.page_number == 2]
    assert all(section.origin != SectionOrigin.CONTINUED for section in page2)
    assert all(section.semantic_label != "EXPERIENCE" for section in page2)


def test_geometry_mismatch_does_not_continue():
    """H. Right-column EXPERIENCE does not continue onto an unrelated left column.

    Geometry uses horizontal overlap of the narrower column, not nearest x-center.
    """
    page1 = [
        _line("h", "EXPERIENCE", 10, x0=320, font_size=14, bold=True),
        _line("t1", "Engineer", 30, x0=320, font_size=12, bold=True),
        _line("c1", "Example Corp LLC", 50, x0=320),
        _line("d1", "2020-2022", 70, x0=320),
    ]
    page2 = [
        _line("x", "Unrelated left-column notes about volunteering.", 20, page=2, x0=10),
        _line("y", "Additional commentary without employment dates.", 40, page=2, x0=10),
    ]
    document = Document(
        pages=[
            Page(1, regions=[Region("page-1-region-0", "column", BoundingBox(300, 0, 520, 100), page1, 0, 1)]),
            Page(2, regions=[Region("page-2-region-0", "column", BoundingBox(0, 0, 180, 80), page2, 0, 0)]),
        ]
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    page2_sections = [section for section in sections if section.page_number == 2]
    assert all(section.origin != SectionOrigin.CONTINUED for section in page2_sections)
    assert all(section.semantic_label != "EXPERIENCE" for section in page2_sections)


def test_multicolumn_only_compatible_path_continues():
    """I. On a two-column page 2, only the overlapping column continues EXPERIENCE."""
    page1_left = [
        _line("hs", "SKILLS", 10, x0=10, font_size=14, bold=True),
        _line("s1", "Excel", 30, x0=10),
        _line("s2", "Filing", 50, x0=10),
    ]
    page1_right = [
        _line("h", "EXPERIENCE", 10, x0=320, font_size=14, bold=True),
        _line("t1", "Clerk", 30, x0=320, font_size=12, bold=True),
        _line("c1", "Example Corp LLC", 50, x0=320),
        _line("d1", "2020-2022", 70, x0=320),
    ]
    page2_left = [
        _line("ls1", "Java", 20, page=2, x0=10),
        _line("ls2", "Python", 40, page=2, x0=10),
        _line("ls3", "Docker", 60, page=2, x0=10),
    ]
    page2_right = [
        _line("d2", "2022-2024", 20, page=2, x0=320),
        _line("t2", "Coordinator", 40, page=2, x0=320, font_size=12, bold=True),
        _line("c2", "Harbor Partners Inc", 60, page=2, x0=320),
    ]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region("page-1-region-0", "column", BoundingBox(0, 0, 180, 80), page1_left, 0, 0),
                    Region("page-1-region-1", "column", BoundingBox(300, 0, 520, 100), page1_right, 1, 1),
                ],
            ),
            Page(
                2,
                regions=[
                    Region("page-2-region-0", "column", BoundingBox(0, 0, 180, 80), page2_left, 0, 0),
                    Region("page-2-region-1", "column", BoundingBox(300, 0, 520, 90), page2_right, 1, 1),
                ],
            ),
        ]
    )
    sections = build_candidate_sections(build_structural_blocks(document))
    continued = [
        section
        for section in sections
        if section.origin == SectionOrigin.CONTINUED and section.semantic_label == "EXPERIENCE"
    ]
    assert len(continued) == 1
    assert continued[0].page_number == 2
    texts = {block.text for block in continued[0].content}
    assert "2022-2024" in texts
    assert "Java" not in texts
    assert continued[0].bbox.x0 >= 200


def test_continued_section_preserves_page_two_provenance():
    """J. Continued membership keeps page-2 line_ids and source_span_ids."""
    page2 = [
        _line("d2", "2022-2024", 20, page=2),
        _line("t2", "Program Manager", 40, page=2, font_size=12, bold=True),
        _line("c2", "Contoso Holdings Inc", 60, page=2),
    ]
    document = _two_page_document(
        [
            _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
            _line("t1", "Operations Lead", 30, font_size=12, bold=True),
            _line("c1", "Northwind Partners LLC", 50),
            _line("d1", "2020-2022", 70),
        ],
        page2,
    )
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    continued = next(
        section
        for section in sections
        if section.origin == SectionOrigin.CONTINUED and section.semantic_label == "EXPERIENCE"
    )
    page2_blocks = [block for block in blocks if block.page_number == 2]
    expected_line_ids = [line_id for block in page2_blocks for line_id in block.line_ids]
    expected_span_ids = [span_id for block in page2_blocks for span_id in block.source_span_ids]
    assert list(continued.line_ids) == expected_line_ids
    assert list(continued.source_span_ids) == expected_span_ids
    assert all(block.reconstruction_method for block in continued.content)
    assert continued.path_id == page2_blocks[0].path_id
    assert continued.region_id == page2_blocks[0].region_id


def test_resume_2_page_two_date_stack_is_continued_experience():
    """K. Specimen: the headingless page-2 date stack continues EXPERIENCE.

    Production code must not key off this path id or employer string.
    """
    raw = PDFExtractor.extract(require_fixture("resume_2.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    continued = [
        section
        for section in sections
        if section.origin == SectionOrigin.CONTINUED and section.semantic_label == "EXPERIENCE"
    ]
    assert continued
    date_stack = next(
        section
        for section in continued
        if any("June 2015" in block.text and "August 2017" in block.text for block in section.content)
    )
    assert date_stack.heading is None
    assert date_stack.continuation_of
    assert date_stack.page_number == 2
    assert all(block.page_number == 2 for block in date_stack.content)
    first = date_stack.content[0]
    assert first.role == StructuralRole.DATE
    assert first.line_ids
    assert first.source_span_ids
    parent = next(section for section in sections if section.section_id == date_stack.continuation_of)
    assert parent.semantic_label == "EXPERIENCE"
    assert parent.page_number == 1

