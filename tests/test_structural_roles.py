from pathlib import Path

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.domain.structural import StructuralRole
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.structural_roles import build_structural_blocks, classify_structural_role
from app.pipeline.stages.text_extraction import PDFExtractor


def _line(line_id: str, text: str, y0: float, *, x0: float = 10.0, font_size: float = 11.0, bold: bool = False) -> Line:
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


def _document_from_lines(lines: list[Line], *, kind: str = "physical_region", region_id: str = "page-1-region-0") -> Document:
    bbox = BoundingBox(
        min(line.bbox.x0 for line in lines),
        min(line.bbox.y0 for line in lines),
        max(line.bbox.x1 for line in lines),
        max(line.bbox.y1 for line in lines),
    )
    return Document(
        pages=[
            Page(
                1,
                regions=[Region(region_id, kind, bbox, lines=lines, reading_order=0, column_id=0)],
            )
        ]
    )


def _roles_by_text(blocks):
    return {block.text: block.role for block in blocks}


def test_single_column_experience_roles():
    lines = [
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("l", "Boston, MA", 70),
        _line("d", "2018-09 - Present", 90),
        _line("b", "• Schedule and coordinate meetings", 110),
    ]
    blocks = build_structural_blocks(_document_from_lines(lines))
    roles = _roles_by_text(blocks)

    assert roles["EXPERIENCE"] == StructuralRole.SECTION_HEADING
    assert roles["Administrative Assistant"] == StructuralRole.ENTRY_TITLE
    assert roles["Redford & Sons"] == StructuralRole.ORGANIZATION
    assert roles["Boston, MA"] == StructuralRole.LOCATION
    assert roles["2018-09 - Present"] == StructuralRole.DATE
    assert roles["• Schedule and coordinate meetings"] == StructuralRole.BULLET


def test_uppercase_role_title_is_not_section_heading():
    role, _, reasons = classify_structural_role(
        "ADMINISTRATIVE ASSISTANT",
        font_size=14,
        bold=True,
        following_texts=("Redford & Sons, Boston, MA / 2018 - Present", "Schedule meetings"),
    )
    assert role == StructuralRole.ENTRY_TITLE
    assert role != StructuralRole.SECTION_HEADING
    assert "entry_title_structure" in reasons


def test_highlights_can_be_section_heading_with_list_body():
    role, _, reasons = classify_structural_role(
        "HIGHLIGHTS",
        font_size=14,
        bold=True,
        following_texts=("Team player", "Safety-conscious", "Problem solving"),
    )
    assert role == StructuralRole.SECTION_HEADING
    assert "section_boundary_geometry" in reasons


def test_wrapped_experience_description_is_description():
    role, _, reasons = classify_structural_role(
        "arrangements for supervisors and managers",
        previous_text="Schedule and coordinate meetings, appointments, and travel",
        font_size=11,
        bold=False,
    )
    assert role == StructuralRole.DESCRIPTION
    assert "wrapped_continuation" in reasons


def test_two_column_layout_preserves_separate_paths_and_neighbors():
    left = [
        _line("l0", "EXPERIENCE", 40, x0=20, font_size=14, bold=True),
        _line("l1", "Secretary", 70, x0=20, font_size=12, bold=True),
    ]
    right = [
        _line("r0", "SKILLS", 40, x0=320, font_size=14, bold=True),
        _line("r1", "Team player", 70, x0=320),
    ]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region("page-1-region-0", "column", BoundingBox(0, 0, 200, 200), left, reading_order=0, column_id=0),
                    Region("page-1-region-1", "column", BoundingBox(300, 0, 500, 200), right, reading_order=1, column_id=1),
                ],
            )
        ]
    )
    blocks = build_structural_blocks(document)
    by_text = {block.text: block for block in blocks}

    assert by_text["EXPERIENCE"].path_id != by_text["SKILLS"].path_id
    assert by_text["EXPERIENCE"].region_id != by_text["SKILLS"].region_id
    assert by_text["Secretary"].previous_block_id == by_text["EXPERIENCE"].block_id
    assert by_text["Secretary"].next_block_id is None
    assert by_text["Team player"].previous_block_id == by_text["SKILLS"].block_id


def test_unknown_heading_without_body_stays_unknown_or_non_forced():
    role, score, _ = classify_structural_role("RANDOM INFORMATION", font_size=14, bold=True, following_texts=())
    assert role == StructuralRole.UNKNOWN
    assert score == 0.0


def test_bullet_list_roles():
    lines = [
        _line("h", "HIGHLIGHTS", 10, font_size=14, bold=True),
        _line("m", "•", 30),
        _line("c", "Team player", 32),
        _line("m2", "•", 50),
        _line("c2", "Safety-conscious", 52),
    ]
    roles = _roles_by_text(build_structural_blocks(_document_from_lines(lines)))
    assert roles["HIGHLIGHTS"] == StructuralRole.SECTION_HEADING
    assert roles["•"] == StructuralRole.BULLET


def test_multiline_entry_title_does_not_become_section_heading():
    lines = [
        _line("t1", "Senior Administrative", 10, font_size=12, bold=True),
        _line("t2", "Assistant", 25, font_size=12, bold=True),
        _line("c", "Bright Spot LTD, Boston, MA / 2015 - 2018", 45),
    ]
    blocks = build_structural_blocks(_document_from_lines(lines))
    roles = _roles_by_text(blocks)
    assert roles["Senior Administrative"] != StructuralRole.SECTION_HEADING
    assert roles["Assistant"] != StructuralRole.SECTION_HEADING
    assert roles["Senior Administrative"] == StructuralRole.ENTRY_TITLE or roles["Assistant"] == StructuralRole.ENTRY_TITLE


def test_organization_and_date_metadata_roles():
    role_org, _, _ = classify_structural_role("Bright Spot LTD")
    role_date, _, _ = classify_structural_role("2015-06 - 2018-08")
    assert role_org == StructuralRole.ORGANIZATION
    assert role_date == StructuralRole.DATE


def test_adjacent_unrelated_regions_do_not_share_neighbors():
    left = [_line("a", "EXPERIENCE", 10, x0=10, font_size=14, bold=True)]
    right = [_line("b", "Python", 10, x0=300)]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region("page-1-region-0", "column", BoundingBox(0, 0, 150, 50), left, 0, 0),
                    Region("page-1-region-1", "column", BoundingBox(250, 0, 400, 50), right, 1, 1),
                ],
            )
        ]
    )
    blocks = build_structural_blocks(document)
    assert all(block.previous_block_id is None and block.next_block_id is None for block in blocks)
    assert blocks[0].path_id != blocks[1].path_id


def test_provenance_survives_structural_block_creation():
    line = _line("line-9", "Team player", 40)
    blocks = build_structural_blocks(_document_from_lines([line]))
    block = blocks[0]
    assert block.line_ids == ("line-9",)
    assert block.source_span_ids == ("line-9-span",)
    assert block.bbox == line.bbox
    assert block.reconstruction_method == "physical"
    assert block.page_number == 1
    assert block.region_id == "page-1-region-0"
    assert block.path_id.startswith("page-1-path-")


def test_resume_1_role_titles_are_entry_titles_not_section_headings():
    raw = PDFExtractor.extract(Path("tests/fixtures/resume_1.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    blocks = build_structural_blocks(document)
    titles = [
        block
        for block in blocks
        if block.text.strip().upper() in {"ADMINISTRATIVE ASSISTANT", "SECRETARY"}
        or (
            len(block.text.split()) <= 4
            and "ADMINISTRATIVE ASSISTANT" == block.text.strip().upper()
        )
    ]
    assert titles
    for block in titles:
        assert block.role != StructuralRole.SECTION_HEADING
        assert block.role in {StructuralRole.ENTRY_TITLE, StructuralRole.UNKNOWN, StructuralRole.DESCRIPTION}


def test_resume_6_highlights_is_structural_section_heading():
    raw = PDFExtractor.extract(Path("tests/fixtures/resume_6.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    blocks = build_structural_blocks(document)
    highlights = [block for block in blocks if block.text.strip().upper() == "HIGHLIGHTS"]
    assert highlights
    assert highlights[0].role == StructuralRole.SECTION_HEADING


def test_layout_parser_resume_1_experience_invariant_unchanged():
    resume = ResumeParser().parse_with_layout_pipeline(Path("tests/fixtures/resume_1.pdf").read_bytes())
    assert [(item.designation, item.company, item.location, item.startDate, item.endDate, item.current) for item in resume.experience] == [
        ("Administrative Assistant", "Redford & Sons", "Boston, MA", "2018-09", None, True),
        ("Secretary", "Bright Spot LTD", "Boston, MA", "2015-06", "2018-08", False),
    ]


def test_layout_parser_resume_6_highlights_skills_invariant_unchanged():
    resume = ResumeParser().parse_with_layout_pipeline(Path("tests/fixtures/resume_6.pdf").read_bytes())
    assert {
        "Warehouse Equipment Operation",
        "Resourceful Problem Solver",
        "Friendly and Helpful",
        "Good Physical Condition",
        "Safety-Conscious",
        "Team Player",
    }.issubset(set(resume.skills))
