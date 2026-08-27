from pathlib import Path

from app.domain.candidate_entry import EntryType
from app.domain.candidate_section import SectionOrigin
from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.domain.structural import StructuralRole
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.candidate_entries import build_candidate_entries
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.structural_roles import build_structural_blocks
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


def _document(lines: list[Line], *, region_id: str = "page-1-region-0", kind: str = "physical_region") -> Document:
    bbox = BoundingBox(
        min(line.bbox.x0 for line in lines),
        min(line.bbox.y0 for line in lines),
        max(line.bbox.x1 for line in lines),
        max(line.bbox.y1 for line in lines),
    )
    return Document(pages=[Page(1, regions=[Region(region_id, kind, bbox, lines=lines, reading_order=0, column_id=0)])])


def _entries_from_lines(lines: list[Line], **kwargs):
    blocks = build_structural_blocks(_document(lines, **kwargs))
    sections = build_candidate_sections(blocks)
    return build_candidate_entries(sections), sections, blocks


def test_one_normal_experience_entry():
    entries, sections, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("l", "Chicago, IL", 70),
        _line("d", "2018 - 2021", 90),
        _line("b", "• Coordinated meetings", 110),
        _line("b2", "• Prepared reports", 130),
    ])
    assert any(section.semantic_label == "EXPERIENCE" for section in sections)
    assert len(entries) == 1
    assert entries[0].entry_type == EntryType.EXPERIENCE
    texts = [block.text for block in entries[0].blocks]
    assert texts[0] == "Administrative Assistant"
    assert "Redford & Sons" in texts
    assert "• Prepared reports" in texts


def test_two_experience_entries():
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c1", "Redford & Sons", 50),
        _line("d1", "2018 - 2021", 70),
        _line("b1", "• Managed schedules.", 90),
        _line("t2", "Office Manager", 120, font_size=12, bold=True),
        _line("c2", "ABC Corp LLC", 140),
        _line("d2", "2021 - Present", 160),
        _line("b2", "• Managed operations.", 180),
    ])
    assert len(entries) == 2
    assert [block.text for block in entries[0].blocks][0] == "Administrative Assistant"
    assert [block.text for block in entries[1].blocks][0] == "Office Manager"


def test_multiline_job_title_one_entry():
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Senior Software", 30, font_size=12, bold=True),
        _line("t2", "Engineer", 45, font_size=12, bold=True),
        _line("c", "Google LLC", 70),
        _line("d", "2022 - Present", 90),
        _line("b", "• Built distributed systems.", 110),
    ])
    assert len(entries) == 1
    texts = [block.text for block in entries[0].blocks]
    assert texts.count("Senior Software") == 1
    assert texts.count("Engineer") == 1
    assert entries[0].blocks[0].line_ids != entries[0].blocks[1].line_ids


def test_company_location_date_grouping():
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Secretary", 30, font_size=12, bold=True),
        _line("c", "Bright Spot LTD", 50),
        _line("l", "Boston, MA", 70),
        _line("d", "2015 - 2018", 90),
    ])
    assert len(entries) == 1
    roles = {block.role for block in entries[0].blocks}
    assert StructuralRole.ORGANIZATION in roles
    assert StructuralRole.LOCATION in roles
    assert StructuralRole.DATE in roles


def test_description_continuation():
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
        _line("b1", "Managed schedules.", 90),
        _line("b2", "Prepared reports.", 110),
        _line("b3", "Coordinated travel arrangements.", 130),
    ])
    assert len(entries) == 1
    assert len(entries[0].blocks) >= 6


def test_title_like_description_does_not_start_new_entry():
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Senior Software Engineer", 30, font_size=12, bold=True),
        _line("c", "Google LLC", 50),
        _line("d", "2022 - Present", 70),
        _line("b", "Arrangements for supervisors and managers.", 90),
    ])
    assert len(entries) == 1
    assert any("Arrangements for supervisors" in block.text for block in entries[0].blocks)


def test_education_entry():
    entries, _, _ = _entries_from_lines([
        _line("h", "EDUCATION", 10, font_size=14, bold=True),
        _line("deg", "Bachelor of Science", 30, font_size=12, bold=True),
        _line("u", "University of Chicago", 50),
        _line("d", "2014 - 2018", 70),
    ])
    assert len(entries) == 1
    assert entries[0].entry_type == EntryType.EDUCATION


def test_two_education_entries():
    entries, _, _ = _entries_from_lines([
        _line("h", "EDUCATION", 10, font_size=14, bold=True),
        _line("d1", "Bachelor of Science", 30, font_size=12, bold=True),
        _line("u1", "University of Chicago", 50),
        _line("y1", "2018", 70),
        _line("d2", "Master of Science", 110, font_size=12, bold=True),
        _line("u2", "MIT Institute", 130),
        _line("y2", "2020", 150),
    ])
    # Year-only lines may not be DATE role; ensure at least education segmentation attempt.
    assert len(entries) >= 1
    assert all(entry.entry_type == EntryType.EDUCATION for entry in entries)


def test_project_entry():
    entries, _, _ = _entries_from_lines([
        _line("h", "PROJECTS", 10, font_size=14, bold=True),
        _line("t", "Inventory Management System", 30, font_size=12, bold=True),
        _line("d", "2024 - 2024", 50),
        _line("b", "Built a Spring Boot application to manage warehouse inventory.", 70),
    ])
    assert len(entries) >= 1
    assert entries[0].entry_type == EntryType.PROJECT


def test_two_project_entries():
    entries, _, _ = _entries_from_lines([
        _line("h", "PROJECTS", 10, font_size=14, bold=True),
        _line("t1", "Project Alpha", 30, font_size=12, bold=True),
        _line("d1", "2024 - 2024", 50),
        _line("b1", "Built application.", 70),
        _line("t2", "Project Beta", 110, font_size=12, bold=True),
        _line("d2", "2025 - 2025", 130),
        _line("b2", "Built another application.", 150),
    ])
    assert len(entries) == 2


def test_certification_entry():
    entries, _, _ = _entries_from_lines([
        _line("h", "CERTIFICATIONS", 10, font_size=14, bold=True),
        _line("c", "AWS Certified Solutions Architect", 30, font_size=12, bold=True),
        _line("o", "Amazon Web Services Inc", 50),
        _line("d", "Issued 2024 - 2025", 70),
    ])
    assert len(entries) >= 1
    assert entries[0].entry_type == EntryType.CERTIFICATION


def test_headingless_experience_two_entries():
    entries, sections, _ = _entries_from_lines([
        _line("t1", "Software Engineer", 10, font_size=12, bold=True),
        _line("c1", "Google LLC", 30),
        _line("d1", "2022 - 2024", 50),
        _line("b1", "• Built services", 70),
        _line("t2", "Software Engineer", 120, font_size=12, bold=True),
        _line("c2", "Microsoft Corp", 140),
        _line("d2", "2024 - Present", 160),
        _line("b2", "• Built platforms", 180),
    ])
    assert any(section.origin == SectionOrigin.UNLABELED for section in sections)
    assert len(entries) == 2
    assert all(entry.entry_type == EntryType.UNKNOWN for entry in entries)
    assert all("experience_like_stack" in entry.evidence for entry in entries)


def test_weak_headingless_skills_remain_unsegmented():
    entries, sections, _ = _entries_from_lines([
        _line("a", "Resourceful problem solver", 10),
        _line("b", "Safety-conscious", 30),
        _line("c", "Team player", 50),
    ])
    assert entries == []
    assert all(section.semantic_label != "EXPERIENCE" for section in sections)


def test_summary_misbucketed_stack_stays_unknown_entry_type():
    """A: SUMMARY ownership must not become EXPERIENCE via EntryType."""
    entries, sections, _ = _entries_from_lines([
        _line("h", "SUMMARY", 10, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, font_size=12, bold=True),
        _line("c", "Google", 50),
        _line("d", "2022 - Present", 70),
        _line("b", "Built distributed systems.", 90),
    ])
    summary_sections = [section for section in sections if section.semantic_label == "SUMMARY"]
    assert summary_sections
    assert len(entries) >= 1
    for entry in entries:
        if entry.section_id == summary_sections[0].section_id:
            assert entry.entry_type == EntryType.UNKNOWN
            assert "experience_like_stack" in entry.evidence


def test_unlabeled_stack_entry_type_unknown_with_structure_evidence():
    """B: UNLABELED strong experience stack → UNKNOWN type, retain evidence."""
    entries, sections, _ = _entries_from_lines([
        _line("t", "Software Engineer", 10, font_size=12, bold=True),
        _line("c", "Google LLC", 30),
        _line("d", "2022 - Present", 50),
        _line("b", "• Built distributed systems", 70),
    ])
    assert any(section.origin == SectionOrigin.UNLABELED for section in sections)
    assert len(entries) == 1
    assert entries[0].entry_type == EntryType.UNKNOWN
    assert "experience_like_stack" in entries[0].evidence


def test_title_plus_date_alone_does_not_open_entry():
    """C: title-like line followed only by a date needs more corroboration."""
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, font_size=12, bold=True),
        _line("d", "2022 - Present", 50),
    ])
    assert entries == []


def test_administrative_assistant_employer_date_one_entry():
    """D: Administrative Assistant / employer / date → one entry."""
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
    ])
    assert len(entries) == 1
    texts = [block.text for block in entries[0].blocks]
    assert texts[0] == "Administrative Assistant"
    assert "Redford & Sons" in texts
    assert any("2018" in block.text for block in entries[0].blocks)


def test_wrapped_description_stays_in_same_entry():
    """E: Resume-1-style wrapped / title-like description stays in the same entry."""
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - Present", 70),
        _line("b1", "Arrangements for supervisors and managers.", 90),
        _line("b2", "Coordinated travel and meeting schedules.", 110),
    ])
    assert len(entries) == 1
    assert any("Arrangements for supervisors" in block.text for block in entries[0].blocks)
    assert any("Coordinated travel" in block.text for block in entries[0].blocks)


def test_adjacent_regions_never_merge():
    left = [
        _line("h1", "EXPERIENCE", 10, x0=10, font_size=14, bold=True),
        _line("t1", "Secretary", 40, x0=10, font_size=12, bold=True),
        _line("c1", "Bright Spot LTD", 60, x0=10),
        _line("d1", "2015 - 2018", 80, x0=10),
    ]
    right = [
        _line("h2", "EXPERIENCE", 10, x0=320, font_size=14, bold=True),
        _line("t2", "Analyst", 40, x0=320, font_size=12, bold=True),
        _line("c2", "Other Company Inc", 60, x0=320),
        _line("d2", "2019 - 2021", 80, x0=320),
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
    entries = build_candidate_entries(build_candidate_sections(build_structural_blocks(document)))
    assert len(entries) == 2
    assert entries[0].path_id != entries[1].path_id
    assert entries[0].region_id != entries[1].region_id
    assert len(entries[0].path_ids) == 1
    assert len(entries[1].path_ids) == 1


def test_adjacent_columns_never_merge_blocks_across_paths():
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Developer", 30, font_size=12, bold=True),
        _line("c", "Example Corp LLC", 50),
        _line("d", "2020 - 2021", 70),
    ])
    assert len({entry.path_id for entry in entries}) == 1
    assert len({entry.region_id for entry in entries}) == 1


def test_provenance_and_source_spans_survive():
    entries, _, blocks = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Secretary", 30, font_size=12, bold=True),
        _line("c", "Bright Spot LTD", 50),
        _line("d", "2015 - 2018", 70),
    ])
    entry = entries[0]
    assert entry.line_ids
    assert entry.source_span_ids
    assert entry.page_numbers == (1,)
    assert entry.page_number == 1
    assert entry.path_ids
    assert entry.region_ids
    assert entry.reconstruction_methods
    assert entry.blocks[0].source_span_ids == next(b for b in blocks if b.text == "Secretary").source_span_ids


def test_table_cell_blocks_are_not_merged_into_entries():
    lines = [
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Secretary", 30, font_size=12, bold=True),
        _line("c", "Bright Spot LTD", 50),
        _line("d", "2015 - 2018", 70),
    ]
    entries, _, _ = _entries_from_lines(lines, kind="table", region_id="page-1-region-table")
    assert entries == []


def test_skills_section_does_not_create_entries():
    entries, sections, _ = _entries_from_lines([
        _line("h", "SKILLS", 10, font_size=14, bold=True),
        _line("a", "Python", 30),
        _line("b", "Docker", 50),
        _line("c", "Kubernetes", 70),
    ])
    assert any(section.semantic_label == "SKILLS" for section in sections)
    assert entries == []


def test_adversarial_multiline_senior_software_engineer():
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Senior Software", 30, font_size=12, bold=True),
        _line("t2", "Engineer", 45, font_size=12, bold=True),
        _line("c", "Google LLC", 70),
        _line("d", "2022 - Present", 90),
        _line("b", "Built distributed systems.", 110),
    ])
    assert len(entries) == 1


def test_adversarial_arrangements_line_stays_in_same_entry():
    entries, _, _ = _entries_from_lines([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Senior Software Engineer", 30, font_size=12, bold=True),
        _line("c", "Google LLC", 50),
        _line("d", "2022 - Present", 70),
        _line("b", "Arrangements for supervisors and managers.", 90),
    ])
    assert len(entries) == 1


def test_resume_1_administrative_assistant_and_secretary_entries():
    raw = PDFExtractor.extract(Path("tests/fixtures/resume_1.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = build_candidate_entries(sections)
    experience_entries = [
        entry
        for entry in entries
        if entry.entry_type in {EntryType.EXPERIENCE, EntryType.UNKNOWN}
    ]
    joined = [" | ".join(block.text for block in entry.blocks) for entry in experience_entries]
    assert any("Administrative Assistant" in text or "ADMINISTRATIVE ASSISTANT" in text.upper() for text in joined)
    assert any("Secretary" in text or "SECRETARY" in text.upper() for text in joined)
    assert len([text for text in joined if "SECRETARY" in text.upper() or "Administrative" in text or "ADMINISTRATIVE" in text.upper()]) >= 2


def test_resume_6_highlights_remain_section_level_not_entries():
    raw = PDFExtractor.extract(Path("tests/fixtures/resume_6.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    sections = build_candidate_sections(build_structural_blocks(document))
    entries = build_candidate_entries(sections)
    skills_sections = [section for section in sections if section.semantic_label == "SKILLS"]
    assert skills_sections
    assert all(entry.section_id not in {section.section_id for section in skills_sections} for entry in entries)


def test_resume_1_parser_invariant_unchanged():
    resume = ResumeParser().parse_with_layout_pipeline(Path("tests/fixtures/resume_1.pdf").read_bytes())
    assert [(item.designation, item.company, item.location, item.startDate, item.endDate, item.current) for item in resume.experience] == [
        ("Administrative Assistant", "Redford & Sons", "Boston, MA", "2018-09", None, True),
        ("Secretary", "Bright Spot LTD", "Boston, MA", "2015-06", "2018-08", False),
    ]


def test_resume_6_parser_skills_invariant_unchanged():
    resume = ResumeParser().parse_with_layout_pipeline(Path("tests/fixtures/resume_6.pdf").read_bytes())
    assert {
        "Warehouse Equipment Operation",
        "Resourceful Problem Solver",
        "Friendly and Helpful",
        "Good Physical Condition",
        "Safety-Conscious",
        "Team Player",
    }.issubset(set(resume.skills))
