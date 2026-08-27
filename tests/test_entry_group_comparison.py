from pathlib import Path

from app.domain.candidate_entry import EntryType
from app.domain.candidate_section import SectionOrigin
from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle, document_from_text_blocks
from app.domain.structural import StructuralRole
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.block_classification import classify_block
from app.pipeline.stages.candidate_entries import build_candidate_entries
from app.pipeline.stages.candidate_grouping import group_candidates
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.entry_compat import candidate_entries_to_extractor_views
from app.pipeline.stages.entry_comparison import (
    ENTRY_ONLY_INFORMATION,
    GROUP_ONLY_INFORMATION,
    compare_document,
    compare_entries_to_section_compat_groups,
    compare_entry_views_and_groups,
    disagreement_categories,
)
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.structural_roles import build_structural_blocks
from app.pipeline.stages.text_extraction import PDFExtractor, TextBlock


def _line(
    line_id: str,
    text: str,
    y0: float,
    *,
    page: int = 1,
    x0: float = 10.0,
    font_size: float = 11.0,
    bold: bool = False,
    reconstruction_method: str = "physical",
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
        reconstruction_method=reconstruction_method,
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


def _views_from_document(document: Document):
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    entries = build_candidate_entries(sections)
    return candidate_entries_to_extractor_views(entries, sections), sections, entries, blocks


def test_information_loss_inventories_are_explicit():
    report = compare_document(_single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
    ]))
    assert report.group_only_information == GROUP_ONLY_INFORMATION
    assert report.entry_only_information == ENTRY_ONLY_INFORMATION


def test_multiple_experience_entries_comparison():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c1", "Redford & Sons", 50),
        _line("d1", "2018 - 2021", 70),
        _line("b1", "• Managed schedules.", 90),
        _line("t2", "Office Manager", 140, font_size=12, bold=True),
        _line("c2", "ABC Corp LLC", 160),
        _line("d2", "2021 - Present", 180),
        _line("b2", "• Managed operations.", 200),
    ])
    _, _, entries, _ = _views_from_document(document)
    production = compare_document(document)
    report = compare_entries_to_section_compat_groups(document)
    assert len(entries) == 2
    assert all(entry.entry_type == EntryType.EXPERIENCE for entry in entries)
    assert report.entry_count == 2
    # Production semantic_paths may leave this synthetic stack UNASSIGNED.
    if production.group_count == 0:
        assert "ENTRY_COUNT_DIFFERENCE" in disagreement_categories(production)
        assert "MEMBERSHIP_DIFFERENCE" in disagreement_categories(production)
    assert report.group_count >= 1


def test_multiple_education_entries_comparison():
    document = _single_column([
        _line("h", "EDUCATION", 10, font_size=14, bold=True),
        _line("d1", "Bachelor of Science", 30, font_size=12, bold=True),
        _line("u1", "University of Chicago", 50),
        _line("y1", "2014 - 2018", 70),
        _line("d2", "Master of Science", 120, font_size=12, bold=True),
        _line("u2", "MIT Institute", 140),
        _line("y2", "2018 - 2020", 160),
    ])
    _, sections, entries, _ = _views_from_document(document)
    assert any(section.semantic_label == "EDUCATION" for section in sections)
    assert all(entry.entry_type == EntryType.EDUCATION for entry in entries)
    report = compare_entries_to_section_compat_groups(document)
    assert report.entry_count >= 1
    assert report.group_count >= 1


def test_project_title_and_date_comparison():
    document = _single_column([
        _line("h", "PROJECTS", 10, font_size=14, bold=True),
        _line("t1", "Project Alpha", 30, font_size=12, bold=True),
        _line("d1", "2024 - 2024", 50),
        _line("b1", "Built application.", 70),
        _line("t2", "Project Beta", 120, font_size=12, bold=True),
        _line("d2", "2025 - 2025", 140),
        _line("b2", "Built another application.", 160),
    ])
    _, _, entries, _ = _views_from_document(document)
    assert len(entries) == 2
    assert all(entry.entry_type == EntryType.PROJECT for entry in entries)
    production = compare_document(document)
    report = compare_entries_to_section_compat_groups(document)
    assert report.entry_count == 2
    assert production.entry_count == 2


def test_certification_entries_comparison():
    document = _single_column([
        _line("h", "CERTIFICATIONS", 10, font_size=14, bold=True),
        _line("c", "AWS Certified Solutions Architect", 30, font_size=12, bold=True),
        _line("o", "Amazon Web Services Inc", 50),
        _line("d", "Issued 2024 - 2025", 70),
    ])
    views, _, entries, _ = _views_from_document(document)
    assert entries
    assert entries[0].entry_type == EntryType.CERTIFICATION
    report = compare_entries_to_section_compat_groups(document)
    assert report.entry_count >= 1
    assert views[0].members


def test_multiline_title_provenance_survives_adapter():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Senior Software", 30, font_size=12, bold=True),
        _line("t2", "Engineer", 45, font_size=12, bold=True, reconstruction_method="adjacent_physical_fragments"),
        _line("c", "Google LLC", 70),
        _line("d", "2022 - Present", 90),
    ])
    views, _, entries, _ = _views_from_document(document)
    assert len(entries) == 1
    view = views[0]
    assert "t1" in view.line_ids
    assert "t2" in view.line_ids
    assert set(view.line_ids) == {lid for member in view.members for lid in member.line_ids}
    assert "adjacent_physical_fragments" in view.reconstruction_methods
    assert list(getattr(view.members[0].text_block, "source_line_ids")) == list(view.members[0].line_ids)


def test_wrapped_description_stays_one_entry_on_entry_side():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - Present", 70),
        _line("b1", "Arrangements for supervisors and managers.", 90),
        _line("b2", "Coordinated travel and meeting schedules.", 110),
    ])
    views, _, entries, _ = _views_from_document(document)
    assert len(entries) == 1
    texts = [member.text for member in views[0].members]
    assert any("Arrangements" in text for text in texts)
    assert any("Coordinated travel" in text for text in texts)


def test_entries_spanning_pages_scope_is_recorded():
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
    report = compare_document(document)
    categories = disagreement_categories(report)
    assert categories.intersection({
        "SCOPE_DIFFERENCE",
        "ENTRY_COUNT_DIFFERENCE",
        "ENTRY_BOUNDARY_DIFFERENCE",
        "MEMBERSHIP_DIFFERENCE",
    })


def test_two_column_layouts_never_merge_on_entry_side():
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
    views, _, entries, _ = _views_from_document(document)
    assert len(entries) == 2
    assert views[0].path_ids != views[1].path_ids
    report = compare_document(document)
    assert report.entry_count == 2


def test_summary_experience_like_does_not_override_ownership():
    document = _single_column([
        _line("h", "SUMMARY", 10, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, font_size=12, bold=True),
        _line("c", "Google", 50),
        _line("d", "2022 - Present", 70),
        _line("b", "Built distributed systems.", 90),
    ])
    views, sections, entries, _ = _views_from_document(document)
    assert any(section.semantic_label == "SUMMARY" for section in sections)
    assert entries
    assert all(entry.entry_type == EntryType.UNKNOWN for entry in entries)
    assert all(view.section_label == "SUMMARY" for view in views)
    assert all("experience_like_stack" in view.evidence for view in views)
    compare_document(document)


def test_unlabeled_section_entry_type_unknown():
    document = _single_column([
        _line("t", "Software Engineer", 10, font_size=12, bold=True),
        _line("c", "Google LLC", 30),
        _line("d", "2022 - Present", 50),
        _line("b", "• Built distributed systems", 70),
    ])
    views, sections, _, _ = _views_from_document(document)
    assert any(section.origin == SectionOrigin.UNLABELED for section in sections)
    assert views
    assert all(view.entry_type == EntryType.UNKNOWN for view in views)
    assert all(view.section_origin == SectionOrigin.UNLABELED for view in views)
    compare_document(document)


def test_unknown_heading_section_comparison():
    document = _single_column([
        _line("h", "MISCELLANEOUS NOTES", 10, font_size=14, bold=True),
        _line("a", "Independent research notes.", 30),
        _line("b", "Not a job stack.", 50),
    ])
    _, sections, _, _ = _views_from_document(document)
    assert all(section.semantic_label != "EXPERIENCE" for section in sections)
    report = compare_document(document)
    assert report.group_only_information


def test_date_plus_organization_opens_entry():
    views, _, entries, _ = _views_from_document(_single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
    ]))
    assert len(entries) == 1
    assert "t" in views[0].line_ids


def test_date_alone_does_not_open_candidate_entry():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Software Engineer", 30, font_size=12, bold=True),
        _line("d", "2022 - Present", 50),
    ])
    _, _, entries, _ = _views_from_document(document)
    assert entries == []
    report = compare_document(document)
    assert report.entry_count == 0
    if report.group_count:
        assert "ENTRY_COUNT_DIFFERENCE" in disagreement_categories(report)
        assert "MEMBERSHIP_DIFFERENCE" in disagreement_categories(report)


def test_repeated_role_titles_are_compared_not_forced():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "SECRETARY", 30, font_size=12, bold=True),
        _line("c", "Bright Spot LTD", 50),
        _line("d", "2015 - 2018", 70),
        _line("t2", "SECRETARY", 90, font_size=12, bold=True),
        _line("b", "• Filed documents", 110),
    ])
    report = compare_entries_to_section_compat_groups(document)
    _, _, entries, _ = _views_from_document(document)
    assert report.entry_count == len(entries)
    assert report.group_count >= 1


def test_education_degree_and_institution_on_entry_side():
    views, _, entries, _ = _views_from_document(_single_column([
        _line("h", "EDUCATION", 10, font_size=14, bold=True),
        _line("deg", "Bachelor of Science", 30, font_size=12, bold=True),
        _line("u", "University of Chicago", 50),
        _line("d", "2014 - 2018", 70),
    ]))
    assert len(entries) == 1
    assert views[0].members


def test_candidate_group_right_column_date_does_not_merge_entry_paths():
    left = [
        _line("h", "EDUCATION", 10, x0=10, font_size=14, bold=True),
        _line("deg", "Bachelor of Science", 40, x0=10, font_size=12, bold=True),
        _line("u", "University of Chicago", 60, x0=10),
    ]
    right = [_line("dt", "2018 - 2022", 40, x0=280)]
    document = Document(
        pages=[
            Page(
                1,
                regions=[
                    Region("page-1-region-0", "column", BoundingBox(0, 0, 200, 90), left, 0, 0),
                    Region("page-1-region-1", "column", BoundingBox(250, 0, 420, 60), right, 1, 1),
                ],
            )
        ]
    )
    _, _, entries, _ = _views_from_document(document)
    if entries:
        assert all(len(entry.region_ids) == 1 for entry in entries)
    report = compare_document(document)
    assert report.group_only_information


def test_candidate_group_large_gap_splits_group_not_entry():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
        _line("b1", "• Managed schedules.", 90),
        _line("b2", "• Prepared long-range reports for leadership.", 200),
    ])
    views, _, entries, _ = _views_from_document(document)
    assert len(entries) == 1
    classified = [classify_block(member.text_block) for member in views[0].members]
    groups = group_candidates(classified, "EXPERIENCE")
    report = compare_entry_views_and_groups(views, groups)
    assert report.entry_count == 1
    if report.group_count > 1:
        assert "ENTRY_COUNT_DIFFERENCE" in disagreement_categories(report)
        assert "ENTRY_BOUNDARY_DIFFERENCE" in disagreement_categories(report)


def test_candidate_group_page_split_behavior():
    blocks = [
        TextBlock("Software Engineer", 1, 10, 30, 200, 42, font_size=12, bold=True),
        TextBlock("Google LLC", 1, 10, 50, 200, 62, font_size=11, bold=False),
        TextBlock("2022 - Present", 1, 10, 70, 200, 82, font_size=11, bold=False),
        TextBlock("• Continued on next page", 2, 10, 20, 200, 32, font_size=11, bold=False),
    ]
    for index, block in enumerate(blocks):
        block.source_line_id = f"p{block.page_number}-{index}"
    groups = group_candidates([classify_block(block) for block in blocks], "EXPERIENCE")
    assert len(groups) >= 2
    assert {group.page_number for group in groups} == {1, 2}
    report = compare_entry_views_and_groups([], groups)
    assert report.entry_count == 0
    assert "ENTRY_COUNT_DIFFERENCE" in disagreement_categories(report)


def test_adapter_does_not_map_structural_roles_to_classifier_labels():
    views, _, _, _ = _views_from_document(_single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons", 50),
        _line("d", "2018 - 2021", 70),
    ]))
    dumped = " ".join(member.structural_role.value for member in views[0].members)
    assert "JOB_TITLE" not in dumped
    assert "DEGREE" not in dumped
    assert all(isinstance(member.structural_role, StructuralRole) for member in views[0].members)


def test_group_classifier_labels_surface_in_comparison_pairs():
    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t", "Administrative Assistant", 30, font_size=12, bold=True),
        _line("c", "Redford & Sons LLC", 50),
        _line("d", "2018 - 2021", 70),
    ])
    report = compare_document(document)
    labels = [label for pair in report.pairs for label in pair.classification_labels]
    assert labels


def test_provenance_not_derived_from_first_block_only():
    views, _, _, _ = _views_from_document(_single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("t1", "Senior Software", 30, font_size=12, bold=True),
        _line("t2", "Engineer", 45, font_size=12, bold=True, reconstruction_method="adjacent_physical_fragments"),
        _line("c", "Google LLC", 70),
        _line("d", "2022 - Present", 90),
    ]))
    view = views[0]
    assert view.line_ids[0] != view.line_ids[-1]
    assert view.reading_order == min(member.reading_order for member in view.members)
    assert set(view.source_span_ids) == {sid for member in view.members for sid in member.source_span_ids}


def test_resume_1_parser_invariant_unchanged():
    resume = ResumeParser().parse_with_layout_pipeline(Path("tests/fixtures/resume_1.pdf").read_bytes())
    assert [(item.designation, item.company, item.location, item.startDate, item.endDate, item.current) for item in resume.experience] == [
        ("Administrative Assistant", "Redford & Sons", "Boston, MA", "2018-09", None, True),
        ("Secretary", "Bright Spot LTD", "Boston, MA", "2015-06", "2018-08", False),
    ]
    raw = PDFExtractor.extract(Path("tests/fixtures/resume_1.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    report = compare_document(document)
    assert report.entry_count >= 1
    assert report.group_only_information


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
