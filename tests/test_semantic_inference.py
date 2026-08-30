import pytest

from app.pipeline.stages.semantic_inference import infer_section
from app.pipeline.stages.sections import SectionDetector
from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.semantic_paths import _is_unknown_heading_line, detect_region_aware_sections


def _line(line_id, text, y0, font_size=11.0, bold=False, x0=10.0):
    bbox = BoundingBox(x0, y0, x0 + 200.0, y0 + 10.0)
    span = Span(f"{line_id}-span", text, bbox, font_size=font_size, bold=bold)
    return Line(line_id, 1, bbox, [span], text, TextStyle(font_size=font_size, bold=bold), source_span_ids=[span.span_id])


def test_unknown_skill_section_uses_content_over_heading():
    result = infer_section("HIGHLIGHTS", ["Team player", "Safety-conscious", "Problem solving"])

    assert result.section == "SKILLS"
    assert "competency_phrase" in result.signals["SKILLS"]


def test_unknown_achievement_section_uses_metrics_and_recognition():
    result = infer_section("HIGHLIGHTS", ["Employee of the Month", "Reduced processing time by 35%"])

    assert result.section == "ACHIEVEMENTS"


def test_unknown_experience_section_uses_role_company_and_dates():
    result = infer_section("PROFESSIONAL BACKGROUND", ["Software Engineer", "ABC Corp", "2020 - 2023"])

    assert result.section == "EXPERIENCE"


def test_unknown_education_section_uses_academic_patterns():
    result = infer_section("ACADEMIC BACKGROUND", ["Bachelor of Science", "University", "2018"])

    assert result.section == "EDUCATION"


def test_unknown_certification_section_uses_credential_content():
    result = infer_section("CREDENTIALS", ["AWS Certified Solutions Architect"])

    assert result.section == "CERTIFICATIONS"


def test_unknown_project_section_uses_project_content():
    result = infer_section("SELECTED WORK", ["Project title", "Built a web application"])

    assert result.section == "PROJECTS"


def test_unknown_language_section_uses_proficiency_content():
    result = infer_section("LANGUAGE PROFICIENCY", ["English - Native", "Spanish - Fluent"])

    assert result.section == "LANGUAGES"


def test_ambiguous_unknown_section_remains_unknown():
    result = infer_section("RANDOM INFORMATION", ["Managed systems and improved processes"])

    assert result.section == "UNKNOWN"


def test_valid_profile_prose_can_infer_summary():
    result = infer_section("PROFILE", ["Experienced software engineer with 8 years of experience leading platform teams and delivering reliable services."])

    assert result.section == "SUMMARY"


def test_generic_background_paragraph_remains_unknown():
    result = infer_section("BACKGROUND", ["Worked in various professional environments."])

    assert result.section == "UNKNOWN"


def test_content_overrides_conflicting_heading_hint():
    result = infer_section("CAREER HIGHLIGHTS", ["Team player", "Safety-conscious", "Problem solving"])

    assert result.section == "SKILLS"


@pytest.mark.parametrize("content", [
    "Managed systems and improved processes",
    "Improved processes and maintained documentation",
    "Managed projects and improved workflows",
    "Worked with teams to improve processes",
])
def test_generic_action_prose_remains_unknown(content):
    assert infer_section("RANDOM INFORMATION", [content]).section == "UNKNOWN"


@pytest.mark.parametrize("content", [
    "Reduced processing time by 35%",
    "Reduced operating costs by 25%",
    "Improved customer satisfaction by 20%",
    "Employee of the Month",
    "Awarded Employee of the Year",
    "Increased productivity by 20%",
])
def test_corroborated_achievement_content_infers_achievements(content):
    assert infer_section("RANDOM INFORMATION", [content]).section == "ACHIEVEMENTS"


@pytest.mark.parametrize("heading,content", [
    ("Major Initiatives", ["Inventory Management System", "Built a web application to manage warehouse inventory.", "Implemented search, reporting, and authentication."]),
    ("Selected Work", ["Customer Portal", "Developed a React and Spring Boot application for customer self-service."]),
    ("Key Initiatives", ["Cloud Migration", "Migrated legacy services to AWS and automated deployment."]),
])
def test_project_like_content_infers_projects(heading, content):
    assert infer_section(heading, content).section == "PROJECTS"


@pytest.mark.parametrize("heading,content", [
    ("Language Proficiency Profile", ["English - Native", "Spanish - Fluent"]),
    ("Spoken Languages", ["English", "Hindi", "Spanish - Professional"]),
    ("Communication Languages", ["English — Fluent", "French — Intermediate"]),
])
def test_language_like_content_infers_languages(heading, content):
    assert infer_section(heading, content).section == "LANGUAGES"


def test_action_prose_does_not_infer_projects():
    assert infer_section("Major Initiatives", ["Managed systems and improved processes"]).section == "UNKNOWN"

    def test_lone_generic_content_does_not_infer_projects():
        result = infer_section("RANDOM INFORMATION", ["generic text"])

        assert result.section == "UNKNOWN"
        assert result.scores["PROJECTS"] == 0.0


    def test_migration_project_pair_infers_projects_but_lone_migration_stays_unknown():
        assert infer_section(
            "Key Initiatives",
            ["Cloud Migration", "Migrated legacy services to AWS and automated deployment."],
        ).section == "PROJECTS"
        assert infer_section(
            "RANDOM INFORMATION",
            ["Migrated legacy services to AWS."],
        ).section == "UNKNOWN"


def test_project_entry_pair_infers_projects_without_project_heading():
    result = infer_section("Selected Work", ["Customer Portal", "Developed a React and Spring Boot application for customer self-service."])

    assert result.section == "PROJECTS"
    assert "project_entry_pair" in result.signals["PROJECTS"]


def test_action_prose_does_not_infer_languages():
    assert infer_section("Language Proficiency Profile", ["Managed systems and improved processes"]).section == "UNKNOWN"


def test_known_section_heading_remains_deterministic():
    from app.pipeline.stages.sections import SectionDetector

    assert SectionDetector()._find_section_header("SKILLS") == "SKILLS"


def test_wrapped_title_case_content_is_not_an_unknown_heading():
    previous = _line("previous", "Schedule and coordinate meetings, appointments, and travel", 10)
    current = _line("current", "Arrangements for Supervisors and Managers", 20)

    assert _is_unknown_heading_line(current, [previous, current], 1, SectionDetector()) is False


def test_unknown_heading_followed_by_bullets_is_detectable():
    heading = _line("heading", "Selected Contributions", 10, font_size=14)
    bullet = _line("bullet", "Team player", 25)
    marker = _line("marker", "•", 24)
    document = Document(pages=[Page(1, regions=[Region("region", "column", BoundingBox(0, 0, 300, 100), [heading, marker, bullet])])])

    result = detect_region_aware_sections(document)

    assert result.unknown_candidates[0].heading.line_id == "heading"
    assert [line.line_id for line in result.unknown_candidates[0].content] == ["marker", "bullet"]


def test_provenance_survives_unknown_section_inference():
    heading = _line("heading", "Selected Contributions", 10, font_size=14)
    content = _line("content", "Team player", 25)
    document = Document(pages=[Page(1, regions=[Region("region", "column", BoundingBox(0, 0, 300, 100), [heading, content])])])

    result = detect_region_aware_sections(document)

    assert result.unknown_candidates[0].heading.line_id == "heading"
    assert result.unknown_candidates[0].content[0].source_span_ids == ["content-span"]


def test_item_heading_with_employment_metadata_stays_in_active_section():
    from app.pipeline.stages.semantic_paths import _is_unknown_heading_line

    lines = [
        _line("role", "ADMINISTRATIVE ASSISTANT", 10, font_size=12),
        _line("company", "Example Company, Boston, MA / September 2018 - Present", 30),
        _line("description", "Schedule and coordinate meetings", 50),
    ]

    assert _is_unknown_heading_line(lines[0], lines, 0, SectionDetector(), "EXPERIENCE") is False


def test_resume_6_highlights_are_inferred_as_skills():
    from app.domain.document import document_from_text_blocks
    from app.pipeline.parser import ResumeParser
    from app.pipeline.stages.layout import interpret_layout
    from app.pipeline.stages.reconstruction import reconstruct_document
    from app.pipeline.stages.semantic_compat import semantic_sections_to_text_blocks
    from app.pipeline.stages.semantic_paths import detect_region_aware_sections
    from app.pipeline.stages.text_extraction import PDFExtractor
    from tests.conftest import require_fixture

    raw = require_fixture("resume_6.pdf").read_bytes()
    document = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(raw))))
    semantic = detect_region_aware_sections(document)
    skills = semantic_sections_to_text_blocks(semantic)["SKILLS"]

    assert {"Warehouse equipment", "operation", "Resourceful problem solver", "Friendly and helpful", "Good physical condition", "Safety-conscious", "Team player"} <= {block.text for block in skills}

    result = ResumeParser().parse_with_layout_pipeline(raw)
    assert {
        "Warehouse Equipment Operation",
        "Resourceful Problem Solver",
        "Friendly and Helpful",
        "Good Physical Condition",
        "Safety-Conscious",
        "Team Player",
    }.issubset(set(result.skills))