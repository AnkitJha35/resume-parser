import pytest

from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.semantic_inference import infer_section


@pytest.mark.parametrize("heading", [
    "Summary", "Professional Summary", "Profile", "Career Objective",
    "Experience", "Work Experience", "Employment",
    "Education", "Skills", "Projects", "Certifications", "Credentials",
    "Achievements", "Awards", "Languages",
])
def test_known_and_common_aliases_are_deterministic(heading):
    assert SectionDetector()._find_section_header(heading) is not None


@pytest.mark.parametrize("heading,content,expected", [
    ("Professional Background", ["Software Engineer", "ABC Corp", "2020 - 2023"], "EXPERIENCE"),
    ("Academic Record", ["Bachelor of Science", "University", "2018"], "EDUCATION"),
    ("Core Competencies", ["Team player", "Safety-conscious", "Problem solving"], "SKILLS"),
    ("Professional Credentials", ["AWS Certified Solutions Architect"], "CERTIFICATIONS"),
    ("Major Initiatives", ["Inventory Management System", "Built Spring Boot application"], "PROJECTS"),
    ("Language Proficiency Profile", ["English - Native", "Spanish - Fluent"], "LANGUAGES"),
    ("Career Highlights", ["Employee of the Month", "Reduced processing time by 35%"], "ACHIEVEMENTS"),
])
def test_unknown_heading_equivalents_infer_from_content(heading, content, expected):
    assert SectionDetector()._find_section_header(heading) is None
    assert infer_section(heading, content).section == expected


@pytest.mark.parametrize("content", [
    "Managed systems and improved processes",
    "Developed REST APIs using Spring Boot",
])
def test_ambiguous_unknown_content_remains_unknown(content):
    assert infer_section("RANDOM INFORMATION", [content]).section == "UNKNOWN"


def test_profile_content_infers_summary_without_alias_lookup():
    assert SectionDetector()._find_section_header("PROFILE") == "SUMMARY"
    assert infer_section("PROFILE", ["Experienced software engineer with expertise in distributed systems"]).section == "SUMMARY"


def test_content_overrides_conflicting_heading_hint():
    assert infer_section("CAREER HIGHLIGHTS", ["Team player", "Safety-conscious", "Problem solving"]).section == "SKILLS"
