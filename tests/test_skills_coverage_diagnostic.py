"""Phase 6F skills coverage: vocabulary-expanded acceptance expectations."""

from __future__ import annotations

from pathlib import Path

from app.extractors.skills import SkillsExtractor
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.text_extraction import TextBlock


def _accept(text: str) -> list[str]:
    return [item["value"] for item in SkillsExtractor().extract([TextBlock(text, 1, 0, 0, 100, 10)], section_name="SKILLS")]


def test_A_common_tech_aliases_java_python_excel():
    assert _accept("Java") == ["Java"]
    assert _accept("Python") == ["Python"]
    assert _accept("Excel") == ["Excel"]
    assert _accept("MS Excel") == ["Excel"]


def test_B_employee_engagement_in_vocab():
    assert _accept("Employee Engagement") == ["Employee Engagement"]


def test_C_recruitment_and_selection_in_vocab():
    assert _accept("Recruitment & Selection") == ["Recruitment"]
    assert _accept("Recruitment & Selection Basics") == ["Recruitment"]


def test_D_teamwork_in_vocab():
    assert _accept("Teamwork") == ["Teamwork"]


def test_E_communication_in_vocab():
    assert _accept("Communication") == ["Communication"]
    assert _accept("Strong Verbal & Written Communication") == ["Communication"]


def test_F_arbitrary_domain_skill_rejected():
    assert _accept("Arbitrary Widget Orchestration") == []


def test_G_comma_list_emits_known_aliases_including_excel():
    assert set(_accept("Java, Python, Excel")) == {"Java", "Python", "Excel"}


def test_H_bullet_prefix_still_matches_known_alias():
    assert _accept("• Python") == ["Python"]


def test_I_fontawesome_prefixed_skill_line_recovers_engagement():
    assert _accept("\\faUsers : Employee Engagement") == ["Employee Engagement"]


def test_J_numeric_rating_alone_rejected():
    assert _accept("4") == []


def test_K_heading_plus_list_emits_vocab_hits():
    text = "Soft Skills\nTeamwork\nTime Management\nAdaptability"
    assert set(_accept(text)) == {"Teamwork", "Time Management", "Adaptability"}


def test_L_multiple_skill_sections_recover_tools_and_soft_skills():
    text = (
        "Human Resource Skills\nEmployee Engagement Concepts\n"
        "Tools & Technical Skills\nMS Excel (VLOOKUP, Pivot Tables, Filters)\n"
        "Soft Skills\nTeam Collaboration\nTime Management"
    )
    assert set(_accept(text)) >= {"Employee Engagement", "Excel", "Collaboration", "Time Management"}


def test_resume_a_skills_improve_from_skills_stream():
    path = Path("tests/fixtures/AditCV_SOL.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    skills = set(resume.skills)
    assert {"Time Management", "Problem Solving", "Adaptability", "Collaboration"} <= skills
    assert {"Excel", "Google Sheets", "Google Docs", "HRMS", "Communication", "Teamwork"} <= skills
    assert "Employee Engagement" in skills or "Recruitment" in skills
    assert len(resume.skills) >= 10
    # CORE HR grid remains EDUCATION-owned in 6F; do not require those cells here.


def test_resume_b_skills_improve_from_skills_stream():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    skills = set(resume.skills)
    assert {"Collaboration", "Time Management", "Adaptability"} <= skills
    assert {"Excel", "Recruitment", "Employee Engagement", "Onboarding", "Communication"} <= skills
    assert len(resume.skills) >= 8


def test_resume_c_skills_remain_dense():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert len(resume.skills) >= 30
    assert {"Java", "Spring Boot", "PostgreSQL", "Docker"} <= set(resume.skills)
