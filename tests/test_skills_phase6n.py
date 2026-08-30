"""Phase 6N: narrow HR Policy / Labor Law extractor aliases."""

from __future__ import annotations

from pathlib import Path

from app.extractors.skills import SkillsExtractor
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.text_extraction import TextBlock


def _accept(text: str) -> list[str]:
    return [
        item["value"]
        for item in SkillsExtractor().extract(
            [TextBlock(text, 1, 0, 0, 100, 10)], section_name="SKILLS"
        )
    ]


def test_hr_policy_singular_and_plural_share_canonical():
    assert _accept("HR Policy") == ["HR Policy"]
    assert _accept("HR Policies") == ["HR Policy"]
    assert _accept("HR Policy\nHR Policies") == ["HR Policy"]


def test_labor_law_family_maps_to_labour_laws_once():
    assert _accept("Labor Law") == ["Labour Laws"]
    assert _accept("Labour Law") == ["Labour Laws"]
    assert _accept("Labor Laws") == ["Labour Laws"]
    assert _accept("Labour Laws") == ["Labour Laws"]
    assert _accept("Basic Labor Law Knowledge") == ["Labour Laws"]
    assert _accept("Basic Labour Law Knowledge") == ["Labour Laws"]
    assert set(
        _accept(
            "Labor Law\nLabour Law\nLabor Laws\nLabour Laws\n"
            "Basic Labor Law Knowledge\nBasic Labour Law Knowledge"
        )
    ) == {"Labour Laws"}


def test_ultra_broad_tokens_do_not_become_skills():
    for token in ("law", "policy", "labor", "labour"):
        assert _accept(token) == []


def test_section_headings_do_not_become_skills():
    for heading in (
        "Technical Skills",
        "Soft Skills",
        "Human Resource Skills",
        "Tools & Technical Skills",
        "Key Competencies",
    ):
        assert _accept(heading) == []


def test_numeric_ratings_do_not_become_skills():
    for rating in ("3", "4", "5"):
        assert _accept(rating) == []


def test_resume_a_recovers_hr_policy_once():
    path = Path("tests/fixtures/AditCV_SOL.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.skills.count("HR Policy") == 1
    assert "HR Policies" not in resume.skills
    assert len(resume.skills) >= 19


def test_resume_b_recovers_labour_laws_once():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.skills.count("Labour Laws") == 1
    for variant in ("Labor Law", "Labour Law", "Labor Laws"):
        assert variant not in resume.skills
    assert len(resume.skills) >= 16


def test_resume_c_engineering_skills_stable():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert len(resume.skills) >= 39
    assert {
        "Java",
        "SQL",
        "MySQL",
        "PostgreSQL",
        "Spring Boot",
        "Docker",
        "Kubernetes",
        "AWS",
    } <= set(resume.skills)
