"""Phase 6V: experience-description skill mode suppresses soft-skill FPs."""

from __future__ import annotations

from pathlib import Path

from app.extractors.experience import ExperienceExtractor
from app.extractors.skills import SkillsExtractor
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.block_classification import ClassifiedBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup
from app.pipeline.stages.text_extraction import TextBlock


def _blk(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=100, y1=10)


def _values(text: str, *, mode: str | None = None, section_name: str | None = None) -> list[str]:
    return [
        item["value"]
        for item in SkillsExtractor().extract(
            [_blk(text)],
            section_name=section_name,
            mode=mode,
        )
    ]


def _experience_skills_from_description(description: str) -> list[str]:
    group = CandidateGroup(
        section="EXPERIENCE",
        blocks=[
            ClassifiedBlock(original=_blk("Software Engineer"), label="JOB_TITLE", score=1.0, reasons=[]),
            ClassifiedBlock(original=_blk("Acme Corp"), label="COMPANY", score=1.0, reasons=[]),
            ClassifiedBlock(original=_blk("Jan 2022 - Present"), label="DATE", score=1.0, reasons=[]),
            ClassifiedBlock(original=_blk(description), label="BULLET", score=1.0, reasons=[]),
        ],
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=3,
        summary_text="",
    )
    entries = ExperienceExtractor().extract(groups=[group])
    return list(entries[0]["skills"] or [])


def test_fp_improved_communication_suppressed():
    skills = _experience_skills_from_description("Improved communication between teams.")
    assert "Communication" not in skills


def test_fp_communication_and_teamwork_suppressed():
    skills = _experience_skills_from_description("Strong communication and teamwork.")
    assert "Communication" not in skills
    assert "Teamwork" not in skills


def test_fp_willingness_to_learn_suppressed():
    skills = _experience_skills_from_description(
        "Worked closely with the HR team while demonstrating responsibility, "
        "professionalism, and a willingness to learn."
    )
    assert "Willingness to Learn" not in skills


def test_fp_asynchronous_communication_keeps_tech():
    skills = _experience_skills_from_description(
        "Used Kafka and MQTT for asynchronous communication."
    )
    assert "Apache Kafka" in skills or "Kafka" in skills
    # Alias map uses Apache Kafka for kafka
    assert "MQTT" in skills
    assert "Communication" not in skills


def test_tp_python_fastapi_rest():
    skills = _experience_skills_from_description(
        "Built REST APIs using Python and FastAPI."
    )
    assert set(skills) >= {"REST", "Python", "FastAPI"}


def test_tp_java_spring_boot():
    skills = _experience_skills_from_description(
        "Developed services using Java and Spring Boot."
    )
    assert set(skills) >= {"Java", "Spring Boot"}


def test_tp_docker_kubernetes():
    skills = _experience_skills_from_description(
        "Deployed Docker containers to Kubernetes."
    )
    assert set(skills) >= {"Docker", "Kubernetes"}


def test_top_level_soft_skills_still_extract():
    blocks = [
        _blk("Communication"),
        _blk("Teamwork"),
        _blk("Willingness to Learn"),
    ]
    values = [
        item["value"]
        for item in SkillsExtractor().extract(blocks, section_name="SKILLS")
    ]
    assert set(values) >= {"Communication", "Teamwork", "Willingness to Learn"}


def test_experience_mode_direct_api():
    assert "Communication" not in _values(
        "asynchronous communication", mode="experience_description"
    )
    assert "Communication" in _values("Communication", section_name="SKILLS")


def test_resume_a_b_c_acceptance():
    fixtures = Path("tests/fixtures")
    a_path = fixtures / "AditCV_SOL.pdf"
    b_path = fixtures / "fresher_hr_resume.pdf"
    c_path = fixtures / "swe_experienced_resume.pdf"
    if not (a_path.exists() and b_path.exists() and c_path.exists()):
        return

    parser = ResumeParser()
    a = parser.parse_with_layout_pipeline(a_path.read_bytes())
    b = parser.parse_with_layout_pipeline(b_path.read_bytes())
    c = parser.parse_with_layout_pipeline(c_path.read_bytes())

    assert len(a.skills) == 19
    assert len(a.experience) == 1
    assert set(a.experience[0].skills or []) == {
        "Recruitment",
        "Onboarding",
        "Employee Engagement",
    }
    assert "Willingness to Learn" not in (a.experience[0].skills or [])

    assert len(b.experience) == 0
    assert len(b.skills) == 16

    assert len(c.skills) == 39
    exp0 = set(c.experience[0].skills or [])
    assert "Communication" not in exp0
    assert exp0 >= {
        "Spring Boot",
        "SOLID",
        "Saga Pattern",
        "Spring Cloud",
        "Apache Kafka",
        "MQTT",
        "Apache Ignite",
        "Hibernate",
        "Jenkins",
        "GitHub",
    }
    assert set(c.experience[1].skills or []) >= {
        "Java",
        "Spring Boot",
        "PostgreSQL",
        "REST",
        "Git",
    }
