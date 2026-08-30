"""Phase 7A: locked golden acceptance for resumes A/B/C.

These assertions encode the Phase 6W-accepted production contract.
Do not weaken them to accommodate unrelated parser experiments.
"""

from __future__ import annotations

from pathlib import Path

from app.pipeline.parser import ResumeParser

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_LIGATURES = "\ufb00\ufb01\ufb02\ufb03\ufb04"


def _parse(name: str):
    path = _FIXTURES / name
    if not path.exists():
        import pytest

        pytest.skip(f"missing golden fixture: {name}")
    return ResumeParser().parse_with_layout_pipeline(path.read_bytes())


def test_golden_resume_a_aditcv_sol():
    resume = _parse("AditCV_SOL.pdf")
    assert resume.personal.name == "ADITI ANAND"
    assert resume.personal.email == "aditianand136@gmail.com"
    assert resume.personal.phone == "+91 9241242699"
    assert resume.personal.linkedin == "https://www.linkedin.com/in/aditianand99"
    assert resume.personal.location is None
    assert resume.summary
    assert len(resume.experience) == 1
    exp = resume.experience[0]
    assert exp.designation == "HR Intern"
    assert exp.company == "ECE Industries Ltd."
    assert exp.location == "Sonipat, Haryana"
    assert exp.startDate == "2025-10"
    assert exp.endDate == "2025-12"
    assert len(resume.education) == 1
    edu = resume.education[0]
    assert edu.degree == "Master of Business Administration"
    assert edu.fieldOfStudy == "Human Resource Management"
    assert "School of Open Learning" in (edu.institution or "")
    assert len(resume.skills) == 19
    assert resume.languages == ["English", "Hindi"]
    assert all(ch not in (resume.summary or "") for ch in _LIGATURES)
    assert "\u200b" not in (resume.summary or "")


def test_golden_resume_b_fresher_hr():
    resume = _parse("fresher_hr_resume.pdf")
    assert resume.summary
    assert resume.personal.linkedin == "https://www.linkedin.com/in/aditianand99/"
    assert resume.personal.location is None
    assert len(resume.experience) == 0
    assert len(resume.education) == 2
    mba, ba = resume.education
    assert mba.degree == "Master of Business Administration"
    assert mba.fieldOfStudy == "Human Resource Management"
    assert ba.degree == "Bachelor of Arts (General)"
    assert len(resume.skills) == 16


def test_golden_resume_c_swe_experienced():
    resume = _parse("swe_experienced_resume.pdf")
    assert resume.personal.location == "Noida, U.P, India"
    assert resume.summary
    assert len(resume.experience) == 2
    assert resume.experience[0].designation == "SDE 2"
    assert resume.experience[0].current is True
    assert resume.experience[0].endDate is None
    assert resume.experience[1].designation == "SDE Intern"
    assert len(resume.education) == 2
    assert len(resume.projects) == 3
    assert resume.projects[0].current is True
    assert resume.projects[0].endDate is None
    assert len(resume.skills) == 39
    desc = resume.projects[0].description or ""
    assert "synchronous and" in desc
    assert "synchronousand" not in desc
    assert "\u200b" not in desc
    assert all(ch not in (resume.summary or "") for ch in _LIGATURES)
