"""Phase 6P: parenthetical DEGREE (FIELD) → fieldOfStudy."""

from __future__ import annotations

from pathlib import Path

from app.extractors.education import EducationExtractor
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.block_classification import ClassifiedBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup
from app.pipeline.stages.text_extraction import TextBlock


def _blk(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=100, y1=10)


def _cb(text: str, label: str) -> ClassifiedBlock:
    return ClassifiedBlock(original=_blk(text), label=label, score=1.0, reasons=[])


def _group(*blocks: ClassifiedBlock) -> CandidateGroup:
    return CandidateGroup(
        section="EDUCATION",
        blocks=list(blocks),
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=max(len(blocks) - 1, 0),
        summary_text=" ".join(b.original.text for b in blocks),
    )


def test_parenthetical_fields_extracted():
    ext = EducationExtractor()
    # Lines recognized as degree lines go through extract() + normalization.
    cases = {
        "MBA (Human Resource Management)": (
            "Master of Business Administration",
            "Human Resource Management",
        ),
        "B.Tech (Computer Science)": ("Bachelor of Technology", "Computer Science"),
        "B.Sc (Information Technology)": ("Bachelor of Science", "Information Technology"),
    }
    for raw, (degree, field) in cases.items():
        entries = ext.extract([raw])
        assert len(entries) == 1, raw
        assert entries[0]["degree"] == degree, raw
        assert entries[0]["fieldOfStudy"] == field, raw
        assert entries[0]["institution"] is None, raw

    # MCA may not be in the degree alias map; still peel field before normalize.
    deg, inst, field = ext._extract_degree_line("MCA (Computer Applications)")
    assert deg == "MCA"
    assert field == "Computer Applications"
    assert inst is None
    assert ext._normalize_degree(deg) in (None, "MCA") or "Computer" not in (
        ext._normalize_degree(deg) or ""
    )


def test_degree_in_field_still_works():
    entries = EducationExtractor().extract(["MBA in Human Resource Management"])
    assert entries[0]["degree"] == "Master of Business Administration"
    assert entries[0]["fieldOfStudy"] == "Human Resource Management"


def test_pipe_institution_still_works():
    ext = EducationExtractor()
    for raw in ("MBA | School Of Open Learning", "MBA | University Name"):
        entries = ext.extract([raw])
        assert entries[0]["degree"] == "Master of Business Administration"
        assert entries[0]["fieldOfStudy"] is None
        assert "School" in (entries[0]["institution"] or "") or "University" in (
            entries[0]["institution"] or ""
        )


def test_parenthetical_non_fields_rejected():
    ext = EducationExtractor()
    rejected = [
        "MBA (Hons.)",
        "MBA (Honours)",
        "B.Sc. (Hons.)",
        "MBA (2024-2026)",
        "MBA (2024 – 2026)",
        "MBA (University Name)",
        "MBA (ABC University)",
        "Bachelor of Arts (General)",
    ]
    for raw in rejected:
        deg, inst, field = ext._extract_degree_line(raw)
        assert field is None, raw
        # Parenthetical must not become a fake institution.
        assert inst is None or not str(inst).startswith("("), raw


def test_group_path_peels_field_before_normalize():
    ext = EducationExtractor()
    group = _group(
        _cb("MBA (Human Resource Management)", "UNKNOWN"),
        _cb("School of Open Learning, Delhi University", "INSTITUTION"),
        _cb("Date: 2024 – 2026", "DATE"),
    )
    entries = ext.extract(groups=[group])
    assert len(entries) == 1
    assert entries[0]["degree"] == "Master of Business Administration"
    assert entries[0]["fieldOfStudy"] == "Human Resource Management"
    assert entries[0]["institution"] == "School of Open Learning, Delhi University"


def test_pipe_line_with_parenthetical_field_and_institution():
    ext = EducationExtractor()
    entries = ext.extract(
        ["MBA (Human Resource Management) | School Of Open Learning, Delhi University (DU-SOL)"]
    )
    assert entries[0]["degree"] == "Master of Business Administration"
    assert entries[0]["fieldOfStudy"] == "Human Resource Management"
    assert "School Of Open Learning" in entries[0]["institution"]


def test_resume_a_field_of_study():
    path = Path("tests/fixtures/AditCV_SOL.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert len(resume.education) == 1
    edu = resume.education[0]
    assert edu.degree == "Master of Business Administration"
    assert edu.fieldOfStudy == "Human Resource Management"
    assert "School of Open Learning" in (edu.institution or "")


def test_resume_b_mba_field_and_ba_unchanged():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert len(resume.education) == 2
    mba, ba = resume.education
    assert mba.degree == "Master of Business Administration"
    assert mba.fieldOfStudy == "Human Resource Management"
    assert "School Of Open Learning" in (mba.institution or "")
    assert ba.degree == "Bachelor of Arts (General)"
    assert ba.fieldOfStudy is None
    assert ba.institution == "BIR Tikendrajit University"


def test_resume_c_no_accidental_fields():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert len(resume.education) == 2
    assert all(e.fieldOfStudy is None for e in resume.education)
    assert len(resume.experience) == 2
    assert len(resume.projects) == 3
    assert len(resume.skills) == 39
