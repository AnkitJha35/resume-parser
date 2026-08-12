from pathlib import Path

from app.extractors.education import EducationExtractor
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.text_extraction import PDFExtractor


def test_extracts_simple_degree_entry():
    lines = [
        "Bachelor of Technology in Computer Science",
        "National Institute of Technology",
        "Jul 2019 - May 2023",
        "GPA: 8.7/10"
    ]

    extractor = EducationExtractor()
    entries = extractor.extract(lines)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["degree"] == "Bachelor of Technology"
    assert entry["institution"] == "National Institute of Technology"
    assert entry["fieldOfStudy"] == "Computer Science"
    assert entry["startDate"] == "2019-07"
    assert entry["endDate"] == "2023-05"
    assert entry["grade"] == "GPA: 8.7/10"


def test_extracts_multiple_education_entries():
    lines = [
        "Master of Business Administration",
        "Asian School of Business",
        "Aug 2023 - Present",
        "B.Tech in Electronics",
        "Indian Institute of Technology",
        "2018 - 2022",
        "Percentage: 74%"
    ]

    extractor = EducationExtractor()
    entries = extractor.extract(lines)

    assert len(entries) == 2
    assert entries[0]["degree"] == "Master of Business Administration"
    assert entries[0]["institution"] == "Asian School of Business"
    assert entries[0]["startDate"] == "2023-08"
    assert entries[0]["endDate"] == "Present"

    assert entries[1]["degree"] == "Bachelor of Technology"
    assert entries[1]["institution"] == "Indian Institute of Technology"
    assert entries[1]["startDate"] == "2018"
    assert entries[1]["endDate"] == "2022"


def test_extracts_fresher_hr_resume_education_entries():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    raw_pdf = path.read_bytes()
    blocks = PDFExtractor.extract(raw_pdf)
    sections = SectionDetector().detect(blocks)
    lines = [block.text for block in sections["EDUCATION"]]

    extractor = EducationExtractor()
    entries = extractor.extract(lines)

    assert len(entries) >= 2

    mba = entries[0]
    assert mba["degree"] == "Master of Business Administration"
    assert mba["institution"] == "School Of Open Learning, Delhi University (DU-SOL)"
    assert mba["startDate"] == "2024"
    assert mba["endDate"] == "2026"

    bachelor = entries[1]
    assert bachelor["degree"] == "Bachelor of Arts (General)"
    assert bachelor["institution"] == "BIR Tikendrajit University"
    assert bachelor["startDate"] == "2020"
    assert bachelor["endDate"] == "2023"
