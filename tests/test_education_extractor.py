from tests.conftest import require_fixture
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


def test_extracts_resume_7_trailing_year_from_degree():
    from app.pipeline.parser import ResumeParser

    resume = ResumeParser().parse(require_fixture("resume_7.pdf").read_bytes())
    education = resume.education[0]

    assert education.degree == "Bachelor of Science: Computer Information Systems"
    assert education.institution == "Columbia University , NY"
    assert education.startDate == "2014"
    assert education.endDate is None


def test_extracts_parenthesized_date_from_resume_2_layout_path():
    from app.pipeline.parser import ResumeParser

    resume = ResumeParser().parse_with_layout_pipeline(require_fixture("resume_2.pdf").read_bytes())
    education = resume.education[0]

    assert education.degree == "Bachelor Of Arts in History,"
    assert education.startDate == "2015-05"
    assert education.institution == "RIVER BROOK UNIVERSITY"


def test_extracts_honors_grade_from_grouped_education_entry():
    from types import SimpleNamespace
    from app.pipeline.stages.block_classification import ClassifiedBlock
    from app.pipeline.stages.candidate_grouping import CandidateGroup

    group = CandidateGroup(
        section="EDUCATION",
        blocks=[
            ClassifiedBlock(original=SimpleNamespace(text="Bachelor of Arts in History"), label="DEGREE", score=1.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="Graduated magna cum laude"), label="UNKNOWN", score=0.0, reasons=[]),
        ],
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=1,
        summary_text="",
    )

    entries = EducationExtractor().extract(groups=[group])

    assert entries[0]["grade"] == "Graduated magna cum laude"


def test_extracts_from_candidate_groups():
    # Simulate two CandidateGroups produced by grouping for the resume
    from types import SimpleNamespace
    from app.pipeline.stages.block_classification import ClassifiedBlock
    from app.pipeline.stages.candidate_grouping import CandidateGroup

    # GROUP 1
    g1_blocks = [
        ClassifiedBlock(original=SimpleNamespace(text="M.C.A NIT Calicut"), label="UNKNOWN", score=1.0, reasons=[]),
        ClassifiedBlock(original=SimpleNamespace(text="Date : 07/2019 - 06/2022"), label="DATE", score=1.0, reasons=[]),
        ClassifiedBlock(original=SimpleNamespace(text="Location: Kerala, India"), label="LOCATION", score=1.0, reasons=[]),
    ]
    group1 = CandidateGroup(section="EDUCATION", blocks=g1_blocks, page_number=1, column_id=0, start_index=0, end_index=2, summary_text="")

    # GROUP 2
    g2_blocks = [
        ClassifiedBlock(original=SimpleNamespace(text="B.SC-IT Magadh University"), label="DEGREE", score=1.0, reasons=[]),
        ClassifiedBlock(original=SimpleNamespace(text="Date : 07/2015 - 08/2018"), label="DATE", score=1.0, reasons=[]),
        ClassifiedBlock(original=SimpleNamespace(text="Location: Patna, India"), label="LOCATION", score=1.0, reasons=[]),
    ]
    group2 = CandidateGroup(section="EDUCATION", blocks=g2_blocks, page_number=1, column_id=0, start_index=3, end_index=5, summary_text="")

    extractor = EducationExtractor()
    entries = extractor.extract(groups=[group1, group2])

    assert len(entries) == 2
    assert entries[0]["institution"] in ("M.C.A NIT Calicut", "NIT Calicut", None)
    assert entries[0]["startDate"] == "2019-07"
    assert entries[1]["institution"] in ("B.SC-IT Magadh University", "Magadh University", None)
    assert entries[1]["startDate"] == "2015-07"


def test_education_institution_fallback_for_acronym_and_location():
    from types import SimpleNamespace
    from app.pipeline.stages.block_classification import ClassifiedBlock
    from app.pipeline.stages.candidate_grouping import CandidateGroup

    # 1. VIT, Chennai inside candidate group
    g1 = CandidateGroup(
        section="EDUCATION",
        blocks=[
            ClassifiedBlock(original=SimpleNamespace(text="B.Tech in Computer Science and Engineering"), label="DEGREE", score=1.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="VIT, Chennai"), label="UNKNOWN", score=0.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="July 2018 – July 2022"), label="DATE", score=1.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="CGPA:8.43"), label="UNKNOWN", score=0.0, reasons=[]),
        ],
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=3,
        summary_text="",
    )

    extractor = EducationExtractor()
    entries = extractor.extract(groups=[g1])
    assert len(entries) == 1
    assert entries[0]["degree"] == "Bachelor of Technology"
    assert entries[0]["fieldOfStudy"] == "Computer Science and Engineering"
    assert entries[0]["institution"] == "VIT, Chennai"
    assert entries[0]["grade"] == "CGPA:8.43"


def test_education_institution_fallback_for_generic_synthetic_case():
    from types import SimpleNamespace
    from app.pipeline.stages.block_classification import ClassifiedBlock
    from app.pipeline.stages.candidate_grouping import CandidateGroup

    # 2. BITS, Pilani inside candidate group
    g2 = CandidateGroup(
        section="EDUCATION",
        blocks=[
            ClassifiedBlock(original=SimpleNamespace(text="Master of Science in Mathematics"), label="DEGREE", score=1.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="BITS, Pilani"), label="UNKNOWN", score=0.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="2016 – 2020"), label="DATE", score=1.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="GPA: 9.1/10"), label="UNKNOWN", score=0.0, reasons=[]),
        ],
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=3,
        summary_text="",
    )

    extractor = EducationExtractor()
    entries = extractor.extract(groups=[g2])
    assert len(entries) == 1
    assert entries[0]["degree"] == "Master of Science"
    assert entries[0]["fieldOfStudy"] == "Mathematics"
    assert entries[0]["institution"] == "BITS, Pilani"
    assert entries[0]["grade"] == "GPA: 9.1/10"


def test_education_institution_fallback_does_not_capture_date_or_grade_as_institution():
    from types import SimpleNamespace
    from app.pipeline.stages.block_classification import ClassifiedBlock
    from app.pipeline.stages.candidate_grouping import CandidateGroup

    # Group with only degree, date, grade — institution must remain None
    g3 = CandidateGroup(
        section="EDUCATION",
        blocks=[
            ClassifiedBlock(original=SimpleNamespace(text="Bachelor of Science in Physics"), label="DEGREE", score=1.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="2015 – 2019"), label="DATE", score=1.0, reasons=[]),
            ClassifiedBlock(original=SimpleNamespace(text="Graduated with First Class"), label="UNKNOWN", score=0.0, reasons=[]),
        ],
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=2,
        summary_text="",
    )

    extractor = EducationExtractor()
    entries = extractor.extract(groups=[g3])
    assert len(entries) == 1
    assert entries[0]["institution"] is None
    assert entries[0]["grade"] == "Graduated with First Class"


def test_shubham_production_education_institution_extracted():
    import glob
    from app.pipeline.parser import ResumeParser

    matches = sorted(glob.glob("tests/fixtures/*Shubham*.pdf"))
    if not matches:
        return
    raw = Path(matches[0]).read_bytes()
    resume = ResumeParser().parse_with_layout_pipeline(raw)

    assert len(resume.education) == 1
    edu = resume.education[0]
    assert edu.degree == "Bachelor of Technology"
    assert edu.fieldOfStudy == "Computer Science and Engineering"
    assert edu.institution == "VIT, Chennai"
    assert edu.grade == "CGPA:8.43"
