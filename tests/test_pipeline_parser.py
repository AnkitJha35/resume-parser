from tests.conftest import require_fixture
from pathlib import Path

import fitz

from app.pipeline.parser import PipelineError, ResumeParser
from app.pipeline.stages.block_classification import classify_block
from app.pipeline.stages.normalization import TextNormalizer
from app.pipeline.stages.reading_order import ReadingOrder
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.text_extraction import PDFExtractor


def _create_pdf_bytes(lines: list[str]) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 16
    return doc.tobytes()


def test_resume_parser_parses_simple_resume_pdf():
    lines = [
        "John Doe",
        "john.doe@example.com",
        "+1 555-555-5555",
        "Skills",
        "Python",
        "FastAPI",
        "Experience",
        "Jan 2022 - Present",
        "Company A, Inc.",
        "Software Engineer",
        "Built REST APIs using Python and FastAPI.",
        "Education",
        "Bachelor of Technology in Computer Science",
        "National Institute of Technology",
        "2020 - 2024",
        "Projects",
        "Resume Parser",
        "Built a parser using Python.",
        "https://example.com/resume-parser",
        "Certifications",
        "AWS Certified Solutions Architect",
        "Amazon",
        "Issued: Jun 2022",
        "Credential ID: ABCD-1234",
        "https://www.credly.com/badges/12345",
    ]
    pdf_bytes = _create_pdf_bytes(lines)

    parser = ResumeParser()
    resume = parser.parse(pdf_bytes)

    assert resume.personal.email == "john.doe@example.com"
    assert "Python" in resume.skills
    assert resume.experience[0].company == "Company A, Inc."
    assert resume.education[0].degree == "Bachelor of Technology"
    assert resume.projects[0].name == "Resume Parser"
    assert resume.certifications[0].credentialId == "ABCD-1234"


def test_resume_parser_rejects_invalid_pdf_bytes():
    parser = ResumeParser()
    try:
        parser.parse(b"not a pdf")
        assert False, "Expected PipelineError for invalid PDF"
    except PipelineError as err:
        assert err.code == "INVALID_PDF"


def test_real_fixture_parser_contracts():
    parser = ResumeParser()

    resume_1 = parser.parse(require_fixture("resume_1.pdf").read_bytes())
    assert [(entry.designation, entry.company, entry.location, entry.startDate, entry.endDate, entry.current) for entry in resume_1.experience] == [
        ("Administrative Assistant", "Redford & Sons", "Boston, MA", "2018-09", None, True),
        ("Secretary", "Bright Spot LTD", "Boston, MA", "2015-06", "2018-08", False),
    ]
    assert resume_1.certifications[0].name == "CERTIFICATION #1"
    assert resume_1.certifications[0].issuingOrganization == "University, Location"

    resume_7 = parser.parse(require_fixture("resume_7.pdf").read_bytes())
    assert (resume_7.experience[0].company, resume_7.experience[0].designation, resume_7.experience[0].location) == (
        "Luna Web Design", "Web Developer", "New York"
    )
    assert (resume_7.experience[0].startDate, resume_7.experience[0].endDate, resume_7.experience[0].current) == (
        "2015-09", "2019-05", False
    )
    assert (resume_7.education[0].degree, resume_7.education[0].institution, resume_7.education[0].startDate, resume_7.education[0].endDate) == (
        "Bachelor of Science: Computer Information Systems", "Columbia University , NY", "2014", None
    )
    assert resume_7.certifications[0].name == "PHP Framework (certificate): Zend, Codeigniter, Symfony ."
    assert resume_7.certifications[0].issuingOrganization is None

    fresher_path = Path("tests/fixtures/fresher_hr_resume.pdf")
    fresher = parser.parse(fresher_path.read_bytes())
    assert [(entry.degree, entry.institution, entry.startDate, entry.endDate) for entry in fresher.education] == [
        ("Master of Business Administration", "School Of Open Learning, Delhi University (DU-SOL)", "2024", "2026"),
        ("Bachelor of Arts (General)", "BIR Tikendrajit University", "2020", "2023"),
    ]
    assert {"Collaboration", "Time Management", "Adaptability"} <= set(fresher.skills)

    normalized = TextNormalizer.normalize_blocks(ReadingOrder.reorder(PDFExtractor.extract(fresher_path.read_bytes())))
    sections = SectionDetector().detect(normalized)
    education_texts = {block.text.strip() for block in sections["EDUCATION"]}
    skill_texts = {block.text.strip() for block in sections["SKILLS"]}
    assert {"2024 – 2026", "2020 – 2023"} <= education_texts
    assert not {"2024 – 2026", "2020 – 2023"} & skill_texts
    ms_excel = next(block for block in sections["SKILLS"] if block.text == "MS Excel (VLOOKUP, Pivot Tables, Filters)")
    assert classify_block(ms_excel).label != "DEGREE"
