import fitz

from app.pipeline.parser import PipelineError, ResumeParser


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
