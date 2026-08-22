from pathlib import Path
import fitz

from app.pipeline.parser import ResumeParser


def _create_pdf_bytes(lines: list[str]) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 16
    return doc.tobytes()


def test_header_and_summary_extraction_from_real_fixture():
    p = Path('tests/fixtures/swe_experienced_resume.pdf')
    pdf_bytes = p.read_bytes()

    parser = ResumeParser()
    resume = parser.parse(pdf_bytes)

    # personal checks
    assert resume.personal.name == 'ANKIT JHA'
    assert resume.personal.email == 'ankitjha6035@gmail.com'
    assert resume.personal.phone == '9570716035'
    assert 'Noida' in resume.personal.location
    assert resume.personal.linkedin == 'https://linkedin.com/in/ankitjhajavafullstackdeveloper'

    # summary checks
    assert resume.summary is not None
    assert resume.summary.startswith('I am a results-driven Software Development Engineer II')
    assert resume.summary.endswith('REST APIs, Kafka, and cloud deployment.' )
