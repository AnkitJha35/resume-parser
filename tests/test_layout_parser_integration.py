from tests.conftest import require_fixture
from pathlib import Path

from app.domain.document import document_from_text_blocks
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.block_classification import classify_block
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.semantic_compat import semantic_sections_to_text_blocks
from app.pipeline.stages.semantic_paths import detect_region_aware_sections
from app.pipeline.stages.text_extraction import PDFExtractor


def test_opt_in_layout_parser_preserves_existing_fixture_contracts():
    parser = ResumeParser()

    resume_1 = parser.parse_with_layout_pipeline(require_fixture("resume_1.pdf").read_bytes())
    assert [(item.designation, item.company, item.location, item.startDate, item.endDate, item.current) for item in resume_1.experience] == [
        ("Administrative Assistant", "Redford & Sons", "Boston, MA", "2018-09", None, True),
        ("Secretary", "Bright Spot LTD", "Boston, MA", "2015-06", "2018-08", False),
    ]
    assert resume_1.experience[0].designation == "Administrative Assistant"

    resume_7 = parser.parse_with_layout_pipeline(require_fixture("resume_7.pdf").read_bytes())
    assert (resume_7.experience[0].company, resume_7.experience[0].designation, resume_7.experience[0].location) == ("Luna Web Design", "Web Developer", "New York")
    assert (resume_7.experience[0].startDate, resume_7.experience[0].endDate) == ("2015-09", "2019-05")
    assert resume_7.experience[0].startDate == "2015-09"

    fresher = parser.parse_with_layout_pipeline(Path("tests/fixtures/fresher_hr_resume.pdf").read_bytes())
    assert [(item.degree, item.institution, item.startDate, item.endDate) for item in fresher.education] == [
        ("Master of Business Administration", "School Of Open Learning, Delhi University (DU-SOL)", "2024", "2026"),
        ("Bachelor of Arts (General)", "BIR Tikendrajit University", "2020", "2023"),
    ]


def test_resume_1_layout_parser_extracts_education_after_truncated_heading():
    resume = ResumeParser().parse_with_layout_pipeline(require_fixture("resume_1.pdf").read_bytes())

    assert resume.education


def test_resume_2_layout_parser_exposes_correct_semantic_inputs():
    path = require_fixture("resume_2.pdf")
    document = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes()))))
    semantic = detect_region_aware_sections(document)
    sections = semantic_sections_to_text_blocks(semantic)

    assert any("Administrative Assistant with 6+ years" in block.text for block in sections["SUMMARY"])
    assert any("September 2019" in block.text for block in sections["EXPERIENCE"])
    assert any("June 2015" in block.text and block.page_number == 2 for block in sections["EXPERIENCE"])
    assert any("Bachelor Of Arts" in block.text for block in sections["EDUCATION"])
    assert any("Microsoft Office" in block.text for block in sections["SKILLS"])
    assert any("AWARD TITLE / Brand" in block.text for block in sections["ACHIEVEMENTS"])
    assert not any("SUNTRUST FINANCIAL" in block.text for block in sections["ACHIEVEMENTS"])
    assert any(block.text == "david.perez@gmail.com" for block in sections["UNASSIGNED"])
    assert any(block.text == "linkedin.com/in/davidperez" for block in sections["UNASSIGNED"])

    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.personal.name == "DAVID PÉREZ"
    assert resume.personal.email == "david.perez@gmail.com"
    assert resume.personal.linkedin == "linkedin.com/in/davidperez"

    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.personal.name == "DAVID PÉREZ"
    assert resume.personal.email == "david.perez@gmail.com"
    assert resume.personal.linkedin == "linkedin.com/in/davidperez"