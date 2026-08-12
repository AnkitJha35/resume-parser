from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_section_detection_single_column_resume():
    blocks = [
        _make_block("Profile"),
        _make_block("Experienced engineer"),
        _make_block("Experience"),
        _make_block("Company A"),
        _make_block("Education"),
        _make_block("University X"),
    ]

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert sections["SUMMARY"] == [blocks[1]]
    assert sections["EXPERIENCE"] == [blocks[3]]
    assert sections["EDUCATION"] == [blocks[5]]


def test_section_detection_two_column_resume():
    blocks = [
        _make_block("Professional Experience"),
        _make_block("Company A"),
        _make_block("Skills"),
        _make_block("Python"),
        _make_block("Projects"),
        _make_block("Resume Parser"),
    ]

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert sections["EXPERIENCE"] == [blocks[1]]
    assert sections["SKILLS"] == [blocks[3]]
    assert sections["PROJECTS"] == [blocks[5]]


def test_section_detection_nonstandard_header_aliases():
    blocks = [
        _make_block("Career Summary"),
        _make_block("Skilled at APIs"),
        _make_block("Work Experience"),
        _make_block("Company B"),
    ]

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert sections["SUMMARY"] == [blocks[1]]
    assert sections["EXPERIENCE"] == [blocks[3]]


def test_section_detection_missing_sections():
    blocks = [
        _make_block("Skills"),
        _make_block("Python"),
        _make_block("Certifications"),
        _make_block("AWS Certified"),
    ]

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert sections["SKILLS"] == [blocks[1]]
    assert sections["CERTIFICATIONS"] == [blocks[3]]
    assert sections["EXPERIENCE"] == []
    assert sections["EDUCATION"] == []


def test_section_detection_fresher_hr_resume_fixture():
    from pathlib import Path
    from app.pipeline.stages.text_extraction import PDFExtractor

    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    raw_pdf = path.read_bytes()
    blocks = PDFExtractor.extract(raw_pdf)

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert any("MBA (Human Resource Management)" in block.text for block in sections["EDUCATION"])
    assert any("Bachelor of Arts (General)" in block.text for block in sections["EDUCATION"])
    assert not any(
        "An ambitious MBA student specializing in Human Resource Management" in block.text
        for block in sections["EDUCATION"]
    )
    assert sections["EXPERIENCE"] == []
