from tests.conftest import require_fixture
from pathlib import Path

from app.pipeline.stages.pdf_detection import PDFDetector
from app.pipeline.stages.text_extraction import PDFExtractor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> bytes:
    return require_fixture(name).read_bytes()


def test_pdf_detection_single_column_has_text():
    data = load_fixture("single-column.pdf")
    result = PDFDetector.detect(data)

    assert result.is_pdf is True
    assert result.has_text is True
    assert result.page_count == 1


def test_pdf_detection_image_only_no_text():
    data = load_fixture("image-only.pdf")
    result = PDFDetector.detect(data)

    assert result.is_pdf is True
    assert result.has_text is False
    assert result.page_count == 1


def test_pdf_extractor_single_column_extracts_blocks():
    data = load_fixture("single-column.pdf")
    blocks = PDFExtractor.extract(data)

    assert len(blocks) > 0
    assert any("John Doe" in block.text for block in blocks)
    assert any(block.page_number == 1 for block in blocks)
    assert all(block.text.strip() for block in blocks)


def test_pdf_extractor_two_column_extracts_blocks():
    data = load_fixture("two-column.pdf")
    blocks = PDFExtractor.extract(data)

    assert len(blocks) > 1
    assert any("Profile" in block.text for block in blocks)
    assert any("Skills" in block.text for block in blocks)
    assert any(block.font_size is not None for block in blocks)


def test_pdf_extractor_fresher_hr_resume_line_grouping():
    data = load_fixture("fresher_hr_resume.pdf")
    blocks = PDFExtractor.extract(data)

    assert any("MBA (Human Resource Management)" in block.text for block in blocks)
    assert any("Bachelor of Arts (General)" in block.text for block in blocks)
