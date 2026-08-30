"""Phase 6T: NFKC ligature normalization at PDF extraction cleanup."""

from __future__ import annotations

from pathlib import Path

from app.pipeline.parser import ResumeParser
from app.pipeline.stages.text_extraction import PDFExtractor


_LIGATURES = {
    "\uFB00": "ff",
    "\uFB01": "fi",
    "\uFB02": "fl",
    "\uFB03": "ffi",
    "\uFB04": "ffl",
}


def test_compatibility_ligatures_expand_via_nfkc():
    for lig, ascii_form in _LIGATURES.items():
        assert PDFExtractor.clean_extracted_text(lig) == ascii_form


def test_ligature_words_normalized():
    assert PDFExtractor.clean_extracted_text("con\uFB02ict") == "conflict"
    assert PDFExtractor.clean_extracted_text("e\uFB00ectively") == "effectively"
    assert PDFExtractor.clean_extracted_text("o\uFB03ce") == "office"
    assert PDFExtractor.clean_extracted_text("\uFB01le") == "file"


def test_ordinary_text_and_symbols_unchanged():
    assert PDFExtractor.clean_extracted_text("Noida, U.P, India") == "Noida, U.P, India"
    assert PDFExtractor.clean_extracted_text("Hello World") == "Hello World"
    assert PDFExtractor.clean_extracted_text("café – résumé") == "café – résumé"
    assert PDFExtractor.clean_extracted_text("2025-02") == "2025-02"
    assert PDFExtractor.clean_extracted_text("2022 – Present") == "2022 – Present"
    assert PDFExtractor.clean_extracted_text("2024 – 2026") == "2024 – 2026"


def test_urls_and_emails_not_corrupted():
    url = "https://www.linkedin.com/in/example"
    email = "user@example.com"
    assert PDFExtractor.clean_extracted_text(url) == url
    assert PDFExtractor.clean_extracted_text(email) == email


def test_zwsp_still_removed_without_inserted_space():
    assert PDFExtractor.clean_extracted_text("India\u200b") == "India"
    # Trailing space before ZWSP is preserved by cleanup; extract() strips fragments.
    assert PDFExtractor.clean_extracted_text("India \u200b").strip() == "India"
    assert PDFExtractor.clean_extracted_text("\u200b") == ""
    assert PDFExtractor.keep_extracted_fragment("\u200b") is False
    assert PDFExtractor.clean_extracted_text("A\u200bB") == "AB"


def test_ligature_plus_zwsp_both_cleaned():
    assert PDFExtractor.clean_extracted_text("con\uFB02ict\u200b") == "conflict"
    assert PDFExtractor.clean_extracted_text("\u200be\uFB00ectively") == "effectively"


def test_resume_a_summary_has_no_ligatures():
    path = Path("tests/fixtures/AditCV_SOL.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    summary = resume.summary or ""
    assert "conflict" in summary
    assert "effectively" in summary
    for lig in _LIGATURES:
        assert lig not in summary
    assert resume.personal.linkedin == "https://www.linkedin.com/in/aditianand99"
    assert resume.languages == ["English", "Hindi"]
    assert resume.education[0].fieldOfStudy == "Human Resource Management"
    assert len(resume.skills) == 19
    assert len(resume.experience) == 1
    assert len(resume.education) == 1


def test_resume_b_c_no_regression():
    b_path = Path("tests/fixtures/fresher_hr_resume.pdf")
    c_path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not b_path.exists() or not c_path.exists():
        return
    parser = ResumeParser()
    b = parser.parse_with_layout_pipeline(b_path.read_bytes())
    c = parser.parse_with_layout_pipeline(c_path.read_bytes())

    assert b.summary
    assert len(b.education) == 2
    assert len(b.skills) == 16
    assert len(b.experience) == 0
    assert b.personal.linkedin == "https://www.linkedin.com/in/aditianand99/"

    assert c.summary
    assert len(c.experience) == 2
    assert len(c.education) == 2
    assert len(c.projects) == 3
    assert len(c.skills) == 39
    assert c.personal.location == "Noida, U.P, India"
    assert c.projects[0].endDate is None
    assert c.projects[0].current is True
    assert "synchronous and" in (c.projects[0].description or "")
