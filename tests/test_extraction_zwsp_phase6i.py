"""Phase 6I: U+200B (ZWSP) cleanup at PDF extraction boundary."""

from __future__ import annotations

from pathlib import Path

from app.domain.document import document_from_text_blocks
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor


def test_a_trailing_zwsp_removed_from_location_like_text():
    assert PDFExtractor.clean_extracted_text("Noida\u200b") == "Noida"


def test_b_trailing_zwsp_removed_from_institution_like_text():
    assert PDFExtractor.clean_extracted_text("NIT Calicut\u200b") == "NIT Calicut"


def test_c_zwsp_only_becomes_empty_and_is_not_kept_as_fragment():
    assert PDFExtractor.clean_extracted_text("\u200b") == ""
    assert PDFExtractor.keep_extracted_fragment("\u200b") is False


def test_d_internal_zwsp_joins_without_inserted_space():
    assert PDFExtractor.clean_extracted_text("A\u200bB") == "AB"


def test_e_ordinary_spaces_unchanged():
    assert PDFExtractor.clean_extracted_text("Noida, U.P, India") == "Noida, U.P, India"
    assert PDFExtractor.clean_extracted_text("Hello World") == "Hello World"


def test_f_multiple_zwsp_characters_removed():
    assert PDFExtractor.clean_extracted_text("Foo\u200b\u200bBar\u200b") == "FooBar"


def test_g_ordinary_unicode_unaffected():
    assert PDFExtractor.clean_extracted_text("café – résumé") == "café – résumé"
    assert PDFExtractor.clean_extracted_text("日本語") == "日本語"


def test_h_resume_c_extraction_and_downstream_lines_have_no_zwsp():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    blocks = PDFExtractor.extract(path.read_bytes())
    assert all("\u200b" not in block.text for block in blocks)
    assert all(block.text.strip() for block in blocks)

    document = reconstruct_document(document_from_text_blocks(blocks))
    for page in document.pages:
        for region in page.regions:
            for line in region.lines:
                assert "\u200b" not in (line.text or "")
                assert (line.text or "").strip()
