"""Phase 6L: LANGUAGES numeric rating cleanup at assembly."""

from __future__ import annotations

from pathlib import Path

from app.extractors.languages import clean_language_texts, clean_language_value
from app.pipeline.parser import ResumeParser


def test_numeric_ratings_filtered_from_language_list():
    assert clean_language_texts(["4", "English", "5", "Hindi"]) == ["English", "Hindi"]


def test_plain_languages_unchanged():
    assert clean_language_texts(["English", "Hindi"]) == ["English", "Hindi"]


def test_lone_numeric_ratings_become_empty():
    assert clean_language_texts(["4"]) == []
    assert clean_language_texts(["5"]) == []
    assert clean_language_texts(["10"]) == []


def test_same_line_trailing_rating_stripped():
    assert clean_language_value("English 4") == "English"
    assert clean_language_value("English - 4") == "English"


def test_digit_containing_token_not_dropped_merely_for_digits():
    assert clean_language_value("Python3") == "Python3"
    assert clean_language_texts(["Python3", "English"]) == ["Python3", "English"]


def test_resume_a_languages_are_english_hindi_only():
    path = Path("tests/fixtures/AditCV_SOL.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.languages == ["English", "Hindi"]


def test_resume_b_languages_remain_empty():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.languages == []


def test_resume_c_languages_remain_empty():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.languages == []
