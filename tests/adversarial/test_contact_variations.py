import pytest

from app.extractors.contact import ContactExtractor
from app.pipeline.stages.text_extraction import TextBlock


def block(text):
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=300, y1=10)


@pytest.mark.parametrize("name,expected", [
    ("JOHN", "JOHN"), ("JOHN SMITH", "JOHN SMITH"),
    ("J O H N", "JOHN"), ("J O H N   S M I T H", "JOHN SMITH"),
    ("María José García", "María José García"), ("Jean-Luc Picard", "Jean-Luc Picard"),
    ("Mary-Jane Watson", "Mary-Jane Watson"), ("O'Connor", "O'Connor"),
    ("Anne-Marie O'Connor", "Anne-Marie O'Connor"),
])
def test_name_variations(name, expected):
    assert ContactExtractor.extract([block(name)])["name"]["value"] == expected


@pytest.mark.parametrize("phone", ["+1 970 333 3833", "9876543210", "(970) 333-3833", "970.333.3833", "895 555\xa0555"])
def test_phone_variations(phone):
    assert ContactExtractor.extract([block(phone)])["phone"]["value"] is not None


@pytest.mark.parametrize("url", ["https://linkedin.com/in/name", "linkedin.com/in/name", "www.linkedin.com/in/name"])
def test_linkedin_variations(url):
    assert ContactExtractor.extract([block(url)])["linkedin"]["value"] is not None


@pytest.mark.parametrize("candidate", ["CAREER OBJECTIVE", "ADMINISTRATIVE", "EXPERIENCE", "SKILLS", "University", "Location", "2018", "12345", "john@example.com"])
def test_structural_and_non_name_candidates_are_rejected(candidate):
    assert ContactExtractor._looks_like_name(candidate) is False
