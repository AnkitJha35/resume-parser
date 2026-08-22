from app.extractors.experience import ExperienceExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _b(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_experience_location_label_is_removed_after_extraction():
    blocks = [
        _b("Jan 2022 - Present"),
        _b("Company A, Inc."),
        _b("Senior Software Engineer"),
        _b("Location: Noida, UP, India"),
        _b("Built REST APIs using Python and FastAPI."),
    ]

    extractor = ExperienceExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    assert entries[0]["location"] == "Noida, UP, India"
    assert blocks[3].text == "Location: Noida, UP, India"


def test_experience_normalize_location_preserves_plain_location_text():
    extractor = ExperienceExtractor()
    assert extractor._normalize_location("Noida, UP, India") == "Noida, UP, India"
    assert extractor._normalize_location("LOCATION: Delhi, India") == "Delhi, India"
