from app.extractors.experience import ExperienceExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_experience_extractor_single_job():
    blocks = [
        _make_block("Jan 2022 - Present"),
        _make_block("Company A, Inc."),
        _make_block("Senior Software Engineer"),
        _make_block("San Francisco, CA"),
        _make_block("Built REST APIs using Python and FastAPI."),
    ]

    extractor = ExperienceExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["company"] == "Company A, Inc."
    assert entry["designation"] == "Senior Software Engineer"
    assert entry["location"] == "San Francisco, CA"
    assert entry["startDate"] == "2022-01"
    assert entry["current"] is True
    assert "Built REST APIs" in entry["description"]
    assert set(entry["skills"]) == {"Python", "FastAPI", "REST"}


def test_experience_extractor_multiple_jobs():
    blocks = [
        _make_block("Jan 2020 - Dec 2021"),
        _make_block("Company A, Inc."),
        _make_block("Software Engineer"),
        _make_block("Built backend services."),
        _make_block("Jan 2022 - Present"),
        _make_block("Company B LLC"),
        _make_block("Senior Software Engineer"),
        _make_block("Building APIs."),
    ]

    extractor = ExperienceExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 2
    assert entries[0]["company"] == "Company A, Inc."
    assert entries[1]["company"] == "Company B LLC"


def test_experience_extractor_missing_location():
    blocks = [
        _make_block("Jan 2020 - Dec 2021"),
        _make_block("Company A"),
        _make_block("Software Engineer"),
        _make_block("Built backend services."),
    ]

    extractor = ExperienceExtractor()
    entries = extractor.extract(blocks)

    assert entries[0]["location"] is None


def test_experience_extractor_description_spanning_multiple_lines():
    blocks = [
        _make_block("Jan 2020 - Dec 2021"),
        _make_block("Company A Inc"),
        _make_block("Software Engineer"),
        _make_block("Built backend services."),
        _make_block("Led a team of 3 engineers."),
    ]

    extractor = ExperienceExtractor()
    entries = extractor.extract(blocks)

    assert "Built backend services." in entries[0]["description"]
    assert "Led a team of 3 engineers." in entries[0]["description"]
