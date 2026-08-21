from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_skills_extractor_normalizes_aliases_and_dedupes():
    extractor = SkillsExtractor()
    blocks = [
        _make_block("Python, FastAPI, postgres"),
        _make_block("Experience with PostgreSQL and Python."),
    ]

    skills = extractor.extract(blocks, section_name="SKILLS")
    skill_values = {item["value"] for item in skills}

    assert skill_values == {"Python", "FastAPI", "PostgreSQL"}
    assert all(item["confidence"] == 0.90 for item in skills)
    assert all(item["source"] == "skills_section" for item in skills)


def test_skills_extractor_finds_skills_outside_skills_section():
    extractor = SkillsExtractor()
    blocks = [
        _make_block("Built REST APIs using Python and Django."),
    ]

    skills = extractor.extract(blocks, section_name="EXPERIENCE")
    assert {item["value"] for item in skills} == {"Python", "Django", "REST"}
    assert all(item["source"] == "full_text" for item in skills)


def test_skills_extractor_handles_none_block_text():
    extractor = SkillsExtractor()
    # Construct a TextBlock with None text to simulate upstream None propagation
    blocks = [
        TextBlock(text=None, page_number=1, x0=0, y0=0, x1=0, y1=0),
    ]

    skills = extractor.extract(blocks, section_name="EXPERIENCE")
    # Should not raise and should return an empty list
    assert isinstance(skills, list)
    assert skills == []
