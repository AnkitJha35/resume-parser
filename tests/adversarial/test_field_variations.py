from app.extractors.education import EducationExtractor
from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.text_extraction import TextBlock


def block(text):
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=300, y1=10)


def test_education_variants_and_honors():
    entry = EducationExtractor().extract(["Bachelor of Science in Computer Science", "Example University", "2018 - 2022", "Graduated with honors"])[0]
    assert entry["degree"] == "Bachelor of Science"
    assert entry["institution"] == "Example University"
    assert entry["startDate"] == "2018"
    assert entry["endDate"] == "2022"
    assert entry["grade"] == "Graduated with honors"


def test_skill_aliases_and_proficiency_terms_are_distinct_from_section_recovery():
    values = {item["value"] for item in SkillsExtractor().extract([
        block("WPM, 120 words per minute, Fluent Spanish, Quick Books")
    ], section_name="SKILLS")}
    assert "Typing" in values
    assert "Spanish" in values
    assert "QuickBooks" in values
