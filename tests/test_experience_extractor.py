from pathlib import Path
from types import SimpleNamespace

from app.extractors.experience import ExperienceExtractor
from app.pipeline.stages.text_extraction import TextBlock
from app.pipeline.stages.block_classification import ClassifiedBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup


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


def test_experience_extractor_handles_title_company_date_order():
    # Title -> Company -> Date ordering should be handled correctly
    blocks = [
        _make_block("SDE 2"),
        _make_block("Probus Smart Things!"),
        _make_block("Date : 07/2022 - Present"),
        _make_block("Worked on feature X."),
        _make_block("SDE Intern"),
        _make_block("Navyug Infosolutions Pvt. Ltd"),
        _make_block("Date : 02/2022 - 06/2022"),
        _make_block("Worked on internship tasks."),
    ]

    extractor = ExperienceExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 2
    first, second = entries[0], entries[1]
    assert first["designation"] == "SDE 2"
    assert first["company"] == "Probus Smart Things!"
    assert first["startDate"] == "2022-07"
    assert first["current"] is True

    assert second["designation"] == "SDE Intern"
    assert second["company"] == "Navyug Infosolutions Pvt. Ltd"
    assert second["startDate"] == "2022-02"
    assert second["endDate"] == "2022-06"


def _make_classified_block(text: str, label: str) -> ClassifiedBlock:
    """Helper to create a ClassifiedBlock directly for testing."""
    block = SimpleNamespace(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)
    return ClassifiedBlock(original=block, label=label, score=1.0, reasons=[])


def _make_candidate_group(blocks: list[ClassifiedBlock], section: str = "EXPERIENCE") -> CandidateGroup:
    """Helper to create a CandidateGroup directly for testing."""
    return CandidateGroup(
        section=section,
        blocks=blocks,
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=len(blocks) - 1,
        summary_text=""
    )


def test_experience_extractor_from_candidate_group_title_first():
    """Test A: JOB_TITLE, COMPANY, DATE, LOCATION, DESCRIPTION order."""
    classified = [
        _make_classified_block("SDE 2", "JOB_TITLE"),
        _make_classified_block("Probus Smart Things!", "COMPANY"),
        _make_classified_block("Date : 07/2022 - Present", "DATE"),
        _make_classified_block("Noida, India", "LOCATION"),
        _make_classified_block("Built backend systems.", "UNKNOWN"),
    ]
    group = _make_candidate_group(classified)

    extractor = ExperienceExtractor()
    entries = extractor.extract(groups=[group])

    assert len(entries) == 1
    entry = entries[0]
    assert entry["designation"] == "SDE 2"
    assert entry["company"] == "Probus Smart Things!"
    assert entry["startDate"] == "2022-07"
    assert entry["current"] is True
    assert entry["location"] == "Noida, India"
    assert "Built backend systems" in entry["description"]


def test_experience_extractor_from_candidate_group_date_first():
    """Test B: DATE, JOB_TITLE, COMPANY, LOCATION, DESCRIPTION order."""
    classified = [
        _make_classified_block("Date : 01/2021 - 12/2021", "DATE"),
        _make_classified_block("Senior Engineer", "JOB_TITLE"),
        _make_classified_block("TechCorp Inc", "COMPANY"),
        _make_classified_block("San Francisco, CA", "LOCATION"),
        _make_classified_block("Led API development.", "UNKNOWN"),
    ]
    group = _make_candidate_group(classified)

    extractor = ExperienceExtractor()
    entries = extractor.extract(groups=[group])

    assert len(entries) == 1
    entry = entries[0]
    assert entry["designation"] == "Senior Engineer"
    assert entry["company"] == "TechCorp Inc"
    assert entry["startDate"] == "2021-01"
    assert entry["endDate"] == "2021-12"
    assert entry["location"] == "San Francisco, CA"
    assert "Led API development" in entry["description"]


def test_experience_extractor_from_candidate_group_company_first():
    """Test C: COMPANY, JOB_TITLE, DATE, DESCRIPTION (no explicit location)."""
    classified = [
        _make_classified_block("Acme Corp Ltd", "COMPANY"),
        _make_classified_block("Software Developer", "JOB_TITLE"),
        _make_classified_block("Date : 06/2020 - 05/2022", "DATE"),
        _make_classified_block("Developed microservices.", "UNKNOWN"),
        _make_classified_block("Worked on CI/CD pipelines.", "UNKNOWN"),
    ]
    group = _make_candidate_group(classified)

    extractor = ExperienceExtractor()
    entries = extractor.extract(groups=[group])

    assert len(entries) == 1
    entry = entries[0]
    assert entry["company"] == "Acme Corp Ltd"
    assert entry["designation"] == "Software Developer"
    assert entry["startDate"] == "2020-06"
    assert entry["endDate"] == "2022-05"
    assert entry["location"] is None
    assert "Developed microservices" in entry["description"]
    assert "Worked on CI/CD pipelines" in entry["description"]


def test_experience_extractor_from_multiple_candidate_groups():
    """Test D: Two CandidateGroups should produce two entries with no bleed."""
    # Group 1
    classified1 = [
        _make_classified_block("SDE 2", "JOB_TITLE"),
        _make_classified_block("Company A Inc", "COMPANY"),
        _make_classified_block("Date : 01/2022 - Present", "DATE"),
        _make_classified_block("Built APIs.", "UNKNOWN"),
    ]
    group1 = _make_candidate_group(classified1)

    # Group 2
    classified2 = [
        _make_classified_block("SDE Intern", "JOB_TITLE"),
        _make_classified_block("Company B LLC", "COMPANY"),
        _make_classified_block("Date : 06/2021 - 12/2021", "DATE"),
        _make_classified_block("Worked on backend.", "UNKNOWN"),
    ]
    group2 = _make_candidate_group(classified2)

    extractor = ExperienceExtractor()
    entries = extractor.extract(groups=[group1, group2])

    assert len(entries) == 2

    # First entry
    assert entries[0]["designation"] == "SDE 2"
    assert entries[0]["company"] == "Company A Inc"
    assert entries[0]["startDate"] == "2022-01"
    assert entries[0]["current"] is True
    assert "Built APIs" in entries[0]["description"]
    assert "Worked on backend" not in entries[0]["description"]  # No bleed

    # Second entry
    assert entries[1]["designation"] == "SDE Intern"
    assert entries[1]["company"] == "Company B LLC"
    assert entries[1]["startDate"] == "2021-06"
    assert entries[1]["endDate"] == "2021-12"
    assert "Worked on backend" in entries[1]["description"]
    assert "Built APIs" not in entries[1]["description"]  # No bleed


def test_resume_1_experience_grouping_splits_two_jobs():
    from app.pipeline.stages.text_extraction import PDFExtractor
    from app.pipeline.stages.reading_order import ReadingOrder
    from app.pipeline.stages.normalization import TextNormalizer
    from app.pipeline.stages.sections import SectionDetector
    from app.pipeline.stages.block_classification import classify_block
    from app.pipeline.stages.candidate_grouping import group_candidates

    raw_pdf = PDFExtractor.extract(Path("tests/fixtures/resume_1.pdf").read_bytes())
    blocks = TextNormalizer.normalize_blocks(ReadingOrder.reorder(raw_pdf))
    sections = SectionDetector().detect(blocks)
    candidate_groups = group_candidates([classify_block(b) for b in sections["EXPERIENCE"]], "EXPERIENCE")

    assert len(candidate_groups) == 2
    first = candidate_groups[0]
    second = candidate_groups[1]

    first_text = "\n".join(cb.original.text for cb in first.blocks)
    second_text = "\n".join(cb.original.text for cb in second.blocks)

    assert "ADMINISTRATIVE ASSISTANT" in first_text
    assert "Redford & Sons, Boston, MA / September 2018 - Present" in first_text
    assert "SECRETARY" in second_text
    assert "Bright Spot LTD, Boston, MA / June 2015 – August 2018" in second_text

    entries = ExperienceExtractor().extract(sections["EXPERIENCE"], groups=candidate_groups)
    assert len(entries) == 2


def test_resume_1_experience_extractor_parses_actual_group_headers():
    from app.pipeline.stages.text_extraction import PDFExtractor
    from app.pipeline.stages.reading_order import ReadingOrder
    from app.pipeline.stages.normalization import TextNormalizer
    from app.pipeline.stages.sections import SectionDetector
    from app.pipeline.stages.block_classification import classify_block
    from app.pipeline.stages.candidate_grouping import group_candidates

    raw_pdf = PDFExtractor.extract(Path("tests/fixtures/resume_1.pdf").read_bytes())
    blocks = TextNormalizer.normalize_blocks(ReadingOrder.reorder(raw_pdf))
    sections = SectionDetector().detect(blocks)
    groups = group_candidates([classify_block(b) for b in sections["EXPERIENCE"]], "EXPERIENCE")

    assert len(groups) == 2

    entries = ExperienceExtractor().extract(groups=groups)
    assert len(entries) == 2

    first, second = entries

    assert first["designation"] == "Administrative Assistant"
    assert first["company"] == "Redford & Sons"
    assert first["location"] == "Boston, MA"
    assert first["startDate"] == "2018-09"
    assert first["endDate"] is None
    assert first["current"] is True
    assert first["designation"] != "arrangements for supervisors and managers"
    assert first["company"] != "Trained 2 administrative assistants during a period of company"
    assert first["location"] != "Redford & Sons, Boston, MA / September 2018 - Present"
    assert "arrangements for supervisors and managers" in first["description"]

    assert second["designation"] == "Secretary"
    assert second["company"] == "Bright Spot LTD"
    assert second["location"] == "Boston, MA"
    assert second["startDate"] == "2015-06"
    assert second["endDate"] == "2018-08"
    assert second["current"] is False
    assert second["designation"] != "SECRETARY"
    assert second["company"] != "Type documents such as correspondence, drafts, memos, and"
    assert second["location"] != "Type documents such as correspondence, drafts, memos, and"


def test_resume_7_experience_extractor_parses_combined_title_date_header():
    from app.pipeline.parser import ResumeParser

    resume = ResumeParser().parse(Path("tests/fixtures/resume_7.pdf").read_bytes())
    experience = resume.experience[0]

    assert experience.company == "Luna Web Design"
    assert experience.designation == "Web Developer"
    assert experience.location == "New York"
    assert experience.startDate == "2015-09"
    assert experience.endDate == "2019-05"
    assert experience.current is False
