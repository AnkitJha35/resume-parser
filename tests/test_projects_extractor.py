from app.extractors.projects import ProjectExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_project_extractor_with_url_and_technologies():
    blocks = [
        _make_block("AI Chatbot"),
        _make_block("Built a chatbot using Python, FastAPI, and TensorFlow."),
        _make_block("https://example.com/chatbot"),
    ]

    extractor = ProjectExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "AI Chatbot"
    assert "Built a chatbot" in entry["description"]
    assert entry["url"] == "https://example.com/chatbot"
    assert set(entry["technologies"]) >= {"Python", "FastAPI", "TensorFlow"}
    assert entry["confidence"] == 0.85


def test_project_extractor_without_dates():
    blocks = [
        _make_block("Resume Parser"),
        _make_block("Created a parser for PDF resumes using Python and regex."),
    ]

    extractor = ProjectExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "Resume Parser"
    assert entry["startDate"] is None
    assert entry["endDate"] is None
    assert "Python" in entry["technologies"]


def test_project_extractor_extracts_multiple_technologies():
    blocks = [
        _make_block("Analytics Dashboard"),
        _make_block("Developed dashboards with React, Django, and PostgreSQL."),
    ]

    extractor = ProjectExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    assert set(entries[0]["technologies"]) >= {"React", "Django", "PostgreSQL"}


def test_project_extractor_handles_missing_description():
    # Project with no description lines should not crash; technologies should be empty
    blocks = [
        _make_block("Lonely Project"),
    ]

    extractor = ProjectExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "Lonely Project"
    assert entry["description"] is None or entry["description"] == ""
    assert isinstance(entry["technologies"], list)
    assert entry["technologies"] == []


def test_project_extractor_accepts_candidate_groups():
    from app.pipeline.stages.block_classification import ClassifiedBlock
    from app.pipeline.stages.candidate_grouping import CandidateGroup
    b1 = _make_block("Resume Parser")
    b2 = _make_block("Date : 01/2020 - 12/2020")
    b3 = _make_block("Created a parser for PDF resumes using Python and regex.")

    cb1 = ClassifiedBlock(original=b1, label="UNKNOWN", score=0.0, reasons=[])
    cb2 = ClassifiedBlock(original=b2, label="DATE", score=1.0, reasons=[])
    cb3 = ClassifiedBlock(original=b3, label="DESCRIPTION", score=0.6, reasons=[])

    group = CandidateGroup(section="PROJECTS", blocks=[cb1, cb2, cb3], page_number=1, column_id=0, start_index=0, end_index=2, summary_text="Resume Parser")

    extractor = ProjectExtractor()
    entries = extractor.extract(groups=[group])

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "Resume Parser"
    assert entry["startDate"] == "2020-01"
    assert entry["endDate"] == "2020-12"
    assert "Python" in entry["technologies"]
