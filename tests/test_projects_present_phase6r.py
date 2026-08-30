"""Phase 6R: project Present dates → endDate=None, current=True."""

from __future__ import annotations

from pathlib import Path

from app.extractors.date_parser import DateRangeParser
from app.extractors.experience import ExperienceExtractor
from app.extractors.projects import ProjectExtractor
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.block_classification import ClassifiedBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup
from app.pipeline.stages.text_extraction import TextBlock


def _blk(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=100, y1=10)


def _cb(text: str, label: str) -> ClassifiedBlock:
    return ClassifiedBlock(original=_blk(text), label=label, score=1.0, reasons=[])


def _project_group(title: str, date_line: str, body: str = "Built something with Python.") -> CandidateGroup:
    return CandidateGroup(
        section="PROJECTS",
        blocks=[
            _cb(title, "UNKNOWN"),
            _cb(date_line, "DATE"),
            _cb(body, "DESCRIPTION"),
        ],
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=2,
        summary_text=title,
    )


def test_present_lowercase_sets_current_not_literal():
    entries = ProjectExtractor().extract(
        groups=[_project_group("Head End System", "Date : 02/2025 – present")]
    )
    assert len(entries) == 1
    assert entries[0]["startDate"] == "2025-02"
    assert entries[0]["endDate"] is None
    assert entries[0]["current"] is True


def test_present_capitalized_sets_current_not_literal():
    entries = ProjectExtractor().extract(
        groups=[_project_group("Head End System", "Date : 02/2025 – Present")]
    )
    assert entries[0]["startDate"] == "2025-02"
    assert entries[0]["endDate"] is None
    assert entries[0]["current"] is True


def test_finite_range_unchanged():
    entries = ProjectExtractor().extract(
        groups=[_project_group("Network Monitor", "Date : 02/2024 – 01/2025")]
    )
    assert entries[0]["startDate"] == "2024-02"
    assert entries[0]["endDate"] == "2025-01"
    assert entries[0]["current"] is False


def test_start_only_not_a_range():
    # DateRangeParser only accepts ranges; a lone month/year is not applied as dates.
    date_line = "Date : 02/2025"
    assert DateRangeParser.parse(date_line) is None
    entries = ProjectExtractor().extract(
        groups=[_project_group("Solo Start", date_line)]
    )
    assert entries[0]["startDate"] is None
    assert entries[0]["endDate"] is None
    assert entries[0]["current"] is False


def test_date_range_parser_current_semantics():
    parsed = DateRangeParser.parse("Date : 02/2025 – present")
    assert parsed is not None
    assert parsed.startDate == "2025-02"
    assert parsed.endDate is None
    assert parsed.current is True


def test_experience_current_behavior_unchanged():
    blocks = [
        _blk("Software Engineer"),
        _blk("Acme Corp"),
        _blk("Jan 2022 - Present"),
        _blk("Built APIs."),
    ]
    entries = ExperienceExtractor().extract(blocks)
    assert entries[0]["startDate"] == "2022-01"
    assert entries[0]["endDate"] is None
    assert entries[0]["current"] is True


def test_block_path_present_also_normalized():
    # Block path attaches dates to the entry opened by the date line.
    entries = ProjectExtractor().extract(
        [
            _blk("Date : 02/2025 – present"),
            _blk("Ongoing Tool"),
            _blk("Uses Docker and Kubernetes."),
        ]
    )
    assert entries[0]["startDate"] == "2025-02"
    assert entries[0]["endDate"] is None
    assert entries[0]["current"] is True
    assert entries[0]["name"] == "Ongoing Tool"


def test_resume_c_ongoing_project_normalized():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert len(resume.projects) == 3
    first = resume.projects[0]
    assert first.startDate == "2025-02"
    assert first.endDate is None
    assert first.current is True
    assert resume.projects[1].endDate == "2025-01"
    assert resume.projects[1].current is False
    assert "synchronous and" in (first.description or "")
    assert "synchronousand" not in (first.description or "")
    assert len(resume.experience) == 2
    assert resume.experience[0].designation == "SDE 2"
    assert resume.experience[1].designation == "SDE Intern"
    assert len(resume.education) == 2
    assert len(resume.skills) == 39
    assert resume.personal.location == "Noida, U.P, India"
