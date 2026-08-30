"""Phase 6E: CandidateGroup EXPERIENCE job-open fix and DATE-boundary safety tests."""

from __future__ import annotations

from pathlib import Path

from app.domain.document import document_from_text_blocks
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.block_classification import classify_block
from app.pipeline.stages.candidate_grouping import group_candidates
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.semantic_compat import semantic_sections_to_text_blocks
from app.pipeline.stages.semantic_paths import detect_region_aware_sections
from app.pipeline.stages.text_extraction import PDFExtractor, TextBlock


def _block(
    text: str,
    y0: float,
    *,
    x0: float = 35.0,
    x1: float = 200.0,
    font_size: float = 10.0,
    bold: bool = False,
) -> TextBlock:
    return TextBlock(
        text=text,
        page_number=1,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y0 + 12.0,
        font_size=font_size,
        bold=bold,
    )


def _groups(blocks: list[TextBlock], section: str = "EXPERIENCE") -> list[list[tuple[str, str]]]:
    classified = [classify_block(block) for block in blocks]
    return [
        [(cb.label, (cb.original.text or "").strip()) for cb in group.blocks]
        for group in group_candidates(classified, section)
    ]


def _labels(groups: list[list[tuple[str, str]]]) -> list[list[str]]:
    return [[label for label, _ in group] for group in groups]


def _texts(groups: list[list[tuple[str, str]]]) -> list[list[str]]:
    return [[text for _, text in group] for group in groups]


def _completed_job1_prefix() -> list[TextBlock]:
    return [
        _block("Software Engineer", 100, font_size=12),
        _block("Example Corp LLC", 120),
        _block("Date : 2018 – 2020", 140),
        _block("Location: Boston, MA", 160),
        _block("• Built internal scheduling tools for operations teams.", 180),
        _block("Delivered reliability improvements across regional services.", 200),
    ]


def test_A_single_job_title_company_date_location_body_one_group():
    groups = _groups([
        _block("Platform Engineer", 100, font_size=12),
        _block("Example Corp LLC", 120),
        _block("Date : 02/2022 – 06/2022", 140),
        _block("Location: Boston, MA", 160),
        _block("Developed backend services using Java and Spring Boot.", 180),
    ])
    assert len(groups) == 1
    assert _labels(groups)[0][:4] == ["JOB_TITLE", "COMPANY", "DATE", "LOCATION"]
    assert "Platform Engineer" in _texts(groups)[0]
    assert "Example Corp LLC" in _texts(groups)[0]


def test_B_completed_job_then_mixed_case_title_company_date_body_two_groups():
    groups = _groups(
        _completed_job1_prefix()
        + [
            _block("Platform Engineer", 240, font_size=12),
            _block("Northwind Partners LLC", 260),
            _block("Date : 2021 – 2022", 280),
            _block("Built APIs for partner integrations.", 300),
        ]
    )
    assert len(groups) == 2
    assert _texts(groups)[0][0] == "Software Engineer"
    assert "Platform Engineer" in _texts(groups)[1]
    assert "Northwind Partners LLC" in _texts(groups)[1]
    assert any("2021" in text for text in _texts(groups)[1])
    assert "Platform Engineer" not in _texts(groups)[0]


def test_C_completed_job_then_title_without_corroboration_stays_one_group():
    groups = _groups(
        _completed_job1_prefix()
        + [
            _block("Platform Engineer", 240, font_size=12),
            _block("Continued ownership of regional delivery workstreams.", 260),
        ]
    )
    assert len(groups) == 1
    assert "Platform Engineer" in _texts(groups)[0]


def test_D_completed_job_then_mixed_case_title_company_date_two_groups():
    groups = _groups(
        _completed_job1_prefix()
        + [
            _block("Platform Engineer", 240, font_size=12),
            _block("Northwind Partners LLC", 260),
            _block("Date : 2021 – 2022", 280),
        ]
    )
    assert len(groups) == 2
    assert _texts(groups)[1][:3] == [
        "Platform Engineer",
        "Northwind Partners LLC",
        "Date : 2021 – 2022",
    ]


def test_E_completed_job_then_title_location_body_opens_with_location_corroboration():
    groups = _groups(
        _completed_job1_prefix()
        + [
            _block("Platform Engineer", 240, font_size=12),
            _block("Location: Austin, TX", 260),
            _block("Owned release coordination for partner services.", 280),
        ]
    )
    assert len(groups) == 2
    assert _texts(groups)[1][0] == "Platform Engineer"
    assert "Location: Austin, TX" in _texts(groups)[1]


def test_F_date_alone_does_not_invent_new_group_from_empty_stream():
    groups = _groups([_block("Date : 02/2022 – 06/2022", 100)])
    assert len(groups) == 1
    assert _labels(groups)[0] == ["DATE"]


def test_G_projects_title_date_body_unchanged():
    groups = _groups(
        [
            _block("Network Monitoring System", 100, font_size=12),
            _block("Date : 11/2023 – 01/2025", 120),
            _block("Developed a scalable backend for device telemetry.", 140),
        ],
        section="PROJECTS",
    )
    assert len(groups) == 1
    assert "DATE" in _labels(groups)[0]
    assert _texts(groups)[0][0] == "Network Monitoring System"


def test_H_education_degree_date_unchanged():
    groups = _groups(
        [
            _block("Bachelor of Science | State University", 100),
            _block("2020-2024", 100, x0=450.0, x1=530.0),
            _block("Completed coursework in statistics.", 130),
        ],
        section="EDUCATION",
    )
    assert len(groups) == 1
    assert "DATE" in _labels(groups)[0]


def test_I_resume_a_exactly_one_experience_internship():
    path = Path("tests/fixtures/AditCV_SOL.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert len(resume.experience) == 1
    assert resume.experience[0].designation == "HR Intern"
    assert resume.experience[0].company == "ECE Industries Ltd."


def test_J_resume_b_no_invented_experience():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        return
    layout = interpret_layout(
        reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes())))
    )
    sections = semantic_sections_to_text_blocks(detect_region_aware_sections(layout))
    assert sections.get("EXPERIENCE", []) == []
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.experience == []


def test_K_resume_c_two_experience_groups_with_complete_intern_membership():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    layout = interpret_layout(
        reconstruct_document(document_from_text_blocks(PDFExtractor.extract(path.read_bytes())))
    )
    blocks = semantic_sections_to_text_blocks(detect_region_aware_sections(layout)).get("EXPERIENCE", [])
    classified = [classify_block(block) for block in blocks]
    groups = group_candidates(classified, "EXPERIENCE")
    assert len(groups) == 2

    g0 = [(cb.label, (cb.original.text or "").replace("\u200b", "")) for cb in groups[0].blocks]
    g1 = [(cb.label, (cb.original.text or "").replace("\u200b", "")) for cb in groups[1].blocks]
    g0_texts = [text for _, text in g0]
    g1_texts = [text for _, text in g1]

    assert any("SDE 2" in text for text in g0_texts)
    assert any("Probus" in text for text in g0_texts)
    assert not any("SDE Intern" in text for text in g0_texts)
    assert not any("Navyug" in text for text in g0_texts)

    assert g1[0][1].strip().startswith("SDE Intern")
    assert any("Navyug" in text for text in g1_texts)
    assert any("02/2022" in text for text in g1_texts)
    assert any("Location: Noida" in text for text in g1_texts)
    assert any("Developed backend services" in text for text in g1_texts)

    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert len(resume.experience) == 2
    assert resume.experience[0].designation == "SDE 2"
    assert resume.experience[0].company == "Probus Smart Things!"
    assert resume.experience[1].designation == "SDE Intern"
    assert resume.experience[1].company == "Navyug Infosolutions Pvt. Ltd"
    assert resume.experience[1].startDate == "2022-02"
    assert resume.experience[1].endDate == "2022-06"
    assert resume.experience[1].location and "Noida" in resume.experience[1].location
    assert resume.experience[1].description and "backend services" in resume.experience[1].description
