"""Tests for Phase 8-1 benchmark runner, expectation checks, and report generators."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.domain.resume import EducationItem, PersonalInfo, Resume
from tests.benchmark.expectations import GOLDEN_EXPECTATIONS, detect_structural_anomalies, evaluate_status
from tests.benchmark.metadata import BENCHMARK_FIXTURES
from tests.benchmark.report import format_terminal_table, generate_markdown_report, save_benchmark_report
from tests.benchmark.runner import BenchmarkRunner, ResumeParseResult

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_fixture_discovery():
    runner = BenchmarkRunner(fixtures_dir=FIXTURES_DIR)
    discovered = runner.discover_fixtures()
    assert len(discovered) == len(BENCHMARK_FIXTURES)
    assert len(discovered) == 12
    discovered_names = {p.name for p in discovered}
    assert "AditCV_SOL.pdf" in discovered_names
    assert "swe_experienced_resume.pdf" in discovered_names
    assert "Sendrick Costa CV.pdf" in discovered_names


def test_runner_single_parse_golden_success():
    runner = BenchmarkRunner(fixtures_dir=FIXTURES_DIR)
    pdf_path = FIXTURES_DIR / "AditCV_SOL.pdf"
    result = runner.run_single(pdf_path)

    assert result.filename == "AditCV_SOL.pdf"
    assert result.parser_success is True
    assert result.status == "PASS"
    assert result.elapsed_seconds > 0.0
    assert result.personal["name"] == "ADITI ANAND"
    assert result.personal["email"] == "aditianand136@gmail.com"
    assert result.experience_count == 1
    assert result.education_count == 1
    assert result.skills_count == 19
    assert len(result.diagnostics) == 0


def test_runner_handles_parser_exceptions_gracefully(tmp_path: Path):
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"not a real pdf content")

    runner = BenchmarkRunner(fixtures_dir=tmp_path)
    result = runner.run_single(corrupt_pdf)

    assert result.parser_success is False
    assert result.status == "FAIL"
    assert result.error_message is not None
    assert len(result.diagnostics) > 0
    assert "PARSER_EXCEPTION" in result.diagnostics[0]


def test_structural_anomaly_detection_rules():
    # Synthetic resume with known failure signatures
    bad_resume = Resume(
        parserVersion="1.0.0",
        personal=PersonalInfo(
            name="APPLICATION FORM",
            email="test@example.com",
            location="QUALIFICATION",
        ),
        skills=[],
        experience=[],
        education=[],
        projects=[],
    )

    anomalies = detect_structural_anomalies(bad_resume)
    assert any("NAME_IS_FORM_OR_DOC_TITLE" in a for a in anomalies)
    assert any("SECTION_HEADER_IN_LOCATION" in a for a in anomalies)

    status, diag = evaluate_status(bad_resume, "dummy.pdf")
    assert status in ("VALIDATION_FAILED", "FAIL")
    assert len(diag) >= 2


def test_report_generation_and_saving(tmp_path: Path):
    runner = BenchmarkRunner(fixtures_dir=FIXTURES_DIR)
    summary = runner.run_all([FIXTURES_DIR / "AditCV_SOL.pdf"])

    assert summary.total_resumes == 1
    assert summary.success_count == 1
    assert summary.pass_count == 1

    table = format_terminal_table(summary)
    assert "Resume" in table
    assert "AditCV_SOL.pdf" in table
    assert "PASS" in table

    md = generate_markdown_report(summary)
    assert "# Phase 8-1 Real Resume Benchmark Baseline Report" in md
    assert "AditCV_SOL.pdf" in md

    json_p, md_p = save_benchmark_report(summary, tmp_path)
    assert json_p.exists()
    assert md_p.exists()

    data = json.loads(json_p.read_text(encoding="utf-8"))
    assert data["total_resumes"] == 1
    assert data["results"][0]["filename"] == "AditCV_SOL.pdf"


def test_fresher_hr_resume_semantic_expectations():
    """Verify fresher_hr_resume.pdf golden expectation reflects 14 explicit source-grounded skills."""
    golden = GOLDEN_EXPECTATIONS["fresher_hr_resume.pdf"]
    assert golden.exact_skills_count is None
    assert golden.expected_skills is not None
    assert len(golden.expected_skills) == 14

    expected_14 = [
        "Recruitment & Selection Basics",
        "Employee Engagement Concepts",
        "Onboarding Process Understanding",
        "HR Policy Awareness",
        "Training & Development Support",
        "Basic Labor Law Knowledge",
        "MS Excel (VLOOKUP, Pivot Tables, Filters)",
        "Google Sheets & Docs",
        "HRMS (Basic understanding)",
        "Email & Calendar Management",
        "Strong Verbal & Written Communication",
        "Team Collaboration",
        "Time Management",
        "Adaptability & Willingness to Learn",
    ]
    assert golden.expected_skills == set(expected_14)

    # Construct clean semantic Resume matching ground truth
    resume = Resume(
        parserVersion="1.0.0",
        personal=PersonalInfo(
            name="Aditi Anand",
            email="aditianand136@gmail.com",
            phone="9241242699",
        ),
        skills=list(expected_14),
        experience=[],
        education=[
            EducationItem(
                degree="Master of Business Administration",
                institution="School Of Open Learning , Delhi University (DU-SOL)",
            ),
            EducationItem(
                degree="Bachelor of Arts (General)",
                institution="BIR Tikendrajit University",
            ),
        ],
        projects=[],
    )

    status, diag = evaluate_status(resume, "fresher_hr_resume.pdf")
    assert status == "PASS"
    assert diag == []

    # Verify missing skill triggers failure
    incomplete_resume = Resume(
        parserVersion="1.0.0",
        personal=PersonalInfo(
            name="Aditi Anand",
            email="aditianand136@gmail.com",
            phone="9241242699",
        ),
        skills=list(expected_14[:-1]),  # missing "Adaptability & Willingness to Learn"
        experience=[],
        education=resume.education,
        projects=[],
    )
    inc_status, inc_diag = evaluate_status(incomplete_resume, "fresher_hr_resume.pdf")
    assert inc_status == "PARTIAL"
    assert any("missing expected skills" in d and "Adaptability & Willingness to Learn" in d for d in inc_diag)


def test_fresher_hr_resume_semantic_input_contains_14_source_grounded_skills():
    """Verify upstream SemanticInput produces exactly the 14 explicit source-grounded skills."""
    from app.pipeline.stages.text_extraction import PDFExtractor
    from app.domain.document import document_from_text_blocks
    from app.pipeline.stages.reconstruction import reconstruct_document
    from app.pipeline.stages.layout import interpret_layout
    from app.domain.semantic_contract import build_semantic_input

    pdf_path = FIXTURES_DIR / "fresher_hr_resume.pdf"
    raw = PDFExtractor.extract(pdf_path.read_bytes())
    doc = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    s_input = build_semantic_input(doc)

    skill_blocks = [b for b in s_input.blocks if b.suggested_role == "SKILL"]
    assert len(skill_blocks) == 14

    extracted_skills = {b.text.strip() for b in skill_blocks}
    golden = GOLDEN_EXPECTATIONS["fresher_hr_resume.pdf"]
    assert extracted_skills == golden.expected_skills
