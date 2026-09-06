"""Phase 9-4: Deterministic unit tests for the Production Semantic Quality Gate."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from tests.benchmark.quality_gate import (
    BenchmarkOutcome,
    QualityGateSummary,
    QualityGateThresholds,
    evaluate_quality_gate,
    export_quality_gate_json,
    format_quality_gate_markdown,
)


def _make_mock_results(
    pass_count: int = 10,
    partial_count: int = 2,
    fail_count: int = 0,
    error_count: int = 0,
    validation_violations: list[str] | None = None,
    diagnostics: list[str] | None = None,
) -> list[dict]:
    results: list[dict] = []
    idx = 0

    for _ in range(pass_count):
        idx += 1
        results.append({
            "filename": f"resume_{idx}.pdf",
            "archetype": "standard_cv" if idx <= 5 else "maritime_cv",
            "status": BenchmarkOutcome.PASS.value,
            "semantic_success": True,
            "passed_validation": True,
            "validation_violations": [],
            "elapsed_seconds": 1.25,
            "usage": {
                "prompt_tokens": 1500,
                "output_tokens": 300,
                "total_tokens": 1800,
                "fallback_invoked": False,
            },
            "skills_count": 15,
            "experience_count": 2,
            "education_count": 1,
            "projects_count": 2,
            "diagnostics": [],
        })

    for _ in range(partial_count):
        idx += 1
        results.append({
            "filename": f"resume_{idx}.pdf",
            "archetype": "structured_form",
            "status": BenchmarkOutcome.PARTIAL.value,
            "semantic_success": True,
            "passed_validation": True,
            "validation_violations": [],
            "elapsed_seconds": 2.10,
            "usage": {
                "prompt_tokens": 3200,
                "output_tokens": 450,
                "total_tokens": 3650,
                "fallback_invoked": False,
            },
            "skills_count": 8,
            "experience_count": 1,
            "education_count": 2,
            "projects_count": 0,
            "diagnostics": diagnostics or ["MINOR_FORMAT_DEVIATION: non-standard tabular grid"],
        })

    for _ in range(fail_count):
        idx += 1
        results.append({
            "filename": f"resume_{idx}.pdf",
            "archetype": "maritime_tabular",
            "status": BenchmarkOutcome.FAIL.value,
            "semantic_success": True,
            "passed_validation": False,
            "validation_violations": validation_violations or ["UNKNOWN_BLOCK_ID in personal.name"],
            "elapsed_seconds": 3.0,
            "usage": {
                "prompt_tokens": 4000,
                "output_tokens": 500,
                "total_tokens": 4500,
                "fallback_invoked": False,
            },
            "diagnostics": diagnostics or ["TABLE_HEADER_IN_EXPERIENCE: s.no"],
        })

    for _ in range(error_count):
        idx += 1
        results.append({
            "filename": f"resume_{idx}.pdf",
            "archetype": "unknown",
            "status": BenchmarkOutcome.ERROR.value,
            "semantic_success": False,
            "passed_validation": False,
            "validation_violations": [],
            "error_message": "Upstream 503 error",
            "elapsed_seconds": 0.5,
            "usage": None,
            "diagnostics": ["PROVIDER_ERROR: 503"],
        })

    return results


def test_quality_gate_passes_when_meeting_all_thresholds():
    """10 PASS + 2 PARTIAL (12 resumes) with 0 validation violations passes the default release gate."""
    results = _make_mock_results(pass_count=10, partial_count=2, fail_count=0, error_count=0)
    summary = evaluate_quality_gate(results)

    assert summary.total_resumes == 12
    assert summary.pass_count == 10
    assert summary.partial_count == 2
    assert summary.fail_count == 0
    assert summary.error_count == 0
    assert summary.pass_rate_pct == pytest.approx(83.33, 0.01)
    assert summary.partial_rate_pct == pytest.approx(16.67, 0.01)
    assert summary.fail_rate_pct == 0.0
    assert summary.error_rate_pct == 0.0
    assert summary.validation_failure_count == 0
    assert summary.provider_error_count == 0
    assert summary.total_tokens == (10 * 1800) + (2 * 3650)
    assert summary.passed_gate is True
    assert len(summary.failure_reasons) == 0


def test_quality_gate_fails_when_any_error_occurs():
    """Even a single provider error or crash immediately fails the quality gate."""
    results = _make_mock_results(pass_count=11, partial_count=0, fail_count=0, error_count=1)
    summary = evaluate_quality_gate(results)

    assert summary.total_resumes == 12
    assert summary.error_count == 1
    assert summary.passed_gate is False
    assert any("Errors exceed threshold" in r for r in summary.failure_reasons)


def test_quality_gate_fails_when_validation_violations_present():
    """Provenance/validation violations fail the release gate."""
    results = _make_mock_results(
        pass_count=10,
        partial_count=1,
        fail_count=1,
        validation_violations=["UNKNOWN_BLOCK_ID in personal.name"],
    )
    summary = evaluate_quality_gate(results)

    assert summary.validation_failure_count == 1
    assert summary.passed_gate is False
    assert any("Validation failures exceed threshold" in r for r in summary.failure_reasons)


def test_quality_gate_fails_when_pass_rate_below_threshold():
    """Low pass rate fails the quality gate."""
    # 7 PASS + 5 PARTIAL = 58.3% pass rate < 80.0%
    results = _make_mock_results(pass_count=7, partial_count=5, fail_count=0, error_count=0)
    summary = evaluate_quality_gate(results)

    assert summary.pass_rate_pct == pytest.approx(58.33, 0.01)
    assert summary.passed_gate is False
    assert any("Pass rate below threshold" in r for r in summary.failure_reasons)


def test_quality_gate_detects_historical_regression_signatures():
    """Known historical regressions fail the regression checks."""
    results = _make_mock_results(
        pass_count=9,
        partial_count=3,
        diagnostics=["NAME_IS_FORM_OR_DOC_TITLE: 'APPLICATION FORM'"],
    )
    summary = evaluate_quality_gate(results)

    assert summary.regression_checks["no_doc_title_as_name"] is False
    assert summary.passed_gate is False
    assert any("Historical regression signatures detected" in r for r in summary.failure_reasons)


def test_quality_gate_tracks_archetype_breakdown():
    """Archetype counters are accurately segmented."""
    results = _make_mock_results(pass_count=10, partial_count=2, fail_count=0, error_count=0)
    summary = evaluate_quality_gate(results)

    archs = summary.archetype_breakdown
    assert "standard_cv" in archs
    assert "maritime_cv" in archs
    assert "structured_form" in archs
    assert archs["standard_cv"]["total"] == 5
    assert archs["maritime_cv"]["total"] == 5
    assert archs["structured_form"]["total"] == 2


def test_quality_gate_markdown_generation_has_zero_pii():
    """Markdown report contains full summary metrics and zero PII."""
    secret_key = "AIzaSyTEST_KEY_12345"
    results = _make_mock_results(pass_count=10, partial_count=2)
    summary = evaluate_quality_gate(results)

    md_report = format_quality_gate_markdown(summary)
    assert "# Production Semantic Quality Gate Report" in md_report
    assert "GATE PASSED" in md_report
    assert "83.3%" in md_report
    assert "Historical Regression Protections" in md_report

    # Zero PII / secret assertions
    assert secret_key not in md_report
    assert "x-goog-api-key" not in md_report
    assert "Bearer " not in md_report


def test_quality_gate_json_export(tmp_path: Path):
    """Machine-readable JSON export serializes all metrics."""
    results = _make_mock_results(pass_count=10, partial_count=2)
    summary = evaluate_quality_gate(results)

    json_path = tmp_path / "quality_gate_summary.json"
    exported = export_quality_gate_json(summary, path=json_path)

    assert json_path.exists()
    assert exported["passed_gate"] is True
    assert exported["total_resumes"] == 12
    assert exported["pass_count"] == 10
    assert exported["partial_count"] == 2
    assert exported["pass_rate_pct"] == 83.33

    with open(json_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["passed_gate"] is True


def test_quality_gate_fails_when_completeness_failure_present():
    """Unrecovered body completeness failure immediately fails the release gate."""
    results = _make_mock_results(pass_count=11, partial_count=0, fail_count=0, error_count=0)
    results.append({
        "filename": "unrecovered_dense.pdf",
        "archetype": "standard_cv",
        "status": BenchmarkOutcome.COMPLETENESS_FAILED.value,
        "failure_type": "COMPLETENESS_ERROR",
        "semantic_success": False,
        "passed_validation": False,
        "body_completeness_failure": True,
        "final_body_empty": True,
        "body_recovery_invoked": True,
        "validation_violations": [],
        "elapsed_seconds": 2.5,
        "usage": {
            "prompt_tokens": 2500,
            "output_tokens": 100,
            "total_tokens": 2600,
            "body_recovery_invoked": True,
            "body_completeness_failure": True,
        },
        "skills_count": 0,
        "experience_count": 0,
        "education_count": 0,
        "projects_count": 0,
        "diagnostics": ["COMPLETENESS_ERROR: Empty body output on evidence-rich document"],
    })

    summary = evaluate_quality_gate(results)

    assert summary.total_resumes == 12
    assert summary.completeness_failure_count == 1
    assert summary.fail_count == 1
    assert summary.passed_gate is False
    assert any("Completeness failures exceed threshold" in r for r in summary.failure_reasons)
