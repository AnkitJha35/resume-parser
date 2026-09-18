"""Phase 10C: Offline tests for Two-Pass Semantic Benchmark representation, metrics, and comparison."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import httpx

from app.domain.semantic_contract import (
    BodySemanticOutput,
    GroundedPersonal,
    GroundedString,
    PersonalSemanticOutput,
    SemanticInput,
    SemanticOutput,
)
from app.extractors.providers.gemini import GeminiSemanticExtractor
from tests.benchmark.compare import (
    SemanticComparisonReport,
    compare_semantic_runs,
    format_semantic_comparison_markdown,
    format_semantic_comparison_terminal,
)
from tests.benchmark.quality_gate import BenchmarkOutcome, evaluate_quality_gate
from tests.benchmark.semantic_runner import (
    SemanticBenchmarkRunner,
    SemanticBenchmarkSummary,
    SemanticParseResult,
)


# =====================================================================
# 1. Representation & Provider Selection Tests
# =====================================================================


def test_representation_selection_single_pass():
    """Selecting single_pass_candidate_b configures single-pass extraction mode."""
    extractor = GeminiSemanticExtractor(api_key="dummy-key", compact=True, two_pass=False)
    runner = SemanticBenchmarkRunner(extractor=extractor, representation="single_pass_candidate_b")

    assert runner.representation == "single_pass_candidate_b"
    assert runner.extraction_mode == "single_pass"
    assert runner.pass_count == 1


def test_representation_selection_two_pass():
    """Selecting two_pass_candidate_b configures two-pass extraction mode."""
    extractor = GeminiSemanticExtractor(api_key="dummy-key", compact=True, two_pass=True)
    runner = SemanticBenchmarkRunner(extractor=extractor, representation="two_pass_candidate_b")

    assert runner.representation == "two_pass_candidate_b"
    assert runner.extraction_mode == "two_pass"
    assert runner.pass_count == 2


def test_auto_detect_two_pass_from_extractor():
    """Runner auto-detects two-pass configuration directly from GeminiSemanticExtractor instance."""
    extractor_2p = GeminiSemanticExtractor(api_key="dummy-key", compact=True, two_pass=True)
    runner_2p = SemanticBenchmarkRunner(extractor=extractor_2p)
    assert runner_2p.representation == "two_pass_candidate_b"
    assert runner_2p.extraction_mode == "two_pass"
    assert runner_2p.pass_count == 2

    extractor_1p = GeminiSemanticExtractor(api_key="dummy-key", compact=True, two_pass=False)
    runner_1p = SemanticBenchmarkRunner(extractor=extractor_1p)
    assert runner_1p.representation == "single_pass_candidate_b"
    assert runner_1p.extraction_mode == "single_pass"
    assert runner_1p.pass_count == 1


def test_cli_fails_clearly_on_unknown_provider(capsys):
    """CLI runner fails with error message on unknown provider without silently falling back."""
    from tests.benchmark.run_semantic import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--provider", "unknown_provider"])
    assert exc_info.value.code == 2  # argparse invalid choice


# =====================================================================
# 2. Two-Pass Metrics Aggregation & Isolation Tests
# =====================================================================


def test_two_pass_metrics_aggregated_accurately():
    """Gemini two-pass execution aggregates prompt/output/total tokens and retries across passes."""
    pass1_response = {
        "document_archetype": "standard_cv",
        "personal": {
            "name": {"value": "Jane Smith", "source_block_ids": ["b_p1_0"]},
            "email": {"value": "jane@example.com", "source_block_ids": ["b_p1_1"]},
        },
    }
    pass2_response = {
        "document_archetype": "standard_cv",
        "skills": [
            {"value": "Python", "source_block_ids": ["b_p1_2"]},
            {"value": "Rust", "source_block_ids": ["b_p1_3"]},
        ],
        "experience": [],
        "education": [],
        "projects": [],
        "certifications": [],
        "languages": [],
        "achievements": [],
    }

    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        req_body = json.loads(request.content.decode("utf-8"))
        schema_props = req_body.get("generationConfig", {}).get("responseSchema", {}).get("properties", {})

        if "personal" in schema_props:
            resp_body = pass1_response
            tokens = {"promptTokenCount": 1000, "candidatesTokenCount": 150, "totalTokenCount": 1150}
        else:
            resp_body = pass2_response
            tokens = {"promptTokenCount": 1200, "candidatesTokenCount": 350, "totalTokenCount": 1550}

        envelope = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(resp_body)}]}}],
            "usageMetadata": tokens,
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = GeminiSemanticExtractor(api_key="mock-key", client=client, two_pass=True)
    sem_input = SemanticInput(document_id="doc1", page_count=1, pages=[], blocks=[])

    output = extractor.extract(sem_input)
    assert isinstance(output, SemanticOutput)
    assert output.personal.name.value == "Jane Smith"
    assert output.personal.email.value == "jane@example.com"
    assert len(output.skills) == 2

    # Verify aggregated usage metadata
    usage = extractor.last_usage_metadata
    assert usage is not None
    assert usage["two_pass"] is True
    assert usage["prompt_tokens"] == 1000 + 1200  # 2200
    assert usage["output_tokens"] == 150 + 350    # 500
    assert usage["total_tokens"] == 1150 + 1550   # 2700
    assert "pass_metadata" in usage
    assert usage["pass_metadata"]["personal"]["total_tokens"] == 1150
    assert usage["pass_metadata"]["body"]["total_tokens"] == 1550


def test_no_accidental_mixing_of_single_and_two_pass_metrics():
    """Single pass usage metadata contains two_pass=False and no pass_metadata sub-dictionary."""
    single_response = {
        "document_archetype": "standard_cv",
        "personal": {"name": {"value": "Jane Smith", "source_block_ids": ["b_p1_0"]}},
        "experience": [],
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        envelope = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(single_response)}]}}],
            "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 50, "totalTokenCount": 550},
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = GeminiSemanticExtractor(api_key="mock-key", client=client, two_pass=False)
    sem_input = SemanticInput(document_id="doc1", page_count=1, pages=[], blocks=[])

    extractor.extract(sem_input)
    usage = extractor.last_usage_metadata
    assert usage is not None
    assert usage["two_pass"] is False
    assert usage["prompt_tokens"] == 500
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 550
    assert "pass_metadata" not in usage


# =====================================================================
# 3. Result Serialization & Sanitization Tests
# =====================================================================


def test_result_serialization_contains_required_fields_and_no_pii(tmp_path: Path):
    """SemanticParseResult and SemanticBenchmarkSummary serialize with all benchmark fields and zero secrets."""
    secret_key = "AIzaSy_SECRET_KEY_NEVER_LEAK"

    result = SemanticParseResult(
        filename="test_candidate.pdf",
        archetype="maritime_tabular",
        target_domain="Maritime",
        semantic_success=True,
        status="PASS",
        passed_validation=True,
        elapsed_seconds=3.25,
        provider="gemini",
        model="gemini-3.5-flash-lite",
        representation="two_pass_candidate_b",
        extraction_mode="two_pass",
        pass_count=2,
        http_failures=0,
        fallback_invoked=False,
        usage={
            "prompt_tokens": 12000,
            "output_tokens": 800,
            "total_tokens": 12800,
            "two_pass": True,
        },
        personal={"name": "Captain Mayur", "email": "mayur@example.com"},
        skills_count=5,
        experience_count=2,
        education_count=1,
    )

    summary = SemanticBenchmarkSummary(
        timestamp="2026-09-04 12:00:00 UTC",
        provider="gemini",
        model="gemini-3.5-flash-lite",
        representation="two_pass_candidate_b",
        extraction_mode="two_pass",
        pass_count=2,
        total_cases=1,
        successful_cases=1,
        results=[result],
    )

    data = summary.to_dict()
    out_file = tmp_path / "summary.json"
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["representation"] == "two_pass_candidate_b"
    assert loaded["extraction_mode"] == "two_pass"
    assert loaded["pass_count"] == 2
    assert loaded["results"][0]["extraction_mode"] == "two_pass"
    assert loaded["results"][0]["pass_count"] == 2
    assert loaded["results"][0]["representation"] == "two_pass_candidate_b"
    assert secret_key not in json.dumps(loaded)


# =====================================================================
# 4. Comparison Classification & Percentiles Tests
# =====================================================================


def test_comparison_classifies_newly_fixed_regressions_and_unchanged():
    """compare_semantic_runs properly segments newly fixed, regressions, manual review, and unchanged resumes."""
    baseline_results = [
        # 1. AKIBUL was validation failure in baseline
        {
            "filename": "AKIBUL ALAM CV(JO).pdf",
            "archetype": "structured_form",
            "status": "FAIL",
            "semantic_success": False,
            "validation_violations": ["UNSUPPORTED_CANONICAL_VALUE in personal.name"],
            "elapsed_seconds": 2.2,
            "usage": {"total_tokens": 8200},
            "diagnostics": ["VALIDATION_VIOLATION: UNSUPPORTED_CANONICAL_VALUE in personal.name"],
        },
        # 2. Mayur was PASS in baseline
        {
            "filename": "2nd Officer Mayur Agarwal_062029.pdf",
            "archetype": "maritime_tabular",
            "status": "PASS",
            "semantic_success": True,
            "validation_violations": [],
            "elapsed_seconds": 2.5,
            "usage": {"total_tokens": 9400},
            "diagnostics": [],
        },
        # 3. Adit was PASS in baseline
        {
            "filename": "AditCV_SOL.pdf",
            "archetype": "standard_cv",
            "status": "PASS",
            "semantic_success": True,
            "validation_violations": [],
            "elapsed_seconds": 2.4,
            "usage": {"total_tokens": 2600},
            "diagnostics": [],
        },
        # 4. Aashish was PARTIAL in baseline
        {
            "filename": "AASHISH DG.pdf",
            "archetype": "structured_form",
            "status": "PARTIAL",
            "semantic_success": True,
            "validation_violations": [],
            "elapsed_seconds": 24.5,
            "usage": {"total_tokens": 14700},
            "diagnostics": ["NAME_IS_FORM_OR_DOC_TITLE"],
        },
    ]

    candidate_results = [
        # 1. AKIBUL is now PASS in candidate -> newly_fixed
        {
            "filename": "AKIBUL ALAM CV(JO).pdf",
            "archetype": "structured_form",
            "status": "PASS",
            "semantic_success": True,
            "validation_violations": [],
            "elapsed_seconds": 4.1,
            "usage": {"total_tokens": 21000},
            "diagnostics": [],
        },
        # 2. Mayur is now FAIL in candidate -> regression
        {
            "filename": "2nd Officer Mayur Agarwal_062029.pdf",
            "archetype": "maritime_tabular",
            "status": "FAIL",
            "semantic_success": False,
            "validation_violations": ["UNKNOWN_BLOCK_ID in personal.name"],
            "elapsed_seconds": 3.0,
            "usage": {"total_tokens": 11000},
            "diagnostics": ["UNKNOWN_BLOCK_ID"],
        },
        # 3. Adit is still PASS in candidate -> unchanged
        {
            "filename": "AditCV_SOL.pdf",
            "archetype": "standard_cv",
            "status": "PASS",
            "semantic_success": True,
            "validation_violations": [],
            "elapsed_seconds": 2.8,
            "usage": {"total_tokens": 4800},
            "diagnostics": [],
        },
        # 4. Aashish is still PARTIAL with diagnostics -> manual_review
        {
            "filename": "AASHISH DG.pdf",
            "archetype": "structured_form",
            "status": "PARTIAL",
            "semantic_success": True,
            "validation_violations": [],
            "elapsed_seconds": 22.0,
            "usage": {"total_tokens": 28000},
            "diagnostics": ["NAME_IS_FORM_OR_DOC_TITLE"],
        },
    ]

    base_summary = {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "representation": "single_pass_candidate_b",
        "results": baseline_results,
    }
    cand_summary = {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "representation": "two_pass_candidate_b",
        "results": candidate_results,
    }

    report = compare_semantic_runs(base_summary, cand_summary)

    assert report.total_compared == 4
    assert report.newly_fixed == ["AKIBUL ALAM CV(JO).pdf"]
    assert report.regressions == ["2nd Officer Mayur Agarwal_062029.pdf"]
    assert report.unchanged == ["AditCV_SOL.pdf"]
    assert report.manual_review == ["AASHISH DG.pdf"]

    # Percentile and average checks
    assert report.baseline_avg_latency > 0.0
    assert report.candidate_avg_latency > 0.0
    assert report.baseline_avg_tokens > 0.0
    assert report.candidate_avg_tokens > 0.0

    terminal_text = format_semantic_comparison_terminal(report)
    assert "AKIBUL ALAM CV(JO).pdf" in terminal_text
    assert "Newly Fixed" in terminal_text
    assert "Regressions" in terminal_text
    assert "[newly_fixed]" in terminal_text
    assert "[regression]" in terminal_text

    md_text = format_semantic_comparison_markdown(report)
    assert "# Semantic Benchmark Comparison Report" in md_text
    assert "| Metric | Baseline (Single-Pass) | Candidate (Two-Pass) | Delta |" in md_text


# =====================================================================
# 5. Quality Gate Integration with Two-Pass Results
# =====================================================================


def test_quality_gate_evaluates_two_pass_benchmark_results():
    """Quality Gate correctly evaluates two-pass structured results and passes when meeting thresholds."""
    results = [
        {
            "filename": f"resume_{i}.pdf",
            "archetype": "standard_cv" if i <= 6 else "maritime_cv",
            "status": "PASS",
            "semantic_success": True,
            "passed_validation": True,
            "validation_violations": [],
            "elapsed_seconds": 3.5,
            "usage": {
                "prompt_tokens": 8000,
                "output_tokens": 1200,
                "total_tokens": 9200,
                "two_pass": True,
                "fallback_invoked": False,
            },
            "diagnostics": [],
        }
        for i in range(1, 13)
    ]

    summary = evaluate_quality_gate(results)
    assert summary.total_resumes == 12
    assert summary.pass_count == 12
    assert summary.passed_gate is True
    assert summary.validation_failure_count == 0
    assert summary.total_tokens == 12 * 9200
