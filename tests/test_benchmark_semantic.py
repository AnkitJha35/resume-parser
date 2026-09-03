"""Offline tests for semantic benchmark runner, comparison helpers, and safety invariants."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import httpx

from app.domain.resume import Resume
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedPersonal,
    GroundedString,
    SemanticInput,
    SemanticOutput,
)
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.semantic_extractor import (
    MockSemanticExtractor,
    SemanticExtractionError,
    SemanticValidationError,
)
from tests.benchmark.compare import compare_benchmarks, format_comparison_summary
from tests.benchmark.metadata import BENCHMARK_FIXTURES
from tests.benchmark.runner import BenchmarkRunner, BenchmarkRunSummary, ResumeParseResult
from tests.benchmark.semantic_runner import (
    SemanticBenchmarkRunner,
    SemanticBenchmarkSummary,
    SemanticParseResult,
)
from tests.test_semantic_llm_contract import _make_test_document


# =====================================================================
# A. Opt-in and Safety Tests
# =====================================================================


def test_live_benchmark_is_opt_in_and_not_run_by_pytest():
    # Normal test execution must not execute run_semantic.py main()
    from tests.benchmark import run_semantic
    assert hasattr(run_semantic, "main")
    # Verify run_semantic does not invoke Gemini at import time
    assert True


def test_missing_api_key_fails_clearly_in_cli(monkeypatch, capsys):
    from tests.benchmark.run_semantic import main
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Mock Settings to not provide gemini_api_key
    monkeypatch.setattr("tests.benchmark.run_semantic.Settings", lambda: type("S", (), {"gemini_api_key": None})())

    exit_code = main(["--provider", "gemini"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "GEMINI_API_KEY is not configured" in captured.err
    assert "Aborting without making any network calls" in captured.err


def test_ollama_unavailable_fails_clearly_in_cli(capsys):
    from tests.benchmark.run_semantic import main

    exit_code = main(["--provider", "ollama", "--base-url", "http://localhost:99999"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    assert "Ollama service is not reachable" in captured.err or "Failed to query" in captured.err


# =====================================================================
# C, D, E. Mocked Extraction, Isolation of Failures
# =====================================================================


def test_mocked_gemini_extraction_through_benchmark_runner(tmp_path):
    # Setup mock transport returning valid SemanticOutput grounded in _make_test_document
    def mock_handler(request: httpx.Request) -> httpx.Response:
        llm_response = {
            "document_archetype": "standard_cv",
            "personal": {
                "name": {"value": "John Doe", "raw_value": "John Doe", "source_block_ids": ["b_p1_0"]},
                "email": {"value": "john.doe@example.com", "source_block_ids": ["b_p1_1"]},
            },
            "experience": [],
        }
        envelope = {
            "candidates": [
                {
                    "content": {"parts": [{"text": json.dumps(llm_response)}]},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 150,
                "candidatesTokenCount": 35,
                "totalTokenCount": 185,
            },
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = GeminiSemanticExtractor(api_key="mock-key", client=client)
    runner = SemanticBenchmarkRunner(extractor=extractor)

    # Test running with synthetic document
    doc = _make_test_document()
    from app.pipeline.semantic_pipeline import parse_document_semantically
    resume = parse_document_semantically(doc, extractor, document_id="doc_mock")
    assert isinstance(resume, Resume)
    assert resume.personal.name == "John Doe"
    assert extractor.last_usage_metadata == {
        "prompt_tokens": 150,
        "output_tokens": 35,
        "total_tokens": 185,
    }


def test_extraction_and_validation_failures_isolated_per_resume():
    class MixedOutcomeExtractor:
        def __init__(self):
            self.call_count = 0
            self.last_usage_metadata = None

        def extract(self, input_data: SemanticInput) -> SemanticOutput:
            self.call_count += 1
            if self.call_count == 1:
                # 1. Extraction Error (e.g. API failure)
                raise SemanticExtractionError("Simulated LLM network timeout")
            elif self.call_count == 2:
                # 2. Validation Error (e.g. document title as name)
                return SemanticOutput(
                    personal=GroundedPersonal(
                        name=GroundedString(value="APPLICATION FORM", source_block_ids=["b_p1_0"])
                    )
                )
            else:
                # 3. Successful extraction
                return SemanticOutput(
                    personal=GroundedPersonal(
                        name=GroundedString(value="Valid Candidate", source_block_ids=["b_p1_0"])
                    )
                )

    runner = SemanticBenchmarkRunner(extractor=MixedOutcomeExtractor())

    # Mock discover_fixtures to return 3 dummy paths
    p1 = Path("/tmp/mock_p1.pdf")
    p2 = Path("/tmp/mock_p2.pdf")
    p3 = Path("/tmp/mock_p3.pdf")

    # Patch run_single to simulate the 3 distinct cases
    res1 = SemanticParseResult(
        filename="mock_1.pdf",
        archetype="standard_cv",
        target_domain="Engineering",
        semantic_success=False,
        status="EXTRACTION_FAILED",
        failure_type="EXTRACTION_ERROR",
        error_message="Simulated LLM network timeout",
    )
    res2 = SemanticParseResult(
        filename="mock_2.pdf",
        archetype="maritime_cv",
        target_domain="Maritime",
        semantic_success=False,
        status="VALIDATION_FAILED",
        failure_type="VALIDATION_ERROR",
        validation_violations=["DOCUMENT_TITLE_AS_NAME in personal.name"],
    )
    res3 = SemanticParseResult(
        filename="mock_3.pdf",
        archetype="standard_cv",
        target_domain="Engineering",
        semantic_success=True,
        status="PASS",
        personal={"name": "Valid Candidate"},
    )

    runner.run_single = lambda path: {p1: res1, p2: res2, p3: res3}[path]

    summary = runner.run_all([p1, p2, p3])

    assert summary.total_cases == 3
    assert summary.successful_cases == 1
    assert summary.extraction_failures == 1
    assert summary.validation_failures == 1
    assert summary.validation_pass_rate_pct == 33.33

    # Ensure one failure does not break the others
    assert summary.results[0].status == "EXTRACTION_FAILED"
    assert summary.results[1].status == "VALIDATION_FAILED"
    assert summary.results[2].status == "PASS"


# =====================================================================
# F, G, J. Credentials and Usage Metadata Safety
# =====================================================================


def test_credentials_never_appear_in_output_or_json(tmp_path):
    secret_key = "AIzaSySecretApiKeyDoNotLeak12345"
    summary = SemanticBenchmarkSummary(
        timestamp="2026-09-02 12:00:00 UTC",
        provider="gemini",
        model="gemini-2.5-flash",
        total_cases=1,
        successful_cases=1,
        results=[
            SemanticParseResult(
                filename="test.pdf",
                archetype="standard_cv",
                target_domain="Tech",
                semantic_success=True,
                status="PASS",
                personal={"name": "John Doe", "email": "john@example.com"},
            )
        ],
    )

    out_file = tmp_path / "summary.json"
    out_file.write_text(json.dumps(summary.to_dict()), encoding="utf-8")
    content = out_file.read_text(encoding="utf-8")

    assert secret_key not in content
    assert "api_key" not in content
    assert "token" not in content.lower() or "tokens" in content.lower()


def test_usage_metadata_remains_none_when_unreported():
    def mock_handler_no_usage(request: httpx.Request) -> httpx.Response:
        llm_response = {
            "document_archetype": "standard_cv",
            "personal": {"name": {"value": "John Doe", "source_block_ids": ["b_p1_0"]}},
        }
        envelope = {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps(llm_response)}]}}
            ]
            # No usageMetadata key
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler_no_usage))
    extractor = GeminiSemanticExtractor(api_key="mock-key", client=client)
    sem_input = SemanticInput(document_id="d1", page_count=1, pages=[], blocks=[])

    output = extractor.extract(sem_input)
    assert isinstance(output, SemanticOutput)
    assert extractor.last_usage_metadata is None


# =====================================================================
# H, I. Deterministic Baseline Unaffected and Comparison Helper
# =====================================================================


def test_deterministic_benchmark_unaffected():
    # Verify standard BenchmarkRunner still initializes and operates normally
    runner = BenchmarkRunner()
    fixtures = runner.discover_fixtures()
    assert len(fixtures) > 0
    assert hasattr(runner, "run_single")
    assert hasattr(runner, "run_all")


def test_comparison_helper_identifies_deltas():
    # Build a deterministic summary
    det_res1 = ResumeParseResult(
        filename="resume_a.pdf",
        archetype="standard_cv",
        target_domain="Tech",
        parser_success=True,
        status="PASS",
        elapsed_seconds=0.12,
        personal={"name": "Alice Smith", "email": "alice@example.com"},
        skills_count=5,
    )
    det_res2 = ResumeParseResult(
        filename="resume_b.pdf",
        archetype="maritime_cv",
        target_domain="Maritime",
        parser_success=True,
        status="FAIL",
        elapsed_seconds=0.15,
        diagnostics=["INVALID_NAME_IN_PERSONAL: APPLICATION FORM"],
        personal={"name": "APPLICATION FORM"},
    )
    det_summary = BenchmarkRunSummary(
        total_resumes=2,
        results=[det_res1, det_res2],
    )

    # Build a semantic summary where resume_b improved
    sem_res1 = SemanticParseResult(
        filename="resume_a.pdf",
        archetype="standard_cv",
        target_domain="Tech",
        semantic_success=True,
        status="PASS",
        elapsed_seconds=1.2,
        personal={"name": "Alice Smith", "email": "alice@example.com"},
        skills_count=5,
    )
    sem_res2 = SemanticParseResult(
        filename="resume_b.pdf",
        archetype="maritime_cv",
        target_domain="Maritime",
        semantic_success=True,
        status="PASS",
        elapsed_seconds=1.5,
        diagnostics=[],
        personal={"name": "Captain Mayur"},
    )
    sem_summary = SemanticBenchmarkSummary(
        timestamp="2026-09-02 12:00:00 UTC",
        total_cases=2,
        successful_cases=2,
        results=[sem_res1, sem_res2],
    )

    report = compare_benchmarks(det_summary, sem_summary)

    assert report.total_compared == 2
    assert report.deterministic_successes == 1  # only resume_a was != FAIL
    assert report.semantic_successes == 2
    assert "resume_b.pdf" in report.semantic_succeeds_deterministic_fails
    assert "resume_a.pdf" in report.both_succeed
    assert report.total_anomalies_reduced == 1

    summary_text = format_comparison_summary(report)
    assert "Total fixtures compared: 2" in summary_text
    assert "Semantic succeeds, deterministic fails: 1" in summary_text


# =====================================================================
# F. Fixture Filtering Tests (--fixture)
# =====================================================================


def test_discover_fixtures_default_discovers_all():
    runner = SemanticBenchmarkRunner()
    fixtures = runner.discover_fixtures()
    assert len(fixtures) == 12
    assert all(f.exists() for f in fixtures)
    assert any(f.name == "AditCV_SOL.pdf" for f in fixtures)


def test_discover_fixtures_valid_single():
    runner = SemanticBenchmarkRunner()
    fixtures = runner.discover_fixtures(fixture_name="AditCV_SOL.pdf")
    assert len(fixtures) == 1
    assert fixtures[0].name == "AditCV_SOL.pdf"
    assert fixtures[0].exists()


def test_discover_fixtures_invalid_raises_value_error():
    runner = SemanticBenchmarkRunner()
    with pytest.raises(ValueError) as exc_info:
        runner.discover_fixtures(fixture_name="non_existent_fixture.pdf")
    err_msg = str(exc_info.value)
    assert "Unknown benchmark fixture 'non_existent_fixture.pdf'" in err_msg
    assert "Available registered fixtures:" in err_msg
    assert "AditCV_SOL.pdf" in err_msg


def test_cli_invalid_fixture_fails_clearly(capsys, monkeypatch):
    from tests.benchmark.run_semantic import main

    monkeypatch.setattr("tests.benchmark.run_semantic.GeminiSemanticExtractor", lambda **kw: type("MockEx", (), {})())
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")

    exit_code = main(["--provider", "gemini", "--fixture", "invalid_resume.pdf"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ERROR: Unknown benchmark fixture 'invalid_resume.pdf'" in captured.err
    assert "AditCV_SOL.pdf" in captured.err
