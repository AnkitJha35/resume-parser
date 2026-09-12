"""Phase 10W benchmark integration tests for the structured_table_two_pass representation.

Assertions:
  A. CLI accepts structured_table_two_pass as a valid --representation choice.
  B. structured_table_two_pass resolves to two_pass extraction mode, pass_count=2.
  C. serialize_structured_table_semantic_input() is actually selected as the serializer.
  D. Existing representations still resolve identically.
  E. Benchmark metadata reports structured_table_two_pass.
  F. No regression - no live API calls in any test.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.domain.semantic_contract import SemanticInput
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.semantic_prompt import (
    build_structured_table_body_prompt,
    build_structured_table_extraction_prompt,
    build_structured_table_personal_prompt,
    serialize_structured_table_semantic_input,
)
from tests.benchmark.semantic_runner import SemanticBenchmarkRunner, SemanticBenchmarkSummary, SemanticParseResult


# =====================================================================
# A. CLI accepts structured_table_two_pass
# =====================================================================


def test_cli_accepts_structured_table_two_pass():
    """CLI accepts structured_table_two_pass without error and configures runner & extractor correctly."""
    from tests.benchmark.run_semantic import main

    with (
        patch("tests.benchmark.run_semantic.SemanticBenchmarkRunner") as mock_runner_cls,
        patch("app.extractors.providers.gemini.GeminiSemanticExtractor") as mock_ext_cls,
    ):
        mock_runner = MagicMock()
        mock_runner.discover_fixtures.return_value = []
        real_summary = SemanticBenchmarkSummary(
            timestamp="2026-09-13 00:00:00 UTC",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            representation="structured_table_two_pass",
            extraction_mode="two_pass",
            pass_count=2,
            total_cases=0,
            successful_cases=0,
            results=[],
        )
        mock_runner.run_all.return_value = real_summary
        mock_runner_cls.return_value = mock_runner

        ret = main([
            "--provider", "gemini",
            "--representation", "structured_table_two_pass",
            "--output-dir", "/tmp/bench_test",
        ])
        assert ret == 0

        # Verify extractor was initialized with structured_table=True and two_pass=True
        mock_ext_cls.assert_called_once()
        _, ext_kwargs = mock_ext_cls.call_args
        assert ext_kwargs.get("structured_table") is True
        assert ext_kwargs.get("two_pass") is True

        # Verify runner was initialized with representation="structured_table_two_pass"
        mock_runner_cls.assert_called_once()
        _, runner_kwargs = mock_runner_cls.call_args
        assert runner_kwargs.get("representation") == "structured_table_two_pass"


def test_cli_rejects_unknown_representation():
    """CLI --representation unknown_rep must be rejected with exit code 2 (argparse invalid choice)."""
    from tests.benchmark.run_semantic import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--provider", "gemini", "--representation", "unknown_rep"])
    assert exc_info.value.code == 2


# =====================================================================
# B. Representation resolves to two-pass mode, pass_count=2
# =====================================================================


def test_structured_table_two_pass_runner_mode():
    """SemanticBenchmarkRunner with structured_table_two_pass resolves to two_pass, pass_count=2."""
    extractor = GeminiSemanticExtractor(api_key="dummy-key", compact=True, two_pass=True, structured_table=True)
    runner = SemanticBenchmarkRunner(extractor=extractor, representation="structured_table_two_pass")
    assert runner.representation == "structured_table_two_pass"
    assert runner.extraction_mode == "two_pass"
    assert runner.pass_count == 2


def test_structured_table_two_pass_extractor_flags():
    """GeminiSemanticExtractor stores structured_table=True and two_pass correctly."""
    extractor = GeminiSemanticExtractor(api_key="dummy-key", compact=True, two_pass=True, structured_table=True)
    assert extractor._structured_table is True
    assert extractor._explicit_two_pass is True
    assert extractor._compact is True


# =====================================================================
# C. Structured serializer is actually selected
# =====================================================================


def test_structured_table_serializer_selected_in_single_pass():
    """When structured_table=True and two_pass=False, build_structured_table_extraction_prompt is called."""
    sem_input = SemanticInput(document_id="test", page_count=1, pages=[], blocks=[])
    single_response = {
        "document_archetype": "standard_cv",
        "personal": {"name": {"value": "Alice", "source_block_ids": []}},
        "skills": [],
        "experience": [],
        "education": [],
        "projects": [],
        "certifications": [],
        "languages": [],
        "achievements": [],
    }
    def mock_handler(request: httpx.Request) -> httpx.Response:
        envelope = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(single_response)}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150},
        }
        return httpx.Response(200, json=envelope, request=request)

    with (
        patch("app.extractors.providers.gemini.build_structured_table_extraction_prompt", wraps=build_structured_table_extraction_prompt) as mock_s,
        patch("app.extractors.providers.gemini.build_compact_extraction_prompt") as mock_c,
        patch("app.extractors.providers.gemini.build_full_extraction_prompt") as mock_f,
    ):
        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        extractor = GeminiSemanticExtractor(api_key="mock-key", client=client, compact=True, two_pass=False, structured_table=True)
        extractor.extract(sem_input)
        mock_s.assert_called_once_with(sem_input)
        mock_c.assert_not_called()
        mock_f.assert_not_called()


def test_structured_table_serializer_selected_in_two_pass():
    """When structured_table=True and two_pass=True, structured-table personal/body prompts are called."""
    sem_input = SemanticInput(document_id="test", page_count=1, pages=[], blocks=[])
    personal_response = {"document_archetype": "standard_cv", "personal": {"name": {"value": "Alice", "source_block_ids": []}}}
    body_response = {"document_archetype": "standard_cv", "skills": [], "experience": [], "education": [], "projects": [], "certifications": [], "languages": [], "achievements": []}
    def mock_handler(request: httpx.Request) -> httpx.Response:
        req_body = json.loads(request.content.decode("utf-8"))
        schema_props = req_body.get("generationConfig", {}).get("responseSchema", {}).get("properties", {})
        body = personal_response if "personal" in schema_props else body_response
        envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}], "usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 80, "totalTokenCount": 280}}
        return httpx.Response(200, json=envelope, request=request)

    with (
        patch("app.extractors.providers.gemini.build_structured_table_personal_prompt", wraps=build_structured_table_personal_prompt) as mp,
        patch("app.extractors.providers.gemini.build_structured_table_body_prompt", wraps=build_structured_table_body_prompt) as mb,
        patch("app.extractors.providers.gemini.build_personal_extraction_prompt") as mcp,
        patch("app.extractors.providers.gemini.build_body_extraction_prompt") as mcb,
    ):
        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        extractor = GeminiSemanticExtractor(api_key="mock-key", client=client, compact=True, two_pass=True, structured_table=True)
        extractor.extract(sem_input)
        mp.assert_called_once_with(sem_input)
        mb.assert_called_once_with(sem_input)
        mcp.assert_not_called()
        mcb.assert_not_called()


def test_structured_table_serializer_function_called_by_prompt_builder():
    """build_structured_table_extraction_prompt calls serialize_structured_table_semantic_input."""
    sem_input = SemanticInput(document_id="doc", page_count=1, pages=[], blocks=[])
    with patch("app.extractors.semantic_prompt.serialize_structured_table_semantic_input", wraps=serialize_structured_table_semantic_input) as mock_s:
        build_structured_table_extraction_prompt(sem_input)
        mock_s.assert_called_once_with(sem_input)


# =====================================================================
# D. Existing representations still resolve identically
# =====================================================================


@pytest.mark.parametrize("rep,expected_mode,expected_count", [
    ("two_pass_candidate_b", "two_pass", 2),
    ("single_pass_candidate_b", "single_pass", 1),
    ("candidate_b_compact", "single_pass", 1),
    ("candidate_b_two_pass", "two_pass", 2),
    ("candidate_b_single_pass", "single_pass", 1),
    ("full", "single_pass", 1),
])
def test_existing_representations_resolve_identically(rep, expected_mode, expected_count):
    """Existing representations resolve to their same expected mode and pass count."""
    two_pass_flag = "two_pass" in rep
    extractor = GeminiSemanticExtractor(api_key="dummy-key", compact=(rep != "full"), two_pass=two_pass_flag, structured_table=False)
    runner = SemanticBenchmarkRunner(extractor=extractor, representation=rep)
    assert runner.representation == rep
    assert runner.extraction_mode == expected_mode
    assert runner.pass_count == expected_count


def test_existing_representation_compact_uses_compact_serializer():
    """When structured_table=False and compact=True, compact serializer is used."""
    import app.extractors.semantic_prompt as _sp
    sem_input = SemanticInput(document_id="doc", page_count=1, pages=[], blocks=[])
    single_response = {"document_archetype": "standard_cv", "personal": {"name": {"value": "Bob", "source_block_ids": []}}, "skills": [], "experience": [], "education": [], "projects": [], "certifications": [], "languages": [], "achievements": []}
    def mock_handler(request: httpx.Request) -> httpx.Response:
        envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(single_response)}]}}], "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150}}
        return httpx.Response(200, json=envelope, request=request)
    with (
        patch("app.extractors.providers.gemini.build_compact_extraction_prompt", wraps=_sp.build_compact_extraction_prompt) as mc,
        patch("app.extractors.providers.gemini.build_structured_table_extraction_prompt") as ms,
    ):
        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        extractor = GeminiSemanticExtractor(api_key="mock-key", client=client, compact=True, two_pass=False, structured_table=False)
        extractor.extract(sem_input)
        mc.assert_called_once()
        ms.assert_not_called()


# =====================================================================
# E. Benchmark metadata reports structured_table_two_pass
# =====================================================================


def test_benchmark_summary_reports_structured_table_two_pass():
    """SemanticBenchmarkSummary and SemanticParseResult correctly report structured_table_two_pass."""
    result = SemanticParseResult(
        filename="AASHISH DG.pdf",
        archetype="structured_form",
        target_domain="Maritime",
        semantic_success=True,
        status="PASS",
        passed_validation=True,
        elapsed_seconds=5.0,
        provider="gemini",
        model="gemini-3.5-flash-lite",
        representation="structured_table_two_pass",
        extraction_mode="two_pass",
        pass_count=2,
        http_failures=0,
        fallback_invoked=False,
        usage={"prompt_tokens": 8000, "output_tokens": 600, "total_tokens": 8600, "two_pass": True},
        personal={"name": "AASHISH"},
        skills_count=0,
        experience_count=3,
        education_count=0,
    )
    summary = SemanticBenchmarkSummary(
        timestamp="2026-09-13 00:00:00 UTC",
        provider="gemini",
        model="gemini-3.5-flash-lite",
        representation="structured_table_two_pass",
        extraction_mode="two_pass",
        pass_count=2,
        total_cases=1,
        successful_cases=1,
        results=[result],
    )
    data = summary.to_dict()
    assert data["representation"] == "structured_table_two_pass"
    assert data["extraction_mode"] == "two_pass"
    assert data["pass_count"] == 2
    assert data["results"][0]["representation"] == "structured_table_two_pass"
    assert data["results"][0]["extraction_mode"] == "two_pass"
    assert data["results"][0]["pass_count"] == 2


def test_gemini_usage_metadata_reports_structured_table_two_pass():
    """GeminiSemanticExtractor.last_usage_metadata reports representation=structured_table_two_pass."""
    personal_response = {"document_archetype": "standard_cv", "personal": {"name": {"value": "Alice", "source_block_ids": []}}}
    body_response = {"document_archetype": "standard_cv", "skills": [], "experience": [], "education": [], "projects": [], "certifications": [], "languages": [], "achievements": []}
    def mock_handler(request: httpx.Request) -> httpx.Response:
        req_body = json.loads(request.content.decode("utf-8"))
        schema_props = req_body.get("generationConfig", {}).get("responseSchema", {}).get("properties", {})
        body = personal_response if "personal" in schema_props else body_response
        envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}], "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 100, "totalTokenCount": 400}}
        return httpx.Response(200, json=envelope, request=request)
    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = GeminiSemanticExtractor(api_key="mock-key", client=client, compact=True, two_pass=True, structured_table=True)
    sem_input = SemanticInput(document_id="test_doc", page_count=1, pages=[], blocks=[])
    extractor.extract(sem_input)
    usage = extractor.last_usage_metadata
    assert usage is not None
    assert usage["representation"] == "structured_table_two_pass"
    assert usage["two_pass"] is True


# =====================================================================
# F. No regression
# =====================================================================


def test_no_regression_two_pass_candidate_b_representation():
    """two_pass_candidate_b still uses compact serializer (not structured-table)."""
    import app.extractors.semantic_prompt as _sp
    personal_response = {"document_archetype": "standard_cv", "personal": {"name": {"value": "Bob", "source_block_ids": []}}}
    body_response = {"document_archetype": "standard_cv", "skills": [], "experience": [], "education": [], "projects": [], "certifications": [], "languages": [], "achievements": []}
    def mock_handler(request: httpx.Request) -> httpx.Response:
        req_body = json.loads(request.content.decode("utf-8"))
        schema_props = req_body.get("generationConfig", {}).get("responseSchema", {}).get("properties", {})
        body = personal_response if "personal" in schema_props else body_response
        envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}], "usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 80, "totalTokenCount": 280}}
        return httpx.Response(200, json=envelope, request=request)
    with (
        patch("app.extractors.providers.gemini.build_personal_extraction_prompt", wraps=_sp.build_personal_extraction_prompt) as mp,
        patch("app.extractors.providers.gemini.build_structured_table_personal_prompt") as msp,
        patch("app.extractors.providers.gemini.build_structured_table_body_prompt") as msb,
    ):
        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        extractor = GeminiSemanticExtractor(api_key="mock-key", client=client, compact=True, two_pass=True, structured_table=False)
        sem_input = SemanticInput(document_id="doc", page_count=1, pages=[], blocks=[])
        extractor.extract(sem_input)
        mp.assert_called_once()
        msp.assert_not_called()
        msb.assert_not_called()
        usage = extractor.last_usage_metadata
        assert usage is not None
        assert usage["representation"] == "candidate_b_compact"
        assert usage["two_pass"] is True
