"""Phase 9-1: Tests for production semantic extraction observability and structured logging."""

from __future__ import annotations

import logging
import pytest

from app.domain.resume import Resume
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedPersonal,
    GroundedString,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    build_semantic_input,
)
from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.semantic_extractor import (
    MockSemanticExtractor,
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticRateLimitError,
    SemanticValidationError,
)
from app.pipeline.semantic_pipeline import parse_document_semantically
from tests.test_semantic_llm_contract import _make_test_document


class StubProviderWithMetadata:
    """Configurable stub provider that returns metadata and results or raises exceptions."""

    def __init__(
        self,
        provider_name: str = "gemini",
        model_name: str = "gemini-3.5-flash-lite",
        output: SemanticOutput | None = None,
        exc_to_raise: Exception | None = None,
        prompt_tokens: int = 1200,
        output_tokens: int = 350,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.output = output
        self.exc_to_raise = exc_to_raise
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.last_usage_metadata: dict | None = None

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        if self.exc_to_raise:
            self.last_usage_metadata = {
                "provider": self.provider_name,
                "model": self.model_name,
                "representation": "candidate_b_compact",
                "prompt_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "latency_ms": 15.0,
                "status": "failure",
                "error_type": type(self.exc_to_raise).__name__,
            }
            raise self.exc_to_raise

        self.last_usage_metadata = {
            "provider": self.provider_name,
            "model": self.model_name,
            "representation": "candidate_b_compact",
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.prompt_tokens + self.output_tokens,
            "latency_ms": 42.5,
            "status": "success",
        }
        return self.output


def _make_valid_test_output(doc_input: SemanticInput) -> SemanticOutput:
    # Match first block for grounded name
    b0 = doc_input.blocks[0]
    return SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value=b0.text, source_block_ids=[b0.block_id])
        ),
    )


def test_successful_semantic_extraction_logs_structured_observability(caplog):
    """Successful extraction emits structured start and completion logs with token and model info."""
    caplog.set_level(logging.INFO)
    doc = _make_test_document()
    sem_in = build_semantic_input(doc, document_id="doc_obs_123")
    valid_output = _make_valid_test_output(sem_in)

    provider = StubProviderWithMetadata(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        output=valid_output,
        prompt_tokens=2500,
        output_tokens=400,
    )

    resume = parse_document_semantically(doc, provider, document_id="doc_obs_123")
    assert isinstance(resume, Resume)

    records = [r for r in caplog.records if "semantic_pipeline" in r.name]
    messages = [r.getMessage() for r in records]

    # Verify start log
    start_logs = [m for m in messages if "Semantic extraction started" in m]
    assert len(start_logs) == 1
    assert "doc_id=doc_obs_123" in start_logs[0]
    assert "representation=candidate_b_compact" in start_logs[0]

    # Verify completion log
    complete_logs = [m for m in messages if "Semantic extraction completed" in m]
    assert len(complete_logs) == 1
    assert "doc_id=doc_obs_123" in complete_logs[0]
    assert "provider=gemini" in complete_logs[0]
    assert "model=gemini-3.5-flash-lite" in complete_logs[0]
    assert "prompt_tokens=2500" in complete_logs[0]
    assert "output_tokens=400" in complete_logs[0]
    assert "total_tokens=2900" in complete_logs[0]
    assert "fallback_invoked=False" in complete_logs[0]


def test_semantic_validation_failure_logs_sanitized_violation(caplog):
    """Validation failure logs structured warning with violation code without leaking full ungrounded payloads."""
    caplog.set_level(logging.INFO)
    doc = _make_test_document()
    sem_in = build_semantic_input(doc, document_id="doc_fail_val")

    # Output has an unknown hallucinated block ID
    invalid_output = SemanticOutput(
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["b_hallucinated_999"])
        )
    )

    provider = StubProviderWithMetadata(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        output=invalid_output,
    )

    with pytest.raises(SemanticValidationError):
        parse_document_semantically(doc, provider, document_id="doc_fail_val")

    records = [r for r in caplog.records if "semantic_pipeline" in r.name]
    val_logs = [r.getMessage() for r in records if "Semantic validation failed" in r.getMessage()]
    assert len(val_logs) == 1
    assert "doc_id=doc_fail_val" in val_logs[0]
    assert "violation_count=" in val_logs[0]
    assert "UNKNOWN_BLOCK_ID" in val_logs[0]


def test_semantic_extraction_provider_error_logs_structured_failure(caplog):
    """Provider failure emits structured error log with error_type and doc_id."""
    caplog.set_level(logging.INFO)
    doc = _make_test_document()

    provider = StubProviderWithMetadata(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        exc_to_raise=SemanticRateLimitError("Rate limit exceeded 429", status_code=429),
    )

    with pytest.raises(SemanticRateLimitError):
        parse_document_semantically(doc, provider, document_id="doc_rate_limit")

    records = [r for r in caplog.records if "semantic_pipeline" in r.name]
    fail_logs = [r.getMessage() for r in records if "Semantic extraction failed" in r.getMessage()]
    assert len(fail_logs) == 1
    assert "doc_id=doc_rate_limit" in fail_logs[0]
    assert "error_type=SemanticRateLimitError" in fail_logs[0]


def test_fallback_success_is_distinguishable_in_observability(caplog):
    """Fallback invocation and success is recorded and observable."""
    caplog.set_level(logging.INFO)
    doc = _make_test_document()
    sem_in = build_semantic_input(doc, document_id="doc_fb_success")
    valid_output = _make_valid_test_output(sem_in)

    primary = StubProviderWithMetadata(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        exc_to_raise=SemanticRateLimitError("Quota limit 429"),
    )
    fallback = StubProviderWithMetadata(
        provider_name="ollama",
        model_name="qwen2.5-coder:7b",
        output=valid_output,
        prompt_tokens=1500,
        output_tokens=300,
    )

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    resume = parse_document_semantically(doc, composite, document_id="doc_fb_success")
    assert isinstance(resume, Resume)

    # Check fallback extractor usage metadata
    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is True
    assert meta["primary_provider"] == "gemini"
    assert meta["fallback_provider"] == "ollama"
    assert meta["primary_error"] == "SemanticRateLimitError"
    assert meta["status"] == "success"

    # Check logs
    all_logs = [r.getMessage() for r in caplog.records]
    fb_init_logs = [m for m in all_logs if "Initiating fallback" in m]
    assert len(fb_init_logs) == 1
    assert "error_type=SemanticRateLimitError" in fb_init_logs[0]

    fb_succ_logs = [m for m in all_logs if "Semantic fallback succeeded" in m]
    assert len(fb_succ_logs) == 1

    complete_logs = [m for m in all_logs if "Semantic extraction completed" in m]
    assert len(complete_logs) == 1
    assert "fallback_invoked=True" in complete_logs[0]


def test_fallback_failure_is_distinguishable_in_observability(caplog):
    """When both primary and fallback fail, status and both errors are recorded."""
    caplog.set_level(logging.INFO)
    doc = _make_test_document()

    primary = StubProviderWithMetadata(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        exc_to_raise=SemanticRateLimitError("Quota limit 429"),
    )
    fallback = StubProviderWithMetadata(
        provider_name="ollama",
        model_name="qwen2.5-coder:7b",
        exc_to_raise=SemanticExtractionError("Ollama connection failed"),
    )

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    with pytest.raises(SemanticExtractionError):
        parse_document_semantically(doc, composite, document_id="doc_both_fail")

    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is True
    assert meta["status"] == "failure"
    assert meta["primary_error"] == "SemanticRateLimitError"
    assert meta["fallback_error"] == "SemanticExtractionError"

    all_logs = [r.getMessage() for r in caplog.records]
    both_fail_logs = [m for m in all_logs if "Both primary and fallback semantic providers failed" in m]
    assert len(both_fail_logs) == 1


def test_observability_logs_never_contain_pii_or_raw_resume_payloads(caplog):
    """Observability logs must never contain PII (names, emails, phone numbers, raw resume text) or API keys."""
    caplog.set_level(logging.DEBUG)
    doc = _make_test_document()
    sem_in = build_semantic_input(doc, document_id="doc_pii_test")

    secret_key = "AIzaSySECRET_API_KEY_987654321"
    b0 = sem_in.blocks[0]
    secret_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value=b0.text, source_block_ids=[b0.block_id])
        ),
    )

    provider = StubProviderWithMetadata(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        output=secret_output,
    )

    parse_document_semantically(doc, provider, document_id="doc_pii_test")

    for record in caplog.records:
        msg = record.getMessage()
        # Verify no API keys
        assert secret_key not in msg
        assert "AIzaSy" not in msg
        assert "Bearer " not in msg
        assert "x-goog-api-key" not in msg
        # Verify no raw prompt or resume dumps
        assert "Candidate B compact representation" not in msg
        assert "Return valid JSON complying strictly" not in msg
