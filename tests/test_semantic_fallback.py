"""Tests for FallbackSemanticExtractor composite provider."""

from __future__ import annotations

from typing import Any
import pytest

from app.domain.semantic_contract import (
    BlockClassification,
    DocumentArchetype,
    GroundedPersonal,
    GroundedString,
    SemanticBlockCategory,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
)
from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.semantic_extractor import (
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticExtractor,
    SemanticRateLimitError,
    SemanticResponseError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
    SemanticValidationError,
)


def _sample_semantic_input() -> SemanticInput:
    return SemanticInput(
        document_id="doc-test-1",
        page_count=1,
        blocks=[
            SemanticBlockInput(
                block_id="b1",
                text="Alice Doe",
                page=1,
                bbox=[10.0, 10.0, 100.0, 20.0],
                region_id="r1",
                region_kind="header",
                reading_order=1,
            )
        ],
    )


def _sample_semantic_output(name: str = "Alice Doe") -> SemanticOutput:
    return SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[BlockClassification(block_id="b1", category=SemanticBlockCategory.PERSONAL)],
        personal=GroundedPersonal(
            name=GroundedString(value=name, raw_value=name, source_block_ids=["b1"])
        ),
    )


class FakeExtractor:
    """Test double for SemanticExtractor with controllable behavior and usage metadata."""

    def __init__(
        self,
        provider_name: str,
        result: SemanticOutput | None = None,
        exception: Exception | None = None,
        usage_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.result = result
        self.exception = exception
        self.call_count = 0
        self.last_usage_metadata = usage_metadata or {
            "provider": provider_name,
            "model": f"{provider_name}-model",
            "prompt_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "latency_ms": 250.0,
            "retry_count": 0,
            "status": "success" if exception is None else "failure",
        }

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        self.call_count += 1
        if self.exception:
            raise self.exception
        return self.result or _sample_semantic_output()


class BareExtractor:
    """Test double lacking any last_usage_metadata attribute."""

    def __init__(self, result: SemanticOutput | None = None, exception: Exception | None = None) -> None:
        self.result = result
        self.exception = exception
        self.call_count = 0

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        self.call_count += 1
        if self.exception:
            raise self.exception
        return self.result or _sample_semantic_output()


def test_fallback_extractor_satisfies_protocol():
    primary = FakeExtractor("gemini")
    fallback = FakeExtractor("ollama")
    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    assert isinstance(composite, SemanticExtractor)


def test_primary_success_bypasses_fallback():
    primary_out = _sample_semantic_output("Alice Primary")
    primary = FakeExtractor("gemini", result=primary_out)
    fallback = FakeExtractor("ollama", result=_sample_semantic_output("Bob Fallback"))

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    sem_input = _sample_semantic_input()
    output = composite.extract(sem_input)

    assert primary.call_count == 1
    assert fallback.call_count == 0
    assert output.personal.name.value == "Alice Primary"

    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is False
    assert meta["primary_provider"] == "gemini"
    assert meta["fallback_provider"] == "ollama"
    assert meta["provider"] == "gemini"
    assert meta["prompt_tokens"] == 100


@pytest.mark.parametrize(
    "eligible_exc",
    [
        SemanticRateLimitError("Rate limit exceeded", status_code=429),
        SemanticServerError("Server error", status_code=500),
        SemanticTimeoutError("Request timed out"),
        SemanticTransportError("Connection failed"),
        SemanticConfigurationError("Missing API key"),
    ],
)
def test_eligible_primary_failures_trigger_fallback(eligible_exc):
    fallback_out = _sample_semantic_output("Fallback Success")
    primary = FakeExtractor("gemini", exception=eligible_exc)
    fallback = FakeExtractor("ollama", result=fallback_out)

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    sem_input = _sample_semantic_input()
    output = composite.extract(sem_input)

    assert primary.call_count == 1
    assert fallback.call_count == 1
    assert output.personal.name.value == "Fallback Success"

    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is True
    assert meta["primary_provider"] == "gemini"
    assert meta["fallback_provider"] == "ollama"
    assert meta["primary_error"] == type(eligible_exc).__name__
    assert meta["provider"] == "ollama"
    assert meta["status"] == "success"


@pytest.mark.parametrize(
    "non_eligible_exc",
    [
        SemanticResponseError("Malformed JSON envelope"),
        SemanticValidationError(["Field violates provenance"]),
        SemanticExtractionError("Generic extraction failure"),
    ],
)
def test_non_eligible_primary_failures_do_not_trigger_fallback(non_eligible_exc):
    primary = FakeExtractor("gemini", exception=non_eligible_exc)
    fallback = FakeExtractor("ollama", result=_sample_semantic_output())

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    sem_input = _sample_semantic_input()

    with pytest.raises(type(non_eligible_exc)):
        composite.extract(sem_input)

    assert primary.call_count == 1
    assert fallback.call_count == 0

    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is False
    assert meta["status"] == "failure"
    assert meta["error_type"] == type(non_eligible_exc).__name__


@pytest.mark.parametrize(
    "eligible_exc",
    [
        SemanticRateLimitError("Rate limit", status_code=429),
        SemanticServerError("Server 503", status_code=503),
        SemanticTimeoutError("Timeout"),
        SemanticTransportError("DNS error"),
        SemanticConfigurationError("No credentials"),
    ],
)
def test_no_fallback_configured_re_raises_primary_error(eligible_exc):
    primary = FakeExtractor("gemini", exception=eligible_exc)
    composite = FallbackSemanticExtractor(primary=primary, fallback=None)
    sem_input = _sample_semantic_input()

    with pytest.raises(type(eligible_exc)):
        composite.extract(sem_input)

    assert primary.call_count == 1
    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is False
    assert meta["status"] == "failure"
    assert meta["fallback_provider"] is None


def test_both_providers_fail_raises_combined_error():
    primary_exc = SemanticServerError("Gemini HTTP 500 error", status_code=500)
    fallback_exc = SemanticTimeoutError("Ollama HTTP request timed out")

    primary = FakeExtractor("gemini", exception=primary_exc)
    fallback = FakeExtractor("ollama", exception=fallback_exc)

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    sem_input = _sample_semantic_input()

    with pytest.raises(SemanticExtractionError) as exc_info:
        composite.extract(sem_input)

    assert primary.call_count == 1
    assert fallback.call_count == 1

    err_msg = str(exc_info.value)
    assert "gemini" in err_msg
    assert "ollama" in err_msg
    assert "SemanticServerError" in err_msg
    assert "SemanticTimeoutError" in err_msg
    assert exc_info.value.__cause__ is fallback_exc

    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["status"] == "failure"
    assert meta["fallback_invoked"] is True
    assert meta["primary_provider"] == "gemini"
    assert meta["fallback_provider"] == "ollama"
    assert meta["primary_error"] == "SemanticServerError"
    assert meta["fallback_error"] == "SemanticTimeoutError"


def test_metadata_handling_when_providers_lack_metadata():
    primary = BareExtractor(exception=SemanticServerError("500", status_code=500))
    fallback = BareExtractor(result=_sample_semantic_output("Bare Success"))

    composite = FallbackSemanticExtractor(primary=primary, fallback=fallback)
    sem_input = _sample_semantic_input()
    output = composite.extract(sem_input)

    assert output.personal.name.value == "Bare Success"
    meta = composite.last_usage_metadata
    assert meta is not None
    assert meta["fallback_invoked"] is True
    assert meta["primary_error"] == "SemanticServerError"
    assert meta["primary_provider"] == "BareExtractor"
    assert meta["fallback_provider"] == "BareExtractor"


def test_constructor_validations():
    with pytest.raises(ValueError, match="Primary semantic extractor is required"):
        FallbackSemanticExtractor(primary=None)  # type: ignore[arg-type]

    same_extractor = FakeExtractor("gemini")
    with pytest.raises(ValueError, match="cannot be the same instance"):
        FallbackSemanticExtractor(primary=same_extractor, fallback=same_extractor)
