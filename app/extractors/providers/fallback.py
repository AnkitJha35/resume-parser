"""Composite semantic extractor providing transparent primary-to-fallback routing."""

from __future__ import annotations

import logging
from typing import Any

from app.domain.semantic_contract import SemanticInput, SemanticOutput
from app.extractors.semantic_extractor import (
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticExtractor,
    SemanticRateLimitError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
)

logger = logging.getLogger(__name__)

# Specific exception types that represent infrastructure, capacity, transport,
# or configuration failures eligible for fallback execution.
FALLBACK_ELIGIBLE_EXCEPTIONS = (
    SemanticRateLimitError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
    SemanticConfigurationError,
)


def _get_provider_identifier(extractor: SemanticExtractor | None, fallback_name: str = "unknown") -> str:
    """Safely extract provider name from usage metadata or class name."""
    if extractor is None:
        return fallback_name
    meta = getattr(extractor, "last_usage_metadata", None)
    if isinstance(meta, dict) and meta.get("provider"):
        return str(meta["provider"])
    cname = type(extractor).__name__.lower()
    if "gemini" in cname:
        return "gemini"
    if "ollama" in cname:
        return "ollama"
    if "mock" in cname:
        return "mock"
    return type(extractor).__name__


class FallbackSemanticExtractor:
    """Composite SemanticExtractor that attempts extraction on a primary provider

    and seamlessly falls back to a secondary provider on eligible transient or
    configuration failures.
    """

    def __init__(
        self,
        primary: SemanticExtractor,
        fallback: SemanticExtractor | None = None,
    ) -> None:
        if primary is None:
            raise ValueError("Primary semantic extractor is required.")
        if fallback is not None and primary is fallback:
            raise ValueError("Primary and fallback extractors cannot be the same instance.")

        self.primary = primary
        self.fallback = fallback
        self.last_usage_metadata: dict[str, Any] | None = None

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        """Extract SemanticOutput from SemanticInput using primary with optional fallback."""
        primary_name = _get_provider_identifier(self.primary, "primary")
        fallback_name = _get_provider_identifier(self.fallback, "fallback") if self.fallback is not None else None

        # 1. Attempt primary provider extraction
        try:
            result = self.primary.extract(input_data)
            self._record_primary_success()
            return result

        except Exception as primary_exc:
            # Check eligibility for fallback
            if not isinstance(primary_exc, FALLBACK_ELIGIBLE_EXCEPTIONS) or self.fallback is None:
                self._record_primary_failure(primary_exc)
                raise

            saved_primary_exc = primary_exc
            logger.warning(
                "Primary semantic provider '%s' failed with error_type=%s. Initiating fallback to '%s'...",
                primary_name,
                type(saved_primary_exc).__name__,
                fallback_name,
            )

        # 2. Attempt fallback provider extraction
        try:
            fallback_result = self.fallback.extract(input_data)
            self._record_fallback_success(saved_primary_exc)
            logger.info(
                "Semantic fallback succeeded: primary '%s' failed (error_type=%s), fallback '%s' succeeded",
                primary_name,
                type(saved_primary_exc).__name__,
                fallback_name,
            )
            return fallback_result

        except Exception as fallback_exc:
            self._record_both_failure(saved_primary_exc, fallback_exc)
            error_msg = (
                f"Primary provider '{primary_name}' failed with {type(saved_primary_exc).__name__}; "
                f"fallback provider '{fallback_name}' failed with {type(fallback_exc).__name__}"
            )
            logger.error(
                "Both primary and fallback semantic providers failed: primary '%s' error_type=%s; fallback '%s' error_type=%s",
                primary_name,
                type(saved_primary_exc).__name__,
                fallback_name,
                type(fallback_exc).__name__,
            )
            raise SemanticExtractionError(error_msg) from fallback_exc

    def _record_primary_success(self) -> None:
        raw_meta = getattr(self.primary, "last_usage_metadata", None)
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        meta["fallback_invoked"] = False
        meta["primary_provider"] = meta.get("provider") or _get_provider_identifier(self.primary)
        meta["fallback_provider"] = (
            _get_provider_identifier(self.fallback) if self.fallback is not None else None
        )
        meta["representation"] = meta.get("representation") or "candidate_b_compact"
        meta["status"] = "success"
        self.last_usage_metadata = meta

    def _record_primary_failure(self, primary_exc: Exception) -> None:
        raw_meta = getattr(self.primary, "last_usage_metadata", None)
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        meta["fallback_invoked"] = False
        meta["primary_provider"] = meta.get("provider") or _get_provider_identifier(self.primary)
        meta["fallback_provider"] = (
            _get_provider_identifier(self.fallback) if self.fallback is not None else None
        )
        meta["representation"] = meta.get("representation") or "candidate_b_compact"
        meta["status"] = "failure"
        meta["error_type"] = type(primary_exc).__name__
        self.last_usage_metadata = meta

    def _record_fallback_success(self, primary_exc: Exception) -> None:
        raw_fallback_meta = getattr(self.fallback, "last_usage_metadata", None)
        meta = dict(raw_fallback_meta) if isinstance(raw_fallback_meta, dict) else {}
        meta["fallback_invoked"] = True
        meta["primary_provider"] = _get_provider_identifier(self.primary)
        meta["fallback_provider"] = meta.get("provider") or _get_provider_identifier(self.fallback)
        meta["primary_error"] = type(primary_exc).__name__
        meta["representation"] = meta.get("representation") or "candidate_b_compact"
        meta["status"] = meta.get("status") or "success"
        self.last_usage_metadata = meta

    def _record_both_failure(self, primary_exc: Exception, fallback_exc: Exception) -> None:
        raw_fallback_meta = getattr(self.fallback, "last_usage_metadata", None)
        meta = dict(raw_fallback_meta) if isinstance(raw_fallback_meta, dict) else {}
        meta["status"] = "failure"
        meta["fallback_invoked"] = True
        meta["primary_provider"] = _get_provider_identifier(self.primary)
        meta["fallback_provider"] = meta.get("provider") or _get_provider_identifier(self.fallback)
        meta["primary_error"] = type(primary_exc).__name__
        meta["fallback_error"] = type(fallback_exc).__name__
        meta["representation"] = meta.get("representation") or "candidate_b_compact"
        meta["error_type"] = "SemanticExtractionError"
        self.last_usage_metadata = meta
