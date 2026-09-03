"""Provider-independent semantic extraction pipeline service."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.domain.document import Document
from app.domain.resume import Resume
from app.domain.semantic_contract import (
    build_semantic_input,
    semantic_output_to_resume,
    validate_semantic_output,
)
from app.extractors.semantic_extractor import (
    MockSemanticExtractor,
    SemanticExtractionError,
    SemanticExtractor,
    SemanticValidationError,
)

logger = logging.getLogger(__name__)


def _extract_usage_metadata(extractor: SemanticExtractor) -> dict[str, Any]:
    """Safely extract provider usage metadata without exposing secrets or PII."""
    raw_meta = getattr(extractor, "last_usage_metadata", None)
    provider = "unknown"
    model = "unknown"

    if isinstance(raw_meta, dict) and raw_meta.get("provider"):
        provider = str(raw_meta["provider"])
    elif isinstance(raw_meta, dict) and raw_meta.get("primary_provider"):
        provider = str(raw_meta["primary_provider"])
    else:
        cname = type(extractor).__name__.lower()
        if "gemini" in cname:
            provider = "gemini"
        elif "ollama" in cname:
            provider = "ollama"
        elif "mock" in cname:
            provider = "mock"
        else:
            provider = type(extractor).__name__

    if isinstance(raw_meta, dict) and raw_meta.get("model"):
        model = str(raw_meta["model"])
    elif hasattr(extractor, "_explicit_model") and getattr(extractor, "_explicit_model"):
        model = str(getattr(extractor, "_explicit_model"))

    if isinstance(raw_meta, dict):
        return {
            "provider": provider,
            "model": model,
            "representation": raw_meta.get("representation") or "candidate_b_compact",
            "prompt_tokens": raw_meta.get("prompt_tokens"),
            "output_tokens": raw_meta.get("output_tokens"),
            "total_tokens": raw_meta.get("total_tokens"),
            "latency_ms": raw_meta.get("latency_ms"),
            "fallback_invoked": raw_meta.get("fallback_invoked", False),
            "fallback_provider": raw_meta.get("fallback_provider"),
            "status": raw_meta.get("status", "success"),
        }
    return {
        "provider": provider,
        "model": model,
        "representation": "candidate_b_compact",
        "prompt_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "latency_ms": None,
        "fallback_invoked": False,
        "fallback_provider": None,
        "status": "success",
    }


def parse_document_semantically(
    document: Document,
    extractor: SemanticExtractor,
    document_id: str = "doc-1",
) -> Resume:
    """Execute provider-independent semantic extraction on a layout Document.

    Pipeline seam:
    Document IR
      -> build_semantic_input(document, document_id)
      -> extractor.extract(semantic_input)
      -> validate_semantic_output(output, semantic_input)
      -> semantic_output_to_resume(output)

    Raises:
        SemanticValidationError: If proposed output violates deterministic validation invariants.
    """
    start_time = time.monotonic()
    page_count = len(document.pages)
    semantic_input = build_semantic_input(document, document_id=document_id)
    block_count = len(semantic_input.blocks)

    logger.info(
        "Semantic extraction started doc_id=%s page_count=%d block_count=%d representation=candidate_b_compact",
        document_id,
        page_count,
        block_count,
    )

    try:
        output = extractor.extract(semantic_input)
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        meta = _extract_usage_metadata(extractor)
        logger.error(
            "Semantic extraction failed doc_id=%s provider=%s model=%s error_type=%s latency_ms=%.2f fallback_invoked=%s",
            document_id,
            meta.get("provider"),
            meta.get("model"),
            type(exc).__name__,
            elapsed_ms,
            meta.get("fallback_invoked"),
        )
        raise

    meta = _extract_usage_metadata(extractor)
    violations = validate_semantic_output(output, semantic_input)
    if violations:
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.warning(
            "Semantic validation failed doc_id=%s provider=%s model=%s violation_count=%d violations=%s latency_ms=%.2f",
            document_id,
            meta.get("provider"),
            meta.get("model"),
            len(violations),
            [v.split(":")[0] for v in violations],
            elapsed_ms,
        )
        raise SemanticValidationError(violations)

    elapsed_ms = (time.monotonic() - start_time) * 1000.0
    logger.info(
        "Semantic extraction completed doc_id=%s provider=%s model=%s representation=%s latency_ms=%.2f prompt_tokens=%s output_tokens=%s total_tokens=%s fallback_invoked=%s",
        document_id,
        meta.get("provider"),
        meta.get("model"),
        meta.get("representation", "candidate_b_compact"),
        elapsed_ms,
        meta.get("prompt_tokens"),
        meta.get("output_tokens"),
        meta.get("total_tokens"),
        meta.get("fallback_invoked"),
    )

    return semantic_output_to_resume(output)
