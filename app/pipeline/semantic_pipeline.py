"""Provider-independent semantic extraction pipeline service."""

from __future__ import annotations

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
    semantic_input = build_semantic_input(document, document_id=document_id)
    output = extractor.extract(semantic_input)
    violations = validate_semantic_output(output, semantic_input)
    if violations:
        raise SemanticValidationError(violations)
    return semantic_output_to_resume(output)
