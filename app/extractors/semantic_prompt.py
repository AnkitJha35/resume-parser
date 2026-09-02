"""Provider-neutral serialization, prompt definitions, and structured output parsing for semantic LLM extraction."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.domain.semantic_contract import SemanticBlockInput, SemanticInput, SemanticOutput
from app.extractors.semantic_extractor import SemanticExtractionError

SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """You are a precise, layout-aware resume data extractor.
Your task is to analyze the provided structured document blocks and extract canonical resume entities strictly grounded in the supplied text blocks.

CRITICAL GROUNDING AND PROVENANCE RULES:
1. Every non-null grounded value MUST reference the exact `source_block_ids` from which it was extracted.
2. EXTRACT ONLY supported information. DO NOT invent, hallucinate, or infer missing values.
3. DO NOT perform semantic renaming or enrichment (e.g., do not rename companies, do not expand job titles).
4. Use exact verbatim text from source blocks in `raw_value` when available. In `value`, only safe canonical normalizations are permitted (e.g., ISO dates 'YYYY-MM', phone digits, whitespace/case cleanup).
5. DO NOT treat document headers, form titles (e.g., 'APPLICATION FORM', 'Curriculum Vitae', 'Surname'), or section labels as personal names.
6. DO NOT treat table column headers (e.g., 'Ship Name', 'Period', 'S.No.', 'Documents Details') as actual company, designation, or degree values.
7. DO NOT classify referee or reference contacts as employment experience. Mark referee and boilerplate blocks explicitly under block_classifications.
8. Every boolean field (e.g., `current`) MUST reference the source block IDs providing evidence (e.g., blocks containing 'Present' or 'Current').
9. If evidence for a field is absent or ambiguous, return null or empty list rather than guessing.
10. Return a single valid JSON object adhering strictly to the SemanticOutput schema."""


def serialize_semantic_input(semantic_input: SemanticInput) -> str:
    """Deterministically serialize SemanticInput into a compact JSON string.

    Preserves exact block IDs, coordinates, style hints, reading order, suggested
    roles, and table metadata without redundant block copies.
    """
    # Sort blocks deterministically by page, reading_order, and block_id
    sorted_blocks = sorted(
        semantic_input.blocks,
        key=lambda b: (b.page, b.reading_order, b.block_id),
    )

    payload: dict[str, Any] = {
        "document_id": semantic_input.document_id,
        "page_count": semantic_input.page_count,
        "pages": [
            {
                "page_number": p.page_number,
                "width": p.width,
                "height": p.height,
            }
            for p in sorted(semantic_input.pages, key=lambda p: p.page_number)
        ],
        "blocks": [
            {
                "block_id": b.block_id,
                "text": b.text,
                "page": b.page,
                "bbox": b.bbox,
                "region_id": b.region_id,
                "region_kind": b.region_kind,
                "reading_order": b.reading_order,
                "column_id": b.column_id,
                "is_bold": b.is_bold,
                "font_size": b.font_size,
                "suggested_role": b.suggested_role,
                "table_id": b.table_id,
                "row_index": b.row_index,
                "column_index": b.column_index,
                "cell_role": b.cell_role,
            }
            for b in sorted_blocks
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_extraction_prompt(semantic_input: SemanticInput) -> str:
    """Construct the complete extraction prompt combining instructions and serialized input."""
    serialized_input = serialize_semantic_input(semantic_input)
    return (
        f"{SEMANTIC_EXTRACTION_SYSTEM_PROMPT}\n\n"
        f"DOCUMENT BLOCKS (JSON):\n"
        f"```json\n{serialized_input}\n```\n\n"
        f"Extract the resume data as a JSON object adhering strictly to the SemanticOutput schema."
    )


def _strip_markdown_code_fences(text: str) -> str:
    """Strip ```json ... ``` code fences if model enclosed JSON response at the boundary.

    Does NOT accept arbitrary conversational prose surrounding the JSON.
    """
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def parse_semantic_output(raw_json_or_text: str | dict[str, Any]) -> SemanticOutput:
    """Parse raw model JSON response into the constrained SemanticOutput domain model.

    Raises:
        SemanticExtractionError: If JSON is malformed or structure violates SemanticOutput schema.
    """
    if isinstance(raw_json_or_text, str):
        cleaned_text = _strip_markdown_code_fences(raw_json_or_text)
        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as err:
            raise SemanticExtractionError(f"Malformed JSON response from LLM: {err}") from err
    elif isinstance(raw_json_or_text, dict):
        parsed = raw_json_or_text
    else:
        raise SemanticExtractionError(
            f"Expected JSON string or dict, got {type(raw_json_or_text).__name__}"
        )

    if not isinstance(parsed, dict):
        raise SemanticExtractionError(
            f"Expected JSON object at root of LLM response, got {type(parsed).__name__}"
        )

    try:
        return SemanticOutput.model_validate(parsed)
    except ValidationError as err:
        raise SemanticExtractionError(f"Invalid SemanticOutput structure from LLM: {err}") from err
