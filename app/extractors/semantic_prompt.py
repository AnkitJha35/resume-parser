"""Provider-neutral serialization, prompt definitions, and structured output parsing for semantic LLM extraction."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

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


def serialize_semantic_input_full(semantic_input: SemanticInput) -> str:
    """Reference / debug serialization preserving all layout, coordinates, and full metadata.

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


# Backward-compatible alias for existing tests and debug inspection
serialize_semantic_input = serialize_semantic_input_full


def build_full_extraction_prompt(semantic_input: SemanticInput) -> str:
    """Construct reference extraction prompt combining instructions and full serialized input."""
    serialized_input = serialize_semantic_input_full(semantic_input)
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


def resolve_schema_defs(pydantic_model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model's JSON schema to a self-contained schema with all $defs inlined.

    Useful for providers (Ollama, Gemini) that support structured JSON output with an inlined schema.
    """
    raw_schema = pydantic_model.model_json_schema()
    defs = raw_schema.pop("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                if ref_key in defs:
                    resolved = _resolve(defs[ref_key])
                    merged = dict(resolved)
                    for k, v in node.items():
                        if k != "$ref":
                            merged[k] = _resolve(v)
                    return merged
            res = {}
            for k, v in node.items():
                if k in ("title", "$defs"):
                    continue
                res[k] = _resolve(v)
            return res
        elif isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(raw_schema)


def serialize_compact_semantic_input(semantic_input: SemanticInput) -> str:
    """Serialize SemanticInput into Candidate B compact JSON representation.

    Preserves deterministic block IDs, verbatim text, structural roles, multi-page indicators,
    and tabular grid coordinates without redundant bounding boxes, style metadata, or explicit nulls.
    """
    sorted_blocks = sorted(
        semantic_input.blocks,
        key=lambda b: (b.page, b.reading_order, b.block_id),
    )

    compact_blocks: list[dict[str, Any]] = []
    for b in sorted_blocks:
        block_dict: dict[str, Any] = {
            "id": b.block_id,
            "text": b.text,
        }
        if b.suggested_role and b.suggested_role != "UNKNOWN":
            block_dict["role"] = b.suggested_role
        if b.page > 1:
            block_dict["page"] = b.page
        if b.is_bold is True:
            block_dict["bold"] = True
        if b.table_id is not None:
            block_dict["table"] = b.table_id
            if b.row_index is not None:
                block_dict["row"] = b.row_index
            if b.column_index is not None:
                block_dict["col"] = b.column_index
            if b.cell_role and b.cell_role != "DATA":
                block_dict["cell_role"] = b.cell_role
        compact_blocks.append(block_dict)

    payload: dict[str, Any] = {
        "doc_id": semantic_input.document_id,
        "blocks": compact_blocks,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def get_compact_schema(pydantic_model: type[BaseModel] = SemanticOutput) -> dict[str, Any]:
    """Generate a compact JSON schema with descriptions and non-structural metadata stripped."""
    raw_schema = resolve_schema_defs(pydantic_model)

    def _strip_metadata(node: Any) -> Any:
        if isinstance(node, dict):
            res: dict[str, Any] = {}
            for k, v in node.items():
                if k in ("title", "description", "default", "examples", "format"):
                    continue
                res[k] = _strip_metadata(v)
            return res
        elif isinstance(node, list):
            return [_strip_metadata(item) for item in node]
        return node

    return _strip_metadata(raw_schema)


def build_compact_extraction_prompt(semantic_input: SemanticInput) -> str:
    """Construct extraction prompt using Candidate B compact input representation."""
    serialized_input = serialize_compact_semantic_input(semantic_input)
    return (
        f"{SEMANTIC_EXTRACTION_SYSTEM_PROMPT}\n\n"
        f"DOCUMENT BLOCKS (JSON):\n"
        f"```json\n{serialized_input}\n```\n\n"
        f"Extract the resume data as a JSON object adhering strictly to the SemanticOutput schema."
    )


# Production default prompt builder (Candidate B compact representation)
build_extraction_prompt = build_compact_extraction_prompt
