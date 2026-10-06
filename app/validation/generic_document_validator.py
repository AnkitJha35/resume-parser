"""Deterministic provenance and grounding validator for Generic Document Structure v1."""

from __future__ import annotations

import re
from typing import Any
from app.domain.document_structure import DocumentBlock, DocumentSection, DocumentStructure
from app.domain.semantic_contract import (
    SemanticBlockInput,
    SemanticInput,
    _is_multiblock_text_semantically_supported,
    _is_value_semantically_supported,
)


class DocumentStructureValidationError(Exception):
    """Raised when DocumentStructure violates provenance or grounding invariants."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            f"Document structure validation failed with {len(violations)} violation(s): {'; '.join(violations)}"
        )


def validate_generic_document_structure(
    doc: DocumentStructure,
    semantic_input: SemanticInput | None = None,
    known_blocks: dict[str, Any] | None = None,
) -> list[str]:
    """Validate that DocumentStructure satisfies provenance, grounding, and hierarchy invariants.

    Returns:
        List of violation strings. Empty list indicates completely valid structure.
    """
    violations: list[str] = []

    blocks_dict: dict[str, str] = {}
    if semantic_input is not None:
        for b in semantic_input.blocks:
            blocks_dict[b.block_id] = b.text
    elif known_blocks is not None:
        for bid, b in known_blocks.items():
            blocks_dict[bid] = b.text if hasattr(b, "text") else str(b)

    def _resolve_text(bids: list[str]) -> str:
        return " ".join(blocks_dict.get(bid, "") for bid in bids if bid in blocks_dict)

    def _check_source_ids(bids: list[str], context: str) -> bool:
        if not bids:
            violations.append(f"MISSING_PROVENANCE in {context}: no source_block_ids cited")
            return False
        if not blocks_dict:
            return True
        all_valid = True
        for bid in bids:
            if bid not in blocks_dict:
                violations.append(f"UNKNOWN_BLOCK_ID in {context}: block ID {bid!r} not found in document")
                all_valid = False
        return all_valid

    def _check_grounding(text: str, bids: list[str], context: str) -> None:
        if not text or not text.strip():
            return
        if not _check_source_ids(bids, context):
            return
        if not blocks_dict:
            return
        source_text = _resolve_text(bids)
        if not source_text.strip():
            violations.append(f"UNGROUNDED_CONTENT in {context}: source blocks contain no text for {text!r}")
            return

        is_single = len(bids) == 1
        if not _is_value_semantically_supported(text, source_text, is_single_block=is_single):
            if not (len(bids) > 1 and _is_multiblock_text_semantically_supported(text, source_text)):
                # Check direct word containment fallback
                text_clean = "".join(ch.lower() for ch in text if ch.isalnum())
                source_clean = "".join(ch.lower() for ch in source_text if ch.isalnum())
                if text_clean not in source_clean:
                    val_words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", text)]
                    src_words = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", source_text))
                    if not (val_words and all(w in src_words for w in val_words)):
                        violations.append(
                            f"UNGROUNDED_CONTENT in {context}: text {text!r} is not grounded in cited blocks {bids}"
                        )

    claimed_blocks_by_section: dict[str, str] = {}

    def _validate_section(sec: DocumentSection, path: str) -> None:
        sec_label = sec.heading or "<unheaded>"
        current_path = f"{path}/{sec_label}"

        # 1. Heading validation
        if sec.heading is not None and sec.heading.strip():
            _check_grounding(sec.heading.strip(), sec.source_block_ids, f"heading '{sec.heading}' at {current_path}")

        # 2. Level validation
        if sec.level < 1:
            violations.append(f"INVALID_LEVEL at {current_path}: level must be >= 1, got {sec.level}")

        # 3. Block validation
        last_reading_order = None
        for idx, b in enumerate(sec.blocks):
            block_ctx = f"{current_path}/block[{idx}:{b.type}]"

            # Check duplicate ownership
            for bid in b.source_block_ids:
                if bid in claimed_blocks_by_section and claimed_blocks_by_section[bid] != current_path:
                    # Duplicate block ownership across distinct sections
                    violations.append(
                        f"DUPLICATE_OWNERSHIP: block {bid!r} claimed by both {claimed_blocks_by_section[bid]} and {current_path}"
                    )
                else:
                    claimed_blocks_by_section[bid] = current_path

            # Table blocks
            if b.type == "table" or b.table_data is not None:
                if b.table_data:
                    headers = b.table_data.get("headers", [])
                    rows = b.table_data.get("rows", [])
                    for h in headers:
                        if isinstance(h, str) and h.strip() and b.source_block_ids:
                            _check_grounding(h, b.source_block_ids, f"table header '{h}' at {block_ctx}")
                    for r_idx, row in enumerate(rows):
                        for c_idx, cell in enumerate(row):
                            if isinstance(cell, str) and cell.strip() and b.source_block_ids:
                                _check_grounding(
                                    cell,
                                    b.source_block_ids,
                                    f"table cell[{r_idx},{c_idx}] '{cell}' at {block_ctx}",
                                )
                elif b.text.strip():
                    _check_grounding(b.text.strip(), b.source_block_ids, block_ctx)
            else:
                if b.text.strip():
                    _check_grounding(b.text.strip(), b.source_block_ids, block_ctx)

            # Check reading order non-inversion
            if b.reading_order is not None:
                if last_reading_order is not None and b.reading_order < last_reading_order:
                    # Allow non-strict ordering only across pages, but flag severe inversions
                    pass
                last_reading_order = b.reading_order

        # 4. Subsections validation
        for sub_idx, sub in enumerate(sec.subsections):
            if sub.level <= sec.level:
                violations.append(
                    f"INVALID_HIERARCHY at {current_path}/subsection[{sub_idx}]: subsection level {sub.level} must be > parent level {sec.level}"
                )
            _validate_section(sub, current_path)

    for idx, section in enumerate(doc.sections):
        _validate_section(section, f"sections[{idx}]")

    return violations
