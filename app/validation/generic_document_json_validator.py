"""Generic JSON Fidelity Validator.

Validates that document_json faithfully represents the extracted source blocks
and DocumentStructure with zero missing blocks, zero duplicate ownership, zero
unsupported blocks, zero provenance errors, and exact text grounding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.domain.document_structure import DocumentBlock, DocumentSection, DocumentStructure
from app.pipeline.stages.generic_document_json import _clean_form_cell_value

_SUPPORTED_BLOCK_TYPES = {
    "paragraph",
    "body",
    "heading",
    "list_item",
    "list",
    "table",
    "field",
    "text",
}


@dataclass
class JsonFidelityReport:
    """Diagnostic fidelity report for a serialized document_json."""

    valid: bool
    source_blocks: int
    json_owned_blocks: int
    missing_blocks: int
    duplicate_blocks: int
    unsupported_blocks: int
    provenance_errors: int
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to standard serializable dictionary."""
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


def validate_document_json(
    document_structure: DocumentStructure,
    document_json: dict[str, Any],
) -> JsonFidelityReport:
    """Validate document_json against DocumentStructure.

    Ensures that every meaningful source block has exactly one logical owner in
    document_json, provenance IDs are known, no blocks are unsupported, text is
    exact, and tables/lists/sections are preserved without mutation.
    """
    # Defensive check for non-mutation guarantee
    doc_before_repr = repr(document_structure)
    json_before_repr = repr(document_json)

    issues: list[str] = []
    unsupported_blocks = 0
    provenance_errors = 0

    # 1. Collect all known source blocks from DocumentStructure
    known_doc_blocks: dict[str, str] = {}  # bid -> text
    all_source_bids: set[str] = set()

    def _collect_from_doc(sections: list[DocumentSection]) -> None:
        nonlocal unsupported_blocks
        for s in sections:
            all_source_bids.update(s.source_block_ids)
            for b in s.blocks:
                if b.type not in _SUPPORTED_BLOCK_TYPES:
                    issues.append(f"UNSUPPORTED_BLOCK_TYPE: DocumentBlock type {b.type!r} is not supported")
                    unsupported_blocks += 1

                for bid in b.source_block_ids:
                    all_source_bids.add(bid)
                    known_doc_blocks[bid] = b.text

            if s.subsections:
                _collect_from_doc(s.subsections)

    _collect_from_doc(document_structure.sections)

    # 2. Extract sections from document_json
    doc_root = document_json.get("document", {}) if isinstance(document_json, dict) else {}
    json_sections = doc_root.get("sections", []) if isinstance(doc_root, dict) else []

    # Check if document_json is Clean Data JSON (no source_block_ids in sections)
    is_clean_data = not any("source_block_ids" in s for s in json_sections)

    # 3. Ownership and Duplicate Tracking
    # Every logical owner is either a section heading or a content block (paragraph, table, list, field, etc.)
    ownership: dict[str, str] = {}  # bid -> owner path
    duplicate_bids: set[str] = set()

    def _register_owner(bid: str, owner_path: str) -> None:
        nonlocal provenance_errors
        if bid not in all_source_bids:
            issues.append(f"UNKNOWN_PROVENANCE_ID: Block ID {bid!r} in {owner_path} not found in source document")
            provenance_errors += 1

        if bid in ownership:
            prior_owner = ownership[bid]
            if prior_owner != owner_path:
                duplicate_bids.add(bid)
                issues.append(
                    f"DUPLICATE_OWNERSHIP: Block {bid!r} claimed by both {prior_owner} and {owner_path}"
                )
        else:
            ownership[bid] = owner_path

    # Check sections count and hierarchy
    def _validate_sections(
        doc_secs: list[DocumentSection],
        json_secs: list[dict[str, Any]],
        path_prefix: str = "section",
    ) -> None:
        nonlocal unsupported_blocks, provenance_errors

        if len(doc_secs) != len(json_secs):
            issues.append(
                f"SECTION_COUNT_MISMATCH at {path_prefix}: DocumentStructure has {len(doc_secs)} sections, JSON has {len(json_secs)}"
            )

        for idx, (d_sec, j_sec) in enumerate(zip(doc_secs, json_secs)):
            curr_path = f"{path_prefix}[{idx}]"

            # Check forbidden metadata in clean mode
            if is_clean_data:
                for forbidden in ("source_block_ids", "geometry", "page"):
                    if forbidden in j_sec:
                        issues.append(f"FORBIDDEN_METADATA_IN_CLEAN_JSON: Section {curr_path} contains {forbidden!r}")

            # Heading text match
            if d_sec.heading != j_sec.get("heading"):
                issues.append(
                    f"HEADING_TEXT_MISMATCH at {curr_path}: expected {d_sec.heading!r}, got {j_sec.get('heading')!r}"
                )

            # Heading level match
            if d_sec.level != j_sec.get("level"):
                issues.append(
                    f"HEADING_LEVEL_MISMATCH at {curr_path}: expected level {d_sec.level}, got {j_sec.get('level')}"
                )

            # In ground truth d_sec, determine heading source block IDs:
            # Source blocks in section that do not belong to child blocks are heading blocks
            d_child_bids = set()
            for db in d_sec.blocks:
                d_child_bids.update(db.source_block_ids)
            heading_bids = set(d_sec.source_block_ids) - d_child_bids

            if is_clean_data:
                # In clean data JSON, section has no source_block_ids, so we register ownership
                # from d_sec directly to verify that every heading source block is accounted for
                for h_bid in heading_bids:
                    _register_owner(h_bid, f"{curr_path}.heading({j_sec.get('heading')})")
            else:
                # Verify that j_sec includes these heading bids in its source_block_ids
                j_sec_bids = set(j_sec.get("source_block_ids", []))
                for h_bid in heading_bids:
                    if h_bid in j_sec_bids:
                        _register_owner(h_bid, f"{curr_path}.heading({j_sec.get('heading')})")
                    else:
                        issues.append(
                            f"MISSING_HEADING_PROVENANCE at {curr_path}: heading block {h_bid!r} omitted from section source_block_ids"
                        )

            # Validate child blocks
            j_blocks = j_sec.get("blocks", [])
            _validate_blocks(d_sec.blocks, j_blocks, curr_path)

            # Recursive subsections
            _validate_sections(d_sec.subsections, j_sec.get("subsections", []), f"{curr_path}.sub")

    def _validate_blocks(
        d_blocks: list[DocumentBlock],
        j_blocks: list[dict[str, Any]],
        section_path: str,
    ) -> None:
        nonlocal unsupported_blocks

        # Build list grouping from d_blocks to match serialized blocks
        grouped_d_blocks: list[DocumentBlock | list[DocumentBlock]] = []
        d_idx = 0
        while d_idx < len(d_blocks):
            db = d_blocks[d_idx]
            if db.type == "list_item":
                items = []
                while d_idx < len(d_blocks) and d_blocks[d_idx].type == "list_item":
                    items.append(d_blocks[d_idx])
                    d_idx += 1
                grouped_d_blocks.append(items)
            else:
                grouped_d_blocks.append(db)
                d_idx += 1

        if len(grouped_d_blocks) != len(j_blocks):
            issues.append(
                f"BLOCK_COUNT_MISMATCH at {section_path}: DocumentStructure has {len(grouped_d_blocks)} logical blocks, JSON has {len(j_blocks)}"
            )

        for b_idx, (d_item, j_blk) in enumerate(zip(grouped_d_blocks, j_blocks)):
            block_path = f"{section_path}.block[{b_idx}:{j_blk.get('type')}]"

            # Check forbidden metadata in clean mode
            if is_clean_data:
                for forbidden in ("source_block_ids", "geometry", "page", "item_details", "records"):
                    if forbidden in j_blk:
                        issues.append(f"FORBIDDEN_METADATA_IN_CLEAN_JSON: Block {block_path} contains {forbidden!r}")

            # Register ownership for every source block id
            if is_clean_data:
                # In clean mode, derive ownership from the corresponding DocumentBlock(s)
                if isinstance(d_item, list):
                    for sub_d in d_item:
                        for bid in sub_d.source_block_ids:
                            _register_owner(bid, block_path)
                else:
                    for bid in d_item.source_block_ids:
                        _register_owner(bid, block_path)
            else:
                for bid in j_blk.get("source_block_ids", []):
                    _register_owner(bid, block_path)

            # If grouped list
            if isinstance(d_item, list):
                if j_blk.get("type") != "list":
                    issues.append(f"TYPE_MISMATCH at {block_path}: expected 'list', got {j_blk.get('type')!r}")
                else:
                    d_texts = [item.text.strip() for item in d_item]
                    j_texts = j_blk.get("items", [])
                    if d_texts != j_texts:
                        issues.append(f"LIST_TEXT_MISMATCH at {block_path}: expected {d_texts!r}, got {j_texts!r}")

                    if not is_clean_data:
                        # Verify item details provenance
                        for item_idx, (d_sub, j_sub) in enumerate(zip(d_item, j_blk.get("item_details", []))):
                            if list(d_sub.source_block_ids) != j_sub.get("source_block_ids", []):
                                issues.append(
                                    f"LIST_ITEM_PROVENANCE_MISMATCH at {block_path}.item[{item_idx}]: expected {d_sub.source_block_ids}, got {j_sub.get('source_block_ids')}"
                                )
                continue

            # Individual block
            d_block = d_item
            j_type = j_blk.get("type")

            if d_block.type == "table":
                if j_type != "table":
                    issues.append(f"TYPE_MISMATCH at {block_path}: expected 'table', got {j_type!r}")
                else:
                    _validate_table_block(d_block, j_blk, block_path)
            elif j_type == "field":
                # Key-value field block
                expected_txt = d_block.text.strip()
                if is_clean_data:
                    # In clean mode, field has 'label' and 'value'
                    field_label = j_blk.get("label", "").strip()
                    field_val = j_blk.get("value", "").strip()
                    if not (field_label in expected_txt and (not field_val or field_val in expected_txt)):
                        issues.append(
                            f"FIELD_TEXT_MISMATCH at {block_path}: label={field_label!r}, val={field_val!r} not in {expected_txt!r}"
                        )
                else:
                    field_txt = j_blk.get("text", "").strip()
                    if expected_txt != field_txt:
                        issues.append(
                            f"FIELD_TEXT_MISMATCH at {block_path}: expected {expected_txt!r}, got {field_txt!r}"
                        )
            elif j_type in ("paragraph", "heading", "text"):
                # Exact text check
                if d_block.text != j_blk.get("text", ""):
                    issues.append(
                        f"TEXT_MISMATCH at {block_path}: expected {d_block.text!r}, got {j_blk.get('text')!r}"
                    )
            else:
                issues.append(f"UNKNOWN_JSON_BLOCK_TYPE at {block_path}: {j_type!r}")
                unsupported_blocks += 1

    def _validate_table_block(
        d_block: DocumentBlock,
        j_block: dict[str, Any],
        block_path: str,
    ) -> None:
        tbl_data = d_block.table_data or {}
        is_form = bool(tbl_data.get("is_form_layout"))

        # 1. Header matching
        if is_form:
            expected_headers = [_clean_form_cell_value(h) for h in tbl_data.get("headers", [])]
        else:
            expected_headers = list(tbl_data.get("headers", []))
        json_headers = list(j_block.get("headers", []))
        if expected_headers != json_headers:
            issues.append(
                f"TABLE_HEADERS_MISMATCH at {block_path}: expected {expected_headers!r}, got {json_headers!r}"
            )

        # 2. Row matching
        if is_form:
            expected_rows = [[_clean_form_cell_value(c) for c in r] for r in tbl_data.get("rows", [])]
        else:
            expected_rows = [list(r) for r in tbl_data.get("rows", [])]
        json_rows = [list(r) for r in j_block.get("rows", [])]
        if expected_rows != json_rows:
            issues.append(
                f"TABLE_ROWS_MISMATCH at {block_path}: expected {len(expected_rows)} rows, got {len(json_rows)} rows"
            )

        # 3. Provenance matching (only in provenance mode)
        if not is_clean_data:
            if list(d_block.source_block_ids) != j_block.get("source_block_ids", []):
                issues.append(
                    f"TABLE_PROVENANCE_MISMATCH at {block_path}: expected {d_block.source_block_ids}, got {j_block.get('source_block_ids')}"
                )

        # 4. Standalone separator colon checks (prevent noise becoming value lines)
        for r_idx, row in enumerate(json_rows):
            for c_idx, cell in enumerate(row):
                if cell == ":":
                    issues.append(f"STANDALONE_COLON_CELL at {block_path}.rows[{r_idx}][{c_idx}]: cell is standalone ':'")
                elif "\n:\n" in cell or cell.startswith(":\n") or cell.endswith("\n:"):
                    issues.append(f"STANDALONE_COLON_LINE at {block_path}.rows[{r_idx}][{c_idx}]: cell contains standalone ':' line")

        # 5. Form fields fidelity, multiline preservation, and text coverage
        if "form_fields" in j_block:
            j_fields = j_block["form_fields"]

            # In clean mode, form_fields should NOT contain source_block_ids or internal metadata
            if is_clean_data:
                for f_idx, ff in enumerate(j_fields):
                    for forbidden in ("source_block_ids", "geometry", "page", "records"):
                        if forbidden in ff:
                            issues.append(
                                f"FORBIDDEN_METADATA_IN_CLEAN_JSON: form_fields[{f_idx}] contains {forbidden!r}"
                            )

            # Single canonical representation check between form_fields and records (if records present in provenance mode)
            if "records" in j_block:
                recs = j_block["records"]
                if recs and isinstance(recs, list) and "fields" in recs[0]:
                    if recs[0]["fields"] != j_fields:
                        issues.append(
                            f"RECORDS_FORM_FIELDS_MISMATCH at {block_path}: records[0]['fields'] differs from form_fields"
                        )

            # Check that no form field value is a standalone colon
            for f_idx, ff in enumerate(j_fields):
                fval = ff.get("value", "")
                if fval == ":":
                    issues.append(f"STANDALONE_COLON_FIELD at {block_path}.form_fields[{f_idx}]: value is standalone ':'")
                elif "\n:\n" in fval or fval.startswith(":\n") or fval.endswith("\n:"):
                    issues.append(f"STANDALONE_COLON_LINE in field at {block_path}.form_fields[{f_idx}]")

            # Check multiline preservation: if raw expected row had multiple meaningful lines in a cell,
            # verify that some form field value also preserves multiline text
            raw_rows = tbl_data.get("rows", [])
            has_multiline_raw = any(
                len([l for l in str(c).split("\n") if l.strip() and l.strip() != ":"]) > 1
                for r in raw_rows
                for c in r
            )
            if has_multiline_raw:
                has_multiline_json = any("\n" in ff.get("value", "") for ff in j_fields)
                if not has_multiline_json:
                    issues.append(
                        f"MULTILINE_FIELD_VALUE_LOST at {block_path}: multiline text in table was truncated to single line"
                    )

            # Validate normalized exact text coverage:
            # Every meaningful source text token in spatial_blocks must be present in the table representation
            table_text_corpus = " ".join(
                [h for h in json_headers]
                + [c for row in json_rows for c in row]
                + [ff.get("label", "") + " " + ff.get("value", "") for ff in j_fields]
            )

            for sb in tbl_data.get("spatial_blocks", []):
                sb_text = sb.get("text", "").strip()
                if not sb_text:
                    continue
                lines = [l.strip() for l in sb_text.split("\n")]
                for line in lines:
                    cleaned_line = line.strip(" :-\t|•▪–")
                    if not cleaned_line:
                        continue
                    if cleaned_line not in table_text_corpus:
                        for w in cleaned_line.split():
                            clean_w = w.strip(" :-\t|•▪–,()")
                            if clean_w and clean_w not in table_text_corpus:
                                issues.append(
                                    f"MISSING_SOURCE_TEXT at {block_path}: meaningful text {clean_w!r} is missing from table representation"
                                )


    # Execute section and block validation
    _validate_sections(document_structure.sections, json_sections)

    # 4. Check Source Coverage (missing blocks)
    missing_bids = sorted(all_source_bids - set(ownership.keys()))
    if missing_bids:
        issues.append(f"MISSING_SOURCE_BLOCKS: {len(missing_bids)} block(s) have no JSON owner: {missing_bids[:10]}")

    # 5. Non-mutation check
    assert repr(document_structure) == doc_before_repr, "DocumentStructure was mutated during validation"
    assert repr(document_json) == json_before_repr, "document_json was mutated during validation"

    total_source = len(all_source_bids)
    json_owned = len(ownership)
    num_missing = len(missing_bids)
    num_duplicates = len(duplicate_bids)

    is_valid = (
        num_missing == 0
        and num_duplicates == 0
        and unsupported_blocks == 0
        and provenance_errors == 0
        and len(issues) == 0
    )

    return JsonFidelityReport(
        valid=is_valid,
        source_blocks=total_source,
        json_owned_blocks=json_owned,
        missing_blocks=num_missing,
        duplicate_blocks=num_duplicates,
        unsupported_blocks=unsupported_blocks,
        provenance_errors=provenance_errors,
        issues=issues,
    )
