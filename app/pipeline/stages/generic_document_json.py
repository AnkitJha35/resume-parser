"""Generic Document JSON Output v1.

Serializes internal DocumentStructure domain models into a clean, lossless,
domain-agnostic generic JSON representation.
"""

from __future__ import annotations

import re
from typing import Any

from app.domain.document_structure import (
    DocumentBlock,
    DocumentSection,
    DocumentStructure,
)

_FIELD_SPLIT_RE = re.compile(r"^([^:\n]{1,45})\s*:\s*(.+)$", re.DOTALL)


def serialize_document_structure_to_json(doc: DocumentStructure) -> dict[str, Any]:
    """Deterministically serialize DocumentStructure to generic JSON without mutation.

    Preserves exact headings, hierarchical levels, blocks, tables, lists, fields,
    geometry, and source block provenance.
    """
    sections = [_serialize_section(s) for s in doc.sections]

    meta = dict(doc.metadata) if doc.metadata else {}
    doc_dict: dict[str, Any] = {
        "pages": doc.page_count,
        "sections": sections,
    }
    if meta:
        doc_dict["metadata"] = meta

    return {"document": doc_dict}


def _serialize_section(section: DocumentSection) -> dict[str, Any]:
    """Serialize a single DocumentSection including blocks and subsections."""
    serialized_blocks = _serialize_blocks(section.blocks)

    sec_dict: dict[str, Any] = {
        "heading": section.heading,
        "level": section.level,
        "blocks": serialized_blocks,
        "source_block_ids": list(section.source_block_ids),
    }

    if section.subsections:
        sec_dict["subsections"] = [_serialize_section(sub) for sub in section.subsections]

    return sec_dict


def _serialize_blocks(blocks: list[DocumentBlock]) -> list[dict[str, Any]]:
    """Serialize a sequence of DocumentBlocks into generic JSON block types."""
    result: list[dict[str, Any]] = []
    idx = 0
    while idx < len(blocks):
        b = blocks[idx]

        # Group consecutive list items into a single list block
        if b.type == "list_item":
            list_items: list[DocumentBlock] = []
            while idx < len(blocks) and blocks[idx].type == "list_item":
                list_items.append(blocks[idx])
                idx += 1
            result.append(_serialize_list_block(list_items))
            continue

        serialized = _serialize_single_block(b)
        if serialized is not None:
            result.append(serialized)
        idx += 1

    return result


def _serialize_list_block(items: list[DocumentBlock]) -> dict[str, Any]:
    """Serialize consecutive list items into a generic list block."""
    all_bids: list[str] = []
    text_items: list[str] = []
    detailed_items: list[dict[str, Any]] = []

    for item in items:
        clean_text = item.text.strip()
        text_items.append(clean_text)
        detailed_items.append({
            "text": clean_text,
            "source_block_ids": list(item.source_block_ids),
        })
        all_bids.extend(item.source_block_ids)

    return {
        "type": "list",
        "items": text_items,
        "item_details": detailed_items,
        "source_block_ids": sorted(set(all_bids)),
        "page": items[0].page_number if items else None,
    }


def _serialize_single_block(block: DocumentBlock) -> dict[str, Any] | None:
    """Serialize an individual DocumentBlock based on its type and content."""
    # 1. Table blocks
    if block.type == "table":
        return _serialize_table_block(block)

    # 2. Heading blocks (within section body)
    if block.type == "heading":
        return {
            "type": "heading",
            "text": block.text,
            "level": block.metadata.get("level", 2),
            "source_block_ids": list(block.source_block_ids),
            "page": block.page_number,
        }

    # 3. Field blocks (key-value patterns)
    txt = block.text.strip()
    field_match = _FIELD_SPLIT_RE.match(txt)
    # Check if this text block is structured as a clear field label: value
    if field_match and not any(bullet in txt for bullet in ("•", "▪", "–", "-")):
        label = field_match.group(1).strip()
        val = field_match.group(2).strip()
        # Verify label looks like a legitimate field key (1-5 words, no newlines)
        if 1 <= len(label.split()) <= 5 and "\n" not in label:
            return {
                "type": "field",
                "label": label,
                "value": val,
                "text": txt,
                "source_block_ids": list(block.source_block_ids),
                "page": block.page_number,
            }

    # 4. Standard paragraph / text block
    b_type = "paragraph" if block.type in ("paragraph", "body") else block.type
    return {
        "type": b_type,
        "text": block.text,
        "source_block_ids": list(block.source_block_ids),
        "page": block.page_number,
    }


def _clean_form_cell_value(val: str) -> str:
    """Clean structural separator colons from form table cells while preserving multiline text.

    Separator colons (e.g. standalone ':' lines, trailing '\n:', leading ': ') are stripped.
    Meaningful source text containing colons (e.g. 'Time: 14:30', ratios, URLs) is preserved.
    """
    if not val:
        return ""
    lines = [line.strip() for line in val.split("\n")]
    meaningful_lines: list[str] = []
    for line in lines:
        if line == ":":
            continue
        if line.startswith(":"):
            line = line.lstrip(":").strip()
        if line:
            meaningful_lines.append(line)
    return "\n".join(meaningful_lines)


def _extract_form_fields_from_spatial(tbl_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract canonical multiline form fields from spatial rows.

    Handles:
    - Continuation lines with leading colons (e.g. ': PRTIKSHA', ': VISHNIUPURI')
      by appending them to the horizontally aligned active field with '\n'.
    - Standalone separator colons (e.g. ':') without creating fake value lines.
    - Same-line key-value pairs (e.g. 'GT' and ': 51747').
    - Accumulates full source block IDs provenance across all line fragments.
    """
    spatial_rows = tbl_data.get("spatial_rows", [])
    headers = tbl_data.get("headers", [])
    headers_set = set(h.strip() for h in headers if h and h.strip())
    active_row_fields: list[dict[str, Any]] | None = None
    all_fields: list[dict[str, Any]] = []

    for r_idx, r in enumerate(spatial_rows):
        fields = r.get("fields", [])
        if not fields:
            continue

        # If row 0 is a pure column header row matching headers, skip from form fields
        if r_idx == 0 and not any(":" in f.get("text", "") for f in fields):
            if all(f.get("text", "").strip() in headers_set for f in fields):
                continue

        # Check if continuation row (all items start with ':' or are purely ':')
        is_continuation = all(
            f.get("text", "").strip().startswith(":") or f.get("text", "").strip() == ":"
            for f in fields
        )

        if is_continuation and active_row_fields is not None:
            for f in fields:
                fx = f.get("relative_x", 0.0)
                closest_field = min(active_row_fields, key=lambda af: abs(af["relative_x"] - fx))
                cont_text = f.get("text", "").strip()
                cont_val = cont_text.lstrip(":").strip()
                if cont_val:
                    if closest_field["value"]:
                        closest_field["value"] += "\n" + cont_val
                    else:
                        closest_field["value"] = cont_val
                closest_field["source_block_ids"].extend(f.get("source_block_ids", []))
            continue

        row_fields: list[dict[str, Any]] = []
        i = 0
        while i < len(fields):
            curr = fields[i]
            ctxt = curr.get("text", "").strip()
            cbids = list(curr.get("source_block_ids", []))
            cx = curr.get("relative_x", 0.0)

            # Case: curr is label without colon, and next field starts with colon (e.g. 'GT' and ': 51747')
            if ":" not in ctxt and (i + 1 < len(fields)) and fields[i + 1].get("text", "").strip().startswith(":"):
                nxt = fields[i + 1]
                label = ctxt
                val = nxt.get("text", "").strip().lstrip(":").strip()
                bids = cbids + list(nxt.get("source_block_ids", []))
                row_fields.append({
                    "label": label,
                    "value": val,
                    "relative_x": cx,
                    "source_block_ids": bids,
                })
                i += 2
            elif ":" in ctxt:
                parts = ctxt.split(":", 1)
                label = parts[0].strip()
                val = parts[1].strip()
                row_fields.append({
                    "label": label,
                    "value": val,
                    "relative_x": cx,
                    "source_block_ids": cbids,
                })
                i += 1
            else:
                row_fields.append({
                    "label": ctxt,
                    "value": "",
                    "relative_x": cx,
                    "source_block_ids": cbids,
                })
                i += 1

        active_row_fields = row_fields
        all_fields.extend(row_fields)

    # Return clean canonical form fields
    return [
        {
            "label": f["label"],
            "value": f["value"],
            "source_block_ids": f["source_block_ids"],
        }
        for f in all_fields
        if f["label"] or f["value"]
    ]


def _serialize_table_block(block: DocumentBlock) -> dict[str, Any]:
    """Serialize a table block preserving headers, rows, columns, and geometry."""
    tbl_data = block.table_data or {}
    raw_headers = list(tbl_data.get("headers", []))
    raw_rows = tbl_data.get("rows", [])
    is_form = bool(tbl_data.get("is_form_layout"))

    if is_form:
        headers = [_clean_form_cell_value(h) for h in raw_headers]
        rows = [[_clean_form_cell_value(c) for c in r] for r in raw_rows]
    else:
        headers = raw_headers
        rows = [list(r) for r in raw_rows]

    num_cols = tbl_data.get("num_columns") or (len(headers) if headers else (max((len(r) for r in rows), default=0)))

    table_dict: dict[str, Any] = {
        "type": "table",
        "headers": headers,
        "rows": rows,
        "columns": num_cols,
        "source_block_ids": list(block.source_block_ids),
        "page": block.page_number,
    }

    # Visual geometry
    vg = tbl_data.get("visual_geometry")
    if vg and isinstance(vg, dict):
        outer = vg.get("outer_bounds")
        if outer and len(outer) == 4:
            x0, y0, x1, y1 = outer
            table_dict["geometry"] = {
                "page": block.page_number,
                "x": round(float(x0), 2),
                "y": round(float(y0), 2),
                "width": round(float(x1 - x0), 2),
                "height": round(float(y1 - y0), 2),
            }

    # Form fields extraction for form-like tables
    form_fields: list[dict[str, Any]] = []
    if is_form and tbl_data.get("spatial_rows"):
        form_fields = _extract_form_fields_from_spatial(tbl_data)
    elif tbl_data.get("spatial_blocks"):
        for sb in tbl_data["spatial_blocks"]:
            stxt = sb.get("text", "").strip()
            sm = _FIELD_SPLIT_RE.match(stxt)
            if sm:
                flabel = sm.group(1).strip()
                fval = _clean_form_cell_value(sm.group(2).strip())
                form_fields.append({
                    "label": flabel,
                    "value": fval,
                    "source_block_ids": sb.get("source_block_ids", []),
                })
    elif rows and not headers and num_cols == 2:
        # 2-column key-value grid without formal headers
        for r in rows:
            if len(r) >= 2 and r[0].strip() and r[1].strip():
                form_fields.append({
                    "label": r[0].rstrip(":").strip(),
                    "value": r[1].strip(),
                })

    if form_fields:
        table_dict["form_fields"] = form_fields
        table_dict["records"] = [{"fields": form_fields}]

    return table_dict


# Alias for explicit internal provenance serialization
serialize_document_structure_to_provenance_json = serialize_document_structure_to_json


# ==============================================================================
# CLEAN USER-FACING DATA JSON SERIALIZATION
# ==============================================================================

def serialize_document_structure_to_data_json(doc: DocumentStructure) -> dict[str, Any]:
    """Deterministically serialize DocumentStructure to clean user-facing Data JSON.

    Contains ONLY extracted document content: sections, headings, paragraphs,
    lists, tables, headers, rows, columns, multiline values, and canonical form fields.
    Contains NO source_block_ids, geometry, block-level page, provenance fields,
    parser diagnostics, or internal records/IDs.
    """
    sections = [_serialize_clean_data_section(s) for s in doc.sections]
    doc_dict: dict[str, Any] = {
        "pages": doc.page_count,
        "sections": sections,
    }
    if doc.metadata:
        doc_dict["metadata"] = dict(doc.metadata)

    return {"document": doc_dict}


def _serialize_clean_data_section(section: DocumentSection) -> dict[str, Any]:
    """Serialize a single DocumentSection without provenance metadata."""
    sec_dict: dict[str, Any] = {
        "heading": section.heading,
        "level": section.level,
        "blocks": _serialize_clean_data_blocks(section.blocks),
    }

    if section.subsections:
        sec_dict["subsections"] = [_serialize_clean_data_section(sub) for sub in section.subsections]

    return sec_dict


def _serialize_clean_data_blocks(blocks: list[DocumentBlock]) -> list[dict[str, Any]]:
    """Serialize blocks into clean user-facing JSON blocks without provenance."""
    result: list[dict[str, Any]] = []
    idx = 0
    while idx < len(blocks):
        b = blocks[idx]

        # Group consecutive list items into a single list block
        if b.type == "list_item":
            list_items: list[DocumentBlock] = []
            while idx < len(blocks) and blocks[idx].type == "list_item":
                list_items.append(blocks[idx])
                idx += 1
            result.append(_serialize_clean_data_list_block(list_items))
            continue

        serialized = _serialize_clean_data_single_block(b)
        if serialized is not None:
            result.append(serialized)
        idx += 1

    return result


def _serialize_clean_data_list_block(items: list[DocumentBlock]) -> dict[str, Any]:
    """Serialize consecutive list items into clean generic list block."""
    return {
        "type": "list",
        "items": [item.text.strip() for item in items],
    }


def _serialize_clean_data_single_block(block: DocumentBlock) -> dict[str, Any] | None:
    """Serialize an individual DocumentBlock into clean user-facing format."""
    # 1. Table blocks
    if block.type == "table":
        return _serialize_clean_data_table_block(block)

    # 2. Heading blocks
    if block.type == "heading":
        return {
            "type": "heading",
            "text": block.text,
            "level": block.metadata.get("level", 2),
        }

    # 3. Field blocks (key-value patterns)
    txt = block.text.strip()
    field_match = _FIELD_SPLIT_RE.match(txt)
    if field_match and not any(bullet in txt for bullet in ("•", "▪", "–", "-")):
        label = field_match.group(1).strip()
        val = field_match.group(2).strip()
        if 1 <= len(label.split()) <= 5 and "\n" not in label:
            return {
                "type": "field",
                "label": label,
                "value": val,
            }

    # 4. Standard paragraph / text block
    b_type = "paragraph" if block.type in ("paragraph", "body") else block.type
    return {
        "type": b_type,
        "text": block.text,
    }


def _serialize_clean_data_table_block(block: DocumentBlock) -> dict[str, Any]:
    """Serialize a table block preserving headers, rows, columns, and canonical form_fields.

    Contains NO geometry, NO page, NO source_block_ids, and NO internal records.
    """
    tbl_data = block.table_data or {}
    raw_headers = list(tbl_data.get("headers", []))
    raw_rows = tbl_data.get("rows", [])
    is_form = bool(tbl_data.get("is_form_layout"))

    if is_form:
        headers = [_clean_form_cell_value(h) for h in raw_headers]
        rows = [[_clean_form_cell_value(c) for c in r] for r in raw_rows]
    else:
        headers = raw_headers
        rows = [list(r) for r in raw_rows]

    num_cols = tbl_data.get("num_columns") or (
        len(headers) if headers else (max((len(r) for r in rows), default=0))
    )

    table_dict: dict[str, Any] = {
        "type": "table",
        "headers": headers,
        "rows": rows,
        "columns": num_cols,
    }

    # Form fields extraction for form-like tables (clean: ONLY label and value)
    if is_form and tbl_data.get("spatial_rows"):
        fields = _extract_form_fields_from_spatial(tbl_data)
        table_dict["form_fields"] = [
            {"label": f["label"], "value": f["value"]}
            for f in fields
        ]
    elif tbl_data.get("spatial_blocks"):
        form_fields = []
        for sb in tbl_data["spatial_blocks"]:
            stxt = sb.get("text", "").strip()
            sm = _FIELD_SPLIT_RE.match(stxt)
            if sm:
                flabel = sm.group(1).strip()
                fval = _clean_form_cell_value(sm.group(2).strip())
                form_fields.append({
                    "label": flabel,
                    "value": fval,
                })
        if form_fields:
            table_dict["form_fields"] = form_fields
    elif rows and not headers and num_cols == 2:
        form_fields = []
        for r in rows:
            if len(r) >= 2 and r[0].strip() and r[1].strip():
                form_fields.append({
                    "label": r[0].rstrip(":").strip(),
                    "value": r[1].strip(),
                })
        if form_fields:
            table_dict["form_fields"] = form_fields

    return table_dict


