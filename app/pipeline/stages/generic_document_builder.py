"""Generic document structure builder from layout Document IR and SemanticInput."""

from __future__ import annotations

import re
from typing import Any

from app.domain.document_structure import DocumentBlock, DocumentSection, DocumentStructure
from app.domain.semantic_contract import SemanticBlockInput, SemanticInput
from app.pipeline.stages.structural_roles import (
    _CANONICAL_SECTION_KEYWORDS,
    _STRUCTURAL_DATE_RE,
    _TITLE_PREFIXES,
)


_BULLET_SYMBOLS_RE = re.compile(r"^[\s•\-\*▪▫‣–—●\uf0b7]+\s*")
_SUBHEADING_NUM_RE = re.compile(r"^(?:\d+\.\d+|[A-Za-z]\.|\([a-z]\))\s+")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[ \-/.]?)?(?:\(\d{2,4}\)|\d{2,4})[ \-/.]?\d{3,4}[ \-/.]?\d{3,4}")

_NON_HEADING_ROLES = frozenset({
    "BULLET", "SKILL", "LOCATION", "DATE", "TABLE_CELL",
    "CONTACT", "FOOTER", "CREDENTIAL",
})

_METADATA_PREFIXES = (
    "name:", "full name:", "candidate name:",
    "phone:", "mobile:", "tel:", "telephone:", "cell:",
    "email:", "e-mail:", "mail:",
    "address:", "permanent address:", "current address:", "present address:",
    "dob:", "date of birth:", "d.o.b:",
    "nationality:", "gender:", "sex:", "marital status:",
    "passport:", "passport no:", "passport number:",
    "visa:", "cdc no:", "rank:",
    "cgpa:", "gpa:", "percentage:", "score:",
)

_LOWERCASE_CONJ_PREP = frozenset({
    "of", "and", "from", "in", "to", "for", "with", "at", "by", "or", "on", "as", "the",
})


def _is_bullet_block(block: SemanticBlockInput) -> bool:
    """Check if block is a bullet or list item."""
    if block.suggested_role == "BULLET":
        return True
    txt = block.text.strip()
    if _BULLET_SYMBOLS_RE.match(txt) and not txt.startswith("--"):
        return True
    return False


def _clean_bullet_text(text: str) -> str:
    """Strip leading bullet marker while preserving wording."""
    return _BULLET_SYMBOLS_RE.sub("", text).strip()


def _is_centered_on_page(mid_x: float, block_width: float, page_blocks: list[SemanticBlockInput]) -> bool:
    """Check if horizontal midpoint is close to page center (standard A4/Letter or page content)."""
    if abs(mid_x - 297.6) < 15.0 or abs(mid_x - 306.0) < 15.0:
        return True
    if page_blocks:
        c_min = min(b.bbox[0] for b in page_blocks)
        c_max = max(b.bbox[2] for b in page_blocks)
        c_span = c_max - c_min
        if c_span > 100.0 and block_width < 0.75 * c_span:
            c_mid = (c_min + c_max) / 2.0
            if abs(mid_x - c_mid) < 15.0:
                return True
    return False


def _is_potential_heading(
    block: SemanticBlockInput,
    following_blocks: list[SemanticBlockInput],
    all_blocks_by_page: dict[int, list[SemanticBlockInput]] | None = None,
) -> bool:
    """Determine if a block is a section heading without relying on a fixed whitelist."""
    if block.table_id is not None:
        return False
    if _is_bullet_block(block):
        return False

    text = block.text.strip()
    if not text:
        return False

    # Punctuation prefix check (e.g. ": CDC", ": DC ENDORSEMENT NAUTICAL", leading bullets)
    if text[0] in ":;,.-—–|#*/" and not (text[0].isdigit() or (len(text) > 1 and text[:2].isdigit())):
        return False

    # Check if this block shares a physical line with preceding siblings situated to its left
    if block.suggested_role != "SECTION_HEADING" and all_blocks_by_page is not None:
        pblocks = all_blocks_by_page.get(block.page, [])
        b_y0, b_y1 = block.bbox[1], block.bbox[3]
        b_x0 = block.bbox[0]
        b_h = max(1.0, b_y1 - b_y0)
        for other in pblocks:
            if other.block_id == block.block_id:
                continue
            o_y0, o_y1 = other.bbox[1], other.bbox[3]
            o_x1 = other.bbox[2]
            o_h = max(1.0, o_y1 - o_y0)
            v_overlap = max(0.0, min(b_y1, o_y1) - max(b_y0, o_y0))
            h_line = abs(b_y0 - o_y0) < 3.0 or v_overlap > 0.4 * min(b_h, o_h)
            if h_line and o_x1 <= b_x0 + 5.0:
                # Preceded by another block on the same horizontal line (e.g. '1 . Ship Name :')
                return False

    # Summary or count lines (e.g. 'Total no of record in ...', 'Page 1 of 2')
    if re.search(r"\btotal\s+(?:no\s+of\s+)?records?\b", text, re.I):
        return False
    if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", text, re.I):
        return False

    core = text.rstrip(".:").strip()
    if not core:
        return False

    words = core.split()
    if not (1 <= len(words) <= 7) or len(core) > 60:
        return False

    # Metadata / contact prefix check
    core_lower = core.lower()
    if any(core_lower.startswith(p) for p in _METADATA_PREFIXES):
        return False

    # Timestamp / date check
    if _STRUCTURAL_DATE_RE.search(text) or re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        return False
    if "@" in text or "http" in text.lower():
        return False

    # Address indicators & contact emojis
    if any(emoji in text for emoji in ["🏠", "📞", "✉", "📱", "🌐"]):
        return False
    if re.search(r"\b(?:vill\b|dist\b|post\b|p\.o\b|pin\b)\b", text, re.IGNORECASE):
        return False

    # Lowercase start check (continuation line, not heading)
    if words[0].lower() in _LOWERCASE_CONJ_PREP and not words[0].isupper():
        return False

    # Document header / candidate name on page 1 check
    if block.page == 1 and block.reading_order <= 3 and block.suggested_role != "SECTION_HEADING":
        # Check if following blocks are contact info or precede the first SECTION_HEADING
        for fb in following_blocks[:8]:
            if fb.suggested_role in ("CONTACT", "EMAIL", "PHONE") or "@" in fb.text or "linkedin.com" in fb.text.lower():
                return False
            if _PHONE_RE.search(fb.text):
                return False
            if fb.suggested_role == "SECTION_HEADING":
                return False

    # Running header across subsequent pages
    if block.page > 1 and block.region_kind == "header" and block.suggested_role != "SECTION_HEADING":
        return False

    first_clean = re.sub(r"[^A-Za-z]", "", words[0]).lower()
    last_clean = re.sub(r"[^A-Za-z0-9]", "", words[-1]).lower()
    has_canonical_keyword = (
        first_clean in _CANONICAL_SECTION_KEYWORDS or last_clean in _CANONICAL_SECTION_KEYWORDS
    )

    # Exclude non-heading roles unless this block is explicitly a canonical section heading
    if block.suggested_role in _NON_HEADING_ROLES and not has_canonical_keyword:
        return False

    if block.suggested_role == "SECTION_HEADING":
        return True

    # Uppercase check (allow up to 7 words)
    is_upper = core.upper() == core and any(ch.isalpha() for ch in core)
    if is_upper and len(words) <= 7:
        if first_clean not in _TITLE_PREFIXES:
            return True

    # Title Case or Capitalized check
    is_title = words[0][0].isupper() and all(
        w[0].isupper() for w in words if w.lower() not in _LOWERCASE_CONJ_PREP
    )
    b_width = block.bbox[2] - block.bbox[0]
    mid_x = (block.bbox[0] + block.bbox[2]) / 2.0
    pblocks = all_blocks_by_page.get(block.page, []) if all_blocks_by_page else []
    is_centered = _is_centered_on_page(mid_x, b_width, pblocks)

    if is_title and (has_canonical_keyword or is_centered):
        if first_clean not in _TITLE_PREFIXES:
            return True

    if block.suggested_role == "HEADING" and block.region_kind not in ("header", "footer"):
        return True

    return False


def _detect_heading_level(block: SemanticBlockInput, parent_heading: SemanticBlockInput | None) -> int:
    """Determine hierarchical heading level (1 for top-level, 2 for subsection)."""
    text = block.text.strip().rstrip(".:")
    if _SUBHEADING_NUM_RE.match(text):
        return 2

    curr_is_upper = text.upper() == text and any(c.isalpha() for c in text)

    # Peer ALL CAPS headings are ALWAYS level 1
    if curr_is_upper:
        return 1

    # If parent was ALL CAPS and this one is Title Case / Mixed Case
    if parent_heading:
        p_text = parent_heading.text.strip().rstrip(".:")
        p_is_upper = p_text.upper() == p_text and any(c.isalpha() for c in p_text)
        if p_is_upper and not curr_is_upper:
            return 2

    return 1


def build_document_structure_from_semantic_input(semantic_input: SemanticInput) -> DocumentStructure:
    """Deterministically reconstruct DocumentStructure from SemanticInput.

    Preserves exact source headings, hierarchical levels, blocks, lists, and tables
    without normalizing or discarding unmapped content.
    """
    blocks = list(semantic_input.blocks)
    if not blocks:
        return DocumentStructure(sections=[], page_count=semantic_input.page_count)

    # Index tables from semantic_input
    tables_by_id: dict[str, Any] = {}
    for tbl in getattr(semantic_input, "tables", []):
        if hasattr(tbl, "table_id"):
            tables_by_id[tbl.table_id] = tbl

    # Pre-index blocks by page for spatial line checks
    all_blocks_by_page: dict[int, list[SemanticBlockInput]] = {}
    for b in blocks:
        all_blocks_by_page.setdefault(b.page, []).append(b)

    # Find heading boundaries
    heading_indices: list[int] = []
    for idx, b in enumerate(blocks):
        following = blocks[idx + 1 : idx + 6]
        if _is_potential_heading(b, following, all_blocks_by_page=all_blocks_by_page):
            # Check if this heading repeats the active section heading on a continuation page
            if heading_indices:
                last_h = blocks[heading_indices[-1]]
                if (
                    b.text.strip().lower() == last_h.text.strip().lower()
                    and b.page > last_h.page
                    and b.bbox[1] < 60.0
                ):
                    continue
            heading_indices.append(idx)

    sections: list[DocumentSection] = []

    # Index all blocks belonging to tables across the entire document
    all_table_blocks_map: dict[str, list[SemanticBlockInput]] = {}
    for b in blocks:
        if b.table_id is not None:
            all_table_blocks_map.setdefault(b.table_id, []).append(b)

    processed_table_ids: set[str] = set()

    # If no headings found in entire document
    if not heading_indices:
        unheaded_blocks = _build_blocks_from_slice(
            blocks,
            tables_by_id,
            all_table_blocks_map=all_table_blocks_map,
            processed_table_ids=processed_table_ids,
        )
        sections.append(
            DocumentSection(
                heading=None,
                level=1,
                blocks=unheaded_blocks,
                source_block_ids=[b.block_id for b in blocks],
            )
        )
        return DocumentStructure(
            sections=sections,
            page_count=semantic_input.page_count,
            metadata={"archetype": getattr(semantic_input.archetype, "value", str(semantic_input.archetype))},
        )

    # Map each table to its preceding heading (or None for unheaded before first heading)
    tables_by_heading_idx: dict[int | None, list[str]] = {}
    for tbl_id, t_blocks in all_table_blocks_map.items():
        t_page = t_blocks[0].page
        t_y0 = min(b.bbox[1] for b in t_blocks)
        best_h_idx = None
        for h_idx in heading_indices:
            h = blocks[h_idx]
            if (h.page < t_page) or (h.page == t_page and h.bbox[1] <= t_y0):
                best_h_idx = h_idx
        tables_by_heading_idx.setdefault(best_h_idx, []).append(tbl_id)

    def _build_section_blocks(
        slice_blocks: list[SemanticBlockInput],
        allowed_tbl_ids: set[str],
    ) -> list[DocumentBlock]:
        filtered = [b for b in slice_blocks if b.table_id is None or b.table_id in allowed_tbl_ids]
        c_blocks = _build_blocks_from_slice(
            filtered,
            tables_by_id,
            all_table_blocks_map=all_table_blocks_map,
            processed_table_ids=processed_table_ids,
        )
        for tid in sorted(allowed_tbl_ids):
            if tid not in processed_table_ids:
                tb = _build_blocks_from_slice(
                    all_table_blocks_map[tid],
                    tables_by_id,
                    all_table_blocks_map=all_table_blocks_map,
                    processed_table_ids=processed_table_ids,
                )
                c_blocks.extend(tb)
        return c_blocks

    # Blocks before the first heading -> unheaded section
    first_h_idx = heading_indices[0]
    if first_h_idx > 0:
        pre_blocks = blocks[:first_h_idx]
        pre_tbl_ids = set(tables_by_heading_idx.get(None, []))
        child_blocks = _build_section_blocks(pre_blocks, pre_tbl_ids)
        if child_blocks:
            sec_bids = set()
            for cb in child_blocks:
                sec_bids.update(cb.source_block_ids)
            sections.append(
                DocumentSection(
                    heading=None,
                    level=1,
                    blocks=child_blocks,
                    source_block_ids=sorted(sec_bids),
                )
            )

    # Process each heading segment
    current_l1_section: DocumentSection | None = None
    last_l1_heading_block: SemanticBlockInput | None = None

    for i, h_idx in enumerate(heading_indices):
        h_block = blocks[h_idx]
        next_boundary = heading_indices[i + 1] if i + 1 < len(heading_indices) else len(blocks)
        content_slice = blocks[h_idx + 1 : next_boundary]

        heading_text = h_block.text.strip()
        level = _detect_heading_level(h_block, last_l1_heading_block)
        sec_tbl_ids = set(tables_by_heading_idx.get(h_idx, []))

        child_blocks = _build_section_blocks(content_slice, sec_tbl_ids)
        sec_bids = set([h_block.block_id])
        for b in content_slice:
            if b.table_id is None or b.table_id in sec_tbl_ids:
                sec_bids.add(b.block_id)
        for cb in child_blocks:
            sec_bids.update(cb.source_block_ids)

        new_section = DocumentSection(
            heading=heading_text,
            level=level,
            blocks=child_blocks,
            source_block_ids=sorted(sec_bids),
        )

        if level == 2 and current_l1_section is not None:
            # Subsection attached to current top-level section
            current_l1_section.subsections.append(new_section)
        else:
            current_l1_section = new_section
            last_l1_heading_block = h_block
            sections.append(new_section)

    return DocumentStructure(
        sections=sections,
        page_count=semantic_input.page_count,
        metadata={"archetype": getattr(semantic_input.archetype, "value", str(semantic_input.archetype))},
    )


def _build_spatial_data(
    tbl_blocks: list[SemanticBlockInput],
    vg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float]:
    """Build 2D spatial blocks and rows with normalized coordinates for form-like tables.

    Uses geometry-first physical line grouping (Y overlap / baseline proximity)
    followed by within-line horizontal ordering and token association.
    """
    if not tbl_blocks:
        return [], [], 1.0, 100.0

    outer_bounds = vg.get("outer_bounds") if vg else None
    if outer_bounds and len(outer_bounds) == 4:
        tbl_x0, tbl_y0, tbl_x1, tbl_y1 = outer_bounds
    else:
        tbl_x0 = min(b.bbox[0] for b in tbl_blocks)
        tbl_y0 = min(b.bbox[1] for b in tbl_blocks)
        tbl_x1 = max(b.bbox[2] for b in tbl_blocks)
        tbl_y1 = max(b.bbox[3] for b in tbl_blocks)

    tbl_w = max(1.0, tbl_x1 - tbl_x0)
    tbl_h = max(1.0, tbl_y1 - tbl_y0)
    aspect_ratio = round(tbl_w / tbl_h, 3)

    valid_blocks = [b for b in tbl_blocks if b.text and b.text.strip()]
    if not valid_blocks:
        return [], [], aspect_ratio, 100.0

    # 1. Cluster source blocks into physical lines using Y overlap
    sorted_by_y = sorted(valid_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    physical_lines: list[list[SemanticBlockInput]] = []
    for item in sorted_by_y:
        iy0, iy1 = item.bbox[1], item.bbox[3]
        ih = max(1.0, iy1 - iy0)
        assigned_line: list[SemanticBlockInput] | None = None
        for line in physical_lines:
            ly0 = min(x.bbox[1] for x in line)
            ly1 = max(x.bbox[3] for x in line)
            lh = max(1.0, ly1 - ly0)
            overlap = min(iy1, ly1) - max(iy0, ly0)
            if overlap > 0.4 * min(ih, lh) or abs(iy0 - ly0) < 4.0:
                assigned_line = line
                break
        if assigned_line is not None:
            assigned_line.append(item)
        else:
            physical_lines.append([item])

    # Sort physical lines by top Y
    physical_lines.sort(key=lambda l: min(x.bbox[1] for x in l))

    spatial_rows: list[dict[str, Any]] = []
    spatial_blocks: list[dict[str, Any]] = []

    # 2. Within each physical line, sort by X and perform token association
    for line in physical_lines:
        line_sorted = sorted(line, key=lambda b: b.bbox[0])
        fields: list[dict[str, Any]] = []

        for b in line_sorted:
            txt = b.text.strip()
            if not txt:
                continue

            merged = False
            if fields:
                prev = fields[-1]
                prev_txt = prev["text"]
                gap = b.bbox[0] - prev["bbox"][2]

                # Case 1: Current block is a standalone colon ':' following a label on the SAME physical line
                if txt == ":" and 0 <= gap < 100.0:
                    prev["text"] = prev_txt.rstrip() + " :"
                    prev["bbox"][1] = min(prev["bbox"][1], b.bbox[1])
                    prev["bbox"][2] = max(prev["bbox"][2], b.bbox[2])
                    prev["bbox"][3] = max(prev["bbox"][3], b.bbox[3])
                    prev["source_block_ids"].append(b.block_id)
                    merged = True
                # Case 2: Value token closely following a colon on the SAME physical line
                elif prev_txt.endswith(":") and 0 <= gap < 45.0:
                    prev["text"] = prev_txt + " " + txt
                    prev["bbox"][1] = min(prev["bbox"][1], b.bbox[1])
                    prev["bbox"][2] = max(prev["bbox"][2], b.bbox[2])
                    prev["bbox"][3] = max(prev["bbox"][3], b.bbox[3])
                    prev["source_block_ids"].append(b.block_id)
                    merged = True
                # Case 3: Current text starts with ':', e.g. ': Nautical'
                elif txt.startswith(":") and 0 <= gap < 60.0:
                    prev["text"] = prev_txt + " " + txt
                    prev["bbox"][1] = min(prev["bbox"][1], b.bbox[1])
                    prev["bbox"][2] = max(prev["bbox"][2], b.bbox[2])
                    prev["bbox"][3] = max(prev["bbox"][3], b.bbox[3])
                    prev["source_block_ids"].append(b.block_id)
                    merged = True
                # Case 4: Continuous text fragments or numbering split horizontally with small gap
                elif (
                    not prev_txt.endswith(":")
                    and not txt.startswith(":")
                    and (
                        (0 <= gap < 18.0)
                        or (prev_txt.isdigit() and txt == "." and 0 <= gap < 12.0)
                        or (prev_txt.endswith(".") and 0 <= gap < 12.0)
                    )
                ):
                    prev["text"] = (
                        (prev_txt + txt)
                        if (txt == "." or (prev_txt.isdigit() and txt == "."))
                        else (prev_txt + " " + txt)
                    )
                    prev["bbox"][1] = min(prev["bbox"][1], b.bbox[1])
                    prev["bbox"][2] = max(prev["bbox"][2], b.bbox[2])
                    prev["bbox"][3] = max(prev["bbox"][3], b.bbox[3])
                    prev["source_block_ids"].append(b.block_id)
                    merged = True

            if not merged:
                fields.append({
                    "text": txt,
                    "bbox": list(b.bbox),
                    "column_index": getattr(b, "column_index", None),
                    "source_block_ids": [b.block_id],
                    "page_number": b.page,
                    "reading_order": b.reading_order,
                })

        row_fields: list[dict[str, Any]] = []
        for f in fields:
            bx0, by0, bx1, by1 = f["bbox"]
            rel_x = round(((bx0 - tbl_x0) / tbl_w) * 100.0, 2)
            rel_y = round(((by0 - tbl_y0) / tbl_h) * 100.0, 2)
            rel_w = round(((bx1 - bx0) / tbl_w) * 100.0, 2)
            rel_h = round(((by1 - by0) / tbl_h) * 100.0, 2)
            block_dict = {
                "text": f["text"],
                "relative_x": max(0.0, min(100.0, rel_x)),
                "relative_y": max(0.0, min(100.0, rel_y)),
                "relative_width": max(0.0, min(100.0, rel_w)),
                "relative_height": max(0.0, min(100.0, rel_h)),
                "source_block_ids": f["source_block_ids"],
                "page_number": f.get("page_number"),
                "reading_order": f.get("reading_order"),
            }
            row_fields.append(block_dict)
            spatial_blocks.append(block_dict)

        if row_fields:
            spatial_rows.append({"fields": row_fields})

    max_bottom = max((b["relative_y"] + b["relative_height"] for b in spatial_blocks), default=100.0)
    return spatial_blocks, spatial_rows, aspect_ratio, max_bottom


def _build_blocks_from_slice(
    blocks: list[SemanticBlockInput],
    tables_by_id: dict[str, Any],
    all_table_blocks_map: dict[str, list[SemanticBlockInput]] | None = None,
    processed_table_ids: set[str] | None = None,
) -> list[DocumentBlock]:
    """Convert a sequence of SemanticBlockInput items into ordered DocumentBlock items."""
    result: list[DocumentBlock] = []
    if processed_table_ids is None:
        processed_table_ids = set()

    idx = 0
    while idx < len(blocks):
        b = blocks[idx]
        if not b.text.strip() and b.table_id is None:
            idx += 1
            continue

        # Handle table blocks
        if b.table_id is not None:
            tbl_id = b.table_id
            if tbl_id in processed_table_ids:
                idx += 1
                continue
            processed_table_ids.add(tbl_id)

            # Collect all blocks belonging to this table
            if all_table_blocks_map and tbl_id in all_table_blocks_map:
                tbl_blocks = all_table_blocks_map[tbl_id]
            else:
                tbl_blocks = [item for item in blocks if item.table_id == tbl_id]
            table_meta = tables_by_id.get(tbl_id)

            headers: list[str] = []
            rows: list[list[str]] = []
            column_widths: list[float] = list(getattr(table_meta, "column_widths", [])) if table_meta else []
            num_columns: int = getattr(table_meta, "num_columns", 0) if table_meta else 0
            all_tbl_bids = [item.block_id for item in tbl_blocks]

            # Reconstruct table cells strictly grounded in tbl_blocks
            has_cols = any(item.column_index is not None for item in tbl_blocks)
            if has_cols:
                row_col_map: dict[int, dict[int, list[str]]] = {}
                for item in tbl_blocks:
                    r = item.row_index if item.row_index is not None else 0
                    c = item.column_index if item.column_index is not None else 0
                    row_col_map.setdefault(r, {}).setdefault(c, []).append(item.text.strip())
                if not num_columns:
                    max_c = max((max(cols.keys()) for cols in row_col_map.values()), default=0)
                    num_columns = max_c + 1
                for r_idx in sorted(row_col_map):
                    cols = row_col_map[r_idx]
                    row_vals = ["\n".join(cols.get(c_idx, [])) for c_idx in range(num_columns)]
                    if r_idx == 0 and len(row_col_map) > 1:
                        headers = row_vals
                    else:
                        rows.append(row_vals)
            else:
                row_dict: dict[int, list[str]] = {}
                for item in tbl_blocks:
                    r = item.row_index if item.row_index is not None else 0
                    row_dict.setdefault(r, []).append(item.text.strip())
                for r_idx in sorted(row_dict):
                    if r_idx == 0 and len(row_dict) > 1:
                        headers = row_dict[r_idx]
                    else:
                        rows.append(row_dict[r_idx])
                if not num_columns:
                    num_columns = max((len(r) for r in ([headers] + rows)), default=0)

            tbl_data: dict[str, Any] = {"headers": headers, "rows": rows}
            if column_widths:
                tbl_data["column_widths"] = column_widths
            if num_columns:
                tbl_data["num_columns"] = num_columns

            vg = getattr(table_meta, "visual_geometry", None) if table_meta else None
            if vg:
                tbl_data["visual_geometry"] = vg

            # Detect form-like table: outer border present, no internal grid lines
            is_form_layout = (
                vg is not None
                and vg.get("has_outer_border") is True
                and not vg.get("has_horizontal_borders")
                and not vg.get("has_vertical_borders")
                and has_cols
            )

            if is_form_layout:
                spatial_blocks, spatial_rows, aspect_ratio, max_bottom = _build_spatial_data(tbl_blocks, vg)
                if spatial_rows:
                    tbl_data["is_form_layout"] = True
                    tbl_data["spatial_rows"] = spatial_rows
                    tbl_data["spatial_blocks"] = spatial_blocks
                    tbl_data["aspect_ratio"] = aspect_ratio
                    tbl_data["max_bottom"] = max_bottom

            result.append(
                DocumentBlock(
                    id=f"block-{b.block_id}",
                    type="table",
                    text="\n".join([" | ".join(headers)] + [" | ".join(r) for r in rows]),
                    source_block_ids=all_tbl_bids,
                    page_number=b.page,
                    reading_order=b.reading_order,
                    table_data=tbl_data,
                )
            )
            idx += 1
            continue

        # Handle bullet / list item
        if _is_bullet_block(b):
            result.append(
                DocumentBlock(
                    id=f"block-{b.block_id}",
                    type="list_item",
                    text=_clean_bullet_text(b.text),
                    source_block_ids=[b.block_id],
                    page_number=b.page,
                    reading_order=b.reading_order,
                )
            )
        else:
            result.append(
                DocumentBlock(
                    id=f"block-{b.block_id}",
                    type="paragraph",
                    text=b.text.strip(),
                    source_block_ids=[b.block_id],
                    page_number=b.page,
                    reading_order=b.reading_order,
                )
            )
        idx += 1

    return result
