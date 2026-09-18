"""Generic deterministic geometric table/row/column binding stage.

Recovers physical tabular grid relationships (table_id, row_index, column_index,
cell_role) based purely on spatial bounding boxes and alignment geometry without
relying on document-specific text heuristics or semantic entity parsing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any

from app.domain.semantic_contract import DocumentArchetype, SemanticBlockInput, SemanticInput


@dataclass(frozen=True)
class GeometricCell:
    """Physical table cell binding for a single text block."""

    block_id: str
    text: str
    bbox: list[float]
    table_id: str
    row_index: int
    column_index: int
    cell_role: str | None = None  # "HEADER", "DATA", or None
    parent_block_id: str | None = None
    spans: tuple[dict[str, Any], ...] = ()


@dataclass
class GeometricTable:
    """Deterministically bound table structure on a single page."""

    table_id: str
    page: int
    num_columns: int
    num_rows: int
    num_data_rows: int
    column_bands: list[tuple[float, float]]
    cells: list[GeometricCell] = field(default_factory=list)


def _group_fragments_to_cells(
    fragments: list[tuple[str, int, float, float, dict[str, Any] | None]],
    b: SemanticBlockInput,
    table_id: str,
    r_idx: int,
    role: str,
) -> list[GeometricCell]:
    """Group contiguous text fragments belonging to the same column into GeometricCell records."""
    if not fragments:
        return []

    unique_cols = sorted({f[1] for f in fragments})
    if len(unique_cols) <= 1:
        c_idx = unique_cols[0] if unique_cols else 0
        return [
            GeometricCell(
                block_id=b.block_id,
                text=b.text,
                bbox=b.bbox,
                table_id=table_id,
                row_index=r_idx,
                column_index=c_idx,
                cell_role=role,
                parent_block_id=b.parent_block_id,
                spans=tuple(b.spans) if hasattr(b, "spans") and b.spans else (),
            )
        ]

    grouped: list[dict[str, Any]] = []
    current_col = fragments[0][1]
    current_texts = [fragments[0][0]]
    current_x0 = fragments[0][2]
    current_x1 = fragments[0][3]
    current_spans = [fragments[0][4]] if fragments[0][4] is not None else []

    for item in fragments[1:]:
        text, c_idx, x0, x1 = item[0], item[1], item[2], item[3]
        s_dict = item[4] if len(item) > 4 else None
        if c_idx == current_col:
            current_texts.append(text)
            current_x1 = max(current_x1, x1)
            if s_dict is not None:
                current_spans.append(s_dict)
        else:
            grouped.append(
                {
                    "col": current_col,
                    "text": " ".join(current_texts),
                    "bbox": [current_x0, b.bbox[1], current_x1, b.bbox[3]],
                    "spans": tuple(current_spans),
                }
            )
            current_col = c_idx
            current_texts = [text]
            current_x0 = x0
            current_x1 = x1
            current_spans = [s_dict] if s_dict is not None else []

    if current_texts:
        grouped.append(
            {
                "col": current_col,
                "text": " ".join(current_texts),
                "bbox": [current_x0, b.bbox[1], current_x1, b.bbox[3]],
                "spans": tuple(current_spans),
            }
        )

    col_counts = Counter(g["col"] for g in grouped)
    col_seen: Counter[int] = Counter()
    cells: list[GeometricCell] = []
    parent_id = b.parent_block_id or b.block_id

    for g in grouped:
        c = g["col"]
        if col_counts[c] == 1:
            frag_id = f"{b.block_id}_c{c}"
        else:
            frag_id = f"{b.block_id}_c{c}_{col_seen[c]}"
            col_seen[c] += 1

        cells.append(
            GeometricCell(
                block_id=frag_id,
                text=g["text"],
                bbox=g["bbox"],
                table_id=table_id,
                row_index=r_idx,
                column_index=c,
                cell_role=role,
                parent_block_id=parent_id,
                spans=g["spans"],
            )
        )

    return cells


class GeometricTableBinder:
    """Generic geometric table/row/column binder based strictly on spatial bounding boxes."""

    def __init__(
        self,
        row_tolerance: float = 5.0,
        min_columns: int = 2,
        min_rows: int = 2,
        col_gutter_threshold: float = 12.0,
        row_gap_threshold: float = 3.0,
    ) -> None:
        self.row_tolerance = row_tolerance
        self.min_columns = min_columns
        self.min_rows = min_rows
        self.col_gutter_threshold = col_gutter_threshold
        self.row_gap_threshold = row_gap_threshold

    def detect_page_tables(
        self,
        blocks: list[SemanticBlockInput],
        page_number: int,
    ) -> list[GeometricTable]:
        """Detect and bind physical tables on a single page using bounding box geometry."""
        if len(blocks) < (self.min_columns * self.min_rows):
            return []

        sorted_p = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

        # 1. Group blocks into horizontal visual slices (lines sharing y alignment)
        slices: list[dict[str, Any]] = []
        for b in sorted_p:
            y_mid = (b.bbox[1] + b.bbox[3]) / 2.0
            placed = False
            for s in slices:
                s_mids = [(item.bbox[1] + item.bbox[3]) / 2.0 for item in s["blocks"]]
                avg_mid = sum(s_mids) / len(s_mids)
                if abs(y_mid - avg_mid) <= self.row_tolerance or abs(b.bbox[1] - s["min_y0"]) <= 3.0:
                    s["blocks"].append(b)
                    s["min_y0"] = min(s["min_y0"], b.bbox[1])
                    s["max_y1"] = max(s["max_y1"], b.bbox[3])
                    placed = True
                    break
            if not placed:
                slices.append({
                    "min_y0": b.bbox[1],
                    "max_y1": b.bbox[3],
                    "blocks": [b],
                })

        for s in slices:
            s["blocks"] = sorted(s["blocks"], key=lambda b: b.bbox[0])

        # 2. Group adjacent slices into candidate table regions
        # A table region must start on a slice with multiple columns (>= min_columns)
        candidate_tables: list[list[int]] = []
        curr_t: list[int] = []
        for s_idx, s in enumerate(slices):
            if not curr_t:
                if len(s["blocks"]) >= self.min_columns:
                    curr_t.append(s_idx)
            else:
                prev_s = slices[curr_t[-1]]
                if s["min_y0"] - prev_s["max_y1"] <= 25.0:
                    curr_t.append(s_idx)
                else:
                    if len(curr_t) >= self.min_rows and sum(1 for idx in curr_t if len(slices[idx]["blocks"]) >= self.min_columns) >= self.min_rows:
                        candidate_tables.append(curr_t)
                    curr_t = [s_idx] if len(s["blocks"]) >= self.min_columns else []
        if curr_t and len(curr_t) >= self.min_rows and sum(1 for idx in curr_t if len(slices[idx]["blocks"]) >= self.min_columns) >= self.min_rows:
            candidate_tables.append(curr_t)

        detected_tables: list[GeometricTable] = []
        consumed_slice_indices: set[int] = set()

        for t_indices in candidate_tables:
            if any(idx in consumed_slice_indices for idx in t_indices):
                continue

            t_slices = [slices[idx] for idx in t_indices]
            t_blocks: list[SemanticBlockInput] = [b for s in t_slices for b in s["blocks"]]
            if len(t_blocks) < (self.min_columns * self.min_rows):
                continue

            # 3. Group visual slices into logical table rows (handling multi-line cell wraps)
            min_x0 = min(b.bbox[0] for b in t_blocks)
            logical_rows: list[list[dict[str, Any]]] = []
            curr_row_slices = [t_slices[0]]
            for i in range(1, len(t_slices)):
                prev_s = curr_row_slices[-1]
                curr_s = t_slices[i]
                gap = curr_s["min_y0"] - prev_s["max_y1"]
                first_text = curr_s["blocks"][0].text.strip()
                is_serial = bool(re.match(r"^\d+$", first_text))
                starts_table_left = (curr_s["blocks"][0].bbox[0] <= min_x0 + 15.0)
                if is_serial or gap > self.row_gap_threshold or (starts_table_left and len(curr_s["blocks"]) >= self.min_columns):
                    logical_rows.append(curr_row_slices)
                    curr_row_slices = [curr_s]
                else:
                    curr_row_slices.append(curr_s)
            if curr_row_slices:
                logical_rows.append(curr_row_slices)

            if len(logical_rows) < self.min_rows:
                continue

            # 4. Form logical row cells and discover column tracks across rows
            # For each logical row, the main slice defines the initial cells;
            # subsequent multi-line continuation slices merge into the closest/overlapping column cell.
            row_cells: list[list[list[Any]]] = []
            for r_slices in logical_rows:
                main_s = r_slices[0]
                main_blocks = sorted(main_s["blocks"], key=lambda b: b.bbox[0])
                cells: list[list[Any]] = []
                for b in main_blocks:
                    if not cells:
                        cells.append([b.bbox[0], b.bbox[2], [b]])
                    else:
                        if b.bbox[0] <= cells[-1][1] + self.col_gutter_threshold:
                            cells[-1][1] = max(cells[-1][1], b.bbox[2])
                            cells[-1][2].append(b)
                        else:
                            cells.append([b.bbox[0], b.bbox[2], [b]])

                for sub_s in r_slices[1:]:
                    for b in sub_s["blocks"]:
                        bx_mid = (b.bbox[0] + b.bbox[2]) / 2.0
                        if cells:
                            best_idx, best_c = min(
                                enumerate(cells),
                                key=lambda item: min(abs(bx_mid - (item[1][0] + item[1][1]) / 2.0), abs(b.bbox[0] - item[1][0])),
                            )
                            min_x = 0.0 if best_idx == 0 else cells[best_idx - 1][1] + 1.0
                            max_x = 10000.0 if best_idx == len(cells) - 1 else cells[best_idx + 1][0] - 1.0
                            best_c[0] = max(min(best_c[0], b.bbox[0]), min_x)
                            best_c[1] = min(max(best_c[1], b.bbox[2]), max_x)
                            best_c[2].append(b)
                        else:
                            cells.append([b.bbox[0], b.bbox[2], [b]])
                row_cells.append(cells)

            # Consensus column template from the most complete row
            header_cells = row_cells[0]
            max_row_cells = max(row_cells, key=lambda r: len(r))
            col_template = (
                header_cells
                if len(header_cells) >= max(self.min_columns, int(len(max_row_cells) * 0.75))
                else max_row_cells
            )
            col_template = sorted(col_template, key=lambda c: c[0])

            if len(col_template) < self.min_columns:
                continue

            # Compute column boundary separators
            seps: list[float] = []
            for i in range(len(col_template) - 1):
                c_curr = col_template[i]
                c_next = col_template[i + 1]
                seps.append((c_curr[1] + c_next[0]) / 2.0)

            col_bands: list[tuple[float, float]] = []
            for i in range(len(col_template)):
                b_left = 0.0 if i == 0 else seps[i - 1]
                b_right = 10000.0 if i == len(col_template) - 1 else seps[i]
                col_bands.append((b_left, b_right))

            table_id = f"table_p{page_number}_{len(detected_tables)}"

            # Build cell bindings with word-level geometric column assignment
            cells: list[GeometricCell] = []
            num_data_rows = 0
            for r_idx, r_slices in enumerate(logical_rows):
                is_header = (r_idx == 0)
                role = "HEADER" if is_header else "DATA"
                if not is_header:
                    num_data_rows += 1
                for s in r_slices:
                    for b in s["blocks"]:
                        split_cells = self._split_block_by_columns(b, seps, r_idx, table_id, role)
                        cells.extend(split_cells)

            # Conservative row density check: ensure candidate is not a single-column block run with an isolated date
            row_multi_cols = [r for r in logical_rows if len(r) >= 2 or sum(len(sl["blocks"]) for sl in r) >= 2]
            if len(col_template) >= 3:
                if len(logical_rows) > 4 and len(row_multi_cols) / len(logical_rows) < 0.40:
                    continue
            elif len(col_template) == 2:
                if len(logical_rows) > 3 and len(row_multi_cols) / len(logical_rows) < 0.60:
                    continue

            table = GeometricTable(
                table_id=table_id,
                page=page_number,
                num_columns=len(col_template),
                num_rows=len(logical_rows),
                num_data_rows=num_data_rows,
                column_bands=col_bands,
                cells=cells,
            )

            consumed_slice_indices.update(t_indices)
            last_idx = max(t_indices)
            self._extend_table_continuation(
                table,
                slices,
                last_idx + 1,
                consumed_slice_indices,
                seps,
                candidate_tables=candidate_tables,
            )

            detected_tables.append(table)

        return detected_tables

    def _extend_table_continuation(
        self,
        table: GeometricTable,
        slices: list[dict[str, Any]],
        start_slice_idx: int,
        consumed_slice_indices: set[int],
        seps: list[float],
        candidate_tables: list[list[int]] | None = None,
        max_inter_row_gap: float = 55.0,
    ) -> None:
        """Deterministically extend a verified table across subsequent variable-height rows.

        Uses stable consensus column tracks and boundaries (seps) established by the table.
        Preserves row grouping, wrapped cell blocks, and handles variable inter-row spacing
        without identifying fixtures by filename or assuming specific section semantics.
        """
        if not seps or table.num_columns < 3 or not table.cells:
            return

        # Map candidate table start indices to their candidate slice indices
        cand_by_start: dict[int, list[int]] = {}
        if candidate_tables:
            for ct in candidate_tables:
                if ct:
                    cand_by_start[ct[0]] = ct

        prev_max_y = max(c.bbox[3] for c in table.cells)
        idx = start_slice_idx

        while idx < len(slices):
            if idx in consumed_slice_indices:
                idx += 1
                continue

            # If this slice starts another candidate table that represents an independent multi-column structure, do not absorb it
            if idx in cand_by_start:
                ct_slices = [slices[i] for i in cand_by_start[idx]]
                ct_blocks = [b for sl in ct_slices for b in sl["blocks"]]
                # If candidate table has distinct column count or forms an independent multi-row structure with its own header
                first_slice_blocks = slices[idx]["blocks"]
                if len(first_slice_blocks) != table.num_columns and len(first_slice_blocks) >= self.min_columns:
                    break

            s = slices[idx]
            gap = s["min_y0"] - prev_max_y

            # Table continuation halts if inter-row spacing exceeds threshold or is inverted
            if gap < -2.0 or gap > max_inter_row_gap:
                break

            # Table continuation halts immediately if any block is a section heading, heading candidate, or table header
            if any(
                b.suggested_role in ("SECTION_HEADING", "TABLE_HEADER") or getattr(b, "heading_candidate", False)
                for b in s["blocks"]
            ):
                break

            # A continuation row cannot have more blocks in its main slice than table columns
            if len(s["blocks"]) > table.num_columns:
                break

            # Map blocks in candidate row start slice to columns
            assigned_cols = []
            for b in s["blocks"]:
                bx_mid = (b.bbox[0] + b.bbox[2]) / 2.0
                c_idx = next((i for i, sep in enumerate(seps) if bx_mid < sep), len(seps))
                assigned_cols.append(c_idx)

            unique_cols = set(assigned_cols)
            # A genuine table continuation row must span multiple columns conforming to the table
            if len(unique_cols) < min(table.num_columns, 2):
                break

            # Form logical row with wrapped continuation slices
            row_slices = [s]
            consumed_slice_indices.add(idx)
            idx += 1

            while idx < len(slices):
                if idx in consumed_slice_indices:
                    idx += 1
                    continue
                next_s = slices[idx]
                sub_gap = next_s["min_y0"] - row_slices[-1]["max_y1"]
                if sub_gap > self.row_gap_threshold:
                    break
                if any(
                    b.suggested_role in ("SECTION_HEADING", "TABLE_HEADER") or getattr(b, "heading_candidate", False)
                    for b in next_s["blocks"]
                ):
                    break
                row_slices.append(next_s)
                consumed_slice_indices.add(idx)
                idx += 1

            # Append new logical row to table
            r_idx = table.num_rows
            new_cells: list[GeometricCell] = []
            for r_s in row_slices:
                for b in r_s["blocks"]:
                    split = self._split_block_by_columns(b, seps, r_idx, table.table_id, "DATA")
                    new_cells.extend(split)

            table.cells.extend(new_cells)
            table.num_rows += 1
            table.num_data_rows += 1
            prev_max_y = max(c.bbox[3] for c in new_cells)

    def _split_block_by_columns(
        self,
        b: SemanticBlockInput,
        seps: list[float],
        r_idx: int,
        table_id: str,
        role: str,
    ) -> list[GeometricCell]:
        """Assign block or its constituent geometric word fragments to table column(s)."""
        if not seps or not b.text.strip():
            bx_mid = (b.bbox[0] + b.bbox[2]) / 2.0
            c_idx = next((i for i, s in enumerate(seps) if bx_mid < s), len(seps))
            return [
                GeometricCell(
                    block_id=b.block_id,
                    text=b.text,
                    bbox=b.bbox,
                    table_id=table_id,
                    row_index=r_idx,
                    column_index=c_idx,
                    cell_role=role,
                    parent_block_id=b.parent_block_id,
                    spans=tuple(b.spans) if hasattr(b, "spans") and b.spans else (),
                )
            ]

        # Check if block bounding box crosses any separator
        crosses_sep = any(b.bbox[0] < s < b.bbox[2] for s in seps)
        if not crosses_sep:
            bx_mid = (b.bbox[0] + b.bbox[2]) / 2.0
            c_idx = next((i for i, s in enumerate(seps) if bx_mid < s), len(seps))
            return [
                GeometricCell(
                    block_id=b.block_id,
                    text=b.text,
                    bbox=b.bbox,
                    table_id=table_id,
                    row_index=r_idx,
                    column_index=c_idx,
                    cell_role=role,
                    parent_block_id=b.parent_block_id,
                    spans=tuple(b.spans) if hasattr(b, "spans") and b.spans else (),
                )
            ]

        # 1. Check if actual word/span geometry is available on the block
        if hasattr(b, "spans") and b.spans:
            fragments: list[tuple[str, int, float, float, dict[str, Any] | None]] = []
            for s in b.spans:
                s_text = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                s_bbox = s.get("bbox", []) if isinstance(s, dict) else getattr(s, "bbox", [])
                if not s_text or not s_bbox or len(s_bbox) < 4:
                    continue
                s_mid = (s_bbox[0] + s_bbox[2]) / 2.0
                c_idx = next((i for i, sep in enumerate(seps) if s_mid < sep), len(seps))
                s_dict = s if isinstance(s, dict) else getattr(s, "__dict__", None)
                fragments.append((s_text, c_idx, s_bbox[0], s_bbox[2], s_dict))

            if fragments:
                return _group_fragments_to_cells(fragments, b, table_id, r_idx, role)

        # 2. Fallback: Tokenize words with geometric bounding box estimation when span geometry is unavailable
        raw_text = b.text
        tokens: list[tuple[str, int, int]] = []
        for match in re.finditer(r"\S+", raw_text):
            tokens.append((match.group(0), match.start(), match.end()))

        if len(tokens) <= 1:
            bx_mid = (b.bbox[0] + b.bbox[2]) / 2.0
            c_idx = next((i for i, s in enumerate(seps) if bx_mid < s), len(seps))
            return [
                GeometricCell(
                    block_id=b.block_id,
                    text=b.text,
                    bbox=b.bbox,
                    table_id=table_id,
                    row_index=r_idx,
                    column_index=c_idx,
                    cell_role=role,
                    parent_block_id=b.parent_block_id,
                    spans=tuple(b.spans) if hasattr(b, "spans") and b.spans else (),
                )
            ]

        total_len = len(raw_text)
        total_width = max(0.0, b.bbox[2] - b.bbox[0])
        char_w = total_width / max(1, total_len)

        fallback_fragments: list[tuple[str, int, float, float, dict[str, Any] | None]] = []
        for word_text, start_idx, end_idx in tokens:
            wx0 = b.bbox[0] + (start_idx * char_w)
            wx1 = b.bbox[0] + (end_idx * char_w)
            w_mid = (wx0 + wx1) / 2.0
            c_idx = next((i for i, s in enumerate(seps) if w_mid < s), len(seps))
            fallback_fragments.append((word_text, c_idx, wx0, wx1, None))

        return _group_fragments_to_cells(fallback_fragments, b, table_id, r_idx, role)

    def detect_document_tables(
        self,
        blocks: list[SemanticBlockInput],
    ) -> list[GeometricTable]:
        """Detect all geometric tables across all pages of a document."""
        by_page: dict[int, list[SemanticBlockInput]] = {}
        for b in blocks:
            by_page.setdefault(b.page, []).append(b)

        all_tables: list[GeometricTable] = []
        for page_num in sorted(by_page.keys()):
            p_blocks = by_page[page_num]
            tables = self.detect_page_tables(p_blocks, page_num)
            all_tables.extend(tables)
        return all_tables

    def bind_document_tables(
        self,
        blocks: list[SemanticBlockInput],
        tables: list[GeometricTable] | None = None,
    ) -> list[SemanticBlockInput]:
        """Populate table_id, row_index, column_index, and cell_role on verifiable table blocks."""
        by_page: dict[int, list[SemanticBlockInput]] = {}
        for b in blocks:
            by_page.setdefault(b.page, []).append(b)

        bound_blocks: list[SemanticBlockInput] = []
        self.last_detected_tables: list[GeometricTable] = []

        for page_num in sorted(by_page.keys()):
            p_blocks = by_page[page_num]
            if tables is not None:
                p_tables = [t for t in tables if t.page == page_num]
            else:
                p_tables = self.detect_page_tables(p_blocks, page_num)
            if not p_tables:
                bound_blocks.extend(p_blocks)
                continue

            self.last_detected_tables.extend(p_tables)
            binding_map: dict[str, list[GeometricCell]] = {}
            for t in p_tables:
                for cell in t.cells:
                    src_id = cell.parent_block_id or cell.block_id
                    binding_map.setdefault(src_id, []).append(cell)

            for b in p_blocks:
                if b.block_id in binding_map:
                    matched_cells = binding_map[b.block_id]
                    for cell in matched_cells:
                        update_kwargs: dict[str, Any] = {
                            "block_id": cell.block_id,
                            "parent_block_id": cell.parent_block_id or b.parent_block_id,
                            "text": cell.text,
                            "bbox": cell.bbox,
                            "table_id": cell.table_id,
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "cell_role": cell.cell_role,
                        }
                        if cell.spans:
                            update_kwargs["spans"] = list(cell.spans)
                        bound_blocks.append(b.model_copy(update=update_kwargs))
                else:
                    bound_blocks.append(b)

        return bound_blocks
