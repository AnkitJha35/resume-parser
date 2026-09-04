"""Generic deterministic geometric table/row/column binding stage.

Recovers physical tabular grid relationships (table_id, row_index, column_index,
cell_role) based purely on spatial bounding boxes and alignment geometry without
relying on document-specific text heuristics or semantic entity parsing.
"""

from __future__ import annotations

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
    fragments: list[tuple[str, int, float, float]],
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
            )
        ]

    cells: list[GeometricCell] = []
    current_col = fragments[0][1]
    current_texts = [fragments[0][0]]
    current_x0 = fragments[0][2]
    current_x1 = fragments[0][3]

    for text, c_idx, x0, x1 in fragments[1:]:
        if c_idx == current_col:
            current_texts.append(text)
            current_x1 = max(current_x1, x1)
        else:
            cells.append(
                GeometricCell(
                    block_id=b.block_id,
                    text=" ".join(current_texts),
                    bbox=[current_x0, b.bbox[1], current_x1, b.bbox[3]],
                    table_id=table_id,
                    row_index=r_idx,
                    column_index=current_col,
                    cell_role=role,
                )
            )
            current_col = c_idx
            current_texts = [text]
            current_x0 = x0
            current_x1 = x1

    if current_texts:
        cells.append(
            GeometricCell(
                block_id=b.block_id,
                text=" ".join(current_texts),
                bbox=[current_x0, b.bbox[1], current_x1, b.bbox[3]],
                table_id=table_id,
                row_index=r_idx,
                column_index=current_col,
                cell_role=role,
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
        candidate_tables: list[list[dict[str, Any]]] = []
        curr_t: list[dict[str, Any]] = []
        for s in slices:
            if not curr_t:
                if len(s["blocks"]) >= self.min_columns:
                    curr_t.append(s)
            else:
                prev_s = curr_t[-1]
                if s["min_y0"] - prev_s["max_y1"] <= 25.0:
                    curr_t.append(s)
                else:
                    if len(curr_t) >= self.min_rows and sum(1 for sl in curr_t if len(sl["blocks"]) >= self.min_columns) >= self.min_rows:
                        candidate_tables.append(curr_t)
                    curr_t = [s] if len(s["blocks"]) >= self.min_columns else []
        if curr_t and len(curr_t) >= self.min_rows and sum(1 for sl in curr_t if len(sl["blocks"]) >= self.min_columns) >= self.min_rows:
            candidate_tables.append(curr_t)

        detected_tables: list[GeometricTable] = []

        for t_idx, t_slices in enumerate(candidate_tables):
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
                            best_c = min(
                                cells,
                                key=lambda c: min(abs(bx_mid - (c[0] + c[1]) / 2.0), abs(b.bbox[0] - c[0])),
                            )
                            best_c[0] = min(best_c[0], b.bbox[0])
                            best_c[1] = max(best_c[1], b.bbox[2])
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

            table_id = f"table_p{page_number}_{t_idx}"

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

            detected_tables.append(
                GeometricTable(
                    table_id=table_id,
                    page=page_number,
                    num_columns=len(col_template),
                    num_rows=len(logical_rows),
                    num_data_rows=num_data_rows,
                    column_bands=col_bands,
                    cells=cells,
                )
            )

        return detected_tables

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
                )
            ]

        # 1. Check if actual word/span geometry is available on the block
        if hasattr(b, "spans") and b.spans:
            fragments: list[tuple[str, int, float, float]] = []
            for s in b.spans:
                s_text = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                s_bbox = s.get("bbox", []) if isinstance(s, dict) else getattr(s, "bbox", [])
                if not s_text or not s_bbox or len(s_bbox) < 4:
                    continue
                s_mid = (s_bbox[0] + s_bbox[2]) / 2.0
                c_idx = next((i for i, sep in enumerate(seps) if s_mid < sep), len(seps))
                fragments.append((s_text, c_idx, s_bbox[0], s_bbox[2]))

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
                )
            ]

        total_len = len(raw_text)
        total_width = max(0.0, b.bbox[2] - b.bbox[0])
        char_w = total_width / max(1, total_len)

        fragments = []
        for word_text, start_idx, end_idx in tokens:
            wx0 = b.bbox[0] + (start_idx * char_w)
            wx1 = b.bbox[0] + (end_idx * char_w)
            w_mid = (wx0 + wx1) / 2.0
            c_idx = next((i for i, s in enumerate(seps) if w_mid < s), len(seps))
            fragments.append((word_text, c_idx, wx0, wx1))

        return _group_fragments_to_cells(fragments, b, table_id, r_idx, role)

    def bind_document_tables(
        self,
        blocks: list[SemanticBlockInput],
    ) -> list[SemanticBlockInput]:
        """Populate table_id, row_index, column_index, and cell_role on verifiable table blocks."""
        by_page: dict[int, list[SemanticBlockInput]] = {}
        for b in blocks:
            by_page.setdefault(b.page, []).append(b)

        bound_blocks: list[SemanticBlockInput] = []

        for page_num in sorted(by_page.keys()):
            p_blocks = by_page[page_num]
            tables = self.detect_page_tables(p_blocks, page_num)
            if not tables:
                bound_blocks.extend(p_blocks)
                continue

            binding_map: dict[str, list[GeometricCell]] = {}
            for t in tables:
                for cell in t.cells:
                    binding_map.setdefault(cell.block_id, []).append(cell)

            for b in p_blocks:
                if b.block_id in binding_map:
                    matched_cells = binding_map[b.block_id]
                    for cell in matched_cells:
                        bound_blocks.append(
                            b.model_copy(
                                update={
                                    "text": cell.text,
                                    "bbox": cell.bbox,
                                    "table_id": cell.table_id,
                                    "row_index": cell.row_index,
                                    "column_index": cell.column_index,
                                    "cell_role": cell.cell_role,
                                }
                            )
                        )
                else:
                    bound_blocks.append(b)

        return bound_blocks
