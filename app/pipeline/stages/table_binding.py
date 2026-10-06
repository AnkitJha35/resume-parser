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
    column_widths: list[float] = field(default_factory=list)
    is_border_defined: bool = False
    visual_geometry: dict[str, Any] | None = None


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


HEADER_WORD_RE = re.compile(
    r"\b(?:no|num|number|date|issue|expiry|place|grade|rank|remark|remarks|type|vessel|ship|name|description|degree|institution|year|company|sign|from|to|total|status|code|valid|issued|qualification|details|courses?|certificates?|col|cols?|columns?)\b",
    re.IGNORECASE,
)


def _extract_outer_table_boxes(
    drawings: list[Any] | None,
    page_height: float = 792.0,
    tol: float = 4.0,
) -> list[tuple[float, float, float, float]]:
    """Detect closed rectangular vector boundaries representing table containers."""
    if not drawings:
        return []

    boxes: list[tuple[float, float, float, float]] = []

    # 1. Direct rectangle vector paths (stroke/rect)
    for d in drawings:
        x0 = round(getattr(d, "x0", 0), 1)
        y0 = round(getattr(d, "y0", 0), 1)
        x1 = round(getattr(d, "x1", 0), 1)
        y1 = round(getattr(d, "y1", 0), 1)
        w, h = x1 - x0, y1 - y0
        if w >= 200.0 and h >= 20.0 and not getattr(d, "is_fill", False):
            if h < 0.92 * page_height:
                boxes.append((x0, y0, x1, y1))

    # 2. Line segments forming closed 4-corner boxes
    h_lines = [
        d for d in drawings
        if getattr(d, "orientation", "") == "horizontal"
        or (getattr(d, "y1", 0) - getattr(d, "y0", 0) < 2.5 and getattr(d, "x1", 0) - getattr(d, "x0", 0) > 20.0)
    ]
    v_lines = [
        d for d in drawings
        if getattr(d, "orientation", "") == "vertical"
        or (getattr(d, "x1", 0) - getattr(d, "x0", 0) < 2.5 and getattr(d, "y1", 0) - getattr(d, "y0", 0) > 5.0)
    ]

    for i, v1 in enumerate(v_lines):
        for j, v2 in enumerate(v_lines):
            if i >= j:
                continue
            x_left = min(v1.x0, v2.x0)
            x_right = max(v1.x0, v2.x0)
            if x_right - x_left < 200.0:
                continue
            y_top = min(v1.y0, v2.y0)
            y_bot = max(v1.y1, v2.y1)
            if abs(v1.y0 - v2.y0) > tol or abs(v1.y1 - v2.y1) > tol:
                continue
            if y_bot - y_top < 20.0 or y_bot - y_top > 0.92 * page_height:
                continue
            has_top = any(
                abs(h.y0 - y_top) <= tol and h.x0 <= x_left + tol and h.x1 >= x_right - tol
                for h in h_lines
            )
            has_bot = any(
                abs(h.y0 - y_bot) <= tol and h.x0 <= x_left + tol and h.x1 >= x_right - tol
                for h in h_lines
            )
            if has_top and has_bot:
                boxes.append((round(x_left, 1), round(y_top, 1), round(x_right, 1), round(y_bot, 1)))

    # Keep only the maximal enclosing box when boxes share vertical span (ignore sub-cell partitions)
    unique_boxes: list[tuple[float, float, float, float]] = []
    for b in sorted(boxes, key=lambda x: (x[1], -(x[2] - x[0]))):
        if not any(
            abs(b[1] - u[1]) <= tol and abs(b[3] - u[3]) <= tol and u[0] <= b[0] + tol and b[2] <= u[2] + tol
            for u in unique_boxes
        ):
            unique_boxes.append(b)

    return unique_boxes


def _compute_table_visual_geometry(
    t_blocks: list[SemanticBlockInput],
    col_seps: list[float],
    logical_rows: list[list[dict[str, Any]]],
    drawings: list[Any] | None,
    enclosing_box: tuple[float, float, float, float] | None = None,
    tol: float = 4.0,
) -> dict[str, Any] | None:
    """Compute visible source border geometry (outer boundary, internal horizontal/vertical lines)."""
    if not drawings or not t_blocks:
        return None

    t_x0 = min(b.bbox[0] for b in t_blocks)
    t_x1 = max(b.bbox[2] for b in t_blocks)
    t_y0 = min(b.bbox[1] for b in t_blocks)
    t_y1 = max(b.bbox[3] for b in t_blocks)
    t_w = max(1.0, t_x1 - t_x0)
    t_h = max(1.0, t_y1 - t_y0)

    h_lines = [
        d for d in drawings
        if getattr(d, "orientation", "") == "horizontal"
        or (getattr(d, "y1", 0) - getattr(d, "y0", 0) < 2.5 and getattr(d, "x1", 0) - getattr(d, "x0", 0) > 20.0)
    ]
    v_lines = [
        d for d in drawings
        if getattr(d, "orientation", "") == "vertical"
        or (getattr(d, "x1", 0) - getattr(d, "x0", 0) < 2.5 and getattr(d, "y1", 0) - getattr(d, "y0", 0) > 5.0)
    ]

    has_outer = False
    outer_bounds = None
    if enclosing_box:
        has_outer = True
        outer_bounds = list(enclosing_box)
    else:
        has_top = any(abs(h.y0 - t_y0) <= 20.0 and h.x0 <= t_x0 + 25.0 and h.x1 >= t_x1 - 25.0 for h in h_lines)
        has_bot = any(abs(h.y0 - t_y1) <= 20.0 and h.x0 <= t_x0 + 25.0 and h.x1 >= t_x1 - 25.0 for h in h_lines)
        has_left = any(abs(v.x0 - t_x0) <= 20.0 and v.y0 <= t_y0 + 25.0 and v.y1 >= t_y1 - 25.0 for v in v_lines)
        has_right = any(abs(v.x0 - t_x1) <= 20.0 and v.y0 <= t_y0 + 25.0 and v.y1 >= t_y1 - 25.0 for v in v_lines)
        if (has_top and has_bot) or (has_left and has_right) or ((has_top or has_bot) and (has_left or has_right)):
            has_outer = True
            outer_bounds = [t_x0, t_y0, t_x1, t_y1]

    # Compute internal horizontal dividers between consecutive logical rows
    row_bounds = []
    for r in logical_rows:
        ry0 = min(b.bbox[1] for sl in r for b in sl["blocks"])
        ry1 = max(b.bbox[3] for sl in r for b in sl["blocks"])
        row_bounds.append((ry0, ry1))

    h_borders: list[bool] = []
    for i in range(len(row_bounds) - 1):
        r_curr_y1 = row_bounds[i][1]
        r_next_y0 = row_bounds[i + 1][0]
        has_h = any(
            r_curr_y1 - 3.0 <= h.y0 <= r_next_y0 + 3.0 and min(h.x1, t_x1) - max(h.x0, t_x0) >= 0.3 * t_w
            for h in h_lines
        )
        h_borders.append(has_h)

    # Compute internal vertical dividers between consecutive columns
    v_borders: list[bool] = []
    for sep in col_seps:
        has_v = any(
            abs(v.x0 - sep) <= 15.0 and min(v.y1, t_y1) - max(v.y0, t_y0) >= 0.3 * t_h
            for v in v_lines
        )
        v_borders.append(has_v)

    return {
        "has_outer_border": has_outer,
        "outer_bounds": outer_bounds,
        "has_horizontal_borders": any(h_borders),
        "has_vertical_borders": any(v_borders),
        "horizontal_borders": h_borders,
        "vertical_borders": v_borders,
    }


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

    def _is_table_header_slice(self, s: dict[str, Any]) -> bool:
        """Heuristically determine if a slice represents a table header row."""
        blocks = s.get("blocks", [])
        if len(blocks) < 2:
            return False

        # Key-value form rows have standalone colons ':' or colons paired with values/numbers
        has_standalone_colon = any(b.text.strip() == ":" for b in blocks)
        has_numbers = any(re.search(r"\b\d{4,}\b", b.text) for b in blocks)
        colon_count = sum(1 for b in blocks if ":" in b.text)
        if has_standalone_colon or (colon_count >= 2 and has_numbers) or (has_numbers and colon_count >= 1):
            return False

        if any(getattr(b, "suggested_role", None) == "TABLE_HEADER" for b in blocks):
            return True
        if any(getattr(b, "is_bold", False) for b in blocks):
            return True
        matches = 0
        for b in blocks:
            text = b.text.strip()
            words = text.split()
            if len(words) <= 5 and HEADER_WORD_RE.search(text):
                matches += 1
        return matches >= max(1, len(blocks) // 2)

    def _are_slices_grid_compatible(
        self,
        s_prev_group: list[dict[str, Any]],
        s_curr: dict[str, Any],
        tol: float = 22.0,
    ) -> bool:
        """Check if candidate slice s_curr conforms to the column grid established by s_prev_group."""
        curr_blocks = s_curr.get("blocks", [])
        if len(curr_blocks) < 2:
            return True

        multi_prev = [s for s in s_prev_group if len(s["blocks"]) >= 2]
        if not multi_prev:
            return True

        ref_s = max(multi_prev, key=lambda s: len(s["blocks"]))
        ref_blocks = sorted(ref_s["blocks"], key=lambda b: b.bbox[0])
        curr_b_sorted = sorted(curr_blocks, key=lambda b: b.bbox[0])

        def _block_matches(b1: Any, b2: Any) -> bool:
            c1 = (b1.bbox[0] + b1.bbox[2]) / 2.0
            c2 = (b2.bbox[0] + b2.bbox[2]) / 2.0
            if abs(b1.bbox[0] - b2.bbox[0]) <= tol:
                return True
            if abs(c1 - c2) <= tol:
                return True
            if b1.bbox[0] >= b2.bbox[0] - 5.0 and b1.bbox[2] <= b2.bbox[2] + 5.0:
                return True
            if b2.bbox[0] >= b1.bbox[0] - 5.0 and b2.bbox[2] <= b1.bbox[2] + 5.0:
                return True
            return False

        if len(curr_b_sorted) <= len(ref_blocks):
            return all(any(_block_matches(cb, rb) for rb in ref_blocks) for cb in curr_b_sorted)
        else:
            return all(any(_block_matches(rb, cb) for cb in curr_b_sorted) for rb in ref_blocks)

    def detect_page_tables(
        self,
        blocks: list[SemanticBlockInput],
        page_number: int,
        drawings: list[Any] | None = None,
    ) -> list[GeometricTable]:
        """Detect and bind physical tables on a single page using bounding box geometry."""
        # Filter out page chrome (running headers, footers, page numbers)
        page_blocks = [
            b for b in blocks
            if getattr(b, "region_kind", None) not in ("header", "footer")
            and getattr(b, "suggested_role", None) not in ("FOOTER", "PAGE_NUMBER")
        ]

        if len(page_blocks) < (self.min_columns * self.min_rows):
            if drawings:
                return self.detect_border_tables(page_blocks, page_number, drawings, [])
            return []

        outer_boxes = _extract_outer_table_boxes(drawings) if drawings else []

        def get_enclosing_box(s_item: dict[str, Any]) -> tuple[float, float, float, float] | None:
            if not outer_boxes or not s_item.get("blocks"):
                return None
            s_x0 = min(b.bbox[0] for b in s_item["blocks"])
            s_x1 = max(b.bbox[2] for b in s_item["blocks"])
            s_y0 = min(b.bbox[1] for b in s_item["blocks"])
            s_y1 = max(b.bbox[3] for b in s_item["blocks"])
            for ob in outer_boxes:
                if ob[0] - 6.0 <= s_x0 and s_x1 <= ob[2] + 6.0 and ob[1] - 6.0 <= s_y0 and s_y1 <= ob[3] + 6.0:
                    return ob
            return None

        sorted_p = sorted(page_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

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
        # A table region must start on a slice with multiple columns (>= min_columns) or inside an outer table box
        candidate_tables: list[list[int]] = []
        curr_t: list[int] = []
        for s_idx, s in enumerate(slices):
            enc = get_enclosing_box(s)
            if not curr_t:
                if len(s["blocks"]) >= self.min_columns:
                    curr_t.append(s_idx)
            else:
                prev_s = slices[curr_t[-1]]
                gap = s["min_y0"] - prev_s["max_y1"]
                prev_group = [slices[i] for i in curr_t]
                compatible = self._are_slices_grid_compatible(prev_group, s, tol=22.0)
                box_curr = enc
                box_prev = get_enclosing_box(prev_s)

                is_new_header = (
                    len(curr_t) >= self.min_rows
                    and self._is_table_header_slice(s)
                    and (gap >= 8.0 or any(getattr(b, "suggested_role", None) == "TABLE_HEADER" for b in s["blocks"]))
                )

                if box_prev is not None:
                    if box_curr == box_prev:
                        curr_t.append(s_idx)
                    else:
                        if len(curr_t) >= self.min_rows and sum(1 for idx in curr_t if len(slices[idx]["blocks"]) >= self.min_columns) >= self.min_rows:
                            candidate_tables.append(curr_t)
                        curr_t = [s_idx] if len(s["blocks"]) >= self.min_columns else []
                elif box_curr is not None:
                    if len(curr_t) >= self.min_rows and sum(1 for idx in curr_t if len(slices[idx]["blocks"]) >= self.min_columns) >= self.min_rows:
                        candidate_tables.append(curr_t)
                    curr_t = [s_idx] if len(s["blocks"]) >= self.min_columns else []
                elif gap <= 25.0 and compatible and not is_new_header:
                    curr_t.append(s_idx)
                else:
                    while len(curr_t) > 2 and len(slices[curr_t[-1]]["blocks"]) < self.min_columns:
                        if slices[curr_t[-1]]["min_y0"] - slices[curr_t[-2]]["max_y1"] >= 10.0:
                            curr_t.pop()
                        else:
                            break
                    if len(curr_t) >= self.min_rows and sum(1 for idx in curr_t if len(slices[idx]["blocks"]) >= self.min_columns) >= self.min_rows:
                        candidate_tables.append(curr_t)
                    curr_t = [s_idx] if len(s["blocks"]) >= self.min_columns else []
        if curr_t:
            while len(curr_t) > 2 and len(slices[curr_t[-1]]["blocks"]) < self.min_columns:
                if get_enclosing_box(slices[curr_t[-1]]) is not None:
                    break
                if slices[curr_t[-1]]["min_y0"] - slices[curr_t[-2]]["max_y1"] >= 10.0:
                    curr_t.pop()
                else:
                    break
            if len(curr_t) >= self.min_rows and sum(1 for idx in curr_t if len(slices[idx]["blocks"]) >= self.min_columns) >= self.min_rows:
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
            max_x1 = max(b.bbox[2] for b in t_blocks)
            min_y0 = min(b.bbox[1] for b in t_blocks)
            max_y1 = max(b.bbox[3] for b in t_blocks)
            t_width = max(1.0, max_x1 - min_x0)
            max_slice_cols = max(len(s["blocks"]) for s in t_slices)

            # Discover explicit horizontal table border/grid lines from PageDrawing geometry
            clustered_h_y: list[float] = []
            if drawings:
                raw_h = [
                    d for d in drawings
                    if getattr(d, "orientation", "") == "horizontal"
                    or (getattr(d, "y1", 0) - getattr(d, "y0", 0) < 2.0 and getattr(d, "x1", 0) - getattr(d, "x0", 0) > 20.0)
                ]
                by_y: dict[float, list[tuple[float, float]]] = {}
                for d in raw_h:
                    ry = round(getattr(d, "y0", 0), 1)
                    by_y.setdefault(ry, []).append((getattr(d, "x0", 0), getattr(d, "x1", 0)))
                table_h_lines: list[float] = []
                for ry, segs in by_y.items():
                    if not (min_y0 - 15.0 <= ry <= max_y1 + 15.0):
                        continue
                    segs_sorted = sorted(segs, key=lambda s: s[0])
                    merged_span = 0.0
                    cur_x0, cur_x1 = segs_sorted[0]
                    for sx0, sx1 in segs_sorted[1:]:
                        if sx0 <= cur_x1 + 3.0:
                            cur_x1 = max(cur_x1, sx1)
                        else:
                            overlap = min(cur_x1, max_x1) - max(cur_x0, min_x0)
                            if overlap > 0:
                                merged_span += overlap
                            cur_x0, cur_x1 = sx0, sx1
                    overlap = min(cur_x1, max_x1) - max(cur_x0, min_x0)
                    if overlap > 0:
                        merged_span += overlap
                    if merged_span >= 0.45 * t_width:
                        table_h_lines.append(ry)
                for y in sorted(set(table_h_lines)):
                    if not clustered_h_y or y - clustered_h_y[-1] > 3.0:
                        clustered_h_y.append(y)

            # Internal horizontal divider lines (strictly inside the table vertical span)
            internal_h_lines = [
                y for y in clustered_h_y
                if min_y0 + 5.0 < y < max_y1 - 5.0
            ]

            logical_rows: list[list[dict[str, Any]]] = []
            curr_row_slices = [t_slices[0]]
            for i in range(1, len(t_slices)):
                prev_s = curr_row_slices[-1]
                curr_s = t_slices[i]
                gap = curr_s["min_y0"] - prev_s["max_y1"]
                first_text = curr_s["blocks"][0].text.strip()
                is_serial = bool(re.match(r"^\d+$", first_text))
                starts_table_left = (curr_s["blocks"][0].bbox[0] <= min_x0 + 15.0)

                active_row_max_y1 = max(sl["max_y1"] for sl in curr_row_slices)
                is_overlapping_active_row = (curr_s["min_y0"] < active_row_max_y1 - 1.0)

                has_border_between = None
                if internal_h_lines:
                    active_min_y0 = min(sl["min_y0"] for sl in curr_row_slices)
                    has_border_between = any(
                        active_min_y0 + 1.0 < h_y < curr_s["max_y1"] - 1.0
                        for h_y in internal_h_lines
                    )

                is_new_row = False
                if has_border_between is True:
                    # Priority 1: An explicit internal horizontal border line separates the active row from curr_s
                    is_new_row = True
                elif has_border_between is False:
                    # Inside the same horizontal border band (table has internal dividers):
                    # Wrapped text lines inside a cell must NOT become separate rows.
                    min_req = max(self.min_columns, max_slice_cols - 1)
                    if starts_table_left and len(curr_s["blocks"]) >= min_req and gap > -2.0:
                        is_new_row = True
                    else:
                        is_new_row = False
                else:
                    # Fallback for tables without internal horizontal dividers (e.g. outer-box forms or sparse tables)
                    if is_serial:
                        is_new_row = True
                    elif is_overlapping_active_row:
                        is_new_row = False
                    elif starts_table_left:
                        if len(curr_s["blocks"]) >= self.min_columns:
                            is_new_row = True
                        elif any(":" in b.text for b in curr_s["blocks"]) or gap >= self.row_gap_threshold:
                            prev_col0_b = None
                            for p_b in curr_row_slices[-1]["blocks"]:
                                if p_b.bbox[0] <= min_x0 + 15.0:
                                    prev_col0_b = p_b
                                    break
                            if prev_col0_b:
                                prev_txt = prev_col0_b.text.strip()
                                words = prev_txt.split()
                                last_word = words[-1].lower() if words else ""
                                is_wrap = (
                                    prev_txt.endswith(("-", "/", ",", ":"))
                                    or last_word in ("and", "&", "of", "for", "in", "to", "with", "the")
                                    or (first_text and first_text[0].islower())
                                )
                                if not is_wrap:
                                    is_new_row = True
                            else:
                                is_new_row = True

                if is_new_row:
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

            # Compute robust column boundary separators
            col_count = len(col_template)
            multi_slices = [s for s in t_slices if len(s["blocks"]) == col_count]
            seps: list[float] = []
            if multi_slices and len(multi_slices) >= 2:
                col_clusters: list[list[SemanticBlockInput]] = [[] for _ in range(col_count)]
                for s in multi_slices:
                    for c_i, b in enumerate(s["blocks"]):
                        col_clusters[c_i].append(b)
                for i in range(col_count - 1):
                    c_curr_max_x1 = max(b.bbox[2] for b in col_clusters[i])
                    c_next_min_x0 = min(b.bbox[0] for b in col_clusters[i + 1])
                    if c_curr_max_x1 < c_next_min_x0:
                        seps.append((c_curr_max_x1 + c_next_min_x0) / 2.0)
                    else:
                        c_curr = col_template[i]
                        c_next = col_template[i + 1]
                        seps.append((c_curr[1] + c_next[0]) / 2.0)
            else:
                for i in range(len(col_template) - 1):
                    c_curr = col_template[i]
                    c_next = col_template[i + 1]
                    seps.append((c_curr[1] + c_next[0]) / 2.0)

            col_bands: list[tuple[float, float]] = []
            for i in range(len(col_template)):
                b_left = 0.0 if i == 0 else seps[i - 1]
                b_right = 10000.0 if i == len(col_template) - 1 else seps[i]
                col_bands.append((b_left, b_right))

            # Compute relative percentage column widths
            t_min_x = min(b.bbox[0] for b in t_blocks)
            t_max_x = max(b.bbox[2] for b in t_blocks)
            total_w = max(1.0, t_max_x - t_min_x)
            column_widths: list[float] = []
            for i in range(len(col_template)):
                cx0 = t_min_x if i == 0 else seps[i - 1]
                cx1 = t_max_x if i == len(col_template) - 1 else seps[i]
                w_pct = round(((cx1 - cx0) / total_w) * 100.0, 1)
                column_widths.append(w_pct)

            # Conservative row density check: ensure candidate is not a single-column block run with an isolated date
            has_header_row = self._is_table_header_slice(t_slices[0])
            row_multi_cols = [r for r in logical_rows if len(r) >= 2 or sum(len(sl["blocks"]) for sl in r) >= 2]
            if not has_header_row:
                if len(col_template) >= 3:
                    if len(logical_rows) > 4 and len(row_multi_cols) / len(logical_rows) < 0.40:
                        continue
                elif len(col_template) == 2:
                    if len(logical_rows) > 3 and len(row_multi_cols) / len(logical_rows) < 0.60:
                        continue

            table_id = f"table_p{page_number}_{len(detected_tables)}"

            # Build cell bindings with word-level geometric column assignment
            cells: list[GeometricCell] = []
            num_data_rows = 0
            for r_idx, r_slices in enumerate(logical_rows):
                is_header = (r_idx == 0 and has_header_row)
                role = "HEADER" if is_header else "DATA"
                if not is_header:
                    num_data_rows += 1
                for s in r_slices:
                    for b in s["blocks"]:
                        split_cells = self._split_block_by_columns(b, seps, r_idx, table_id, role)
                        cells.extend(split_cells)

            # Determine enclosing box and visual geometry
            table_enclosing_box = None
            for ob in outer_boxes:
                if ob[0] - 8.0 <= min_x0 and max_x1 <= ob[2] + 8.0 and ob[1] - 8.0 <= min_y0 and max_y1 <= ob[3] + 8.0:
                    table_enclosing_box = ob
                    break

            visual_geom = _compute_table_visual_geometry(
                t_blocks,
                seps,
                logical_rows,
                drawings,
                enclosing_box=table_enclosing_box,
            )

            table = GeometricTable(
                table_id=table_id,
                page=page_number,
                num_columns=len(col_template),
                num_rows=len(logical_rows),
                num_data_rows=num_data_rows,
                column_bands=col_bands,
                cells=cells,
                column_widths=column_widths,
                is_border_defined=bool(internal_h_lines or table_enclosing_box or len(clustered_h_y) >= 2),
                visual_geometry=visual_geom,
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

        if drawings:
            border_tables = self.detect_border_tables(blocks, page_number, drawings, detected_tables)
            detected_tables.extend(border_tables)

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
                first_slice_blocks = slices[idx]["blocks"]
                if len(first_slice_blocks) >= self.min_columns:
                    if self._is_table_header_slice(slices[idx]) or len(first_slice_blocks) != table.num_columns:
                        break
                    assigned = [next((i for i, sep in enumerate(seps) if ((b.bbox[0] + b.bbox[2]) / 2.0) < sep), len(seps)) for b in first_slice_blocks]
                    if len(set(assigned)) != len(first_slice_blocks):
                        break

            s = slices[idx]
            gap = s["min_y0"] - prev_max_y

            # Table continuation halts if inter-row spacing exceeds threshold or is inverted
            if gap < -2.0 or gap > max_inter_row_gap:
                break

            # If table is bounded by an outer box, continuation cannot exceed outer box bottom
            if table.visual_geometry and table.visual_geometry.get("outer_bounds"):
                ob_bottom = table.visual_geometry["outer_bounds"][3]
                if s["min_y0"] > ob_bottom + 4.0:
                    break

            # Table continuation halts immediately if any block is a section heading, heading candidate, table header, or page footer/number
            if any(
                b.suggested_role in ("SECTION_HEADING", "TABLE_HEADER", "FOOTER", "PAGE_NUMBER")
                or getattr(b, "region_kind", None) in ("header", "footer")
                or re.search(r"\bpage\s+\d+\s+of\s+\d+\b", b.text, re.IGNORECASE)
                or getattr(b, "heading_candidate", False)
                for b in s["blocks"]
            ) or self._is_table_header_slice(s):
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
                ) or self._is_table_header_slice(next_s):
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

    def detect_border_tables(
        self,
        blocks: list[SemanticBlockInput],
        page_number: int,
        page_drawings: list[Any],
        existing_tables: list[GeometricTable],
        tol: float = 4.0,
    ) -> list[GeometricTable]:
        """Detect tables defined by explicit vector drawing borders/grid lines."""
        if not page_drawings:
            return []

        h_lines = [
            d for d in page_drawings
            if getattr(d, "orientation", "") == "horizontal"
            or (getattr(d, "y1", 0) - getattr(d, "y0", 0) < 2.0 and getattr(d, "x1", 0) - getattr(d, "x0", 0) > 20.0)
        ]
        v_lines = [
            d for d in page_drawings
            if getattr(d, "orientation", "") == "vertical"
            or (getattr(d, "x1", 0) - getattr(d, "x0", 0) < 2.0 and getattr(d, "y1", 0) - getattr(d, "y0", 0) > 5.0)
        ]

        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        candidate_boxes: list[tuple[float, float, float, float]] = []
        for i, v1 in enumerate(v_lines):
            for j, v2 in enumerate(v_lines):
                if i >= j:
                    continue
                v1_x0, v1_y0, v1_y1 = v1.x0, v1.y0, v1.y1
                v2_x0, v2_y0, v2_y1 = v2.x0, v2.y0, v2.y1
                if v1_x0 > v2_x0:
                    v1_x0, v2_x0 = v2_x0, v1_x0
                    v1_y0, v2_y0 = v2_y0, v1_y0
                    v1_y1, v2_y1 = v2_y1, v1_y1

                if v2_x0 - v1_x0 > 80.0 and abs(v1_y0 - v2_y0) <= tol and abs(v1_y1 - v2_y1) <= tol:
                    y_top = min(v1_y0, v2_y0)
                    y_bot = max(v1_y1, v2_y1)
                    x_left = v1_x0
                    x_right = v2_x0
                    has_top = any(
                        abs(h.y0 - y_top) <= tol and h.x0 <= x_left + tol and h.x1 >= x_right - tol
                        for h in h_lines
                    )
                    has_bot = any(
                        abs(h.y0 - y_bot) <= tol and h.x0 <= x_left + tol and h.x1 >= x_right - tol
                        for h in h_lines
                    )
                    if has_top and has_bot:
                        candidate_boxes.append((x_left, y_top, x_right, y_bot))

        table_boxes: list[tuple[float, float, float, float]] = []
        for box in sorted(candidate_boxes, key=lambda b: (b[1], -(b[2] - b[0]))):
            x0, y0, x1, y1 = box
            if not any(abs(y0 - tb[1]) <= tol and abs(y1 - tb[3]) <= tol for tb in table_boxes):
                table_boxes.append(box)

        existing_y_ranges: list[tuple[float, float]] = [
            (min(c.bbox[1] for c in t.cells), max(c.bbox[3] for c in t.cells))
            for t in existing_tables if t.cells
        ]
        existing_block_ids: set[str] = set()
        for t in existing_tables:
            for c in t.cells:
                existing_block_ids.add(c.block_id)
                if c.parent_block_id:
                    existing_block_ids.add(c.parent_block_id)

        new_tables: list[GeometricTable] = []
        for box in table_boxes:
            x_left, y_top, x_right, y_bot = box
            if any(min(y_bot, ey1) - max(y_top, ey0) > 0.4 * (y_bot - y_top) for ey0, ey1 in existing_y_ranges):
                continue

            b_in_box = [
                b for b in blocks
                if b.page == page_number
                and y_top - tol <= (b.bbox[1] + b.bbox[3]) / 2.0 <= y_bot + tol
                and x_left - tol <= (b.bbox[0] + b.bbox[2]) / 2.0 <= x_right + tol
            ]
            if not b_in_box:
                continue

            if any(b.block_id in existing_block_ids for b in b_in_box):
                continue

            box_h = [
                h for h in h_lines
                if y_top - tol <= h.y0 <= y_bot + tol
                and not (h.x1 < x_left - tol or h.x0 > x_right + tol)
            ]
            sorted_y = sorted(set(round(h.y0, 1) for h in box_h))
            clustered_y: list[float] = []
            for y in sorted_y:
                if not clustered_y or y - clustered_y[-1] > tol:
                    clustered_y.append(y)
            if len(clustered_y) < 2:
                clustered_y = [y_top, y_bot]

            num_rows = len(clustered_y) - 1
            row_bands = [(clustered_y[r], clustered_y[r + 1]) for r in range(num_rows)]

            inner_v = [
                v for v in v_lines
                if x_left + 10.0 < v.x0 < x_right - 10.0
                and not (v.y1 < y_top - tol or v.y0 > y_bot + tol)
            ]
            row_divs: list[list[float]] = []
            for (ry0, ry1) in row_bands:
                divs = [v.x0 for v in inner_v if min(v.y1, ry1) - max(v.y0, ry0) > 3.0]
                row_divs.append(sorted(divs))

            all_inner_x = sorted(set(round(v.x0, 1) for v in inner_v))
            clustered_divs: list[float] = []
            for x in all_inner_x:
                if not clustered_divs or x - clustered_divs[-1] > 15.0:
                    clustered_divs.append(x)

            max_divs = max((len(d) for d in row_divs), default=0)
            if max_divs == 0:
                num_cols = 1
                seps = []
            elif max_divs == 1:
                num_cols = 2
                div_vals = [d[0] for d in row_divs if d]
                seps = [sum(div_vals) / len(div_vals)] if div_vals else (clustered_divs[:1] if clustered_divs else [])
            else:
                num_cols = max(max_divs + 1, len(clustered_divs) + 1)
                seps = clustered_divs

            # A table must have at least 2 columns; a 1-column box is simply a framed paragraph/callout
            if num_cols < 2:
                continue

            total_w = max(1.0, x_right - x_left)
            col_bounds = [x_left] + seps + [x_right]
            col_bands = [(col_bounds[i], col_bounds[i + 1]) for i in range(len(col_bounds) - 1)]
            column_widths = [round(((cb[1] - cb[0]) / total_w) * 100.0, 1) for cb in col_bands]

            table_id = f"table_p{page_number}_{len(existing_tables) + len(new_tables)}"
            cells: list[GeometricCell] = []
            for b in b_in_box:
                b_ymid = (b.bbox[1] + b.bbox[3]) / 2.0
                r_idx = 0
                for idx, (ry0, ry1) in enumerate(row_bands):
                    if ry0 - tol <= b_ymid <= ry1 + tol:
                        r_idx = idx
                        break

                r_seps = [row_divs[r_idx][0]] if (num_cols == 2 and row_divs[r_idx]) else seps
                b_cells = self._split_block_by_columns(b, r_seps, r_idx, table_id, "DATA")
                cells.extend(b_cells)

            border_visual_geom = _compute_table_visual_geometry(
                b_in_box,
                seps,
                [[{"blocks": [c]} for c in cells if c.row_index == r_i] for r_i in range(num_rows)],
                page_drawings,
                enclosing_box=(x_left, y_top, x_right, y_bot),
            )

            t = GeometricTable(
                table_id=table_id,
                page=page_number,
                num_columns=num_cols,
                num_rows=num_rows,
                num_data_rows=num_rows,
                column_bands=col_bands,
                cells=cells,
                column_widths=column_widths,
                is_border_defined=True,
                visual_geometry=border_visual_geom,
            )
            new_tables.append(t)

        return new_tables

    def detect_document_tables(
        self,
        blocks: list[SemanticBlockInput],
        drawings: list[Any] | None = None,
    ) -> list[GeometricTable]:
        """Detect all geometric tables across all pages of a document."""
        by_page: dict[int, list[SemanticBlockInput]] = {}
        for b in blocks:
            by_page.setdefault(b.page, []).append(b)

        drawings_by_page: dict[int, list[Any]] = {}
        if drawings:
            for d in drawings:
                p_num = getattr(d, "page_number", 1)
                drawings_by_page.setdefault(p_num, []).append(d)

        all_tables: list[GeometricTable] = []
        for page_num in sorted(by_page.keys()):
            p_blocks = by_page[page_num]
            p_drawings = drawings_by_page.get(page_num, [])
            tables = self.detect_page_tables(p_blocks, page_num, drawings=p_drawings)
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
