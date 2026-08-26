from __future__ import annotations

from dataclasses import replace

from app.domain.document import BoundingBox, Document, Line, Page, Region


_MIN_COLUMN_GAP = 32.0
_COLUMN_GAP_RATIO = 0.08
_HEADER_BAND_FACTOR = 1.75


def interpret_layout(document: Document) -> Document:
    """Derive page-local physical regions and paths without semantic labeling."""
    pages: list[Page] = []
    for page in document.pages:
        pages.append(_interpret_page(page))
    return replace(document, pages=pages)


def _interpret_page(page: Page) -> Page:
    source_lines = [line for region in page.regions for line in region.lines]
    if not source_lines:
        return replace(page, regions=[])

    columns = _column_groups(source_lines)
    if len(columns) < 2:
        region = _make_region(page.page_number, 0, "physical_region", source_lines, None)
        return replace(page, regions=[region])

    body_start = _body_start_y(source_lines, columns)
    page_width = max(line.bbox.x1 for line in source_lines) - min(line.bbox.x0 for line in source_lines)
    header_lines = [
        line
        for line in source_lines
        if line.bbox.y0 < body_start
        or (
            line.bbox.y0 == min(item.bbox.y0 for item in source_lines)
            and _is_wide_header(line, page_width, body_columns=columns)
        )
    ]
    body_lines = [line for line in source_lines if line not in header_lines]

    body_columns = _column_groups(body_lines)
    if len(body_columns) < 2:
        region = _make_region(page.page_number, 0, "physical_region", source_lines, None)
        return replace(page, regions=[region])

    regions: list[Region] = []
    if header_lines:
        regions.append(_make_region(page.page_number, 0, "header", header_lines, None))

    for offset, column_lines in enumerate(body_columns):
        regions.append(
            _make_region(
                page.page_number,
                len(regions),
                "column",
                column_lines,
                offset,
            )
        )

    return replace(page, regions=regions)


def _column_groups(lines: list[Line]) -> list[list[Line]]:
    if len(lines) < 2:
        return [sorted(lines, key=_line_position)] if lines else []

    ordered = sorted(lines, key=lambda line: line.bbox.x0)
    min_x = min(line.bbox.x0 for line in ordered)
    max_x = max(line.bbox.x1 for line in ordered)
    span = max(max_x - min_x, 1.0)
    gap_threshold = max(_MIN_COLUMN_GAP, span * _COLUMN_GAP_RATIO)

    split_points: list[int] = []
    for index in range(1, len(ordered)):
        if ordered[index].bbox.x0 - ordered[index - 1].bbox.x0 >= gap_threshold:
            split_points.append(index)

    if not split_points:
        return [sorted(lines, key=_line_position)]

    groups: list[list[Line]] = []
    start = 0
    for end in split_points + [len(ordered)]:
        groups.append(sorted(ordered[start:end], key=_line_position))
        start = end
    return [group for group in groups if group]


def _body_start_y(lines: list[Line], columns: list[list[Line]]) -> float:
    if len(columns) < 2:
        return min(line.bbox.y0 for line in lines)

    heights = [max(line.bbox.y1 - line.bbox.y0, 1.0) for line in lines]
    heights.sort()
    median_height = heights[len(heights) // 2]
    band = median_height * _HEADER_BAND_FACTOR
    ordered = sorted(lines, key=_line_position)

    for candidate in ordered:
        present = {
            column_index
            for column_index, column in enumerate(columns)
            if any(
                line.page_number == candidate.page_number
                and abs(line.bbox.y0 - candidate.bbox.y0) <= band
                for line in column
            )
        }
        if len(present) >= 2:
            return candidate.bbox.y0

    return min(line.bbox.y0 for line in lines)


def _is_wide_header(line: Line, page_width: float, body_columns: list[list[Line]]) -> bool:
    if line.bbox.x1 - line.bbox.x0 < page_width * 0.65:
        return False

    overlapping_columns = 0
    for column in body_columns:
        column_x0 = min(item.bbox.x0 for item in column)
        column_x1 = max(item.bbox.x1 for item in column)
        if line.bbox.x0 <= column_x1 and line.bbox.x1 >= column_x0:
            overlapping_columns += 1
    return overlapping_columns >= 2


def _make_region(
    page_number: int,
    region_index: int,
    kind: str,
    lines: list[Line],
    column_id: int | None,
) -> Region:
    ordered_lines = sorted(lines, key=_line_position)
    bbox = _bounds(ordered_lines)
    return Region(
        region_id=f"page-{page_number}-region-{region_index}",
        kind=kind,
        bbox=bbox,
        lines=ordered_lines,
        reading_order=region_index,
        column_id=column_id,
    )


def _bounds(lines: list[Line]) -> BoundingBox:
    return BoundingBox(
        x0=min(line.bbox.x0 for line in lines),
        y0=min(line.bbox.y0 for line in lines),
        x1=max(line.bbox.x1 for line in lines),
        y1=max(line.bbox.y1 for line in lines),
    )


def _line_position(line: Line) -> tuple[float, float, str]:
    return line.bbox.y0, line.bbox.x0, line.line_id
