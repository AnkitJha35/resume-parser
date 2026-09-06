from __future__ import annotations

import re
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
    body_start = _body_start_y(source_lines, columns)
    # A section heading must never be swallowed into the header band.
    # Cap body_start at the first section heading found in the candidate header area.
    body_start = _cap_body_start_at_section_heading(source_lines, body_start)
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
    if not body_lines:
        kind = "header" if header_lines else "physical_region"
        region = _make_region(page.page_number, 0, kind, source_lines, None)
        return replace(page, regions=[region])

    body_columns = _column_groups(body_lines)
    regions: list[Region] = []
    if header_lines:
        regions.append(_make_region(page.page_number, 0, "header", header_lines, None))

    if len(body_columns) < 2:
        regions.append(
            _make_region(page.page_number, len(regions), "physical_region", body_lines, None)
        )
        return replace(page, regions=regions)

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
    return _merge_embedded_column_groups([group for group in groups if group])


_EMBEDDED_OVERLAP = 0.60
_EMBEDDED_NARROW = 0.55
_EMBEDDED_SHORT = 0.60
_INDEPENDENT_LEFT_SLACK = 8.0


def _merge_embedded_column_groups(groups: list[list[Line]]) -> list[list[Line]]:
    """Fold short/narrow x0 clusters into a wider containing sibling.

    Tentative splits use consecutive x0 gaps. Intra-column grids, right-aligned
    meta on the same row, rating ticks, and date gutters share a parent column's
    x-span and must not become page columns. Overlapping *real* columns stay
    separate: they are not horizontally contained and have independent x0 modes.
    """
    if len(groups) < 2:
        return groups

    merged = [list(group) for group in groups]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        bounds = [_bounds(group) for group in merged]
        embed_index = None
        parent_index = None
        for index, cluster in enumerate(merged):
            # Candidates are ranked containment-first, so a marginally wider
            # sibling that does not overlap cannot out-rank one that actually
            # contains the cluster. Try each in turn: a rejected first choice
            # must not orphan a cluster a later candidate would accept.
            for parent in _containing_sibling_candidates(index, bounds):
                if _is_embedded_subcolumn(cluster, merged[parent], bounds[index], bounds[parent]):
                    parent_index = parent
                    break
            if parent_index is None:
                continue
            embed_index = index
            break
        if embed_index is None or parent_index is None:
            break
        merged[parent_index] = sorted(merged[parent_index] + merged[embed_index], key=_line_position)
        del merged[embed_index]
        changed = True

    return [sorted(group, key=_line_position) for group in merged]


def _widest_containing_sibling(index: int, bounds: list[BoundingBox]) -> int | None:
    """Best-ranked sibling eligible to adopt ``bounds[index]``, or None."""
    candidates = _containing_sibling_candidates(index, bounds)
    return candidates[0] if candidates else None


def _containing_sibling_candidates(index: int, bounds: list[BoundingBox]) -> list[int]:
    """Sibling indices eligible to adopt ``bounds[index]``, best candidate first.

    Ranked by horizontal containment, then by width. Width alone is not a proxy
    for ownership: a wider sibling with no horizontal overlap is a different page
    column, not a parent. Admission gates below are unchanged -- only the
    ordering, and the fact that all admitted candidates are returned rather than
    just one, is new.
    """
    cluster = bounds[index]
    cluster_width = max(cluster.x1 - cluster.x0, 1.0)
    ranked: list[tuple[float, float, int]] = []
    for sibling_index, sibling in enumerate(bounds):
        if sibling_index == index:
            continue
        sibling_width = max(sibling.x1 - sibling.x0, 1.0)
        if sibling_width <= cluster_width:
            continue
        overlap_ratio = _horizontal_overlap(cluster, sibling) / cluster_width
        if overlap_ratio < _EMBEDDED_OVERLAP:
            indented = sibling.x0 - 4.0 <= cluster.x0 <= sibling.x1
            gutter = cluster.x0 - sibling.x1
            tick_gutter = 0.0 <= gutter <= max(48.0, sibling_width * 0.25)
            sparse_right = (
                cluster.x0 > sibling.x0
                and (sibling.y1 - sibling.y0) > 0
                and (cluster.y1 - cluster.y0) / max(sibling.y1 - sibling.y0, 1.0) <= 0.40
            )
            if not indented and not tick_gutter and not sparse_right:
                continue
        if cluster.x0 + _INDEPENDENT_LEFT_SLACK < sibling.x0:
            continue
        ranked.append((overlap_ratio, sibling_width, sibling_index))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[2] for item in ranked]


def _is_embedded_subcolumn(
    cluster: list[Line],
    sibling: list[Line],
    cluster_box: BoundingBox,
    sibling_box: BoundingBox,
) -> bool:
    _ = sibling
    cluster_width = max(cluster_box.x1 - cluster_box.x0, 1.0)
    sibling_width = max(sibling_box.x1 - sibling_box.x0, 1.0)
    cluster_height = max(cluster_box.y1 - cluster_box.y0, 1.0)
    sibling_height = max(sibling_box.y1 - sibling_box.y0, 1.0)
    if sibling_width <= cluster_width:
        return False
    overlap = _horizontal_overlap(cluster_box, sibling_box)
    overlap_ratio = overlap / cluster_width
    if cluster_box.x0 + _INDEPENDENT_LEFT_SLACK < sibling_box.x0:
        return False

    indented = sibling_box.x0 - 4.0 <= cluster_box.x0 <= sibling_box.x1
    gutter = cluster_box.x0 - sibling_box.x1
    tick_gutter = (
        0.0 <= gutter <= max(48.0, sibling_width * 0.25)
        and cluster_width / sibling_width <= 0.35
        and len(cluster) <= 10
    )
    sparse_fragment = len(cluster) <= 2 and cluster_height / sibling_height <= 0.40
    contained = overlap_ratio >= _EMBEDDED_OVERLAP or (indented and len(cluster) <= 3)
    if not contained and not tick_gutter and not sparse_fragment:
        return False
    # Partial overlap that spills past a sibling is a real adjacent column, not a
    # tick/gutter fragment sitting just outside the sibling's x1.
    extends_past_sibling = cluster_box.x1 > sibling_box.x1 + max(24.0, sibling_width * 0.15)
    if (
        extends_past_sibling
        and overlap_ratio < 0.85
        and len(cluster) > 3
        and not tick_gutter
        and not sparse_fragment
    ):
        return False

    narrow = cluster_width / sibling_width <= _EMBEDDED_NARROW
    short = cluster_height / sibling_height <= _EMBEDDED_SHORT
    sparse = len(cluster) <= 3 and cluster_height / sibling_height <= 0.85
    return narrow or short or sparse or tick_gutter or sparse_fragment


def _horizontal_overlap(left: BoundingBox, right: BoundingBox) -> float:
    return max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))


_CONTACT_LABEL = re.compile(
    r"^\s*(?:e-?mail|phone|tel|mobile|cell|linkedin|github|portfolio|website)\b",
    re.IGNORECASE,
)
_URL_OR_HANDLE = re.compile(r"https?://|www\.|linkedin\.com|github\.com|mailto:", re.IGNORECASE)
_PHONE_DIGITS = re.compile(r"\d+")
_YEAR_RANGE = re.compile(r"\b(?:19|20)\d{2}\s*[-–—/]\s*(?:19|20)\d{2}\b")


# Matches all-caps section heading text: only uppercase letters, spaces, and a small set of
# punctuation characters allowed in heading text such as '&', '/', '(', ')'.
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z\s&/()\-]+$")


def _is_section_heading_line(line: Line, median_size: float) -> bool:
    """Return True when *line* looks like a body section heading.

    Conservatively detects section headings using typography and text shape
    signals only — no fixture-specific strings.  Two independent signals:

    1. **All-caps phrase**: text matches ``_ALL_CAPS_HEADING``, has 2–6 words,
       contains at least two alphabetic characters, and is not a contact line.
    2. **Bold oversized short phrase**: bold=True, font_size ≥ median * 1.15,
       word count 2–6, and not a contact line.

    Name-like lines (very large font ≥ median * 1.35, very short word count)
    are excluded because they are letterhead, not section headings.
    """
    text = (line.text or "").strip()
    if not text or _is_contact_line(line):
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 6:
        return False
    # Exclude name-sized lines: same threshold as _is_name_like_line.
    size = line.style.font_size or 0.0
    if size >= max(12.0, median_size * 1.35):
        return False
    # All-caps heading: letters, spaces, and limited punctuation only.
    if _ALL_CAPS_HEADING.match(text) and sum(c.isalpha() for c in text) >= 4:
        return True
    # Bold oversized heading: bold formatting at a clearly larger font.
    if line.style.bold and size >= max(10.0, median_size * 1.15):
        return True
    return False


def _cap_body_start_at_section_heading(lines: list[Line], body_start: float) -> float:
    """Cap *body_start* so that section headings are never swallowed into the header band.

    Scans lines with ``y0 < body_start`` in top-to-bottom order and returns
    the ``y0`` of the first line that satisfies :func:`_is_section_heading_line`.
    If no such line exists, returns the original *body_start* unchanged.
    """
    candidate_header = [line for line in lines if line.bbox.y0 < body_start]
    if not candidate_header:
        return body_start
    sizes = sorted(line.style.font_size or 10.0 for line in lines)
    median_size = sizes[len(sizes) // 2]
    for line in sorted(candidate_header, key=_line_position):
        if _is_section_heading_line(line, median_size):
            return line.bbox.y0
    return body_start


def _body_start_y(lines: list[Line], columns: list[list[Line]]) -> float:
    letterhead_start = _letterhead_body_start_y(lines)
    coincidence = _column_coincidence_y(lines, columns)
    first_prose = _first_wide_prose_y(lines)
    page_top = min(line.bbox.y0 for line in lines)

    if coincidence is None or abs(coincidence - page_top) < 1.0:
        return letterhead_start
    if letterhead_start < coincidence - 1.0 and _opens_body_content(lines, letterhead_start):
        return letterhead_start
    if first_prose is not None and first_prose < coincidence - 1.0:
        return min(letterhead_start, first_prose)
    return coincidence


def _column_coincidence_y(lines: list[Line], columns: list[list[Line]]) -> float | None:
    if len(columns) < 2:
        return None

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
    return None


def _letterhead_body_start_y(lines: list[Line]) -> float:
    ordered = sorted(lines, key=_line_position)
    page_span = max(line.bbox.x1 for line in ordered) - min(line.bbox.x0 for line in ordered)
    page_span = max(page_span, 1.0)
    sizes = sorted(line.style.font_size or 10.0 for line in ordered)
    median_size = sizes[len(sizes) // 2]
    page_top = min(line.bbox.y0 for line in ordered)
    page_height = max(line.bbox.y1 for line in ordered) - page_top
    name_band = page_top + max(72.0, page_height * 0.10)
    seen_letterhead = False
    used_tagline = False
    seen_contact = False

    for line in ordered:
        if _is_contact_line(line) or _is_name_like_line(line, median_size, name_band):
            seen_letterhead = True
            if _is_contact_line(line):
                seen_contact = True
            continue
        if (
            seen_letterhead
            and not used_tagline
            and not seen_contact
            and _is_tagline_line(line, page_span)
        ):
            used_tagline = True
            continue
        if seen_letterhead and _is_compact_letterhead_line(line, page_span):
            continue
        if seen_letterhead:
            return line.bbox.y0
    return min(line.bbox.y0 for line in ordered)


def _first_wide_prose_y(lines: list[Line]) -> float | None:
    page_span = max(line.bbox.x1 for line in lines) - min(line.bbox.x0 for line in lines)
    page_span = max(page_span, 1.0)
    for line in sorted(lines, key=_line_position):
        if _is_wide_prose_line(line, page_span):
            return line.bbox.y0
    return None


def _is_contact_line(line: Line) -> bool:
    text = (line.text or "").strip()
    if not text:
        return False
    if "@" in text or _URL_OR_HANDLE.search(text) or _CONTACT_LABEL.match(text):
        return True
    if _YEAR_RANGE.search(text):
        return False
    digits = "".join(_PHONE_DIGITS.findall(text))
    return len(digits) >= 10 and len(text.split()) <= 10


def _is_name_like_line(line: Line, median_size: float, name_band: float | None = None) -> bool:
    text = (line.text or "").strip()
    words = text.split()
    if not text or len(words) > 5 or _is_contact_line(line):
        return False
    if name_band is not None and line.bbox.y0 > name_band:
        return False
    size = line.style.font_size or 0.0
    return size >= max(12.0, median_size * 1.35)


def _is_tagline_line(line: Line, page_span: float) -> bool:
    text = (line.text or "").strip()
    words = text.split()
    if not text or len(words) < 2 or len(words) > 12 or _is_contact_line(line):
        return False
    if (line.style.font_size or 0.0) >= 13.0:
        return False
    if _is_wide_prose_line(line, page_span):
        return False
    width = line.bbox.x1 - line.bbox.x0
    return width / page_span < 0.70


def _is_compact_letterhead_line(line: Line, page_span: float) -> bool:
    text = (line.text or "").strip()
    words = text.split()
    if not text or _is_wide_prose_line(line, page_span) or _is_contact_line(line):
        return False
    width = line.bbox.x1 - line.bbox.x0
    if width / page_span >= 0.60 or len(words) > 10:
        return False
    # Short address / meta rows. Do not keep emphasized section-like titles.
    size = line.style.font_size or 0.0
    if size >= 12.5 or (line.style.bold and len(words) <= 3):
        return False
    return True


def _is_wide_prose_line(line: Line, page_span: float) -> bool:
    text = (line.text or "").strip()
    if not text or _is_contact_line(line):
        return False
    width = line.bbox.x1 - line.bbox.x0
    return width / page_span >= 0.70 and len(text.split()) >= 8


def _opens_body_content(lines: list[Line], y0: float) -> bool:
    for line in sorted(lines, key=_line_position):
        if abs(line.bbox.y0 - y0) > 1.0:
            continue
        text = (line.text or "").strip()
        words = text.split()
        if not text or _is_contact_line(line):
            continue
        if len(words) >= 8:
            return True
        if len(words) <= 4 and ((line.style.font_size or 0.0) >= 12.0 or line.style.bold):
            return False
        return len(words) >= 6
    return False


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
