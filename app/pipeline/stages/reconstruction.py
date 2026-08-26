from __future__ import annotations

from dataclasses import replace

from app.domain.document import Document, Line


def reconstruct_document(document: Document) -> Document:
    """Merge only geometrically adjacent physical lines within each region."""
    pages = []
    for page in document.pages:
        regions = []
        for region in page.regions:
            reconstructed_lines: list[Line] = []
            for line in sorted(region.lines, key=lambda item: (item.bbox.y0, item.bbox.x0)):
                merge_index = next(
                    (
                        index
                        for index in range(len(reconstructed_lines) - 1, max(-1, len(reconstructed_lines) - 9), -1)
                        if _can_merge(reconstructed_lines[index], line)
                    ),
                    None,
                )
                if merge_index is None:
                    reconstructed_lines.append(line)
                else:
                    reconstructed_lines[merge_index] = _merge_lines(reconstructed_lines[merge_index], line)
            regions.append(replace(region, lines=reconstructed_lines))
        pages.append(replace(page, regions=regions))
    return replace(document, pages=pages)


def _can_merge(previous: Line, current: Line) -> bool:
    if previous.page_number != current.page_number:
        return False

    previous_height = max(previous.bbox.y1 - previous.bbox.y0, 0.0)
    current_height = max(current.bbox.y1 - current.bbox.y0, 0.0)
    previous_width = max(previous.bbox.x1 - previous.bbox.x0, 0.0)
    current_width = max(current.bbox.x1 - current.bbox.x0, 0.0)
    vertical_tolerance = max(previous_height, current_height, 1.0) * 0.25
    same_baseline = abs(previous.bbox.y0 - current.bbox.y0) <= vertical_tolerance
    wrapped_continuation = (
        abs(current.bbox.y0 - previous.bbox.y1) <= vertical_tolerance
        and abs(current.bbox.x1 - previous.bbox.x1) <= _horizontal_tolerance_for(previous, current)
        and current.bbox.x0 > previous.bbox.x0
        and current_width <= previous_width * 0.8
    )
    if not same_baseline and not wrapped_continuation:
        return False

    horizontal_gap = current.bbox.x0 - previous.bbox.x1
    if wrapped_continuation:
        return True
    horizontal_tolerance = _horizontal_tolerance_for(previous, current)
    return 0.0 <= horizontal_gap <= horizontal_tolerance


def _horizontal_tolerance_for(previous: Line, current: Line) -> float:
    font_size = current.style.font_size or previous.style.font_size or 0.0
    return max(font_size * 1.5, 4.0)


def _merge_lines(previous: Line, current: Line) -> Line:
    text_separator = "" if (
        _is_continuation(previous.text, current.text)
        or _is_compact_adjacent_fragment(previous, current)
        or _is_wrapped_continuation(previous, current)
    ) else " "
    return replace(
        previous,
        bbox=type(previous.bbox)(
            x0=min(previous.bbox.x0, current.bbox.x0),
            y0=min(previous.bbox.y0, current.bbox.y0),
            x1=max(previous.bbox.x1, current.bbox.x1),
            y1=max(previous.bbox.y1, current.bbox.y1),
        ),
        spans=previous.spans + current.spans,
        text=previous.text + text_separator + current.text,
        source_span_ids=previous.source_span_ids + current.source_span_ids,
        reconstruction_method="adjacent_physical_fragments",
    )


def _is_continuation(previous: str, current: str) -> bool:
    return previous.endswith(("/", "-", "@", ".")) or current.startswith(('.', ',', '/', ')', ':', ';'))


def _is_compact_adjacent_fragment(previous: Line, current: Line) -> bool:
    previous_width = max(previous.bbox.x1 - previous.bbox.x0, 0.0)
    current_width = max(current.bbox.x1 - current.bbox.x0, 0.0)
    return current_width <= previous_width * 0.2 and current.bbox.x0 >= previous.bbox.x1


def _is_wrapped_continuation(previous: Line, current: Line) -> bool:
    return (
        current.bbox.y0 != previous.bbox.y0
        and abs(current.bbox.y0 - previous.bbox.y1)
        <= max(previous.bbox.y1 - previous.bbox.y0, current.bbox.y1 - current.bbox.y0, 1.0) * 0.25
        and current.bbox.x0 > previous.bbox.x0
    )