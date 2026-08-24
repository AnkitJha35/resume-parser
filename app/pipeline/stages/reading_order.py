from __future__ import annotations

from typing import Iterable, List

from app.pipeline.stages.text_extraction import TextBlock


class ReadingOrder:
    @staticmethod
    def reorder(blocks: Iterable[TextBlock]) -> list[TextBlock]:
        pages: dict[int, list[TextBlock]] = {}
        for b in blocks:
            pages.setdefault(b.page_number, []).append(b)

        ordered: list[TextBlock] = []
        for page_number in sorted(pages.keys()):
            page_blocks = pages[page_number]
            regions = ReadingOrder.cluster_blocks_by_x0(page_blocks)

            if len(regions) == 1:
                ordered.extend(sorted(page_blocks, key=lambda b: (b.y0, b.x0)))
                continue

            # Preserve visual order within each region and read left-to-right for
            # genuinely separate regions. This keeps a simple single-column page
            # unchanged while allowing multi-column pages to remain stable.
            for region in regions:
                ordered.extend(sorted(region, key=lambda b: (b.y0, b.x0)))

        return ordered

    @staticmethod
    def cluster_blocks_by_x0(blocks: Iterable[TextBlock]) -> list[list[TextBlock]]:
        blocks_list = list(blocks)
        if not blocks_list:
            return []

        def _is_artifact(text: str) -> bool:
            text = (text or "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").strip()
            if not text:
                return True
            if text in {"•", "●", "-", "*", "·", "."}:
                return True
            if len(text) <= 2 and all(not ch.isalnum() for ch in text):
                return True
            return False

        filtered = [b for b in blocks_list if not _is_artifact(getattr(b, "text", "") or "")]
        if not filtered:
            return [sorted(blocks_list, key=lambda b: (b.y0, b.x0))]

        sorted_by_y = sorted(filtered, key=lambda b: (b.y0, b.x0))
        if len(filtered) <= 1:
            return [sorted_by_y]

        sorted_by_x = sorted(filtered, key=lambda b: b.x0)
        strongest_gap = 0.0
        split_index = 0
        for idx in range(1, len(sorted_by_x)):
            gap = sorted_by_x[idx].x0 - sorted_by_x[idx - 1].x0
            if gap > strongest_gap:
                strongest_gap = gap
                split_index = idx

        min_x = min(b.x0 for b in filtered)
        max_x = max(b.x1 for b in filtered)
        page_width = max(max_x - min_x, 1.0)
        gap_threshold = max(60.0, page_width * 0.18)

        if strongest_gap < gap_threshold:
            return [sorted_by_y]

        left = sorted(sorted_by_x[:split_index], key=lambda b: (b.y0, b.x0))
        right = sorted(sorted_by_x[split_index:], key=lambda b: (b.y0, b.x0))
        if not left or not right:
            return [sorted_by_y]

        return [left, right]
