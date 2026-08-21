from __future__ import annotations

from typing import Iterable, List

from app.pipeline.stages.text_extraction import TextBlock


class ReadingOrder:
    @staticmethod
    def reorder(blocks: Iterable[TextBlock]) -> list[TextBlock]:
        # Group blocks by page
        pages: dict[int, list[TextBlock]] = {}
        for b in blocks:
            pages.setdefault(b.page_number, []).append(b)

        ordered: list[TextBlock] = []
        for page_number in sorted(pages.keys()):
            page_blocks = pages[page_number]
            clusters = ReadingOrder.cluster_blocks_by_x0(page_blocks)

            if len(clusters) == 1:
                # single-column: preserve top-to-bottom order
                ordered.extend(sorted(page_blocks, key=lambda b: (b.y0, b.x0)))
            else:
                # multi-column: left-to-right clusters, top-to-bottom within each
                for cluster in clusters:
                    ordered.extend(sorted(cluster, key=lambda b: (b.y0, b.x0)))

        return ordered

    @staticmethod
    def cluster_blocks_by_x0(blocks: Iterable[TextBlock]) -> list[list[TextBlock]]:
        blocks_list = list(blocks)
        if not blocks_list:
            return []

        # Sort by x0 to analyze horizontal distribution
        sorted_by_x = sorted(blocks_list, key=lambda b: b.x0)
        x_values = [b.x0 for b in sorted_by_x]

        if len(x_values) <= 1:
            return [sorted_by_x]

        min_x, max_x = x_values[0], x_values[-1]
        # gap cutoff: adaptive but conservative. Large documents with clear column gaps
        # will exceed this value; single-column layouts won't.
        gap_cutoff = max(50.0, (max_x - min_x) * 0.25)

        clusters: list[list[TextBlock]] = []
        current_cluster: list[TextBlock] = [sorted_by_x[0]]

        for prev, curr in zip(sorted_by_x, sorted_by_x[1:]):
            gap = curr.x0 - prev.x0
            if gap > gap_cutoff:
                clusters.append(current_cluster)
                current_cluster = [curr]
            else:
                current_cluster.append(curr)

        clusters.append(current_cluster)

        # If no meaningful split found, return single cluster
        if len(clusters) == 1:
            return [sorted(blocks_list, key=lambda b: (b.y0, b.x0))]

        # Sort each cluster top-to-bottom, and then sort clusters left-to-right
        clusters = [sorted(c, key=lambda b: (b.y0, b.x0)) for c in clusters]
        clusters = sorted(clusters, key=lambda c: sum(b.x0 for b in c) / len(c))
        return clusters
