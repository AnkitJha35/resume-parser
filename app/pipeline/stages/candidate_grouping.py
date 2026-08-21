from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable, Optional

from app.pipeline.stages.block_classification import ClassifiedBlock


@dataclass
class CandidateGroup:
    section: str
    blocks: List[ClassifiedBlock]
    page_number: int
    column_id: Optional[int]
    start_index: int
    end_index: int
    summary_text: str


def _column_id_for_block(block) -> int:
    # Conservative column id: bucket by x0 using 200pt width
    x0 = getattr(block, "x0", 0) or 0
    try:
        return int(x0 // 200)
    except Exception:
        return 0


def _vertical_gap(prev_block, next_block) -> float:
    if prev_block is None or next_block is None:
        return float("inf")
    try:
        # Use starting Y positions for a more reliable vertical gap when y1
        # (bottom coordinate) may be missing or unreliable in fixtures.
        return abs(getattr(next_block, "y0", 0) - getattr(prev_block, "y0", 0))
    except Exception:
        return float("inf")


def group_candidates(classified_blocks: Iterable[ClassifiedBlock], section: str) -> List[CandidateGroup]:
    """Group nearby ClassifiedBlocks into conservative CandidateGroups.

    The rule is intentionally conservative:
    - preserve strong visual gaps and page boundaries
    - never split on zero-width data
    - keep education blocks together unless a new degree/institution begins
    - split on repeated dates for experience/project entries
    """
    groups: List[CandidateGroup] = []
    current: Optional[CandidateGroup] = None
    last_block = None

    for idx, cb in enumerate(classified_blocks):
        b = cb.original
        page = getattr(b, "page_number", 1)
        col = _column_id_for_block(b)

        text = getattr(b, "text", "") or ""
        meaningful = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").strip()
        if not meaningful:
            # Ignore invisible/empty blocks entirely — do not update last_block
            # so they do not create artificial vertical gaps or group boundaries.
            continue

        if cb.label == "SECTION_HEADER":
            current = None
            last_block = None
            continue

        gap = _vertical_gap(last_block, b)
        need_new = False

        if current is None:
            need_new = True
        else:
            if page != current.page_number:
                need_new = True
            elif gap > 40:
                need_new = True
            elif cb.label == "JOB_TITLE" and any(x.label == "JOB_TITLE" for x in current.blocks):
                need_new = True
            elif section == "EDUCATION":
                # Education is usually a single degree/institution record with dates and location.
                # A new degree or institution name is the canonical start of a new education entry.
                if cb.label in ("DEGREE", "INSTITUTION") and any(
                    x.label in ("DATE", "DEGREE", "INSTITUTION", "LOCATION") for x in current.blocks
                ):
                    need_new = True
            elif cb.label == "DATE" and any(x.label == "DATE" for x in current.blocks):
                if section in ("EXPERIENCE", "PROJECTS"):
                    need_new = True
                elif gap > 30:
                    need_new = True

        if need_new:
            summary = (cb.original.text or "").strip()[:200]
            current = CandidateGroup(
                section=section,
                blocks=[cb],
                page_number=page,
                column_id=col,
                start_index=idx,
                end_index=idx,
                summary_text=summary,
            )
            groups.append(current)
        else:
            current.blocks.append(cb)
            current.end_index = idx
            if len(current.summary_text) < 200:
                add = (" \n" + (cb.original.text or "")).strip()
                current.summary_text = (current.summary_text + add)[:200]

        last_block = b

    return groups
