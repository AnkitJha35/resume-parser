from __future__ import annotations

import re
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


def _is_obvious_contact_sidebar_block(block) -> bool:
    text = (getattr(block, "text", "") or "").strip()
    if not text:
        return False

    lowered = text.lower()
    contact_markers = (
        "contact",
        "email:",
        "phone:",
        "address:",
        "linkedin:",
        "github:",
        "portfolio:",
    )
    if lowered in contact_markers or lowered.startswith(contact_markers):
        return True
    if "linkedin.com" in lowered or "github.com" in lowered or re.search(r"@\S+\.\S+", lowered):
        return True
    return False


def _is_strong_new_experience_job(current: Optional["CandidateGroup"], new_block, idx: int, cbs: list[ClassifiedBlock]) -> bool:
    if current is None:
        return False
    if new_block.label not in {"JOB_TITLE", "UNKNOWN"}:
        return False
    if not current.blocks:
        return False

    text = (getattr(new_block.original, "text", "") or "").strip()
    normalized_text = re.sub(r"[\u200b\u200c\u200d]", "", text).strip()
    if normalized_text and any(
        re.sub(r"[\u200b\u200c\u200d]", "", (getattr(existing.original, "text", "") or "")).strip() == normalized_text
        for existing in current.blocks
    ):
        # Resume 1 contains the repeated heading "SECRETARY" for the same role
        # after the company/date block; treat this as the same job rather than a
        # new experience split.
        return False

    # This is intentionally narrow: only split on a short uppercase title-like
    # block that is immediately followed by a company/date line after an
    # existing experience group already has content. This preserves the
    # conservative behavior for regular descriptions while handling the
    # Resume 1 pattern: "SECRETARY" followed by "Bright Spot LTD, ...".
    if not text or len(text) > 40:
        return False

    words = text.split()
    if len(words) > 3:
        return False
    if any(ch.isdigit() for ch in text):
        return False
    if not all(ch.isupper() or ch.isspace() or ch in "-&/" for ch in text):
        return False

    lookahead = cbs[idx + 1 : idx + 4]
    if not lookahead:
        return False

    if any(next_cb.label in {"COMPANY", "DATE", "LOCATION"} for next_cb in lookahead):
        return True

    next_texts = [((getattr(next_cb.original, "text", "") or "").strip()) for next_cb in lookahead]
    if any(re.search(r"\b(?:inc|llc|ltd|corp|corporation|company|pvt|private)\b", t, re.I) for t in next_texts if t):
        return True

    if not any(x.label == "JOB_TITLE" for x in current.blocks):
        return False

    prev_block = current.blocks[-1].original
    new_x0 = float(getattr(new_block.original, "x0", 0) or 0)
    prev_x0 = float(getattr(prev_block, "x0", 0) or 0)
    gap = _vertical_gap(prev_block, new_block.original)
    return gap > 70 and abs(new_x0 - prev_x0) <= 60


def group_candidates(classified_blocks: Iterable[ClassifiedBlock], section: str) -> List[CandidateGroup]:
    """Group nearby ClassifiedBlocks into conservative CandidateGroups.

    The rule is intentionally conservative:
    - preserve strong visual gaps and page boundaries
    - never split on zero-width data
    - keep education blocks together unless a new degree/institution begins
    - split on repeated dates for experience/project entries
    """
    # Convert to list so we can do look-ahead and compute section-level stats
    cbs = list(classified_blocks)

    groups: List[CandidateGroup] = []
    current: Optional[CandidateGroup] = None
    last_block = None

    # Precompute some section-local layout statistics to avoid fixed thresholds
    # Compute typical vertical spacing (median gap) among meaningful blocks
    y0s: list[float] = []
    for cb in cbs:
        b = cb.original
        text = getattr(b, "text", "") or ""
        meaningful = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").strip()
        if meaningful:
            try:
                y0s.append(float(getattr(b, "y0", 0) or 0))
            except Exception:
                pass

    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        mid = n // 2
        return float((s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0))

    def _mad(values: list[float], center: float) -> float:
        if not values:
            return 0.0
        devs = [abs(v - center) for v in values]
        return _median(devs)

    # Compute typical vertical gap as median consecutive difference when possible
    gaps: list[float] = []
    if len(y0s) >= 2:
        sorted_y = sorted(y0s)
        gaps = [sorted_y[i + 1] - sorted_y[i] for i in range(len(sorted_y) - 1)]
    median_gap = _median(gaps) or 12.0

    # Collect x0 positions of DATE-like blocks to infer title/date column alignment
    date_x0s: list[float] = []
    for cb in cbs:
        if getattr(cb, "label", None) == "DATE":
            try:
                date_x0s.append(float(getattr(cb.original, "x0", 0) or 0))
            except Exception:
                pass

    date_x0_median = _median(date_x0s)
    date_x0_mad = _mad(date_x0s, date_x0_median)

    for idx, cb in enumerate(cbs):
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

        if section == "EXPERIENCE" and _is_obvious_contact_sidebar_block(b):
            # Keep obvious contact/sidebar content out of the experience stream.
            # This is kept narrow to explicit contact markers, not a blanket right-column rule.
            current = None
            last_block = b
            continue

        gap = _vertical_gap(last_block, b)
        need_new = False

        if section == "EDUCATION" and cb.label == "DATE":
            matching_group = None
            for candidate_group in groups:
                if any(x.label == "DATE" for x in candidate_group.blocks):
                    continue
                for existing in candidate_group.blocks:
                    if existing.label not in ("DEGREE", "INSTITUTION"):
                        continue
                    existing_block = existing.original
                    if (
                        getattr(existing_block, "page_number", None) == page
                        and abs(getattr(existing_block, "y0", 0) - getattr(b, "y0", 0)) <= 1.0
                        and getattr(b, "x0", 0) > getattr(existing_block, "x1", 0)
                    ):
                        matching_group = candidate_group
                        break
                if matching_group is not None:
                    break

            if matching_group is not None:
                header_index = next(
                    index
                    for index, existing in enumerate(matching_group.blocks)
                    if existing.label in ("DEGREE", "INSTITUTION")
                )
                matching_group.blocks.insert(header_index + 1, cb)
                matching_group.end_index = idx
                last_block = b
                continue

        if current is None:
            need_new = True
        else:
            if page != current.page_number:
                need_new = True
            elif gap > 40:
                need_new = True
            elif section == "EXPERIENCE" and _is_strong_new_experience_job(current, cb, idx, cbs):
                need_new = True
            elif section == "EDUCATION":
                # A repeated institution alone is not enough to start a new education entry.
                # Only a completed entry (DATE already present) or a new explicit DEGREE after
                # an existing DEGREE should trigger a split.
                if cb.label in ("DEGREE", "INSTITUTION"):
                    has_date = any(x.label == "DATE" for x in current.blocks)
                    has_identity = any(x.label in ("DEGREE", "INSTITUTION") for x in current.blocks)
                    starts_degree_like_entry = cb.label == "DEGREE" or (
                        cb.label == "INSTITUTION"
                        and re.search(r"\b(?:bachelor|master|mba|b\.?a\.?|b\.?sc|m\.?a\.?|m\.?sc)\b", text, re.I)
                    )
                    if has_date or (has_identity and starts_degree_like_entry):
                        need_new = True
            elif section == "PROJECTS":
                # Layout-adaptive project grouping.
                # Signals that a new project likely starts here:
                # - an UNKNOWN block that is followed shortly by a DATE (title-like)
                # - a DATE when the current group already has a DATE (split on repeated dates)
                # - a meaningful vertical gap relative to the section's median spacing
                # - horizontal alignment with previously observed DATE/title positions
                # - typographic emphasis (bold or larger font)
                # We combine these signals conservatively to avoid over-splitting.

                # 1) repeated DATEs: keep existing behavior
                if cb.label == "DATE" and any(x.label == "DATE" for x in current.blocks):
                    need_new = True
                # If this is a DATE and the current group was just created for an
                # immediately preceding UNKNOWN title (no DATE yet), prefer to
                # attach the DATE into the existing group so TITLE+DATE remain
                # together rather than splitting the title into its own group.
                elif cb.label == "DATE" and current is not None and not any(x.label == "DATE" for x in current.blocks):
                    try:
                        # only attach if the current group's last block is the
                        # immediately preceding block in the visual sequence
                        if current.end_index == idx - 1 and getattr(current.blocks[-1], "label", None) == "UNKNOWN":
                            last_y = float(getattr(current.blocks[-1].original, "y0", 0) or 0)
                            cur_y = float(getattr(b, "y0", 0) or 0)
                            # allow moderate separation relative to median_gap
                            if abs(cur_y - last_y) <= max(2.0 * median_gap, 48.0):
                                # keep in same group
                                need_new = False
                            else:
                                need_new = True
                        else:
                            need_new = True
                    except Exception:
                        need_new = True

                # 2) unknown block that looks like a title: lookahead for DATE
                is_title_like = False
                try:
                    # Look ahead a small window for a DATE block, but only consider
                    # DATEs that are reasonably close in vertical space to avoid
                    # picking up dates from a following project separated by a
                    # large visual gap.
                    lookahead = cbs[idx + 1 : idx + 4]
                    for n in lookahead:
                        if getattr(n, "label", None) == "DATE":
                            try:
                                ny = float(getattr(n.original, "y0", 0) or 0)
                                by = float(getattr(b, "y0", 0) or 0)
                                # distance tolerance: either a few typical gaps or a moderate pixel value
                                tol = max(3.0 * median_gap, 48.0)
                                if abs(ny - by) <= tol:
                                    is_title_like = True
                                    break
                            except Exception:
                                # if any error, conservatively treat as not title-like
                                continue
                except Exception:
                    pass

                # reject pure bullet/continuation markers as title-like
                def _is_bullet_marker(s: str) -> bool:
                    if not s:
                        return False
                    s = s.strip()
                    # common bullet characters or single-symbol list markers
                    if s in ("•", "\u2022", "\u2023", "\u25E6", "-", "*", "●", "\u00B7"):
                        return True
                    # single punctuation bullets like a lone dot or dash
                    if len(s) <= 3 and not any(ch.isalnum() for ch in s):
                        return True
                    return False

                is_bullet = _is_bullet_marker(text)

                # typography hint (disabled for bullets)
                typography_strong = False if is_bullet else bool(getattr(b, "bold", False) or (getattr(b, "font_size", None) or 0) >= 11)

                # vertical separation relative to median gap
                vertical_separation = gap > max(1.5 * median_gap, 18)

                # horizontal alignment with date column (if available)
                aligned_with_date_column = False
                try:
                    if date_x0s:
                        cb_x0 = float(getattr(b, "x0", 0) or 0)
                        # tolerance driven by observed spread (MAD); fallback to 24pt
                        tol = max(date_x0_mad * 2.0, 24.0)
                        if abs(cb_x0 - date_x0_median) <= tol:
                            aligned_with_date_column = True
                except Exception:
                    pass

                # If current group already has a DATE and this UNKNOWN looks like a
                # title (by lookahead or typography) AND it's either vertically separated
                # or aligned with the date/title column, start a new group.
                if cb.label == "UNKNOWN" and any(x.label == "DATE" for x in current.blocks):
                    if (is_title_like or typography_strong) and (vertical_separation or aligned_with_date_column):
                        need_new = True

                # If an UNKNOWN looks like a title (lookahead/date) and there is
                # a significant vertical separation from the last block, start new.
                if cb.label == "UNKNOWN" and is_title_like and vertical_separation:
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
