from __future__ import annotations

from collections import defaultdict

from app.domain.candidate_section import CandidateSection, SectionOrigin
from app.domain.document import BoundingBox
from app.domain.structural import StructuralBlock, StructuralRole
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.semantic_inference import infer_section


_LARGE_GAP_FACTOR = 2.5
_MIN_LARGE_GAP = 36.0
_TYPOGRAPHY_DELTA = 2.5

# Continuation geometry: prefer horizontal overlap / column alignment over nearest
# x-center. A headingless path continues a previous-page labeled section only when
# overlap with the narrower column is at least _MIN_HORIZONTAL_OVERLAP and the
# score (overlap minus a center-distance penalty) uniquely wins. Ambiguous
# geometry (two parents within _AMBIGUOUS_SCORE_DELTA) yields no continuation.
_MIN_HORIZONTAL_OVERLAP = 0.28
_MIN_WIDTH_RATIO = 0.35
_CENTER_PENALTY = 0.35
_AMBIGUOUS_SCORE_DELTA = 0.12
_MIN_GEOMETRY_SCORE = 0.20

_LABELED_ORIGINS = {
    SectionOrigin.KNOWN_ALIAS,
    SectionOrigin.INFERRED,
    SectionOrigin.CONTINUED,
}
_EXPERIENCE_EVIDENCE_ROLES = {
    StructuralRole.DATE,
    StructuralRole.ENTRY_TITLE,
    StructuralRole.ORGANIZATION,
    StructuralRole.LOCATION,
    StructuralRole.DESCRIPTION,
    StructuralRole.BULLET,
}
_EXCLUDED_REGION_KINDS = {
    "header",
    "footer",
    "sidebar",
    "table",
    "table_cell",
    "margin",
}
_EXCLUDED_ROLES = {
    StructuralRole.HEADER,
    StructuralRole.FOOTER,
    StructuralRole.SIDEBAR,
    StructuralRole.TABLE_CELL,
}


def build_candidate_sections(
    blocks: list[StructuralBlock],
    section_detector: SectionDetector | None = None,
) -> list[CandidateSection]:
    """Propose section boundaries from StructuralBlocks, then attach semantic labels.

    Boundary detection uses structural roles and discontinuities only.
    Semantic labels use known aliases and existing infer_section — never the reverse.
    """
    detector = section_detector or SectionDetector()
    by_page_path: dict[int, dict[str, list[StructuralBlock]]] = defaultdict(lambda: defaultdict(list))
    for block in blocks:
        by_page_path[block.page_number][block.path_id].append(block)

    sections: list[CandidateSection] = []
    previous_labeled: list[CandidateSection] = []
    section_counter = 0
    for page_number in sorted(by_page_path):
        path_map = by_page_path[page_number]
        path_ids = sorted(
            path_map,
            key=lambda path_id: (
                min(item.bbox.y0 for item in path_map[path_id]),
                min(item.bbox.x0 for item in path_map[path_id]),
                path_id,
            ),
        )
        for path_id in path_ids:
            # Path-local visual order. Line.reading_order is a global extraction
            # index and must not precede geometry (it inverts headings vs bodies).
            path_blocks = sorted(
                path_map[path_id],
                key=lambda item: (item.bbox.y0, item.bbox.x0, item.block_id),
            )
            path_sections, section_counter = _sections_for_path(
                path_blocks,
                detector,
                section_counter,
                previous_labeled,
            )
            sections.extend(path_sections)
            previous_labeled.extend(
                section
                for section in path_sections
                if section.semantic_label and section.origin in _LABELED_ORIGINS
            )
    return sections


def _sections_for_path(
    blocks: list[StructuralBlock],
    detector: SectionDetector,
    section_counter: int,
    previous_labeled: list[CandidateSection] | None = None,
) -> tuple[list[CandidateSection], int]:
    if not blocks:
        return [], section_counter

    claimed = [False] * len(blocks)
    headed: list[CandidateSection] = []

    index = 0
    while index < len(blocks):
        block = blocks[index]
        known = detector._find_section_header(block.text)
        is_heading = known is not None or block.role == StructuralRole.SECTION_HEADING
        # ENTRY_TITLE / org / date stacks are never section headings.
        if block.role == StructuralRole.ENTRY_TITLE:
            is_heading = False
            known = None

        if not is_heading:
            index += 1
            continue

        end = _next_headed_boundary(blocks, index + 1, detector)
        content = tuple(blocks[index + 1 : end])
        for claimed_index in range(index, end):
            claimed[claimed_index] = True

        if known is not None:
            section = _make_section(
                section_id=f"section-{section_counter}",
                origin=SectionOrigin.KNOWN_ALIAS,
                semantic_label=known,
                heading=block,
                content=content,
                confidence=1.0,
                evidence=("known_section_alias", f"label:{known}"),
            )
        else:
            inference = infer_section(block.text, [item.text for item in content])
            if inference.section != "UNKNOWN":
                section = _make_section(
                    section_id=f"section-{section_counter}",
                    origin=SectionOrigin.INFERRED,
                    semantic_label=inference.section,
                    heading=block,
                    content=content,
                    confidence=inference.confidence,
                    evidence=("inferred_from_content", f"label:{inference.section}")
                    + tuple(f"signal:{signal}" for signal in inference.signals.get(inference.section, ())),
                )
            else:
                section = _make_section(
                    section_id=f"section-{section_counter}",
                    origin=SectionOrigin.UNKNOWN,
                    semantic_label=None,
                    heading=block,
                    content=content,
                    confidence=inference.confidence,
                    evidence=("ambiguous_or_weak_content", "label:UNKNOWN"),
                )
        headed.append(section)
        section_counter += 1
        index = end

    unlabeled: list[CandidateSection] = []
    leftover = [blocks[i] for i, was_claimed in enumerate(claimed) if not was_claimed]
    if leftover and not headed:
        continued, section_counter = _try_continue_section(
            leftover,
            previous_labeled or [],
            detector,
            section_counter,
        )
        if continued is not None:
            return [continued], section_counter

    for cluster in _propose_headingless_clusters(leftover):
        if not cluster:
            continue
        # Skip pure contact/header margin noise as sections.
        if all(item.role in {StructuralRole.CONTACT, StructuralRole.HEADER, StructuralRole.FOOTER} for item in cluster):
            continue
        unlabeled.append(
            _make_section(
                section_id=f"section-{section_counter}",
                origin=SectionOrigin.UNLABELED,
                semantic_label=None,
                heading=None,
                content=tuple(cluster),
                confidence=0.0,
                evidence=_headingless_evidence(cluster),
            )
        )
        section_counter += 1

    if (
        leftover
        and not headed
        and not unlabeled
        and not _path_is_excluded(leftover)
        and _previous_page_labeled(leftover, previous_labeled or [])
    ):
        unlabeled.append(
            _make_section(
                section_id=f"section-{section_counter}",
                origin=SectionOrigin.UNLABELED,
                semantic_label=None,
                heading=None,
                content=tuple(leftover),
                confidence=0.0,
                evidence=("headingless_unlabeled", "continuation_rejected"),
            )
        )
        section_counter += 1

    return headed + unlabeled, section_counter


def _try_continue_section(
    leftover: list[StructuralBlock],
    previous_labeled: list[CandidateSection],
    detector: SectionDetector,
    section_counter: int,
) -> tuple[CandidateSection | None, int]:
    """Inherit a previous-page label onto a headingless, geometrically aligned path.

    Does not infer a new semantic label from the continuation path. Membership is
    path-local: only leftover blocks from the current path are owned.
    """
    if not leftover or _path_is_excluded(leftover):
        return None, section_counter
    if _path_has_known_heading(leftover, detector):
        return None, section_counter
    if _path_has_section_heading(leftover, detector):
        return None, section_counter

    parents = _previous_page_labeled(leftover, previous_labeled)
    if not parents:
        return None, section_counter

    path_bbox = _blocks_bbox(leftover)
    parent = _select_geometrically_compatible_parent(parents, path_bbox, leftover)
    if parent is None or parent.semantic_label is None:
        return None, section_counter

    overlap = _horizontal_overlap_ratio(parent.bbox, path_bbox)
    section = _make_section(
        section_id=f"section-{section_counter}",
        origin=SectionOrigin.CONTINUED,
        semantic_label=parent.semantic_label,
        heading=None,
        content=tuple(leftover),
        confidence=min(parent.confidence, 0.9),
        evidence=(
            "section_continuation",
            f"inherited:{parent.semantic_label}",
            f"from:{parent.section_id}",
            f"horizontal_overlap:{overlap:.2f}",
            "geometry:overlap_then_center_penalty",
        ),
        continuation_of=parent.section_id,
    )
    return section, section_counter + 1


def _previous_page_labeled(
    leftover: list[StructuralBlock],
    previous_labeled: list[CandidateSection],
) -> list[CandidateSection]:
    page_number = leftover[0].page_number
    return [
        section
        for section in previous_labeled
        if section.page_number < page_number
        and section.semantic_label
        and section.origin in _LABELED_ORIGINS
    ]


def _path_has_known_heading(blocks: list[StructuralBlock], detector: SectionDetector) -> bool:
    return any(detector._find_section_header(block.text) is not None for block in blocks)


def _path_has_section_heading(blocks: list[StructuralBlock], detector: SectionDetector) -> bool:
    for block in blocks:
        if block.role == StructuralRole.ENTRY_TITLE:
            continue
        if block.role == StructuralRole.SECTION_HEADING:
            return True
        if detector._find_section_header(block.text) is not None:
            return True
    return False


def _path_is_excluded(blocks: list[StructuralBlock]) -> bool:
    if not blocks:
        return True
    kinds = {(block.region_kind or "").lower() for block in blocks}
    if kinds & _EXCLUDED_REGION_KINDS:
        return True
    roles = {block.role for block in blocks}
    if roles and roles <= {StructuralRole.CONTACT}:
        return True
    if roles and roles <= _EXCLUDED_ROLES:
        return True
    if roles and roles <= (_EXCLUDED_ROLES | {StructuralRole.CONTACT}):
        return True
    return False


def _experience_structural_evidence(blocks: list[StructuralBlock]) -> bool:
    roles = {block.role for block in blocks}
    if not (roles & _EXPERIENCE_EVIDENCE_ROLES):
        return False
    # DATE alone is enough: some specimens classify the title line as DESCRIPTION.
    if StructuralRole.DATE in roles:
        return True
    if StructuralRole.ENTRY_TITLE in roles and roles & {
        StructuralRole.ORGANIZATION,
        StructuralRole.LOCATION,
        StructuralRole.DESCRIPTION,
        StructuralRole.BULLET,
        StructuralRole.DATE,
    }:
        return True
    if StructuralRole.ORGANIZATION in roles and roles & {
        StructuralRole.DESCRIPTION,
        StructuralRole.BULLET,
        StructuralRole.LOCATION,
        StructuralRole.DATE,
    }:
        return True
    if StructuralRole.BULLET in roles and len(blocks) >= 2:
        return True
    if StructuralRole.DESCRIPTION in roles and roles & {
        StructuralRole.ORGANIZATION,
        StructuralRole.LOCATION,
        StructuralRole.DATE,
        StructuralRole.BULLET,
    }:
        return True
    return False


def _skills_structural_evidence(blocks: list[StructuralBlock]) -> bool:
    if StructuralRole.DATE in {block.role for block in blocks}:
        return False
    short = sum(
        1
        for block in blocks
        if block.role in {StructuralRole.UNKNOWN, StructuralRole.TECHNOLOGY}
        and len(block.text.split()) <= 4
    )
    return short >= 3 or sum(1 for block in blocks if block.role == StructuralRole.TECHNOLOGY) >= 2


def _education_structural_evidence(blocks: list[StructuralBlock]) -> bool:
    if StructuralRole.DATE not in {block.role for block in blocks}:
        return False
    return any(
        token in block.text.lower()
        for block in blocks
        for token in ("bachelor", "master", "associate", "phd", "degree", "university", "college", "school", "institute")
    )


def _label_compatible_with_evidence(label: str, blocks: list[StructuralBlock]) -> bool:
    if label == "EXPERIENCE":
        return _experience_structural_evidence(blocks) and not _skills_structural_evidence(blocks)
    if label == "EDUCATION":
        return _education_structural_evidence(blocks)
    if label == "SKILLS":
        return _skills_structural_evidence(blocks) and not _experience_structural_evidence(blocks)
    return False


def _competing_inferred_label(inherited: str, blocks: list[StructuralBlock]) -> bool:
    """Reject inheritance when the new path has structural evidence of a different section.

    A lone date range must not count as EDUCATION competing with EXPERIENCE.
    infer_section is not used to assign a new continuation label.
    """
    if inherited == "EXPERIENCE":
        return _skills_structural_evidence(blocks) or _education_structural_evidence(blocks)
    if inherited == "SKILLS":
        return _experience_structural_evidence(blocks) or _education_structural_evidence(blocks)
    if inherited == "EDUCATION":
        return _skills_structural_evidence(blocks) or (
            _experience_structural_evidence(blocks) and not _education_structural_evidence(blocks)
        )
    return False


def _blocks_bbox(blocks: list[StructuralBlock]) -> BoundingBox:
    return BoundingBox(
        min(block.bbox.x0 for block in blocks),
        min(block.bbox.y0 for block in blocks),
        max(block.bbox.x1 for block in blocks),
        max(block.bbox.y1 for block in blocks),
    )


def _horizontal_overlap_ratio(previous: BoundingBox, current: BoundingBox) -> float:
    overlap = max(0.0, min(previous.x1, current.x1) - max(previous.x0, current.x0))
    narrower = min(previous.x1 - previous.x0, current.x1 - current.x0)
    if narrower <= 0:
        return 0.0
    return overlap / narrower


def _geometry_score(previous: BoundingBox, current: BoundingBox) -> float | None:
    overlap = _horizontal_overlap_ratio(previous, current)
    if overlap < _MIN_HORIZONTAL_OVERLAP:
        return None
    previous_width = max(1e-6, previous.x1 - previous.x0)
    current_width = max(1e-6, current.x1 - current.x0)
    width_ratio = min(previous_width, current_width) / max(previous_width, current_width)
    if width_ratio < _MIN_WIDTH_RATIO and overlap < 0.55:
        return None
    avg_width = (previous_width + current_width) / 2.0
    center_distance = abs(((previous.x0 + previous.x1) / 2.0) - ((current.x0 + current.x1) / 2.0))
    score = overlap - _CENTER_PENALTY * (center_distance / avg_width)
    if score < _MIN_GEOMETRY_SCORE:
        return None
    return score


def _cluster_parents_by_column(parents: list[CandidateSection]) -> list[list[CandidateSection]]:
    clusters: list[list[CandidateSection]] = []
    for parent in parents:
        placed = False
        for cluster in clusters:
            if any(_horizontal_overlap_ratio(parent.bbox, other.bbox) >= 0.5 for other in cluster):
                cluster.append(parent)
                placed = True
                break
        if not placed:
            clusters.append([parent])
    return clusters


def _select_geometrically_compatible_parent(
    parents: list[CandidateSection],
    path_bbox: BoundingBox,
    leftover: list[StructuralBlock],
) -> CandidateSection | None:
    """Choose a previous-page parent using column overlap, not nearest x-center.

    Same-column stacked sections (SUMMARY then EXPERIENCE) are one cluster; the
    last compatible section in that cluster is preferred. Two horizontally
    distinct columns with similar overlap scores are ambiguous → no continuation.
    """
    scored: list[tuple[float, CandidateSection]] = []
    for parent in parents:
        score = _geometry_score(parent.bbox, path_bbox)
        if score is not None:
            scored.append((score, parent))
    if not scored:
        return None

    clusters = _cluster_parents_by_column([parent for _, parent in scored])
    cluster_best: list[tuple[float, list[CandidateSection]]] = []
    for cluster in clusters:
        best_score = max(score for score, parent in scored if parent in cluster)
        cluster_best.append((best_score, cluster))
    cluster_best.sort(key=lambda item: -item[0])
    if len(cluster_best) > 1 and cluster_best[0][0] - cluster_best[1][0] < _AMBIGUOUS_SCORE_DELTA:
        return None

    winning = cluster_best[0][1]
    ordered = sorted(winning, key=lambda parent: (parent.bbox.y1, parent.reading_order), reverse=True)
    for parent in ordered:
        if parent.semantic_label is None:
            continue
        if not _label_compatible_with_evidence(parent.semantic_label, leftover):
            continue
        if _competing_inferred_label(parent.semantic_label, leftover):
            continue
        return parent
    return None


def _next_headed_boundary(blocks: list[StructuralBlock], start: int, detector: SectionDetector) -> int:
    for index in range(start, len(blocks)):
        block = blocks[index]
        if block.role == StructuralRole.ENTRY_TITLE:
            continue
        known = detector._find_section_header(block.text)
        if known is not None or block.role == StructuralRole.SECTION_HEADING:
            return index
    return len(blocks)


def _propose_headingless_clusters(blocks: list[StructuralBlock]) -> list[list[StructuralBlock]]:
    if not blocks:
        return []

    gaps = [
        max(0.0, blocks[index].bbox.y0 - blocks[index - 1].bbox.y1)
        for index in range(1, len(blocks))
    ]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 12.0
    large_gap = max(_MIN_LARGE_GAP, median_gap * _LARGE_GAP_FACTOR)

    clusters: list[list[StructuralBlock]] = []
    current: list[StructuralBlock] = [blocks[0]]
    for index in range(1, len(blocks)):
        previous = blocks[index - 1]
        block = blocks[index]
        gap = max(0.0, block.bbox.y0 - previous.bbox.y1)
        typography_shift = abs((block.style.font_size or 0.0) - (previous.style.font_size or 0.0)) >= _TYPOGRAPHY_DELTA
        role_shift = _role_distribution_shift(current, block)
        if gap >= large_gap or (typography_shift and role_shift) or role_shift and gap >= median_gap:
            if _looks_like_section_cluster(current):
                clusters.append(current)
            current = [block]
        else:
            current.append(block)
    if _looks_like_section_cluster(current):
        clusters.append(current)
    return clusters


def _looks_like_section_cluster(blocks: list[StructuralBlock]) -> bool:
    if len(blocks) < 2:
        return False
    roles = [block.role for block in blocks]
    has_entry = StructuralRole.ENTRY_TITLE in roles
    has_meta = any(role in {StructuralRole.ORGANIZATION, StructuralRole.DATE, StructuralRole.LOCATION} for role in roles)
    bulletish = sum(1 for role in roles if role in {StructuralRole.BULLET, StructuralRole.TECHNOLOGY})
    short_unknown = sum(
        1
        for block in blocks
        if block.role in {StructuralRole.UNKNOWN, StructuralRole.TECHNOLOGY}
        and len(block.text.split()) <= 4
    )
    # Experience-like, education-like (date + identity), or skills-like list clusters.
    if has_entry and has_meta:
        return True
    if StructuralRole.DATE in roles and has_entry:
        return True
    if StructuralRole.DATE in roles and any(
        "university" in block.text.lower()
        or "college" in block.text.lower()
        or "school" in block.text.lower()
        or "bachelor" in block.text.lower()
        or "master" in block.text.lower()
        for block in blocks
    ):
        return True
    if bulletish >= 2 or short_unknown >= 3:
        return True
    return False


def _role_distribution_shift(current: list[StructuralBlock], nxt: StructuralBlock) -> bool:
    if not current:
        return False
    prev_roles = {block.role for block in current[-3:]}
    # List body after non-list prologue, or entry stack after list.
    if nxt.role == StructuralRole.BULLET and StructuralRole.BULLET not in prev_roles:
        return True
    if nxt.role == StructuralRole.ENTRY_TITLE and StructuralRole.BULLET in prev_roles:
        return True
    if nxt.role == StructuralRole.DATE and StructuralRole.BULLET in prev_roles:
        return True
    return False


def _headingless_evidence(blocks: list[StructuralBlock]) -> tuple[str, ...]:
    roles = [block.role.value for block in blocks]
    evidence = ["headingless_cluster"]
    if StructuralRole.ENTRY_TITLE in {block.role for block in blocks}:
        evidence.append("entry_title_present")
    if StructuralRole.DATE in {block.role for block in blocks}:
        evidence.append("date_present")
    if sum(1 for block in blocks if block.role == StructuralRole.BULLET) >= 2:
        evidence.append("list_structure")
    evidence.append("roles:" + ",".join(sorted(set(roles))))
    return tuple(evidence)


def _make_section(
    *,
    section_id: str,
    origin: SectionOrigin,
    semantic_label: str | None,
    heading: StructuralBlock | None,
    content: tuple[StructuralBlock, ...],
    confidence: float,
    evidence: tuple[str, ...],
    continuation_of: str | None = None,
) -> CandidateSection:
    members = ((heading,) if heading is not None else ()) + content
    if not members:
        bbox = BoundingBox(0.0, 0.0, 0.0, 0.0)
        page_number = 0
        region_id = ""
        path_id = ""
        reading_order = 0
    else:
        bbox = BoundingBox(
            min(item.bbox.x0 for item in members),
            min(item.bbox.y0 for item in members),
            max(item.bbox.x1 for item in members),
            max(item.bbox.y1 for item in members),
        )
        page_number = members[0].page_number
        region_id = members[0].region_id
        path_id = members[0].path_id
        reading_order = members[0].reading_order

    line_ids: list[str] = []
    span_ids: list[str] = []
    for item in members:
        line_ids.extend(item.line_ids)
        span_ids.extend(item.source_span_ids)

    return CandidateSection(
        section_id=section_id,
        origin=origin,
        semantic_label=semantic_label,
        heading=heading,
        content=content,
        page_number=page_number,
        region_id=region_id,
        path_id=path_id,
        bbox=bbox,
        reading_order=reading_order,
        confidence=confidence,
        evidence=evidence,
        line_ids=tuple(line_ids),
        source_span_ids=tuple(span_ids),
        continuation_of=continuation_of,
    )
