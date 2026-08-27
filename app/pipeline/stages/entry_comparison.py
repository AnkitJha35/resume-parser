from __future__ import annotations

from dataclasses import dataclass

from app.domain.candidate_entry import CandidateEntry
from app.domain.candidate_section import CandidateSection
from app.domain.document import Document
from app.domain.extraction_input import ExtractorEntryView
from app.pipeline.stages.block_classification import classify_block
from app.pipeline.stages.candidate_entries import build_candidate_entries
from app.pipeline.stages.candidate_grouping import CandidateGroup, group_candidates
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.entry_compat import candidate_entries_to_extractor_views
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.semantic_compat import (
    candidate_sections_to_text_blocks,
    semantic_sections_to_text_blocks,
)
from app.pipeline.stages.semantic_paths import detect_region_aware_sections
from app.pipeline.stages.structural_roles import build_structural_blocks


# Standing information-loss inventory — not bugs, not scores.
GROUP_ONLY_INFORMATION: tuple[str, ...] = (
    "classification labels (JOB_TITLE, COMPANY, DEGREE, INSTITUTION, DATE, …)",
    "classifier scores and reasons",
    "job_titles.json / degrees.json vocabulary results via classify_block",
    "start_index / end_index in the classified section stream",
    "column_id from x0 // 200",
    "forced page splitting",
    "40pt vertical-gap split",
    "experience contact-sidebar filtering",
    "education right-column date attachment",
    "project median-gap / date-column alignment rules",
    "duplicate uppercase-title same-job guard",
    "placeholder degree non-split",
    "always grouping leftover labeled-section text",
)

ENTRY_ONLY_INFORMATION: tuple[str, ...] = (
    "StructuralRole",
    "path_id",
    "region_id",
    "region_kind",
    "source line_ids (full tuple, including wrapped lines)",
    "source_span_ids",
    "reconstruction_methods",
    "multi-page / multi-region / multi-path scope tuples",
    "typography (TextStyle)",
    "neighboring structural block ids",
    "CandidateSection origin",
    "structural evidence (experience_like_stack, …)",
)


@dataclass(frozen=True)
class Disagreement:
    category: str
    detail: str
    entry_ids: tuple[str, ...] = ()
    group_indexes: tuple[int, ...] = ()


@dataclass(frozen=True)
class AlignedPair:
    entry_id: str | None
    group_index: int | None
    entry_line_ids: tuple[str, ...]
    group_line_ids: tuple[str, ...]
    overlap_line_ids: tuple[str, ...]
    entry_section: str | None
    group_section: str | None
    entry_type: str | None
    structural_roles: tuple[str, ...]
    classification_labels: tuple[str, ...]
    classification_reasons: tuple[str, ...]
    entry_pages: tuple[int, ...]
    group_page: int | None
    entry_regions: tuple[str, ...]
    entry_paths: tuple[str, ...]
    group_column_id: int | None
    group_start_index: int | None
    group_end_index: int | None


@dataclass(frozen=True)
class ComparisonReport:
    entry_count: int
    group_count: int
    pairs: tuple[AlignedPair, ...]
    disagreements: tuple[Disagreement, ...]
    group_only_information: tuple[str, ...]
    entry_only_information: tuple[str, ...]


def compare_document(
    document: Document,
    section_detector: SectionDetector | None = None,
) -> ComparisonReport:
    """Run CandidateEntry IR and production CandidateGroup in parallel. Mutates neither."""
    detector = section_detector or SectionDetector()
    structural_blocks = build_structural_blocks(document)
    candidate_sections = build_candidate_sections(structural_blocks, detector)
    entries = build_candidate_entries(candidate_sections)
    views = candidate_entries_to_extractor_views(entries, candidate_sections)

    semantic_document = detect_region_aware_sections(document, detector)
    section_blocks = semantic_sections_to_text_blocks(semantic_document)
    groups: list[CandidateGroup] = []
    for section_name, blocks in section_blocks.items():
        if section_name == "UNASSIGNED" or not blocks:
            continue
        groups.extend(group_candidates([classify_block(block) for block in blocks], section_name))

    return compare_entry_views_and_groups(views, groups, sections=candidate_sections, entries=entries)


def groups_from_candidate_sections(
    sections: list[CandidateSection],
    all_blocks: list | None = None,
) -> list[CandidateGroup]:
    """Diagnostic grouping from CandidateSection TextBlock views — not production Resume path."""
    converted = candidate_sections_to_text_blocks(sections, all_blocks)
    groups: list[CandidateGroup] = []
    for section_name, blocks in converted.items():
        if section_name == "UNASSIGNED" or not blocks:
            continue
        groups.extend(group_candidates([classify_block(block) for block in blocks], section_name))
    return groups


def compare_entries_to_section_compat_groups(
    document: Document,
    section_detector: SectionDetector | None = None,
) -> ComparisonReport:
    """Compare CandidateEntry to groups built from CandidateSection labels (not semantic_paths)."""
    detector = section_detector or SectionDetector()
    structural_blocks = build_structural_blocks(document)
    candidate_sections = build_candidate_sections(structural_blocks, detector)
    entries = build_candidate_entries(candidate_sections)
    views = candidate_entries_to_extractor_views(entries, candidate_sections)
    groups = groups_from_candidate_sections(candidate_sections, structural_blocks)
    return compare_entry_views_and_groups(views, groups, sections=candidate_sections, entries=entries)


def compare_entry_views_and_groups(
    views: list[ExtractorEntryView],
    groups: list[CandidateGroup],
    *,
    sections: list[CandidateSection] | None = None,
    entries: list[CandidateEntry] | None = None,
) -> ComparisonReport:
    """Asymmetric comparison: neither side is expected truth."""
    _ = sections, entries
    disagreements: list[Disagreement] = []
    pairs: list[AlignedPair] = []

    view_line_sets = [_view_line_ids(view) for view in views]
    group_line_sets = [_group_line_ids(group) for group in groups]

    used_groups: set[int] = set()
    used_views: set[int] = set()

    # Greedy max-overlap alignment (not a similarity score for pass/fail).
    while True:
        best: tuple[int, int, int] | None = None
        for view_index, view_ids in enumerate(view_line_sets):
            if view_index in used_views:
                continue
            for group_index, group_ids in enumerate(group_line_sets):
                if group_index in used_groups:
                    continue
                overlap = len(view_ids & group_ids)
                if overlap == 0:
                    continue
                if best is None or overlap > best[2]:
                    best = (view_index, group_index, overlap)
        if best is None:
            break
        view_index, group_index, _ = best
        used_views.add(view_index)
        used_groups.add(group_index)
        pairs.append(_pair_from(views[view_index], view_index, groups[group_index], group_index))

    for view_index, view in enumerate(views):
        if view_index in used_views:
            continue
        pairs.append(_unmatched_view_pair(view))
        disagreements.append(
            Disagreement(
                "MEMBERSHIP_DIFFERENCE",
                f"CandidateEntry {view.entry_id} has no overlapping CandidateGroup",
                entry_ids=(view.entry_id,),
            )
        )

    for group_index, group in enumerate(groups):
        if group_index in used_groups:
            continue
        pairs.append(_unmatched_group_pair(group, group_index))
        disagreements.append(
            Disagreement(
                "MEMBERSHIP_DIFFERENCE",
                f"CandidateGroup[{group_index}] section={group.section} has no overlapping CandidateEntry",
                group_indexes=(group_index,),
            )
        )

    if len(views) != len(groups):
        disagreements.append(
            Disagreement(
                "ENTRY_COUNT_DIFFERENCE",
                f"entry_count={len(views)} group_count={len(groups)}",
            )
        )

    entry_union = set().union(*view_line_sets) if view_line_sets else set()
    group_union = set().union(*group_line_sets) if group_line_sets else set()
    if entry_union != group_union:
        only_entry = tuple(sorted(entry_union - group_union))
        only_group = tuple(sorted(group_union - entry_union))
        disagreements.append(
            Disagreement(
                "MEMBERSHIP_DIFFERENCE",
                f"line_ids only in entries={only_entry} only in groups={only_group}",
            )
        )

    if _partitions_differ(view_line_sets, group_line_sets):
        disagreements.append(
            Disagreement(
                "ENTRY_BOUNDARY_DIFFERENCE",
                "same or overlapping lines are clustered differently",
            )
        )

    for pair in pairs:
        if pair.entry_id is None or pair.group_index is None:
            continue
        if _ownership_differs(pair.entry_section, pair.entry_type, pair.group_section):
            disagreements.append(
                Disagreement(
                    "SECTION_OWNERSHIP_DIFFERENCE",
                    (
                        f"entry section_label={pair.entry_section} entry_type={pair.entry_type} "
                        f"vs group.section={pair.group_section}"
                    ),
                    entry_ids=(pair.entry_id,),
                    group_indexes=(pair.group_index,),
                )
            )
        if set(pair.entry_line_ids) != set(pair.group_line_ids):
            disagreements.append(
                Disagreement(
                    "ENTRY_BOUNDARY_DIFFERENCE",
                    (
                        f"aligned pair line mismatch entry={pair.entry_line_ids} "
                        f"group={pair.group_line_ids}"
                    ),
                    entry_ids=(pair.entry_id,),
                    group_indexes=(pair.group_index,),
                )
            )
        if pair.group_page is not None and pair.entry_pages and pair.group_page not in pair.entry_pages:
            disagreements.append(
                Disagreement(
                    "SCOPE_DIFFERENCE",
                    f"entry pages={pair.entry_pages} vs group page={pair.group_page}",
                    entry_ids=(pair.entry_id,),
                    group_indexes=(pair.group_index,),
                )
            )
        elif len(pair.entry_pages) > 1:
            disagreements.append(
                Disagreement(
                    "SCOPE_DIFFERENCE",
                    f"entry spans pages {pair.entry_pages}; group records a single page_number",
                    entry_ids=(pair.entry_id,),
                    group_indexes=(pair.group_index,),
                )
            )
        if pair.classification_labels and pair.structural_roles:
            disagreements.append(
                Disagreement(
                    "ROLE_DIFFERENCE",
                    (
                        f"structural_roles={pair.structural_roles} "
                        f"classification_labels={pair.classification_labels}"
                    ),
                    entry_ids=(pair.entry_id,),
                    group_indexes=(pair.group_index,),
                )
            )
        group = groups[pair.group_index]
        if not _group_has_full_provenance(group):
            disagreements.append(
                Disagreement(
                    "PROVENANCE_DIFFERENCE",
                    "CandidateGroup TextBlocks lack path/region/full line_id tuples that ExtractorEntryView preserves",
                    entry_ids=(pair.entry_id,),
                    group_indexes=(pair.group_index,),
                )
            )

    # Deduplicate category+detail while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[Disagreement] = []
    for item in disagreements:
        key = (item.category, item.detail)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return ComparisonReport(
        entry_count=len(views),
        group_count=len(groups),
        pairs=tuple(pairs),
        disagreements=tuple(unique),
        group_only_information=GROUP_ONLY_INFORMATION,
        entry_only_information=ENTRY_ONLY_INFORMATION,
    )


def disagreement_categories(report: ComparisonReport) -> set[str]:
    return {item.category for item in report.disagreements}


def _view_line_ids(view: ExtractorEntryView) -> set[str]:
    return set(view.line_ids)


def _group_line_ids(group: CandidateGroup) -> set[str]:
    ids: set[str] = set()
    for classified in group.blocks:
        original = classified.original
        line_id = getattr(original, "source_line_id", None)
        if line_id:
            ids.add(str(line_id))
        extra = getattr(original, "source_line_ids", None)
        if extra:
            ids.update(str(item) for item in extra)
    return ids


def _pair_from(
    view: ExtractorEntryView,
    _view_index: int,
    group: CandidateGroup,
    group_index: int,
) -> AlignedPair:
    entry_ids = tuple(view.line_ids)
    group_ids = tuple(sorted(_group_line_ids(group)))
    overlap = tuple(sorted(set(entry_ids) & set(group_ids)))
    reasons: list[str] = []
    for classified in group.blocks:
        reasons.extend(getattr(classified, "reasons", None) or [])
    return AlignedPair(
        entry_id=view.entry_id,
        group_index=group_index,
        entry_line_ids=entry_ids,
        group_line_ids=group_ids,
        overlap_line_ids=overlap,
        entry_section=view.section_label,
        group_section=group.section,
        entry_type=view.entry_type.value,
        structural_roles=tuple(member.structural_role.value for member in view.members),
        classification_labels=tuple(classified.label for classified in group.blocks),
        classification_reasons=tuple(reasons),
        entry_pages=view.page_numbers,
        group_page=group.page_number,
        entry_regions=view.region_ids,
        entry_paths=view.path_ids,
        group_column_id=group.column_id,
        group_start_index=group.start_index,
        group_end_index=group.end_index,
    )


def _unmatched_view_pair(view: ExtractorEntryView) -> AlignedPair:
    return AlignedPair(
        entry_id=view.entry_id,
        group_index=None,
        entry_line_ids=tuple(view.line_ids),
        group_line_ids=(),
        overlap_line_ids=(),
        entry_section=view.section_label,
        group_section=None,
        entry_type=view.entry_type.value,
        structural_roles=tuple(member.structural_role.value for member in view.members),
        classification_labels=(),
        classification_reasons=(),
        entry_pages=view.page_numbers,
        group_page=None,
        entry_regions=view.region_ids,
        entry_paths=view.path_ids,
        group_column_id=None,
        group_start_index=None,
        group_end_index=None,
    )


def _unmatched_group_pair(group: CandidateGroup, group_index: int) -> AlignedPair:
    group_ids = tuple(sorted(_group_line_ids(group)))
    reasons: list[str] = []
    for classified in group.blocks:
        reasons.extend(getattr(classified, "reasons", None) or [])
    return AlignedPair(
        entry_id=None,
        group_index=group_index,
        entry_line_ids=(),
        group_line_ids=group_ids,
        overlap_line_ids=(),
        entry_section=None,
        group_section=group.section,
        entry_type=None,
        structural_roles=(),
        classification_labels=tuple(classified.label for classified in group.blocks),
        classification_reasons=tuple(reasons),
        entry_pages=(),
        group_page=group.page_number,
        entry_regions=(),
        entry_paths=(),
        group_column_id=group.column_id,
        group_start_index=group.start_index,
        group_end_index=group.end_index,
    )


def _ownership_differs(entry_section: str | None, entry_type: str | None, group_section: str | None) -> bool:
    if group_section is None:
        return False
    if entry_section == group_section:
        return False
    # UNKNOWN/None entry ownership vs a typed Group section is an ownership-context difference.
    return True


def _partitions_differ(view_sets: list[set[str]], group_sets: list[set[str]]) -> bool:
    nonempty_views = [ids for ids in view_sets if ids]
    nonempty_groups = [ids for ids in group_sets if ids]
    if not nonempty_views or not nonempty_groups:
        return False
    return {frozenset(ids) for ids in nonempty_views} != {frozenset(ids) for ids in nonempty_groups}


def _group_has_full_provenance(group: CandidateGroup) -> bool:
    for classified in group.blocks:
        original = classified.original
        has_path = bool(getattr(original, "path_id", None))
        has_region = bool(getattr(original, "region_id", None))
        extra = getattr(original, "source_line_ids", None)
        if has_path and has_region and extra:
            return True
    return False
