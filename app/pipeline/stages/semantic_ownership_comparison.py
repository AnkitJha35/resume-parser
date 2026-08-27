from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.domain.candidate_section import CandidateSection, SectionOrigin
from app.domain.document import Document, Line
from app.domain.structural import StructuralBlock
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.sections import SECTION_NAMES, SectionDetector
from app.pipeline.stages.semantic_compat import (
    candidate_sections_to_text_blocks,
    semantic_sections_to_text_blocks,
)
from app.pipeline.stages.semantic_paths import SemanticDocument, detect_region_aware_sections
from app.pipeline.stages.structural_roles import build_structural_blocks


DISAGREEMENT_CATEGORIES = (
    "SECTION_MISSING",
    "SECTION_EXTRA",
    "SECTION_LABEL_DIFFERENCE",
    "SECTION_BOUNDARY_DIFFERENCE",
    "LINE_UNASSIGNED",
    "LINE_DOUBLE_ASSIGNED",
    "PATH_SCOPE_DIFFERENCE",
    "REGION_SCOPE_DIFFERENCE",
    "CONTINUATION_DIFFERENCE",
    "PROVENANCE_DIFFERENCE",
)


@dataclass(frozen=True)
class LineOwnership:
    """Per-line ownership snapshot. Alignment key is line_id (then source_span_ids)."""

    line_id: str
    text: str
    page_number: int
    region_id: str
    path_id: str
    source_span_ids: tuple[str, ...]
    reconstruction_method: str
    current_labels: tuple[str, ...]
    current_paths: tuple[str, ...]
    current_regions: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    candidate_origins: tuple[str, ...]
    candidate_section_ids: tuple[str, ...]
    candidate_heading_line_ids: tuple[str, ...]
    candidate_paths: tuple[str, ...]
    candidate_regions: tuple[str, ...]
    is_candidate_heading: bool


@dataclass(frozen=True)
class OwnershipDisagreement:
    category: str
    detail: str
    line_id: str | None = None
    text: str | None = None
    section: str | None = None
    semantic_label: str | None = None
    origin: str | None = None
    heading_line_id: str | None = None
    member_line_ids: tuple[str, ...] = ()
    page: int | None = None
    region_id: str | None = None
    path_id: str | None = None
    current_ownership: str = ""
    candidate_ownership: str = ""


@dataclass(frozen=True)
class OwnershipComparison:
    line_records: tuple[LineOwnership, ...]
    disagreements: tuple[OwnershipDisagreement, ...]
    first_loss: OwnershipDisagreement | None
    current_labeled_line_ids: tuple[str, ...]
    candidate_claimed_line_ids: tuple[str, ...]
    current_unassigned_line_ids: tuple[str, ...]
    candidate_unclaimed_line_ids: tuple[str, ...]


def compare_semantic_ownership(
    document: Document,
    section_detector: SectionDetector | None = None,
) -> OwnershipComparison:
    """Compare detect_region_aware_sections vs CandidateSection. Neither side is truth."""
    detector = section_detector or SectionDetector()
    semantic = detect_region_aware_sections(document, detector)
    structural_blocks = build_structural_blocks(document)
    candidate_sections = build_candidate_sections(structural_blocks, detector)
    return compare_ownership_artifacts(document, semantic, candidate_sections, structural_blocks)


def compare_ownership_artifacts(
    document: Document,
    semantic: SemanticDocument,
    candidate_sections: list[CandidateSection],
    structural_blocks: list[StructuralBlock],
) -> OwnershipComparison:
    catalog = _catalog_document_lines(document)
    current_map = _current_assignments(semantic)
    candidate_map = _candidate_assignments(candidate_sections)

    records: list[LineOwnership] = []
    disagreements: list[OwnershipDisagreement] = []

    for line_id, meta in catalog.items():
        current = current_map.get(line_id, [])
        candidate = candidate_map.get(line_id, [])
        record = _line_ownership(meta, current, candidate)
        records.append(record)
        disagreements.extend(_line_disagreements(record, current, candidate, catalog))

    disagreements.extend(_section_inventory_disagreements(semantic, candidate_sections))
    disagreements.extend(_boundary_disagreements(semantic, candidate_sections, catalog))
    disagreements.extend(_double_assignment_disagreements(current_map, candidate_map, catalog))
    disagreements.extend(_compat_provenance_disagreements(semantic, candidate_sections, structural_blocks))

    unique = _dedupe(disagreements)
    first_loss = _first_loss(records, unique)

    current_labeled = tuple(
        sorted(line_id for line_id, hits in current_map.items() if hits)
    )
    candidate_claimed = tuple(sorted(candidate_map))
    unassigned = tuple(sorted(line_id for line_id in catalog if not current_map.get(line_id)))
    unclaimed = tuple(sorted(line_id for line_id in catalog if line_id not in candidate_map))

    return OwnershipComparison(
        line_records=tuple(records),
        disagreements=tuple(unique),
        first_loss=first_loss,
        current_labeled_line_ids=current_labeled,
        candidate_claimed_line_ids=candidate_claimed,
        current_unassigned_line_ids=unassigned,
        candidate_unclaimed_line_ids=unclaimed,
    )


def compare_extraction_input_streams(
    document: Document,
    section_detector: SectionDetector | None = None,
) -> tuple[OwnershipDisagreement, ...]:
    """Diagnostic: section-bucket TextBlock streams without changing parser.py."""
    detector = section_detector or SectionDetector()
    semantic = detect_region_aware_sections(document, detector)
    structural_blocks = build_structural_blocks(document)
    candidate_sections = build_candidate_sections(structural_blocks, detector)

    current_blocks = semantic_sections_to_text_blocks(semantic)
    candidate_blocks = candidate_sections_to_text_blocks(candidate_sections, structural_blocks)

    disagreements: list[OwnershipDisagreement] = []
    labels = set(SECTION_NAMES) | {"UNASSIGNED"}
    for label in sorted(labels):
        current_ids = _compat_line_ids(current_blocks.get(label, []))
        candidate_ids = _compat_line_ids(candidate_blocks.get(label, []))
        if current_ids == candidate_ids:
            continue
        only_current = tuple(sorted(current_ids - candidate_ids))
        only_candidate = tuple(sorted(candidate_ids - current_ids))
        disagreements.append(
            OwnershipDisagreement(
                category="SECTION_BOUNDARY_DIFFERENCE",
                detail=(
                    f"compat stream {label}: current_only={only_current} "
                    f"candidate_only={only_candidate}"
                ),
                section=label,
                semantic_label=label,
                member_line_ids=tuple(sorted(current_ids | candidate_ids)),
                current_ownership=f"{label}:{sorted(current_ids)}",
                candidate_ownership=f"{label}:{sorted(candidate_ids)}",
            )
        )
        for block in current_blocks.get(label, []):
            if not getattr(block, "path_id", None) or not getattr(block, "region_id", None):
                disagreements.append(
                    OwnershipDisagreement(
                        category="PROVENANCE_DIFFERENCE",
                        detail="current compat TextBlock lacks path_id/region_id attributes",
                        line_id=getattr(block, "source_line_id", None),
                        text=block.text,
                        section=label,
                        current_ownership=label,
                    )
                )
                break
        extra = getattr(current_blocks.get(label, [None])[0], "source_line_ids", None) if current_blocks.get(label) else None
        if current_blocks.get(label) and not extra:
            disagreements.append(
                OwnershipDisagreement(
                    category="PROVENANCE_DIFFERENCE",
                    detail="compat TextBlock exposes source_line_id (first id) but not full source_line_ids",
                    section=label,
                    current_ownership=label,
                )
            )
    return tuple(_dedupe(disagreements))


def disagreement_categories(comparison: OwnershipComparison) -> set[str]:
    return {item.category for item in comparison.disagreements}


def _catalog_document_lines(document: Document) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    path_index = 0
    for page in document.pages:
        regions = sorted(
            page.regions,
            key=lambda region: (
                region.reading_order if region.reading_order is not None else 10**9,
                region.region_id,
            ),
        )
        for region_order, region in enumerate(regions):
            path_id = f"page-{page.page_number}-path-{region_order}"
            for line in region.lines:
                catalog[line.line_id] = {
                    "line": line,
                    "path_id": path_id,
                    "region_id": region.region_id,
                    "page_number": page.page_number,
                    "order": path_index,
                }
                path_index += 1
    return catalog


def _current_assignments(semantic: SemanticDocument) -> dict[str, list[dict]]:
    assigned: dict[str, list[dict]] = defaultdict(list)
    for label, sections in semantic.sections.items():
        for section in sections:
            for line in section.lines:
                assigned[line.line_id].append(
                    {
                        "label": label,
                        "path_id": section.path_id,
                        "region_id": section.region_id,
                        "page": section.page_number,
                        "heading_line_id": None,
                        "origin": "CURRENT_SEMANTIC",
                    }
                )
    return assigned


def _candidate_assignments(sections: list[CandidateSection]) -> dict[str, list[dict]]:
    assigned: dict[str, list[dict]] = defaultdict(list)
    for section in sections:
        heading_ids = tuple(section.heading.line_ids) if section.heading is not None else ()
        members: list[tuple[str, bool]] = []
        if section.heading is not None:
            for line_id in section.heading.line_ids:
                members.append((line_id, True))
        for block in section.content:
            for line_id in block.line_ids:
                members.append((line_id, False))
        for line_id, is_heading in members:
            assigned[line_id].append(
                {
                    "label": section.semantic_label,
                    "origin": section.origin.value,
                    "section_id": section.section_id,
                    "path_id": section.path_id,
                    "region_id": section.region_id,
                    "page": section.page_number,
                    "heading_line_id": heading_ids[0] if heading_ids else None,
                    "is_heading": is_heading,
                }
            )
    return assigned


def _line_ownership(meta: dict, current: list[dict], candidate: list[dict]) -> LineOwnership:
    line: Line = meta["line"]
    return LineOwnership(
        line_id=line.line_id,
        text=line.text,
        page_number=meta["page_number"],
        region_id=meta["region_id"],
        path_id=meta["path_id"],
        source_span_ids=tuple(line.source_span_ids),
        reconstruction_method=line.reconstruction_method,
        current_labels=tuple(item["label"] for item in current),
        current_paths=tuple(item["path_id"] for item in current),
        current_regions=tuple(item["region_id"] for item in current),
        candidate_labels=tuple(item["label"] for item in candidate if item["label"] is not None),
        candidate_origins=tuple(item["origin"] for item in candidate),
        candidate_section_ids=tuple(item["section_id"] for item in candidate),
        candidate_heading_line_ids=tuple(
            item["heading_line_id"] for item in candidate if item.get("heading_line_id")
        ),
        candidate_paths=tuple(item["path_id"] for item in candidate),
        candidate_regions=tuple(item["region_id"] for item in candidate),
        is_candidate_heading=any(item.get("is_heading") for item in candidate),
    )


def _line_disagreements(
    record: LineOwnership,
    current: list[dict],
    candidate: list[dict],
    catalog: dict[str, dict],
) -> list[OwnershipDisagreement]:
    items: list[OwnershipDisagreement] = []
    current_assigned = bool(current)
    candidate_claimed = bool(candidate)
    current_label = record.current_labels[0] if record.current_labels else None
    candidate_label = record.candidate_labels[0] if record.candidate_labels else None
    candidate_origin = record.candidate_origins[0] if record.candidate_origins else None
    heading_id = record.candidate_heading_line_ids[0] if record.candidate_heading_line_ids else None
    current_desc = ",".join(record.current_labels) or "UNASSIGNED"
    candidate_desc = (
        f"{candidate_origin}:{candidate_label or 'None'}" if candidate_claimed else "UNCLAIMED"
    )
    base = dict(
        line_id=record.line_id,
        text=record.text,
        page=record.page_number,
        region_id=record.region_id,
        path_id=record.path_id,
        heading_line_id=heading_id,
        current_ownership=current_desc,
        candidate_ownership=candidate_desc,
        origin=candidate_origin,
        semantic_label=candidate_label or current_label,
        section=current_label or candidate_label,
    )

    if current_assigned and not candidate_claimed:
        items.append(OwnershipDisagreement(
            category="LINE_UNASSIGNED",
            detail="line assigned by current semantic path but unclaimed by CandidateSection",
            **base,
        ))
    if candidate_claimed and not current_assigned:
        items.append(OwnershipDisagreement(
            category="LINE_UNASSIGNED",
            detail="line claimed by CandidateSection but UNASSIGNED on current semantic path",
            **base,
        ))

    if current_label and candidate_label and current_label != candidate_label:
        items.append(OwnershipDisagreement(
            category="SECTION_LABEL_DIFFERENCE",
            detail=f"current={current_label} candidate={candidate_label}",
            **base,
        ))
    if current_label and candidate_claimed and not candidate_label and candidate_origin in {
        SectionOrigin.UNLABELED.value,
        SectionOrigin.UNKNOWN.value,
    }:
        items.append(OwnershipDisagreement(
            category="SECTION_LABEL_DIFFERENCE",
            detail=f"current={current_label} candidate_origin={candidate_origin} label=None",
            **base,
        ))

    if current and candidate:
        if record.current_paths and record.candidate_paths and set(record.current_paths) != set(record.candidate_paths):
            items.append(OwnershipDisagreement(
                category="PATH_SCOPE_DIFFERENCE",
                detail=f"current_paths={record.current_paths} candidate_paths={record.candidate_paths}",
                **base,
            ))
        if record.current_regions and record.candidate_regions and set(record.current_regions) != set(record.candidate_regions):
            items.append(OwnershipDisagreement(
                category="REGION_SCOPE_DIFFERENCE",
                detail=f"current_regions={record.current_regions} candidate_regions={record.candidate_regions}",
                **base,
            ))

    if (
        record.page_number > 1
        and current_assigned
        and candidate_origin == SectionOrigin.UNLABELED.value
        and not record.is_candidate_heading
    ):
        items.append(OwnershipDisagreement(
            category="CONTINUATION_DIFFERENCE",
            detail="page>1 line labeled by current path without a CandidateSection heading (likely continuation)",
            **base,
        ))
    if (
        record.page_number > 1
        and current_assigned
        and not candidate_claimed
    ):
        items.append(OwnershipDisagreement(
            category="CONTINUATION_DIFFERENCE",
            detail="page>1 line owned by current path, unclaimed by CandidateSection",
            **base,
        ))

    _ = catalog
    return items


def _section_inventory_disagreements(
    semantic: SemanticDocument,
    candidate_sections: list[CandidateSection],
) -> list[OwnershipDisagreement]:
    items: list[OwnershipDisagreement] = []
    current_labels = {
        label for label, sections in semantic.sections.items() if any(section.lines for section in sections)
    }
    candidate_labels = {
        section.semantic_label
        for section in candidate_sections
        if section.semantic_label
        and section.origin
        in {SectionOrigin.KNOWN_ALIAS, SectionOrigin.INFERRED, SectionOrigin.CONTINUED}
    }
    for label in sorted(current_labels - candidate_labels):
        members = tuple(
            line.line_id
            for section in semantic.sections[label]
            for line in section.lines
        )
        items.append(
            OwnershipDisagreement(
                category="SECTION_MISSING",
                detail=(
                    f"current path has labeled {label} content; "
                    f"CandidateSection has no KNOWN_ALIAS/INFERRED/CONTINUED {label}"
                ),
                section=label,
                semantic_label=label,
                member_line_ids=members,
                current_ownership=label,
                candidate_ownership="absent",
            )
        )
    for label in sorted(candidate_labels - current_labels):
        members = tuple(
            line_id
            for section in candidate_sections
            if section.semantic_label == label
            for line_id in section.line_ids
        )
        origin = next(
            section.origin.value
            for section in candidate_sections
            if section.semantic_label == label
        )
        heading = next(
            (section.heading.line_ids[0] if section.heading and section.heading.line_ids else None)
            for section in candidate_sections
            if section.semantic_label == label
        )
        items.append(
            OwnershipDisagreement(
                category="SECTION_EXTRA",
                detail=f"CandidateSection has {origin} {label}; current path has no lines in {label}",
                section=label,
                semantic_label=label,
                origin=origin,
                heading_line_id=heading,
                member_line_ids=members,
                current_ownership="absent",
                candidate_ownership=f"{origin}:{label}",
            )
        )
    return items


def _boundary_disagreements(
    semantic: SemanticDocument,
    candidate_sections: list[CandidateSection],
    catalog: dict[str, dict],
) -> list[OwnershipDisagreement]:
    items: list[OwnershipDisagreement] = []
    for label in SECTION_NAMES:
        current_ids = {
            line.line_id
            for section in semantic.sections.get(label, [])
            for line in section.lines
        }
        candidate_ids = {
            line_id
            for section in candidate_sections
            if section.semantic_label == label
            for line_id in _candidate_content_line_ids(section)
        }
        if not current_ids and not candidate_ids:
            continue
        if current_ids != candidate_ids and (current_ids & candidate_ids or (current_ids and candidate_ids)):
            items.append(
                OwnershipDisagreement(
                    category="SECTION_BOUNDARY_DIFFERENCE",
                    detail=(
                        f"{label} members differ current_only={tuple(sorted(current_ids - candidate_ids))} "
                        f"candidate_only={tuple(sorted(candidate_ids - current_ids))}"
                    ),
                    section=label,
                    semantic_label=label,
                    member_line_ids=tuple(sorted(current_ids | candidate_ids)),
                    current_ownership=f"{label}:{sorted(current_ids)}",
                    candidate_ownership=f"{label}:{sorted(candidate_ids)}",
                )
            )
    _ = catalog
    return items


def _candidate_content_line_ids(section: CandidateSection) -> tuple[str, ...]:
    ids: list[str] = []
    for block in section.content:
        ids.extend(block.line_ids)
    return tuple(ids)


def _double_assignment_disagreements(
    current_map: dict[str, list[dict]],
    candidate_map: dict[str, list[dict]],
    catalog: dict[str, dict],
) -> list[OwnershipDisagreement]:
    items: list[OwnershipDisagreement] = []
    for line_id, hits in current_map.items():
        labels = [item["label"] for item in hits]
        if len(hits) > 1:
            meta = catalog.get(line_id, {})
            line = meta.get("line")
            items.append(
                OwnershipDisagreement(
                    category="LINE_DOUBLE_ASSIGNED",
                    detail=f"current semantic path assigns {line_id} to {labels}",
                    line_id=line_id,
                    text=getattr(line, "text", None),
                    page=meta.get("page_number"),
                    current_ownership=",".join(labels),
                    candidate_ownership="",
                )
            )
    for line_id, hits in candidate_map.items():
        if len(hits) > 1:
            meta = catalog.get(line_id, {})
            line = meta.get("line")
            items.append(
                OwnershipDisagreement(
                    category="LINE_DOUBLE_ASSIGNED",
                    detail=f"CandidateSection assigns {line_id} to {[item['section_id'] for item in hits]}",
                    line_id=line_id,
                    text=getattr(line, "text", None),
                    page=meta.get("page_number"),
                    current_ownership="",
                    candidate_ownership=",".join(item["section_id"] for item in hits),
                )
            )
    return items


def _compat_provenance_disagreements(
    semantic: SemanticDocument,
    candidate_sections: list[CandidateSection],
    structural_blocks: list[StructuralBlock],
) -> list[OwnershipDisagreement]:
    items: list[OwnershipDisagreement] = []
    current_blocks = semantic_sections_to_text_blocks(semantic)
    for label, blocks in current_blocks.items():
        if not blocks:
            continue
        sample = blocks[0]
        if not getattr(sample, "path_id", None) or not getattr(sample, "region_id", None):
            items.append(
                OwnershipDisagreement(
                    category="PROVENANCE_DIFFERENCE",
                    detail="semantic_sections_to_text_blocks TextBlock has no path_id/region_id",
                    section=label,
                    semantic_label=label if label != "UNASSIGNED" else None,
                    current_ownership=label,
                )
            )
            break
    candidate_blocks = candidate_sections_to_text_blocks(candidate_sections, structural_blocks)
    for label, blocks in candidate_blocks.items():
        if not blocks:
            continue
        sample = blocks[0]
        extra = getattr(sample, "source_line_ids", None)
        if getattr(sample, "source_line_id", None) and not extra:
            items.append(
                OwnershipDisagreement(
                    category="PROVENANCE_DIFFERENCE",
                    detail="candidate_sections_to_text_blocks sets source_line_id to first line only",
                    section=label,
                    candidate_ownership=label,
                )
            )
            break
        if not getattr(sample, "path_id", None):
            items.append(
                OwnershipDisagreement(
                    category="PROVENANCE_DIFFERENCE",
                    detail="candidate_sections_to_text_blocks TextBlock has no path_id/region_id",
                    section=label,
                    candidate_ownership=label,
                )
            )
            break
    return items


def _compat_line_ids(blocks: list) -> set[str]:
    ids: set[str] = set()
    for block in blocks:
        extra = getattr(block, "source_line_ids", None)
        if extra:
            ids.update(str(item) for item in extra)
        line_id = getattr(block, "source_line_id", None)
        if line_id:
            ids.add(str(line_id))
    return ids


def _first_loss(
    records: list[LineOwnership],
    disagreements: list[OwnershipDisagreement],
) -> OwnershipDisagreement | None:
    preferred = (
        "LINE_UNASSIGNED",
        "LINE_DOUBLE_ASSIGNED",
        "SECTION_MISSING",
        "SECTION_EXTRA",
        "CONTINUATION_DIFFERENCE",
        "SECTION_LABEL_DIFFERENCE",
        "SECTION_BOUNDARY_DIFFERENCE",
        "PATH_SCOPE_DIFFERENCE",
        "REGION_SCOPE_DIFFERENCE",
        "PROVENANCE_DIFFERENCE",
    )
    by_line = {item.line_id: item for item in disagreements if item.line_id}
    for record in records:
        hit = by_line.get(record.line_id)
        if hit and hit.category in {"LINE_UNASSIGNED", "LINE_DOUBLE_ASSIGNED", "SECTION_LABEL_DIFFERENCE"}:
            return hit
    for category in preferred:
        for item in disagreements:
            if item.category == category:
                return item
    return disagreements[0] if disagreements else None


def _dedupe(items: list[OwnershipDisagreement]) -> list[OwnershipDisagreement]:
    seen: set[tuple] = set()
    unique: list[OwnershipDisagreement] = []
    for item in items:
        key = (item.category, item.detail, item.line_id, item.section)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
