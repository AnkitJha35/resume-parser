from __future__ import annotations

import re
from collections import defaultdict

from app.domain.candidate_entry import CandidateEntry, EntryType
from app.domain.candidate_section import CandidateSection, SectionOrigin
from app.domain.document import BoundingBox
from app.domain.structural import StructuralBlock, StructuralRole


_SEGMENTABLE_LABELS = {
    "EXPERIENCE": EntryType.EXPERIENCE,
    "EDUCATION": EntryType.EDUCATION,
    "PROJECTS": EntryType.PROJECT,
    "CERTIFICATIONS": EntryType.CERTIFICATION,
}

# Explicitly non-entry sections. SUMMARY is omitted so mis-bucketed job stacks
# (e.g. experience body under an objective heading) can still be segmented —
# but entry_type stays UNKNOWN (never overrides SUMMARY ownership).
_NON_SEGMENTABLE_LABELS = {"SKILLS", "LANGUAGES", "ACHIEVEMENTS"}

_META_ROLES = {
    StructuralRole.ORGANIZATION,
    StructuralRole.LOCATION,
    StructuralRole.DATE,
    StructuralRole.CREDENTIAL,
}

_BODY_ROLES = {
    StructuralRole.DESCRIPTION,
    StructuralRole.BULLET,
}

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def build_candidate_entries(sections: list[CandidateSection]) -> list[CandidateEntry]:
    """Segment CandidateSection content into CandidateEntries where appropriate.

    Does not map blocks to Resume fields. SKILLS/LANGUAGES/ACHIEVEMENTS stay section-level.
    EntryType follows semantic_label only for known entry sections; otherwise UNKNOWN.
    """
    entries: list[CandidateEntry] = []
    entry_counter = 0
    for section in sections:
        if not _should_segment(section):
            continue
        entry_type = _entry_type_for_section(section)
        for cluster in _segment_section_content(list(section.content), section.semantic_label):
            entry = _make_entry(
                entry_id=f"entry-{entry_counter}",
                section_id=section.section_id,
                blocks=tuple(cluster),
                entry_type=entry_type,
                evidence=_entry_evidence(cluster, entry_type),
            )
            entries.append(entry)
            entry_counter += 1
    return entries


def candidate_entries_to_text_blocks(entries: list[CandidateEntry]):
    """Optional adapter: flatten entry blocks to TextBlocks without field mapping."""
    from app.pipeline.stages.semantic_compat import _structural_to_text_block

    return [
        [_structural_to_text_block(block) for block in entry.blocks]
        for entry in entries
    ]


def _should_segment(section: CandidateSection) -> bool:
    if any(_is_table_block(block) for block in section.content):
        return False
    if section.semantic_label in _NON_SEGMENTABLE_LABELS:
        return False
    if section.semantic_label in _SEGMENTABLE_LABELS:
        return True
    if section.origin == SectionOrigin.UNLABELED and _looks_like_entry_section(section.content):
        return True
    # Recover entry stacks that Phase-2 labeling placed under a non-entry heading
    # (commonly SUMMARY/objective) without forcing skills/language lists into entries.
    if _looks_like_entry_section(section.content):
        return True
    if _has_education_or_project_shape(list(section.content)):
        return True
    return False


def _entry_type_for_section(section: CandidateSection) -> EntryType:
    """Map entry_type from section ownership only — never infer a new section label."""
    if section.semantic_label in _SEGMENTABLE_LABELS:
        return _SEGMENTABLE_LABELS[section.semantic_label]
    # SUMMARY, None, UNKNOWN origin, UNLABELED → UNKNOWN (structure goes in evidence).
    return EntryType.UNKNOWN


def _looks_like_entry_section(blocks: tuple[StructuralBlock, ...]) -> bool:
    roles = {block.role for block in blocks}
    if StructuralRole.ENTRY_TITLE in roles and (roles & _META_ROLES):
        return True
    if StructuralRole.ENTRY_TITLE in roles and any(_has_date_signal(block) for block in blocks):
        return True
    if StructuralRole.CREDENTIAL in roles:
        return True
    return False


def _looks_like_experience_entry_shape(blocks: list[StructuralBlock] | tuple[StructuralBlock, ...]) -> bool:
    """Experience-like stack without requiring ENTRY_TITLE (labeled EXPERIENCE sections only)."""
    if not any(_has_date_signal(block) for block in blocks):
        return False
    roles = {block.role for block in blocks}
    has_org = any(
        block.role == StructuralRole.ORGANIZATION or _looks_like_org_candidate(block) for block in blocks
    )
    has_location = StructuralRole.LOCATION in roles
    has_body = any(_looks_like_body_candidate(block) for block in blocks)
    has_title = StructuralRole.ENTRY_TITLE in roles or any(
        _is_title_start(block, list(blocks)[index + 1 : index + 4])
        for index, block in enumerate(blocks)
        if index + 1 < len(blocks)
    )
    if has_org and has_body:
        return True
    if has_location and has_body:
        return True
    if has_org and has_location:
        return True
    if has_title and (has_org or has_location or has_body):
        return True
    if StructuralRole.ENTRY_TITLE in roles and (has_org or has_location):
        return True
    return False


def _looks_like_experience_stack(blocks: tuple[StructuralBlock, ...] | list[StructuralBlock]) -> bool:
    roles = {block.role for block in blocks}
    if StructuralRole.ENTRY_TITLE in roles:
        return any(
            block.role in {StructuralRole.ORGANIZATION, StructuralRole.LOCATION} or _has_date_signal(block)
            for block in blocks
        )
    if any(_has_date_signal(block) for block in blocks):
        return any(
            block.role in {StructuralRole.ORGANIZATION, StructuralRole.LOCATION}
            or _looks_like_org_candidate(block)
            for block in blocks
        )
    return False


def _looks_like_education_stack(blocks: tuple[StructuralBlock, ...] | list[StructuralBlock]) -> bool:
    roles = {block.role for block in blocks}
    if StructuralRole.CREDENTIAL in roles:
        return True
    has_date = any(_has_date_signal(block) for block in blocks)
    has_edu_identity = any(
        block.role == StructuralRole.ENTRY_TITLE or _is_education_text(block.text)
        for block in blocks
    )
    return has_date and has_edu_identity


def _looks_like_project_stack(blocks: tuple[StructuralBlock, ...] | list[StructuralBlock]) -> bool:
    if _looks_like_experience_stack(blocks):
        return False
    roles = {block.role for block in blocks}
    has_title = StructuralRole.ENTRY_TITLE in roles or any(
        block.role in {StructuralRole.UNKNOWN, StructuralRole.TECHNOLOGY} and len(block.text.split()) <= 6
        for block in blocks
    )
    has_date = any(_has_date_signal(block) for block in blocks)
    has_body = any(block.role in _BODY_ROLES for block in blocks)
    has_org = StructuralRole.ORGANIZATION in roles or any(_looks_like_org_candidate(block) for block in blocks)
    # Project-like: title + date + body, without an employer/org stack.
    return has_title and has_date and has_body and not has_org


def _section_has_entry_shape(
    blocks: list[StructuralBlock],
    section_label: str | None,
) -> bool:
    if _looks_like_entry_section(tuple(blocks)):
        return True
    if _has_education_or_project_shape(blocks):
        return True
    if section_label == "EXPERIENCE" and _looks_like_experience_entry_shape(blocks):
        return True
    return False


def _segment_section_content(
    blocks: list[StructuralBlock],
    section_label: str | None = None,
) -> list[list[StructuralBlock]]:
    if not blocks:
        return []
    if any(_is_table_block(block) for block in blocks):
        return []
    if not _section_has_entry_shape(blocks, section_label):
        return []

    # Segment independently per path/region — never merge across unrelated scopes.
    by_scope: dict[tuple[str, str], list[StructuralBlock]] = defaultdict(list)
    for block in blocks:
        by_scope[(block.path_id, block.region_id)].append(block)

    entries: list[list[StructuralBlock]] = []
    for scope_key in sorted(by_scope):
        scope_blocks = sorted(
            by_scope[scope_key],
            key=lambda item: (item.reading_order, item.bbox.y0, item.bbox.x0),
        )
        entries.extend(_segment_single_scope(scope_blocks, section_label))
    return entries


def _segment_single_scope(
    blocks: list[StructuralBlock],
    section_label: str | None = None,
) -> list[list[StructuralBlock]]:
    if not blocks:
        return []
    if not _section_has_entry_shape(blocks, section_label):
        return []

    entries: list[list[StructuralBlock]] = []
    current: list[StructuralBlock] = []

    index = 0
    while index < len(blocks):
        if _is_strong_new_entry(blocks, index, current):
            if current:
                entries.append(current)
            cluster_start = index
            if not current and index > 0 and _has_date_signal(blocks[index - 1]):
                cluster_start = index - 1
            current = [blocks[cluster_start]]
            index = cluster_start + 1
            while index < len(blocks) and _is_title_continuation(blocks, index, current):
                current.append(blocks[index])
                index += 1
            continue

        if current and _is_date_led_new_entry(blocks, index, current):
            entries.append(current)
            current = [blocks[index]]
            index += 1
            continue

        if not current:
            if _can_open_entry_without_title(blocks, index):
                current = [blocks[index]]
                index += 1
                continue
            # Skip non-opening preamble inside this scope; do not abort sibling scopes.
            index += 1
            continue

        current.append(blocks[index])
        index += 1

    if current:
        entries.append(current)

    return [entry for entry in entries if _entry_is_viable(entry)]


def _is_strong_new_entry(
    blocks: list[StructuralBlock],
    index: int,
    current: list[StructuralBlock],
) -> bool:
    block = blocks[index]
    if _is_table_block(block):
        return False
    if block.role in _BODY_ROLES:
        return False
    if block.role == StructuralRole.CONTACT:
        return False
    if block.role == StructuralRole.ENTRY_TITLE and current and not _entry_has_organization(current):
        return False

    preceding = blocks[max(0, index - 2) : index]
    following = blocks[index + 1 : index + 5]
    if not _is_title_start(block, following):
        return False
    if not _has_entry_corroboration(following, preceding):
        return False

    if not current:
        return True
    if not _entry_has_meta(current):
        return False
    if _entry_has_body(current) and not _has_entry_corroboration(following, preceding):
        return False
    return True


def _is_date_led_new_entry(
    blocks: list[StructuralBlock],
    index: int,
    current: list[StructuralBlock],
) -> bool:
    """Close a completed entry when a new DATE-led cluster begins."""
    if not current:
        return False
    block = blocks[index]
    if not _has_date_signal(block):
        return False
    if not _entry_has_meta(current):
        return False
    if not (_entry_has_body(current) or len(current) >= 3):
        return False
    return _has_date_open_corroboration(blocks[index + 1 : index + 5])


def _is_title_start(block: StructuralBlock, following: list[StructuralBlock]) -> bool:
    if block.role == StructuralRole.ENTRY_TITLE:
        return True
    if block.role == StructuralRole.CREDENTIAL:
        return True
    if block.role in {StructuralRole.ORGANIZATION, StructuralRole.DATE, StructuralRole.LOCATION}:
        return False
    if block.role in _BODY_ROLES:
        return False
    words = block.text.split()
    if _is_education_text(block.text) and 1 <= len(words) <= 8:
        return True
    if block.role in {StructuralRole.UNKNOWN, StructuralRole.TECHNOLOGY} and 1 <= len(words) <= 6:
        return _has_entry_corroboration(following, ())
    return False


def _is_title_continuation(
    blocks: list[StructuralBlock],
    index: int,
    current: list[StructuralBlock],
) -> bool:
    if not current:
        return False
    block = blocks[index]
    if block.role == StructuralRole.ENTRY_TITLE and not _entry_has_organization(current):
        return True
    if _entry_has_meta(current) and _entry_has_organization(current):
        return False
    if block.role in _META_ROLES or block.role in _BODY_ROLES:
        return False
    if block.role == StructuralRole.ENTRY_TITLE:
        return True
    words = block.text.split()
    if block.role in {StructuralRole.UNKNOWN, StructuralRole.TECHNOLOGY} and 1 <= len(words) <= 4:
        return _has_entry_corroboration(blocks[index + 1 : index + 4], current)
    return False


def _has_date_open_corroboration(following: list[StructuralBlock]) -> bool:
    """Corroboration for opening on a DATE block — DATE alone is never sufficient."""
    if not following:
        return False

    has_org = any(_looks_like_org_candidate(block) for block in following)
    has_location = any(block.role == StructuralRole.LOCATION for block in following)
    has_body = any(_looks_like_body_candidate(block) for block in following)
    has_credential = any(block.role == StructuralRole.CREDENTIAL for block in following)
    has_title = any(
        _is_title_start(block, following[index + 1 : index + 4])
        for index, block in enumerate(following[:3])
    )

    if has_org and has_body:
        return True
    if has_location and has_body:
        return True
    if has_org and has_location:
        return True
    if has_title and (has_org or has_location or has_body):
        return True
    if has_credential and (has_org or has_location or has_body):
        return True
    return False


def _has_entry_corroboration(
    following: list[StructuralBlock],
    preceding: tuple[StructuralBlock, ...] | list[StructuralBlock] = (),
) -> bool:
    """Require independent structural evidence; preceding DATE may corroborate following title/meta."""
    if not following and not preceding:
        return False

    has_date = any(_has_date_signal(block) for block in following) or any(
        _has_date_signal(block) for block in preceding
    )
    has_org = any(_looks_like_org_candidate(block) for block in following)
    has_location = any(block.role == StructuralRole.LOCATION for block in following)
    has_credential = any(block.role == StructuralRole.CREDENTIAL for block in following)
    has_body = any(_looks_like_body_candidate(block) for block in following)

    meta_flags = [
        has_date,
        has_org,
        has_location,
        has_credential,
    ]
    independent_meta = sum(1 for flag in meta_flags if flag)

    if has_date and has_org:
        return True
    if has_date and has_location:
        return True
    if has_date and has_credential:
        return True
    if has_date and has_body:
        return True
    if has_org and has_location:
        return True
    if independent_meta >= 2:
        return True
    return False


def _looks_like_org_candidate(block: StructuralBlock) -> bool:
    if block.role == StructuralRole.ORGANIZATION:
        return True
    if _is_organization_text(block.text):
        return True
    # Short identity line between title and date (capitalized, no digits) —
    # structural pattern, not a company-name dictionary.
    if block.role not in {StructuralRole.UNKNOWN, StructuralRole.TECHNOLOGY}:
        return False
    words = (block.text or "").split()
    if not (1 <= len(words) <= 6):
        return False
    if any(ch.isdigit() for ch in block.text):
        return False
    if _has_date_signal(block) or _looks_like_body_candidate(block):
        return False
    return any(word[:1].isupper() for word in words if word)


def _looks_like_body_candidate(block: StructuralBlock) -> bool:
    if block.role in _BODY_ROLES:
        return True
    text = (block.text or "").strip()
    words = text.split()
    # Sentence-like prose / description lines that may still be role UNKNOWN.
    if len(words) >= 4 and (text.endswith((".", ";", "!")) or len(text) >= 40):
        return True
    return False


def _can_open_entry_without_title(blocks: list[StructuralBlock], index: int) -> bool:
    block = blocks[index]
    following = blocks[index + 1 : index + 5]
    if block.role == StructuralRole.CREDENTIAL and _has_entry_corroboration(following):
        return True
    if block.role == StructuralRole.CREDENTIAL and any(_has_date_signal(b) for b in following):
        # Credential + date is an allowed credential/degree + date pattern.
        return True
    if _is_education_text(block.text) and _has_entry_corroboration(following):
        return True
    if _has_date_signal(block):
        return _has_date_open_corroboration(following)
    return False


def _has_education_or_project_shape(blocks: list[StructuralBlock]) -> bool:
    has_date = any(_has_date_signal(block) for block in blocks)
    if has_date and any(_is_education_text(block.text) for block in blocks):
        return True
    if any(block.role == StructuralRole.CREDENTIAL for block in blocks):
        return True
    if has_date and any(
        block.role in {StructuralRole.UNKNOWN, StructuralRole.ENTRY_TITLE, StructuralRole.TECHNOLOGY}
        and len(block.text.split()) <= 6
        for block in blocks
    ):
        return True
    return False


def _has_date_signal(block: StructuralBlock) -> bool:
    if block.role == StructuralRole.DATE:
        return True
    text = (block.text or "").strip()
    if _YEAR_RE.fullmatch(text):
        return True
    if _YEAR_RE.search(text) and any(sep in text for sep in ("-", "–", "—", "/")):
        return True
    return False


def _is_education_text(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in (
            "university",
            "college",
            "school",
            "institute",
            "academy",
            "bachelor",
            "master",
            "phd",
            "degree",
        )
    )


def _is_organization_text(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        re.search(rf"\b{re.escape(token)}\b", lowered)
        for token in ("inc", "llc", "ltd", "corp", "corporation", "company", "university", "college", "institute")
    )


def _entry_has_organization(blocks: list[StructuralBlock]) -> bool:
    return any(
        block.role in {StructuralRole.ORGANIZATION, StructuralRole.LOCATION}
        or (block.role == StructuralRole.UNKNOWN and _is_organization_text(block.text))
        for block in blocks
    )


def _entry_has_meta(blocks: list[StructuralBlock]) -> bool:
    return any(block.role in _META_ROLES or _has_date_signal(block) for block in blocks)


def _entry_has_body(blocks: list[StructuralBlock]) -> bool:
    return any(block.role in _BODY_ROLES for block in blocks)


def _entry_is_viable(blocks: list[StructuralBlock]) -> bool:
    if not blocks:
        return False
    if len(blocks) < 2:
        return blocks[0].role == StructuralRole.CREDENTIAL
    roles = {block.role for block in blocks}
    if StructuralRole.ENTRY_TITLE in roles:
        return _entry_has_meta(blocks) or _entry_has_body(blocks)
    if _entry_has_meta(blocks) and _entry_has_body(blocks):
        return True
    return _entry_has_meta(blocks)


def _is_table_block(block: StructuralBlock) -> bool:
    return block.role == StructuralRole.TABLE_CELL or (block.region_kind or "").lower() in {
        "table",
        "table_cell",
    }


def _entry_evidence(blocks: list[StructuralBlock], entry_type: EntryType) -> tuple[str, ...]:
    roles = sorted({block.role.value for block in blocks})
    evidence = [f"entry_type:{entry_type.value}", "roles:" + ",".join(roles)]
    if StructuralRole.ENTRY_TITLE in {block.role for block in blocks}:
        evidence.append("entry_title")
    if any(_has_date_signal(block) for block in blocks):
        evidence.append("date")
    if StructuralRole.ORGANIZATION in {block.role for block in blocks} or any(
        _looks_like_org_candidate(block) for block in blocks
    ):
        evidence.append("organization")
    if _entry_has_body(blocks):
        evidence.append("body_continuation")
    # Structural stack evidence — never promotes UNKNOWN → EXPERIENCE ownership.
    if _looks_like_experience_stack(blocks):
        evidence.append("experience_like_stack")
    if _looks_like_education_stack(blocks):
        evidence.append("education_like_stack")
    if _looks_like_project_stack(blocks):
        evidence.append("project_like_stack")
    page_numbers = {block.page_number for block in blocks}
    region_ids = {block.region_id for block in blocks}
    path_ids = {block.path_id for block in blocks}
    if len(page_numbers) > 1:
        evidence.append("spans_multiple_pages")
    if len(region_ids) > 1:
        evidence.append("spans_multiple_regions")
    if len(path_ids) > 1:
        evidence.append("spans_multiple_paths")
    return tuple(evidence)


def _make_entry(
    *,
    entry_id: str,
    section_id: str,
    blocks: tuple[StructuralBlock, ...],
    entry_type: EntryType,
    evidence: tuple[str, ...],
) -> CandidateEntry:
    bbox = BoundingBox(
        min(block.bbox.x0 for block in blocks),
        min(block.bbox.y0 for block in blocks),
        max(block.bbox.x1 for block in blocks),
        max(block.bbox.y1 for block in blocks),
    )
    line_ids: list[str] = []
    span_ids: list[str] = []
    page_numbers: list[int] = []
    region_ids: list[str] = []
    path_ids: list[str] = []
    reconstruction_methods: list[str] = []
    for block in blocks:
        line_ids.extend(block.line_ids)
        span_ids.extend(block.source_span_ids)
        if block.page_number not in page_numbers:
            page_numbers.append(block.page_number)
        if block.region_id not in region_ids:
            region_ids.append(block.region_id)
        if block.path_id not in path_ids:
            path_ids.append(block.path_id)
        method = block.reconstruction_method or ""
        if method and method not in reconstruction_methods:
            reconstruction_methods.append(method)

    confidence = min(
        1.0,
        0.35
        * sum(
            1
            for token in evidence
            if token in {"entry_title", "date", "organization", "body_continuation"}
        ),
    )

    return CandidateEntry(
        entry_id=entry_id,
        section_id=section_id,
        blocks=blocks,
        entry_type=entry_type,
        page_numbers=tuple(page_numbers),
        region_ids=tuple(region_ids),
        path_ids=tuple(path_ids),
        reconstruction_methods=tuple(reconstruction_methods),
        bbox=bbox,
        reading_order=min(block.reading_order for block in blocks),
        confidence=confidence,
        evidence=evidence,
        line_ids=tuple(line_ids),
        source_span_ids=tuple(span_ids),
    )
