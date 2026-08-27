from __future__ import annotations

from app.domain.candidate_entry import CandidateEntry
from app.domain.candidate_section import CandidateSection
from app.domain.extraction_input import ExtractorEntryView, ExtractorMember
from app.pipeline.stages.semantic_compat import _structural_to_text_block


def candidate_entry_to_extractor_view(
    entry: CandidateEntry,
    section: CandidateSection,
) -> ExtractorEntryView:
    """Project a CandidateEntry into an extractor-compatible observation view.

    Does not classify fields or remap StructuralRole → classify_block labels.
    TextBlock conversion is reused; full line_ids stay on the member (not only first).
    """
    members = tuple(_member_from_block(block) for block in entry.blocks)
    return ExtractorEntryView(
        entry_id=entry.entry_id,
        section_id=entry.section_id,
        section_label=section.semantic_label,
        section_origin=section.origin,
        entry_type=entry.entry_type,
        members=members,
        evidence=entry.evidence,
        line_ids=entry.line_ids,
        source_span_ids=entry.source_span_ids,
        page_numbers=entry.page_numbers,
        region_ids=entry.region_ids,
        path_ids=entry.path_ids,
        reconstruction_methods=entry.reconstruction_methods,
        bbox=entry.bbox,
        reading_order=entry.reading_order,
        confidence=entry.confidence,
    )


def candidate_entries_to_extractor_views(
    entries: list[CandidateEntry],
    sections: list[CandidateSection],
) -> list[ExtractorEntryView]:
    by_id = {section.section_id: section for section in sections}
    views: list[ExtractorEntryView] = []
    for entry in entries:
        section = by_id.get(entry.section_id)
        if section is None:
            continue
        views.append(candidate_entry_to_extractor_view(entry, section))
    return views


def _member_from_block(block) -> ExtractorMember:
    text_block = _structural_to_text_block(block)
    # Compat TextBlock keeps source_line_id as the first line; preserve the full set.
    text_block.source_line_ids = list(block.line_ids)
    return ExtractorMember(
        text=block.text,
        text_block=text_block,
        structural_role=block.role,
        role_score=block.role_score,
        line_ids=block.line_ids,
        source_span_ids=block.source_span_ids,
        page=block.page_number,
        region_id=block.region_id,
        region_kind=block.region_kind,
        path_id=block.path_id,
        bbox=block.bbox,
        style=block.style,
        reconstruction_method=block.reconstruction_method,
        reading_order=block.reading_order,
        previous_block_id=block.previous_block_id,
        next_block_id=block.next_block_id,
    )
