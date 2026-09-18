from __future__ import annotations

from dataclasses import dataclass

from app.domain.candidate_section import CandidateSection, SectionOrigin
from app.domain.structural import StructuralBlock
from app.pipeline.stages.semantic_paths import SemanticDocument
from app.pipeline.stages.sections import SECTION_NAMES
from app.pipeline.stages.text_extraction import TextBlock


@dataclass(frozen=True)
class CompatibilitySection:
    section: str
    page_number: int
    region_id: str
    path_id: str
    blocks: list[TextBlock]


def semantic_sections_to_text_blocks(
    semantic_document: SemanticDocument,
) -> dict[str, list[TextBlock]]:
    """Convert region-aware sections into ordered compatibility TextBlock views."""
    converted: dict[str, list[TextBlock]] = {
        section: [] for section in semantic_document.sections
    }
    for section, semantic_sections in semantic_document.sections.items():
        for semantic_section in semantic_sections:
            converted[section].extend(
                _line_to_text_block(line)
                for line in semantic_section.lines
            )
    if semantic_document.unassigned_lines:
        converted["UNASSIGNED"] = [
            _line_to_text_block(line) for line in semantic_document.unassigned_lines
        ]
    return converted


def semantic_header_to_text_blocks(
    semantic_document: SemanticDocument,
) -> list[TextBlock]:
    """Return unassigned physical header/contact candidates as TextBlocks."""
    return [_line_to_text_block(line) for line in semantic_document.unassigned_lines]


def semantic_sections_to_section_views(
    semantic_document: SemanticDocument,
) -> dict[str, list[CompatibilitySection]]:
    """Retain section/path boundaries while exposing TextBlock-compatible blocks."""
    views: dict[str, list[CompatibilitySection]] = {
        section: [] for section in semantic_document.sections
    }
    for section, semantic_sections in semantic_document.sections.items():
        views[section] = [
            CompatibilitySection(
                section=item.section,
                page_number=item.page_number,
                region_id=item.region_id,
                path_id=item.path_id,
                blocks=[_line_to_text_block(line) for line in item.lines],
            )
            for item in semantic_sections
        ]
    return views


def candidate_sections_to_text_blocks(
    sections: list[CandidateSection],
    all_blocks: list[StructuralBlock] | None = None,
) -> dict[str, list[TextBlock]]:
    """Adapt labeled CandidateSections into the existing section→TextBlock shape.

    Only KNOWN_ALIAS / INFERRED / CONTINUED sections with a concrete semantic_label
    contribute content blocks (headings excluded, matching detect_region_aware_sections).
    UNKNOWN and UNLABELED content remains available via UNASSIGNED when all_blocks
    is provided.
    """
    converted: dict[str, list[TextBlock]] = {name: [] for name in SECTION_NAMES}
    assigned_ids: set[str] = set()

    for section in sections:
        if section.origin in {SectionOrigin.UNKNOWN, SectionOrigin.UNLABELED}:
            continue
        label = section.semantic_label
        if label is None or label not in converted:
            continue
        for block in section.content:
            converted[label].append(_structural_to_text_block(block))
            assigned_ids.add(block.block_id)
        if section.heading is not None:
            assigned_ids.add(section.heading.block_id)

    if all_blocks is not None:
        unassigned = [
            _structural_to_text_block(block)
            for block in all_blocks
            if block.block_id not in assigned_ids
        ]
        if unassigned:
            converted["UNASSIGNED"] = unassigned
    else:
        unassigned = []
        for section in sections:
            for block in section.content:
                if block.block_id not in assigned_ids:
                    unassigned.append(_structural_to_text_block(block))
                    assigned_ids.add(block.block_id)
        if unassigned:
            converted["UNASSIGNED"] = unassigned
    return converted


def candidate_header_to_text_blocks(
    sections: list[CandidateSection],
    all_blocks: list[StructuralBlock],
) -> list[TextBlock]:
    """Return blocks not claimed by labeled CandidateSection content/headings."""
    assigned_ids: set[str] = set()
    for section in sections:
        if section.origin in {SectionOrigin.UNKNOWN, SectionOrigin.UNLABELED}:
            continue
        if section.semantic_label is None:
            continue
        if section.heading is not None:
            assigned_ids.add(section.heading.block_id)
        for block in section.content:
            assigned_ids.add(block.block_id)
    return [
        _structural_to_text_block(block)
        for block in all_blocks
        if block.block_id not in assigned_ids
    ]


def _structural_to_text_block(block: StructuralBlock) -> TextBlock:
    text_block = TextBlock(
        text=block.text,
        page_number=block.page_number,
        x0=block.bbox.x0,
        y0=block.bbox.y0,
        x1=block.bbox.x1,
        y1=block.bbox.y1,
        font_size=block.style.font_size,
        bold=block.style.bold,
    )
    text_block.source_line_id = block.line_ids[0] if block.line_ids else None
    text_block.source_span_ids = list(block.source_span_ids)
    text_block.reconstruction_method = block.reconstruction_method
    return text_block


def _line_to_text_block(line) -> TextBlock:
    block = TextBlock(
        text=line.text,
        page_number=line.page_number,
        x0=line.bbox.x0,
        y0=line.bbox.y0,
        x1=line.bbox.x1,
        y1=line.bbox.y1,
        font_size=line.style.font_size,
        bold=line.style.bold,
    )
    block.source_line_id = line.line_id
    block.source_span_ids = list(line.source_span_ids)
    block.reconstruction_method = line.reconstruction_method
    return block
