from __future__ import annotations

from dataclasses import dataclass

from app.pipeline.stages.semantic_paths import SemanticDocument, SemanticSection
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
