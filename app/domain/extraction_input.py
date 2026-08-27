from __future__ import annotations

from dataclasses import dataclass

from app.domain.candidate_entry import EntryType
from app.domain.candidate_section import SectionOrigin
from app.domain.document import BoundingBox, TextStyle
from app.domain.structural import StructuralRole
from app.pipeline.stages.text_extraction import TextBlock


@dataclass(frozen=True)
class ExtractorMember:
    """One StructuralBlock projected for observation — not a classify_block label."""

    text: str
    text_block: TextBlock
    structural_role: StructuralRole
    role_score: float
    line_ids: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    page: int
    region_id: str
    region_kind: str
    path_id: str
    bbox: BoundingBox
    style: TextStyle
    reconstruction_method: str
    reading_order: int
    previous_block_id: str | None
    next_block_id: str | None


@dataclass(frozen=True)
class ExtractorEntryView:
    """Compatibility/observation view of a CandidateEntry.

    section_label is copied from CandidateSection — never inferred here.
    Structural roles are not mapped to JOB_TITLE/COMPANY/DEGREE/INSTITUTION.
    """

    entry_id: str
    section_id: str
    section_label: str | None
    section_origin: SectionOrigin
    entry_type: EntryType
    members: tuple[ExtractorMember, ...]
    evidence: tuple[str, ...]
    line_ids: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    page_numbers: tuple[int, ...]
    region_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    reconstruction_methods: tuple[str, ...]
    bbox: BoundingBox
    reading_order: int
    confidence: float
