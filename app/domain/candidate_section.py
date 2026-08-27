from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.domain.document import BoundingBox
from app.domain.structural import StructuralBlock


class SectionOrigin(str, Enum):
    """How a CandidateSection boundary/label was established."""

    KNOWN_ALIAS = "KNOWN_ALIAS"
    INFERRED = "INFERRED"
    CONTINUED = "CONTINUED"
    UNLABELED = "UNLABELED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CandidateSection:
    """First-class section candidate: boundary separate from semantic label.

    Does not perform entry segmentation. Heading may be absent (headingless resumes).
    """

    section_id: str
    origin: SectionOrigin
    semantic_label: str | None
    heading: StructuralBlock | None
    content: tuple[StructuralBlock, ...]
    page_number: int
    region_id: str
    path_id: str
    bbox: BoundingBox
    reading_order: int
    confidence: float
    evidence: tuple[str, ...]
    line_ids: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    continuation_of: str | None = None
