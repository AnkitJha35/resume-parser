from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.domain.document import BoundingBox
from app.domain.structural import StructuralBlock


class EntryType(str, Enum):
    """Structural/contextual entry kind — not a public Resume field mapping.

    Must never override CandidateSection.semantic_label ownership.
    """

    EXPERIENCE = "EXPERIENCE"
    EDUCATION = "EDUCATION"
    PROJECT = "PROJECT"
    CERTIFICATION = "CERTIFICATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CandidateEntry:
    """One logical entry inside a CandidateSection.

    Holds StructuralBlocks only — does not assign company/designation/degree fields.
    Provenance is aggregated from all member blocks; plurals are authoritative when
    an entry spans multiple pages/regions/paths.
    """

    entry_id: str
    section_id: str
    blocks: tuple[StructuralBlock, ...]
    entry_type: EntryType
    page_numbers: tuple[int, ...]
    region_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    reconstruction_methods: tuple[str, ...]
    bbox: BoundingBox
    reading_order: int
    confidence: float
    evidence: tuple[str, ...]
    line_ids: tuple[str, ...]
    source_span_ids: tuple[str, ...]

    @property
    def page_number(self) -> int | None:
        """Sole page when unique; None if the entry spans multiple pages."""
        if len(self.page_numbers) == 1:
            return self.page_numbers[0]
        return None

    @property
    def region_id(self) -> str | None:
        """Sole region when unique; None if the entry spans multiple regions."""
        if len(self.region_ids) == 1:
            return self.region_ids[0]
        return None

    @property
    def path_id(self) -> str | None:
        """Sole path when unique; None if the entry spans multiple paths."""
        if len(self.path_ids) == 1:
            return self.path_ids[0]
        return None
