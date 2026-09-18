from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.domain.document import BoundingBox, TextStyle


class StructuralRole(str, Enum):
    """Structural roles — not semantic section types (EXPERIENCE/SKILLS/…)."""

    SECTION_HEADING = "SECTION_HEADING"
    ENTRY_TITLE = "ENTRY_TITLE"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    BULLET = "BULLET"
    DESCRIPTION = "DESCRIPTION"
    TECHNOLOGY = "TECHNOLOGY"
    CREDENTIAL = "CREDENTIAL"
    CONTACT = "CONTACT"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    SIDEBAR = "SIDEBAR"
    TABLE_CELL = "TABLE_CELL"
    UNKNOWN = "UNKNOWN"
    SKILL = "SKILL"


@dataclass(frozen=True)
class StructuralBlock:
    """Provenance-preserving structural unit derived from layout Lines.

    Does not assign semantic section ownership (EXPERIENCE, SKILLS, …).
    """

    block_id: str
    text: str
    line_ids: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    page_number: int
    region_id: str
    region_kind: str
    path_id: str
    bbox: BoundingBox
    style: TextStyle
    reading_order: int
    reconstruction_method: str
    role: StructuralRole
    role_score: float
    role_reasons: tuple[str, ...]
    previous_block_id: str | None = None
    next_block_id: str | None = None
