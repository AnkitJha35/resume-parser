"""Phase 8-2A: SemanticInput and SemanticOutput contracts for LLM-assisted extraction.

These models define:
1. SemanticInput: Compact, non-redundant, JSON-serializable representation of layout-aware
   Document IR including generic table metadata and suggested structural roles.
2. SemanticOutput: Constrained, provenance-grounded semantic structure produced by an LLM,
   supporting conservative canonical normalization backed by source evidence.
3. Validation invariants: Deterministic boundaries enforcing anti-hallucination, document
   title exclusion, structural header-region location scope, table header exclusion,
   referee separation, and source-grounded boolean current-status.
4. Conversion: Projection from SemanticOutput to public Resume domain model.
"""

from __future__ import annotations

from collections import Counter
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar
from pydantic import BaseModel, ConfigDict, Field

from app.domain.document import Document
from app.domain.resume import (
    CertificationItem,
    EducationItem,
    ExperienceItem,
    PersonalInfo,
    ProjectItem,
    Resume,
)
from app.domain.structural import StructuralBlock
from app.pipeline.stages.structural_roles import _GENERIC_DEGREE_RE, build_structural_blocks

# Known document headers/labels that must never be accepted as personal names
INVALID_NAME_PATTERNS = [
    re.compile(r"^application\s+form\b", re.IGNORECASE),
    re.compile(r"^curriculum\s+vitae\b", re.IGNORECASE),
    re.compile(r"^seafarer\s+profile\b", re.IGNORECASE),
    re.compile(r"^surname\b", re.IGNORECASE),
    re.compile(r"^resume\b", re.IGNORECASE),
    re.compile(r"^biodata\b", re.IGNORECASE),
]

# Section headers that must never leak into personal location
INVALID_LOCATION_PATTERNS = [
    re.compile(r"^qualification\b", re.IGNORECASE),
    re.compile(r"^references?\b", re.IGNORECASE),
    re.compile(r"^position\b", re.IGNORECASE),
    re.compile(r"^discipline\b", re.IGNORECASE),
    re.compile(r"^declaration\b", re.IGNORECASE),
]

# Table column headers that must not form company or designation values
TABLE_HEADER_PATTERNS = [
    re.compile(r"^\s*ship\s+name\s*$", re.IGNORECASE),
    re.compile(r"^\s*s\.?\s*no\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*vessel\s+(?:name|type)\s*$", re.IGNORECASE),
    re.compile(r"^\s*documents?\s+details?\s*$", re.IGNORECASE),
    re.compile(r"^\s*period\s*$", re.IGNORECASE),
    re.compile(r"^\s*poi\s*$", re.IGNORECASE),
    re.compile(r"^\s*doi\s*$", re.IGNORECASE),
    re.compile(r"^\s*sea\s+service\s*$", re.IGNORECASE),
    re.compile(r"^\s*endorsements?\s*$", re.IGNORECASE),
    re.compile(r"^\s*vaccinations?\s*$", re.IGNORECASE),
]

# Accepted tokens for derived current-status boolean normalization
ACCEPTED_CURRENT_MARKERS = [
    re.compile(r"\bpresent\b", re.IGNORECASE),
    re.compile(r"\bcurrent(?:ly)?(?:\s+employed)?\b", re.IGNORECASE),
    re.compile(r"\btill\s+date\b", re.IGNORECASE),
    re.compile(r"\bnow\b", re.IGNORECASE),
    re.compile(r"\bongoing\b", re.IGNORECASE),
]

_ISO_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")
_PHONE_CANONICAL_RE = re.compile(r"^\+?\d{7,15}$")

_MONTH_NAMES = {
    "01": "jan",
    "02": "feb",
    "03": "mar",
    "04": "apr",
    "05": "may",
    "06": "jun",
    "07": "jul",
    "08": "aug",
    "09": "sep",
    "10": "oct",
    "11": "nov",
    "12": "dec",
}

_FULL_MONTH_NAMES = {
    "01": "january",
    "02": "february",
    "03": "march",
    "04": "april",
    "05": "may",
    "06": "june",
    "07": "july",
    "08": "august",
    "09": "september",
    "10": "october",
    "11": "november",
    "12": "december",
}


# =====================================================================
# 1. Enums and Categories
# =====================================================================


class DocumentArchetype(str, Enum):
    """Constrained document archetype classification."""

    STANDARD_CV = "standard_cv"
    MARITIME_CV = "maritime_cv"
    MARITIME_TABULAR = "maritime_tabular"
    STRUCTURED_FORM = "structured_form"
    ACADEMIC_CV = "academic_cv"
    UNKNOWN = "unknown"


class SemanticBlockCategory(str, Enum):
    """Explicit semantic category assigned to individual blocks for classification & exclusion."""

    SECTION_HEADING = "SECTION_HEADING"
    PERSONAL = "PERSONAL"
    EXPERIENCE = "EXPERIENCE"
    EDUCATION = "EDUCATION"
    PROJECT = "PROJECT"
    SKILL = "SKILL"
    CERTIFICATION = "CERTIFICATION"
    REFERENCE = "REFERENCE"
    TABLE_HEADER = "TABLE_HEADER"
    TABLE_DATA = "TABLE_DATA"
    BOILERPLATE = "BOILERPLATE"
    UNKNOWN = "UNKNOWN"


EXCLUDED_LOCATION_ROLES: frozenset[str] = frozenset({
    "ORGANIZATION",
    "ENTRY_TITLE",
    "BULLET",
    "TECHNOLOGY",
    "CREDENTIAL",
    "FOOTER",
    "SECTION_HEADING",
})

EXCLUDED_LOCATION_CATEGORIES: frozenset[SemanticBlockCategory] = frozenset({
    SemanticBlockCategory.EXPERIENCE,
    SemanticBlockCategory.EDUCATION,
    SemanticBlockCategory.PROJECT,
    SemanticBlockCategory.REFERENCE,
    SemanticBlockCategory.BOILERPLATE,
    SemanticBlockCategory.TABLE_HEADER,
})


# =====================================================================
# 2. SemanticInput Contract (Document IR -> LLM Prompt Payload)
# =====================================================================


class SemanticBlockInput(BaseModel):
    """Compact representation of a single physical/layout block with table & role metadata."""

    block_id: str
    text: str
    page: int
    bbox: list[float]  # [x0, y0, x1, y1]
    region_id: str
    region_kind: str  # "header", "column", "physical_region"
    reading_order: int
    column_id: int | None = None
    is_bold: bool | None = None
    is_italic: bool | None = None
    font_size: float | None = None
    suggested_role: str | None = None  # from existing StructuralRole
    heading_candidate: bool | None = None
    parent_heading_id: str | None = None

    # Generic table representation (populated when block is in a table structure)
    table_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    cell_role: str | None = None  # e.g. "HEADER", "DATA"
    table_purpose: str | None = None
    column_semantic: str | None = None

    # Traceable parent/original source block relationship for split table fragments
    parent_block_id: str | None = None

    # Underlying span geometry if available from reconstruction/IR
    spans: list[dict[str, Any]] = Field(default_factory=list)


class SemanticPageMeta(BaseModel):
    """Page-level spatial metadata only (canonical blocks reside in top-level blocks list)."""

    page_number: int
    width: float | None = None
    height: float | None = None


class SemanticInput(BaseModel):
    """Non-redundant, canonical JSON-serializable input payload for LLM extraction."""

    document_id: str
    page_count: int
    archetype: DocumentArchetype = DocumentArchetype.STANDARD_CV
    pages: list[SemanticPageMeta] = Field(default_factory=list)
    blocks: list[SemanticBlockInput] = Field(default_factory=list)
    tables: list[Any] = Field(default_factory=list)


# =====================================================================
# 3. SemanticOutput Contract (Constrained LLM Response Schema)
# =====================================================================


def _enforce_provenance_json_schema(schema: dict[str, Any]) -> None:
    """Ensure generated JSON Schema structurally marks source_block_ids as required."""
    req = schema.setdefault("required", [])
    if "source_block_ids" not in req:
        req.append("source_block_ids")


class GroundedString(BaseModel):
    """A field value retaining mandatory source provenance and supporting canonical normalization.

    - value: The canonical/normalized representation (e.g. '2018-07', '+919829519017', 'John Doe').
    - raw_value: Optional verbatim raw text from source blocks when it differs from canonical value.
    - source_block_ids: Mandatory list of referenced block IDs providing evidence.
    """

    model_config = ConfigDict(json_schema_extra=_enforce_provenance_json_schema)

    value: str
    raw_value: str | None = None
    source_block_ids: list[str] = Field(default_factory=list)


class GroundedBool(BaseModel):
    """A boolean field with mandatory source provenance evidence (e.g. current employment)."""

    model_config = ConfigDict(json_schema_extra=_enforce_provenance_json_schema)

    value: bool
    source_block_ids: list[str] = Field(default_factory=list)


class BlockClassification(BaseModel):
    """Explicit classification of a source block into semantic categories or exclusions."""

    block_id: str
    category: SemanticBlockCategory


class GroundedPersonal(BaseModel):
    """Candidate personal identity and contact info strictly grounded in header evidence."""

    name: GroundedString | None = None
    email: GroundedString | None = None
    phone: GroundedString | None = None
    location: GroundedString | None = None
    linkedin: GroundedString | None = None
    github: GroundedString | None = None
    portfolio: GroundedString | None = None


class GroundedExperienceItem(BaseModel):
    """Work experience entry strictly grounded in source blocks with reference/table exclusions."""

    company: GroundedString | None = None
    designation: GroundedString | None = None
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    current: GroundedBool | None = None
    location: GroundedString | None = None
    description: GroundedString | None = None
    technologies: list[GroundedString] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)


class GroundedEducationItem(BaseModel):
    """Education entry grounded in source blocks."""

    institution: GroundedString | None = None
    degree: GroundedString | None = None
    fieldOfStudy: GroundedString | None = None
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    grade: GroundedString | None = None
    source_block_ids: list[str] = Field(default_factory=list)


class GroundedProjectItem(BaseModel):
    """Project entry grounded in source blocks."""

    name: GroundedString | None = None
    description: GroundedString | None = None
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    current: GroundedBool | None = None
    technologies: list[GroundedString] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)


class SemanticOutput(BaseModel):
    """Constrained, provenance-grounded semantic output produced by LLM."""

    document_archetype: DocumentArchetype = DocumentArchetype.UNKNOWN
    block_classifications: list[BlockClassification] = Field(default_factory=list)
    personal: GroundedPersonal = Field(default_factory=GroundedPersonal)
    summary: GroundedString | None = None
    skills: list[GroundedString] = Field(default_factory=list)
    experience: list[GroundedExperienceItem] = Field(default_factory=list)
    education: list[GroundedEducationItem] = Field(default_factory=list)
    projects: list[GroundedProjectItem] = Field(default_factory=list)
    certifications: list[GroundedString] = Field(default_factory=list)
    languages: list[GroundedString] = Field(default_factory=list)
    achievements: list[GroundedString] = Field(default_factory=list)


class PersonalSemanticOutput(BaseModel):
    """Constrained, provenance-grounded semantic output for candidate identity and contact coordinates (Pass 1)."""

    document_archetype: DocumentArchetype = DocumentArchetype.UNKNOWN
    block_classifications: list[BlockClassification] = Field(default_factory=list)
    personal: GroundedPersonal = Field(default_factory=GroundedPersonal)


class BodySemanticOutput(BaseModel):
    """Constrained, provenance-grounded semantic output for core resume body entities and qualification records (Pass 2)."""

    document_archetype: DocumentArchetype = DocumentArchetype.UNKNOWN
    block_classifications: list[BlockClassification] = Field(default_factory=list)
    summary: GroundedString | None = None
    skills: list[GroundedString] = Field(default_factory=list)
    experience: list[GroundedExperienceItem] = Field(default_factory=list)
    education: list[GroundedEducationItem] = Field(default_factory=list)
    projects: list[GroundedProjectItem] = Field(default_factory=list)
    certifications: list[GroundedString] = Field(default_factory=list)
    languages: list[GroundedString] = Field(default_factory=list)
    achievements: list[GroundedString] = Field(default_factory=list)


def merge_semantic_passes(
    personal_output: PersonalSemanticOutput,
    body_output: BodySemanticOutput,
) -> SemanticOutput:
    """Deterministically merge Pass 1 (Personal) and Pass 2 (Body) outputs into canonical SemanticOutput."""
    if body_output.document_archetype != DocumentArchetype.UNKNOWN:
        archetype = body_output.document_archetype
    elif personal_output.document_archetype != DocumentArchetype.UNKNOWN:
        archetype = personal_output.document_archetype
    else:
        archetype = DocumentArchetype.UNKNOWN

    merged_classifications: dict[str, BlockClassification] = {}
    for bc in personal_output.block_classifications:
        merged_classifications[bc.block_id] = bc

    for bc in body_output.block_classifications:
        if bc.block_id in merged_classifications:
            existing = merged_classifications[bc.block_id]
            if existing.category == SemanticBlockCategory.PERSONAL and bc.category != SemanticBlockCategory.PERSONAL:
                continue
        merged_classifications[bc.block_id] = bc

    sorted_classifications = sorted(merged_classifications.values(), key=lambda x: x.block_id)

    return SemanticOutput(
        document_archetype=archetype,
        block_classifications=sorted_classifications,
        personal=personal_output.personal,
        summary=body_output.summary,
        skills=body_output.skills,
        experience=body_output.experience,
        education=body_output.education,
        projects=body_output.projects,
        certifications=body_output.certifications,
        languages=body_output.languages,
        achievements=body_output.achievements,
    )


_BODY_STRUCTURAL_ROLES: frozenset[str] = frozenset({
    "EXPERIENCE",
    "ORGANIZATION",
    "ENTRY",
    "ENTRY_TITLE",
    "EDUCATION",
    "DEGREE",
    "INSTITUTION",
    "SKILL",
    "PROJECT",
    "CERTIFICATION",
    "ACHIEVEMENT",
    "LANGUAGE",
    "CREDENTIAL",
    "TECHNOLOGY",
    "BULLET",
    "DESCRIPTION",
})


def get_body_evidence_category(semantic_input: SemanticInput) -> str | None:
    """Classify the primary reason why semantic_input contains meaningful body evidence, or None if sparse/header-only."""
    # 1. Check for blocks with explicit structural body roles
    explicit_body_roles = sum(
        1
        for b in semantic_input.blocks
        if (b.suggested_role or "").upper() in _BODY_STRUCTURAL_ROLES
    )
    if explicit_body_roles >= 2:
        return "structural_body_roles"

    # 2. Check for structured table data cells (tabular experience/education/certifications)
    table_data_cells = sum(
        1
        for b in semantic_input.blocks
        if b.table_id is not None and (b.cell_role != "HEADER" or (b.row_index is not None and b.row_index > 0))
    )
    if table_data_cells >= 2:
        return "table_data_cells"

    # 3. Check for meaningful non-header body blocks with substantial text
    non_header_body_blocks = [
        b
        for b in semantic_input.blocks
        if b.region_kind != "header"
        and (b.suggested_role or "").upper() not in ("HEADER", "CONTACT", "FOOTER")
        and b.text.strip()
    ]
    total_body_text_len = sum(len(b.text.strip()) for b in non_header_body_blocks)

    if len(non_header_body_blocks) >= 3 and total_body_text_len >= 80:
        return "dense_body_text"

    return None


def summarize_body_evidence(semantic_input: SemanticInput) -> dict[str, Any]:
    """Deterministically summarize structural body evidence present in SemanticInput.

    Derives evidence strictly from existing fields (suggested_role, region_kind, table_id,
    cell_role, block_id, text, page, reading_order).
    Does NOT invent classifications or use ungrounded text heuristics.
    """
    sorted_blocks = sorted(
        semantic_input.blocks,
        key=lambda b: (b.page, b.reading_order, b.block_id),
    )

    roles: dict[str, list[str]] = {}
    section_headings: list[str] = []
    table_blocks: list[str] = []
    evidence_block_ids: list[str] = []

    for b in sorted_blocks:
        if not b.text.strip():
            continue

        role_upper = (b.suggested_role or "").strip().upper()

        # Exclude pure footer and contact blocks
        if b.region_kind == "footer" or role_upper in ("FOOTER", "CONTACT"):
            continue

        # Exclude page 1 personal/header blocks that have no body role or table
        if b.page == 1 and b.region_kind == "header" and b.table_id is None and role_upper in ("HEADER", "UNKNOWN", ""):
            continue

        # Exclude page 2+ running header blocks if they are marked as HEADER with no body role or table
        if b.page > 1 and role_upper == "HEADER" and b.table_id is None:
            continue

        evidence_block_ids.append(b.block_id)

        if b.table_id is not None:
            table_blocks.append(b.block_id)

        if role_upper == "SECTION_HEADING":
            section_headings.append(b.block_id)

        # Record in roles dictionary if suggested_role is present and not UNKNOWN
        if role_upper and role_upper != "UNKNOWN":
            if role_upper not in roles:
                roles[role_upper] = []
            roles[role_upper].append(b.block_id)
        elif b.table_id is not None:
            tbl_role = "TABLE_CELL"
            if tbl_role not in roles:
                roles[tbl_role] = []
            roles[tbl_role].append(b.block_id)

    # Deterministically sort dictionary keys
    sorted_roles = {k: roles[k] for k in sorted(roles.keys())}

    return {
        "total_body_blocks": len(evidence_block_ids),
        "roles": sorted_roles,
        "section_headings": section_headings,
        "table_blocks": table_blocks,
        "evidence_block_ids": evidence_block_ids,
    }


def is_body_output_suspiciously_empty(
    body_output: BodySemanticOutput | SemanticOutput,
    semantic_input: SemanticInput,
) -> bool:
    """Detect if BodySemanticOutput or SemanticOutput returned empty collections despite meaningful body evidence in the input.

    Returns False if any primary body collection (skills, experience, education, projects,
    certifications, languages, achievements) is populated.

    When all body collections are empty, checks whether the input document contains explicit
    structural body evidence (e.g. structural roles, table data cells, or substantial non-header content).
    """
    has_entities = bool(
        body_output.skills
        or body_output.experience
        or body_output.education
        or body_output.projects
        or body_output.certifications
        or body_output.languages
        or body_output.achievements
    )
    if has_entities:
        return False

    return get_body_evidence_category(semantic_input) is not None


# =====================================================================
# 4a. Section-Aware Extraction Helpers (Phase 10M)
# =====================================================================

_DEGREE_PATTERN = re.compile(
    r"\b(?:Ph\.?D|M\.?S|B\.?S|M\.?A|B\.?A|Doctor|Master|Bachelor|Degree|Diploma)\b",
    re.IGNORECASE,
)
_INSTITUTION_PATTERN = re.compile(
    r"\b(?:University|College|School|Institute|Academy)\b",
    re.IGNORECASE,
)
_GRANT_OR_AWARD_PATTERN = re.compile(
    r"[\$€£]|(?:\b(?:grant|award|fellowship|honor|funding|scholarship)\b)",
    re.IGNORECASE,
)
_SKILL_SECTION_HEADING_PATTERN = re.compile(
    r"\b(?:skills?|technolog(?:y|ies)|competenc(?:y|ies)|proficienc(?:y|ies)|"
    r"tools?|tooling|tech\s+stack|technical\s+expertise|technical\s+strengths|"
    r"technical\s+environment|languages\s+(&|and)\s+frameworks)\b",
    re.IGNORECASE,
)
_NON_SKILLS_HEADING_PATTERN = re.compile(
    r"\b(?:clients?|customers?|engagements?|portfolio|experience|employment|history|"
    r"work|projects?|education|academics?|certifications?|credentials?|licenses?|"
    r"achievements?|awards?|publications?|teaching|service|references?|declarations?)\b",
    re.IGNORECASE,
)



class SemanticSection:
    """A logical section derived generically from structural evidence in SemanticInput.

    Attributes:
        heading_block_id: The block_id of the boundary block that opened this section
                          (may be None for blocks before the first explicit boundary).
        heading_text:     Verbatim text of the heading block (empty string if no heading).
        block_ids:        Ordered list of block_ids belonging to this section (includes heading).
        canonical_target: Generic extraction hint: one of 'experience', 'education',
                          'certifications', 'skills', 'projects', 'achievements',
                          'languages', or 'unsupported'.
        page_start:       Page number of the first block in this section.
    """

    __slots__ = ("heading_block_id", "heading_text", "block_ids", "canonical_target", "page_start")

    def __init__(
        self,
        heading_block_id: str | None,
        heading_text: str,
        block_ids: list[str],
        canonical_target: str,
        page_start: int,
    ) -> None:
        self.heading_block_id = heading_block_id
        self.heading_text = heading_text
        self.block_ids = list(block_ids)
        self.canonical_target = canonical_target
        self.page_start = page_start

    def __repr__(self) -> str:
        return (
            f"SemanticSection(heading={self.heading_text!r:.40}, "
            f"target={self.canonical_target!r}, "
            f"blocks={len(self.block_ids)}, page={self.page_start})"
        )


def _is_section_boundary_block(block: SemanticBlockInput) -> bool:
    """Generic structural predicate determining if a block introduces a section boundary.

    Uses typography and structural roles only — no fixture-specific strings:
    1. suggested_role == 'SECTION_HEADING'
    2. Uppercase phrase (2-60 chars) with parenthetical qualifiers stripped
    3. Bold short line (<= 6 words) with structural roles ENTRY_TITLE, ORGANIZATION, or SECTION_HEADING
    """
    if block.region_kind in ("header", "footer"):
        return False

    text = (block.text or "").strip()
    if not text or len(text) < 3 or len(text) > 70:
        return False

    # Narrative clauses starting with subordinate conjunctions/prepositions + participle or article
    # (e.g. "After gathering the experience and knowledge", "While working as", "Having completed")
    # are descriptive sentence openings, never section headings.
    if re.match(r"^(?:after|before|during|while|having|with|as|upon|since)\b\s+(?:the\b|a\b|an\b|\w+ing\b)", text, re.IGNORECASE):
        return False

    if block.suggested_role == "SECTION_HEADING" or getattr(block, "heading_candidate", None) is True:
        return True
    if block.suggested_role in ("SKILL", "BULLET"):
        return False

    # Check for uppercase heading text (ignoring parenthetical clauses e.g. '(Total: $5.8M)')
    cleaned = re.sub(r"\(.*?\)", "", text).strip()
    alphas = [c for c in cleaned if c.isalpha()]
    if len(alphas) >= 4 and all(c.isupper() for c in alphas):
        return True

    # Title Case heading check
    words = cleaned.split()
    if 1 <= len(words) <= 6 and not text.endswith((".", ";", ":", ",")):
        if all(w[:1].isupper() for w in words if w and w[0].isalpha()):
            if block.is_bold or (block.font_size is not None and block.font_size >= 12.0):
                return True

    return False


def _is_qualifying_education_entry_title(block: SemanticBlockInput) -> bool:
    """Predicate determining if a block serves as an education entry title."""
    if block.suggested_role in ("ENTRY_TITLE", "DEGREE", "EDUCATION"):
        text = block.text or ""
        return bool(_GENERIC_DEGREE_RE.search(text) or _DEGREE_PATTERN.search(text))
    return False


def _infer_section_target(
    heading_text: str,
    child_blocks: list[SemanticBlockInput],
    archetype: DocumentArchetype | None = None,
) -> str:
    """Infer the canonical destination for a section using structural signals and generic aliases.

    Uses no fixture-specific strings. Maps to one of the 8 canonical Resume collections:
    'summary', 'experience', 'education', 'skills', 'projects', 'certifications',
    'achievements', 'languages', or 'unsupported' if no recognized canonical destination exists.
    """
    from app.pipeline.stages.sections import _load_section_aliases

    try:
        aliases = _load_section_aliases()
    except Exception:
        aliases = {}

    clean_h = re.sub(r"\(.*?\)", "", heading_text).strip().lower()
    for sec_name, sec_aliases in aliases.items():
        if any(a in clean_h for a in sec_aliases):
            return sec_name.lower()

    child_roles = [b.suggested_role for b in child_blocks if b.suggested_role]
    child_texts = " ".join(b.text for b in child_blocks)

    # 1. Education: degree + institution indicators
    has_degree = any(r in ("DEGREE", "EDUCATION") for r in child_roles) or bool(_DEGREE_PATTERN.search(child_texts))
    has_inst = any(r == "INSTITUTION" for r in child_roles) or bool(_INSTITUTION_PATTERN.search(child_texts))
    if (has_degree and has_inst) or any(r == "DEGREE" for r in child_roles):
        return "education"

    # 2. Achievements / Awards: grant funding or achievement markers
    if any(r == "ACHIEVEMENT" for r in child_roles) or _GRANT_OR_AWARD_PATTERN.search(heading_text) or _GRANT_OR_AWARD_PATTERN.search(child_texts):
        return "achievements"

    # 3. Experience: organization/title accompanied by dates or locations
    has_org_title = any(r in ("ENTRY_TITLE", "ORGANIZATION") for r in child_roles)
    has_date_loc = any(r in ("DATE", "LOCATION") for r in child_roles)
    if has_org_title and has_date_loc:
        return "experience"

    # 4. Skills / Technologies
    if not _NON_SKILLS_HEADING_PATTERN.search(clean_h):
        if any(r == "SKILL" for r in child_roles):
            return "skills"
        if any(r == "TECHNOLOGY" for r in child_roles) and (
            bool(_SKILL_SECTION_HEADING_PATTERN.search(clean_h)) or not clean_h
        ):
            return "skills"

    # 5. Certifications / Credentials
    if any(r in ("CERTIFICATION", "CREDENTIAL") for r in child_roles):
        return "certifications"

    # 6. Languages
    if any(r == "LANGUAGE" for r in child_roles):
        return "languages"

    # 7. Summary / Profile: unheaded text block composed of description/unknown prose
    if not clean_h and child_roles and all(r in ("DESCRIPTION", "UNKNOWN") for r in child_roles):
        return "summary"

    if re.search(r"\b(?:publications?|papers?|bibliography|editorial|peer\s*reviewed|references?|referees?|declarations?)\b", clean_h):
        return "unsupported"

    if archetype == DocumentArchetype.ACADEMIC_CV:
        return "unsupported"

    return "unknown"



def partition_semantic_input_into_sections(
    semantic_input: SemanticInput,
) -> list[SemanticSection]:
    """Partition body blocks in SemanticInput into logical sections using generic structural signals.

    Sections are derived purely from existing fields:
      - suggested_role (SECTION_HEADING, ENTRY_TITLE, ORGANIZATION as boundary signals)
      - region_kind (header/footer blocks are excluded)
      - page, reading_order (ordering)
      - block_id, text (provenance and unsupported-section detection)

    No fixture-specific section names are used. The function is deterministic and makes no LLM calls.

    Returns:
        A list of SemanticSection objects in reading order. Header and footer blocks are excluded.
        Blocks that appear before the first boundary signal are grouped into a preamble section.
        Unsupported sections (publications, teaching/mentorship, editorial service, references,
        declarations) are tagged with canonical_target='unsupported'.
    """
    # Sort body blocks in reading order (page, reading_order, block_id)
    body_blocks = sorted(
        [
            b for b in semantic_input.blocks
            if b.region_kind not in ("header", "footer")
            and b.text.strip()
        ],
        key=lambda b: (b.page, b.reading_order, b.block_id),
    )

    if not body_blocks:
        return []

    # Build an index from block_id -> block for fast lookup
    block_by_id: dict[str, SemanticBlockInput] = {b.block_id: b for b in body_blocks}

    sections: list[SemanticSection] = []
    current_heading_id: str | None = None
    current_heading_text: str = ""
    current_heading_role: str = ""
    current_block_ids: list[str] = []
    current_page_start: int = body_blocks[0].page if body_blocks else 1

    def _flush_section(next_page: int | None = None) -> None:
        nonlocal current_heading_id, current_heading_text, current_heading_role
        nonlocal current_block_ids, current_page_start

        if not current_block_ids:
            return

        # Collect child blocks (all blocks except heading itself)
        child_blocks = [
            block_by_id[bid] for bid in current_block_ids
            if bid != current_heading_id and bid in block_by_id
        ]
        target = _infer_section_target(current_heading_text, child_blocks, archetype=semantic_input.archetype)

        # Implicit education refinement: an unheaded preamble section targeting education
        # may start with introductory summary prose before the first degree title.
        # Split at the first qualifying education entry title to keep summary and education clean.
        if current_heading_id is None and target == "education":
            first_edu_idx: int | None = None
            for idx, bid in enumerate(current_block_ids):
                b = block_by_id.get(bid)
                if b and _is_qualifying_education_entry_title(b):
                    first_edu_idx = idx
                    break
            if first_edu_idx is not None and first_edu_idx > 0:
                pre_bids = current_block_ids[:first_edu_idx]
                edu_bids = current_block_ids[first_edu_idx:]
                pre_children = [block_by_id[bid] for bid in pre_bids if bid in block_by_id]
                pre_target = _infer_section_target("", pre_children, archetype=semantic_input.archetype)
                if pre_target == "unsupported":
                    pre_roles = [b.suggested_role for b in pre_children if b.suggested_role]
                    if pre_roles and all(r in ("DESCRIPTION", "UNKNOWN") for r in pre_roles):
                        pre_target = "summary"
                sections.append(SemanticSection(
                    heading_block_id=None,
                    heading_text="",
                    block_ids=list(pre_bids),
                    canonical_target=pre_target,
                    page_start=current_page_start,
                ))
                sections.append(SemanticSection(
                    heading_block_id=None,
                    heading_text="",
                    block_ids=list(edu_bids),
                    canonical_target="education",
                    page_start=block_by_id[edu_bids[0]].page if edu_bids[0] in block_by_id else current_page_start,
                ))
                # Reset
                current_heading_id = None
                current_heading_text = ""
                current_heading_role = ""
                current_block_ids = []
                current_page_start = next_page if next_page is not None else (body_blocks[-1].page if body_blocks else 1)
                return

        sections.append(SemanticSection(
            heading_block_id=current_heading_id,
            heading_text=current_heading_text,
            block_ids=list(current_block_ids),
            canonical_target=target,
            page_start=current_page_start,
        ))
        # Reset
        current_heading_id = None
        current_heading_text = ""
        current_heading_role = ""
        current_block_ids = []
        current_page_start = next_page if next_page is not None else (body_blocks[-1].page if body_blocks else 1)

    for block in body_blocks:
        if _is_section_boundary_block(block):
            # Flush whatever we have accumulated so far as one section
            _flush_section(next_page=block.page)
            # Start new section with this block as heading
            current_heading_id = block.block_id
            current_heading_text = block.text
            current_heading_role = block.suggested_role or ""
            current_block_ids = [block.block_id]
            current_page_start = block.page
        else:
            current_block_ids.append(block.block_id)


    # Flush the final section
    _flush_section()

    return sections


def filter_semantic_input_to_blocks(
    semantic_input: SemanticInput,
    block_ids: list[str],
    include_headers: bool = True,
) -> SemanticInput:
    """Return a new SemanticInput containing only the specified body block_ids plus optionally header blocks.

    Header blocks are preserved by default so that provenance validators or prompts can reference them.
    The returned SemanticInput shares the same document_id, archetype, page_count, and pages.
    Block ordering within the result is preserved (same as in semantic_input.blocks).

    Args:
        semantic_input: The full SemanticInput to filter.
        block_ids:      Ordered list of block_ids to retain.
        include_headers: Whether to include header blocks (default: True).

    Returns:
        A new SemanticInput with only the selected blocks.
    """
    keep_ids: frozenset[str] = frozenset(block_ids)
    if include_headers:
        filtered_blocks = [
            b for b in semantic_input.blocks
            if b.region_kind in ("header",) or b.block_id in keep_ids
        ]
    else:
        filtered_blocks = [
            b for b in semantic_input.blocks
            if b.block_id in keep_ids
        ]
    return SemanticInput(
        document_id=semantic_input.document_id,
        page_count=semantic_input.page_count,
        archetype=semantic_input.archetype,
        pages=list(semantic_input.pages),
        blocks=filtered_blocks,
    )


def merge_body_outputs(outputs: list[BodySemanticOutput]) -> BodySemanticOutput:
    """Deterministically merge multiple BodySemanticOutput objects into one.

    Rules:
    - document_archetype: first non-UNKNOWN wins.
    - block_classifications: last-writer-wins per block_id (same as merge_semantic_passes).
    - summary: first non-None wins.
    - All list collections (skills, experience, education, projects, certifications,
      languages, achievements): concatenated in input order.

    This merge is order-preserving and does NOT deduplicate entities.
    The caller is responsible for passing outputs in reading-order to preserve document order.
    """
    if not outputs:
        return BodySemanticOutput()

    # Archetype: first non-UNKNOWN wins
    archetype = DocumentArchetype.UNKNOWN
    for o in outputs:
        if o.document_archetype != DocumentArchetype.UNKNOWN:
            archetype = o.document_archetype
            break

    # Block classifications: last-writer-wins (same semantics as merge_semantic_passes)
    merged_classifications: dict[str, BlockClassification] = {}
    for o in outputs:
        for bc in o.block_classifications:
            merged_classifications[bc.block_id] = bc
    sorted_classifications = sorted(merged_classifications.values(), key=lambda x: x.block_id)

    # Summary: first non-None wins
    summary: GroundedString | None = None
    for o in outputs:
        if o.summary is not None:
            summary = o.summary
            break

    return BodySemanticOutput(
        document_archetype=archetype,
        block_classifications=sorted_classifications,
        summary=summary,
        skills=sum((o.skills for o in outputs), []),
        experience=sum((o.experience for o in outputs), []),
        education=sum((o.education for o in outputs), []),
        projects=sum((o.projects for o in outputs), []),
        certifications=sum((o.certifications for o in outputs), []),
        languages=sum((o.languages for o in outputs), []),
        achievements=sum((o.achievements for o in outputs), []),
    )


def should_use_section_aware_body_extraction(input_data: SemanticInput) -> bool:
    """Determine whether section-aware body extraction should be used for a document.

    Decision matrix:
    - ACADEMIC_CV: Section-aware if at least 1 supported section exists.
    - All other archetypes (STANDARD_CV, MARITIME_CV, MARITIME_TABULAR, STRUCTURED_FORM, UNKNOWN):
      monolithic body extraction by default in normal path.
    """
    if input_data.archetype == DocumentArchetype.ACADEMIC_CV:
        sections = partition_semantic_input_into_sections(input_data)
        return any(s.canonical_target != "unsupported" for s in sections)
    return False


def group_sections_by_target(
    sections: list[SemanticSection],
) -> list[SemanticSection]:
    """Deterministically group supported sections by their canonical target.

    Preserves target order of first appearance.
    Merges block_ids from multiple sections sharing the same canonical_target into a single section.
    Unsupported sections (canonical_target == 'unsupported') are excluded.
    """
    grouped: dict[str, list[SemanticSection]] = {}
    for sec in sections:
        if sec.canonical_target == "unsupported":
            continue
        grouped.setdefault(sec.canonical_target, []).append(sec)

    result: list[SemanticSection] = []
    for target, sec_list in grouped.items():
        combined_blocks: list[str] = []
        heading_texts: list[str] = []
        for s in sec_list:
            combined_blocks.extend(s.block_ids)
            if s.heading_text:
                heading_texts.append(s.heading_text)

        first_sec = sec_list[0]
        result.append(
            SemanticSection(
                heading_block_id=first_sec.heading_block_id,
                heading_text=" / ".join(heading_texts) if heading_texts else first_sec.heading_text,
                block_ids=combined_blocks,
                canonical_target=target,
                page_start=first_sec.page_start,
            )
        )
    return result


RECOVERY_OVERVIEW_TARGETS: frozenset[str] = frozenset(
    {"summary", "skills", "certifications", "languages", "achievements"}
)
RECOVERY_ENTITY_TARGETS: frozenset[str] = frozenset(
    {"experience", "education", "projects", "unknown"}
)


def group_sections_for_recovery(
    sections: list[SemanticSection],
) -> list[SemanticSection]:
    """Deterministically group supported sections into at most two bounded recovery groups.

    1. 'overview': Declarative / profile sections (summary, skills, certifications, languages, achievements).
    2. 'entities': Narrative / timeline record sections (experience, education, projects).

    Unsupported sections are excluded. Returns 0, 1, or 2 SemanticSection objects.
    """
    overview_blocks: list[str] = []
    overview_headings: list[str] = []
    entity_blocks: list[str] = []
    entity_headings: list[str] = []
    first_overview: SemanticSection | None = None
    first_entity: SemanticSection | None = None

    for sec in sections:
        if sec.canonical_target == "unsupported":
            continue
        if sec.canonical_target in RECOVERY_OVERVIEW_TARGETS:
            if first_overview is None:
                first_overview = sec
            overview_blocks.extend(sec.block_ids)
            if sec.heading_text:
                overview_headings.append(sec.heading_text)
        elif sec.canonical_target in RECOVERY_ENTITY_TARGETS:
            if first_entity is None:
                first_entity = sec
            entity_blocks.extend(sec.block_ids)
            if sec.heading_text:
                entity_headings.append(sec.heading_text)

    result: list[SemanticSection] = []
    if overview_blocks and first_overview is not None:
        result.append(
            SemanticSection(
                heading_block_id=first_overview.heading_block_id,
                heading_text=" / ".join(overview_headings) if overview_headings else "Overview",
                block_ids=overview_blocks,
                canonical_target="overview",
                page_start=first_overview.page_start,
            )
        )
    if entity_blocks and first_entity is not None:
        result.append(
            SemanticSection(
                heading_block_id=first_entity.heading_block_id,
                heading_text=" / ".join(entity_headings) if entity_headings else "Entities",
                block_ids=entity_blocks,
                canonical_target="entities",
                page_start=first_entity.page_start,
            )
        )
    return result


TOutput = TypeVar("TOutput", SemanticOutput, BodySemanticOutput)


def sanitize_grounded_current_status(
    output: TOutput,
    semantic_input: SemanticInput,
) -> TOutput:
    """Deterministically sanitize ungrounded current=True status on experience and project items.

    If an experience or project item asserts current=True, but the cited source blocks do not
    contain any explicit textual marker matching ACCEPTED_CURRENT_MARKERS (e.g. 'present',
    'currently', 'till date', 'now', 'ongoing'), normalize current = None prior to validation
    and projection.

    Invariants preserved:
    - Does not convert True to False (which would assert an ungrounded non-current fact).
    - Preserves explicitly grounded current=True.
    - Preserves explicitly grounded current=False.
    - Does not alter non-current fields (dates, titles, descriptions, etc.).
    - Strict provenance isolation: un-cited blocks containing current markers cannot validate
      the field.
    """
    known_blocks = {b.block_id: b for b in semantic_input.blocks}

    def _is_current_grounded(gb: GroundedBool) -> bool:
        if not gb.source_block_ids:
            return False
        source_text = " ".join(
            known_blocks[bid].text for bid in gb.source_block_ids if bid in known_blocks
        )
        return any(pat.search(source_text) for pat in ACCEPTED_CURRENT_MARKERS)

    for exp in getattr(output, "experience", []):
        if exp.current is not None and exp.current.value is True:
            if not _is_current_grounded(exp.current):
                exp.current = None

    for prj in getattr(output, "projects", []):
        if prj.current is not None and prj.current.value is True:
            if not _is_current_grounded(prj.current):
                prj.current = None

    return output


def has_explicit_skills_evidence(semantic_input: SemanticInput) -> bool:
    """Determine whether SemanticInput contains structural evidence of an explicit skills inventory.

    Returns True only when supported by existing structural/section information:
    - a section inferred to target 'skills' (excluding non-skills/client headings)
    - SKILL blocks within an appropriate skills-oriented section
    - TECHNOLOGY blocks only when they belong to an appropriate skills/technologies section

    Arbitrary TECHNOLOGY blocks (e.g. within client, project, or engagement sections)
    do not count as skills evidence.
    """
    sections = partition_semantic_input_into_sections(semantic_input)
    if not sections:
        return False

    blocks_by_id = {b.block_id: b for b in semantic_input.blocks}

    for sec in sections:
        heading_clean = re.sub(r"\(.*?\)", "", sec.heading_text).strip().lower()
        is_non_skills = bool(_NON_SKILLS_HEADING_PATTERN.search(heading_clean))
        is_skills_heading = bool(_SKILL_SECTION_HEADING_PATTERN.search(heading_clean))

        child_blocks = [
            blocks_by_id[bid]
            for bid in sec.block_ids
            if bid != sec.heading_block_id and bid in blocks_by_id
        ]
        child_roles = {b.suggested_role for b in child_blocks if b.suggested_role}

        # 1. A section inferred to target "skills" (excluding non-skills/client headings)
        if sec.canonical_target == "skills" and not is_non_skills:
            return True

        # 2. SKILL blocks within an appropriate skills-oriented section
        if (
            "SKILL" in child_roles
            or (sec.heading_block_id in blocks_by_id and blocks_by_id[sec.heading_block_id].suggested_role == "SKILL")
        ) and not is_non_skills:
            return True

        # 3. TECHNOLOGY blocks only when they belong to an appropriate skills/technologies section
        if "TECHNOLOGY" in child_roles:
            if is_skills_heading or (sec.canonical_target == "skills" and not is_non_skills):
                return True

    return False


def sanitize_grounded_skills(
    output: TOutput,
    semantic_input: SemanticInput,
) -> TOutput:
    """Deterministically enforce skills-evidence boundaries on semantic output.

    When the source document contains no explicit skills inventory/section,
    force output.skills = [].
    When explicit skills evidence exists, preserve output.skills for normal
    grounding and provenance validation.
    """
    if not has_explicit_skills_evidence(semantic_input):
        if getattr(output, "skills", None):
            output.skills = []
    return output


normalize_semantic_output_skills = sanitize_grounded_skills


def sanitize_grounded_personal_location(
    output: TOutput,
    semantic_input: SemanticInput,
) -> tuple[TOutput, list[dict[str, Any]]]:
    """Deterministically sanitize personal.location to prevent body/employment location leakage.

    Requirements:
    - personal.location is retained ONLY when its cited source block(s) belong to a header/contact context.
    - If personal.location cites a body block that is also mapped to body collections
      (experience, education, projects, etc.), or belongs to an experience/education/projects/skills section,
      or has an excluded structural role (ORGANIZATION, ENTRY_TITLE, etc.), treat it as an employer/institution
      location rather than a personal residence location and set personal.location = None.
    - Preserve valid personal locations already grounded in header/contact blocks.
    - Emits structured diagnostic records for each sanitized location.
    """
    diagnostics: list[dict[str, Any]] = []
    personal = getattr(output, "personal", None)
    if not personal or not getattr(personal, "location", None):
        return output, diagnostics

    loc: GroundedString | None = personal.location
    if not loc or not loc.value or not loc.source_block_ids:
        return output, diagnostics

    blocks_by_id = {b.block_id: b for b in semantic_input.blocks}

    # 1. Collect block IDs mapped to body collections to detect cross-collection leakage
    body_mapped_bids: set[str] = set()
    for exp in getattr(output, "experience", []):
        body_mapped_bids.update(exp.source_block_ids)
        for f in (exp.company, exp.designation, exp.startDate, exp.endDate, exp.location, exp.description):
            if f and f.source_block_ids:
                body_mapped_bids.update(f.source_block_ids)
        for tech in getattr(exp, "technologies", []):
            if tech.source_block_ids:
                body_mapped_bids.update(tech.source_block_ids)

    for edu in getattr(output, "education", []):
        body_mapped_bids.update(edu.source_block_ids)
        for f in (edu.institution, edu.degree, edu.fieldOfStudy, edu.startDate, edu.endDate, edu.grade):
            if f and f.source_block_ids:
                body_mapped_bids.update(f.source_block_ids)

    for prj in getattr(output, "projects", []):
        body_mapped_bids.update(prj.source_block_ids)
        for f in (prj.name, prj.description, prj.startDate, prj.endDate):
            if f and f.source_block_ids:
                body_mapped_bids.update(f.source_block_ids)
        for tech in getattr(prj, "technologies", []):
            if tech.source_block_ids:
                body_mapped_bids.update(tech.source_block_ids)

    # 2. Collect block IDs belonging to explicit body sections
    sections = partition_semantic_input_into_sections(semantic_input)
    body_section_bids: set[str] = set()
    for sec in sections:
        if sec.canonical_target in ("experience", "education", "projects", "certifications", "skills"):
            body_section_bids.update(sec.block_ids)

    # 3. Check for any condition that disqualifies the personal location
    should_sanitize = False
    sanitize_reason = ""

    for bid in loc.source_block_ids:
        b = blocks_by_id.get(bid)
        if not b:
            should_sanitize = True
            sanitize_reason = f"unknown_block_{bid}"
            break
        if b.page != 1:
            should_sanitize = True
            sanitize_reason = f"page_{b.page}_outside_header"
            break
        if b.region_kind == "footer":
            should_sanitize = True
            sanitize_reason = "footer_region"
            break
        if bid in body_mapped_bids:
            should_sanitize = True
            sanitize_reason = f"mapped_to_body_collection_{bid}"
            break
        if bid in body_section_bids:
            should_sanitize = True
            sanitize_reason = f"in_body_section_{bid}"
            break
        if b.suggested_role in EXCLUDED_LOCATION_ROLES:
            should_sanitize = True
            sanitize_reason = f"excluded_role_{b.suggested_role}"
            break
        if not _is_value_semantically_supported(loc.value, b.text):
            should_sanitize = True
            sanitize_reason = f"unsupported_value_in_source_{bid}"
            break

    if should_sanitize:
        diagnostics.append({
            "field": "personal.location",
            "value": loc.value,
            "source_block_ids": list(loc.source_block_ids),
            "reason": sanitize_reason,
        })
        personal.location = None

    return output, diagnostics


normalize_semantic_output_location = sanitize_grounded_personal_location


_EXCLUDED_HEADER_EMAIL_ROLES = frozenset({
    "DESCRIPTION",
    "ENTRY_TITLE",
    "ORGANIZATION",
    "SKILL",
    "BULLET",
    "FOOTER",
    "SECTION_HEADING",
})

_EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def supplement_high_confidence_semantic_fields(
    output: TOutput,
    semantic_input: SemanticInput,
) -> tuple[TOutput, list[dict[str, Any]]]:
    """Deterministically supplement high-confidence semantic fields omitted by LLM extraction.

    Safety invariants:
    1. EMAIL:
       - Only supplements if personal.email is null or empty.
       - Searches ONLY page-1 header/contact blocks (region_kind in ("header", "contact") or
         suggested_role in ("HEADER", "CONTACT")).
       - Excludes footer and non-contact body roles (DESCRIPTION, ENTRY_TITLE, etc.).
       - Extracts only an explicit, syntactically valid email address.
       - Grounds to the exact source block ID.
       - Preserves existing non-null personal.email unchanged.
    2. SKILLS:
       - Only supplements if output.skills is completely empty.
       - Only supplements if has_explicit_skills_evidence(semantic_input) is True.
       - Only collects blocks with suggested_role == "SKILL" in sections whose canonical target is "skills".
       - Excludes BULLET, DESCRIPTION, ENTRY_TITLE, and other non-SKILL roles.
       - Preserves original reading order and grounds each skill to its exact source block ID.
       - Deterministically deduplicates skills by normalized text (case/whitespace insensitive).
       - Preserves existing non-empty output.skills unchanged.
    """
    diagnostics: list[dict[str, Any]] = []

    # 1. EMAIL SUPPLEMENTATION
    if hasattr(output, "personal"):
        personal = getattr(output, "personal", None)
        if personal is None:
            personal = GroundedPersonal()
            output.personal = personal

        existing_email = getattr(personal, "email", None)
        if not existing_email or not getattr(existing_email, "value", None) or not str(existing_email.value).strip():
            # Inspect page-1 candidate blocks in reading order
            page1_candidate_blocks = [
                b for b in semantic_input.blocks
                if b.page == 1
                and b.region_kind != "footer"
                and (b.region_kind in ("header", "contact") or b.suggested_role in ("HEADER", "CONTACT"))
                and b.suggested_role not in _EXCLUDED_HEADER_EMAIL_ROLES
                and b.text.strip()
            ]
            page1_candidate_blocks.sort(key=lambda b: (b.reading_order, b.block_id))

            for block in page1_candidate_blocks:
                match = _EMAIL_REGEX.search(block.text)
                if match:
                    email_val = match.group(0).strip()
                    personal.email = GroundedString(
                        value=email_val,
                        source_block_ids=[block.block_id],
                    )
                    diagnostics.append({
                        "field": "personal.email",
                        "value": email_val,
                        "source_block_ids": [block.block_id],
                        "reason": "supplemented_from_header_contact_block",
                    })
                    break

    # 2. SKILLS SUPPLEMENTATION
    if hasattr(output, "skills"):
        current_skills = getattr(output, "skills", None)
        if not current_skills and has_explicit_skills_evidence(semantic_input):
            blocks_by_id = {b.block_id: b for b in semantic_input.blocks}
            sections = partition_semantic_input_into_sections(semantic_input)

            supplemented_skills: list[GroundedString] = []
            seen_keys: set[str] = set()

            for sec in sections:
                if sec.canonical_target != "skills":
                    continue
                for bid in sec.block_ids:
                    if bid == sec.heading_block_id:
                        continue
                    b = blocks_by_id.get(bid)
                    if not b or b.suggested_role != "SKILL":
                        continue
                    raw_text = b.text.strip()
                    clean_text = re.sub(r"^[\u2022\u2023\u25E6\-\*\u00B7●\uf0b7]\s*", "", raw_text).strip()
                    if not clean_text or clean_text in ("•", "●", "-", "*", "—"):
                        continue
                    norm_key = re.sub(r"\s+", " ", clean_text).strip().lower()
                    if norm_key in seen_keys:
                        continue
                    seen_keys.add(norm_key)
                    supplemented_skills.append(
                        GroundedString(
                            value=clean_text,
                            source_block_ids=[b.block_id],
                        )
                    )

            if supplemented_skills:
                output.skills = supplemented_skills
                diagnostics.append({
                    "field": "skills",
                    "count": len(supplemented_skills),
                    "values": [s.value for s in supplemented_skills],
                    "reason": "supplemented_from_explicit_skill_blocks",
                })

    return output, diagnostics


@dataclass(frozen=True)
class DeterministicEntitySpan:
    """Represents the bounded block scope of a single entity within an experience section."""

    section_index: int
    entity_index: int
    block_ids: list[str]
    title_block_id: str | None = None


@dataclass
class DeterministicAppointmentGroup:
    """A deterministic appointment group within an academic experience section.

    Attributes:
        section_heading_text: Heading text of the enclosing section (or merged heading).
        section_heading_block_id: Heading block ID if present.
        canonical_target: Canonical target of the section (typically 'experience').
        title_block_id: Block ID of the starting ENTRY_TITLE (or leading boundary block).
        block_ids: Ordered list of block IDs comprising this appointment.
        blocks: List of SemanticBlockInput objects in reading order.
    """

    section_heading_text: str = ""
    section_heading_block_id: str | None = None
    canonical_target: str = "experience"
    title_block_id: str | None = None
    block_ids: list[str] = field(default_factory=list)
    blocks: list[SemanticBlockInput] = field(default_factory=list)


def build_deterministic_appointment_groups(
    section: SemanticSection,
    semantic_input: SemanticInput,
) -> list[DeterministicAppointmentGroup]:
    """Deterministically partition an experience section into discrete appointment groups.

    A group begins at suggested_role == 'ENTRY_TITLE' and contains subsequent blocks
    until the next 'ENTRY_TITLE', preserving original SemanticBlockInput objects,
    block IDs, and section metadata. Also supports organization-before-title layouts
    where an ORGANIZATION block immediately precedes an ENTRY_TITLE.
    """
    if section.canonical_target != "experience":
        return []

    blocks_by_id = {b.block_id: b for b in semantic_input.blocks}
    child_blocks = [
        blocks_by_id[bid]
        for bid in section.block_ids
        if bid in blocks_by_id
        and bid != section.heading_block_id
        and blocks_by_id[bid].region_kind not in ("header", "footer")
        and blocks_by_id[bid].text.strip()
    ]
    if not child_blocks:
        return []

    # Filter out table header rows
    valid_blocks = [
        b for b in child_blocks
        if not (b.table_id is not None and (b.cell_role == "HEADER" or b.row_index == 0))
    ]
    if not valid_blocks:
        return []

    # Group table blocks by (table_id, row_index)
    table_rows: dict[tuple[str, int], list[SemanticBlockInput]] = {}
    non_table_blocks: list[SemanticBlockInput] = []
    for b in valid_blocks:
        if b.table_id is not None and b.row_index is not None:
            table_rows.setdefault((b.table_id, b.row_index), []).append(b)
        else:
            non_table_blocks.append(b)

    units: list[tuple[str, tuple[int, int, float], list[SemanticBlockInput]]] = []
    for (t_id, r_idx), r_blocks in table_rows.items():
        r_blocks.sort(key=lambda x: (x.column_index if x.column_index is not None else 0, x.reading_order, x.block_id))
        min_pos = (min(x.page for x in r_blocks), min(x.reading_order for x in r_blocks), min(x.bbox[1] for x in r_blocks))
        units.append(("table_row", min_pos, r_blocks))

    for b in non_table_blocks:
        pos = (b.page, b.reading_order, b.bbox[1])
        units.append(("non_table", pos, [b]))

    units.sort(key=lambda u: u[1])

    groups: list[list[SemanticBlockInput]] = []
    current_group: list[SemanticBlockInput] = []
    current_has_title = False
    current_has_org = False

    for u_type, _, u_blocks in units:
        if u_type == "table_row":
            if current_group:
                groups.append(current_group)
                current_group = []
                current_has_title = False
                current_has_org = False
            groups.append(u_blocks)
        else:
            b = u_blocks[0]
            is_boundary = False
            if b.suggested_role == "ENTRY_TITLE":
                if current_has_title:
                    is_boundary = True
            elif b.suggested_role == "ORGANIZATION":
                if current_has_org and current_has_title:
                    is_boundary = True

            if is_boundary:
                if current_group:
                    groups.append(current_group)
                current_group = [b]
                current_has_title = (b.suggested_role == "ENTRY_TITLE")
                current_has_org = (b.suggested_role == "ORGANIZATION")
            else:
                current_group.append(b)
                if b.suggested_role == "ENTRY_TITLE":
                    current_has_title = True
                if b.suggested_role == "ORGANIZATION":
                    current_has_org = True

    if current_group:
        groups.append(current_group)

    result: list[DeterministicAppointmentGroup] = []
    for grp in groups:
        title_id = next((b.block_id for b in grp if b.suggested_role == "ENTRY_TITLE"), None)
        if title_id is None and grp:
            title_id = grp[0].block_id
        result.append(
            DeterministicAppointmentGroup(
                section_heading_text=section.heading_text,
                section_heading_block_id=section.heading_block_id,
                canonical_target=section.canonical_target,
                title_block_id=title_id,
                block_ids=[b.block_id for b in grp],
                blocks=grp,
            )
        )

    return result


@dataclass
class SectionAwareExtractionUnit:
    """Represents an isolated extraction unit for section-aware body extraction."""

    pass_name: str
    section_input: SemanticInput
    section_heading: str
    canonical_target: str
    title_block_id: str | None = None
    is_appointment: bool = False
    appt_index: int = 0
    total_appts: int = 0
    appointment_block_ids: list[str] | None = None


def enforce_single_experience_entity(
    body_output: BodySemanticOutput,
    title_block_id: str | None,
) -> BodySemanticOutput:
    """Enforce that an appointment-scoped body output contains at most one experience entity.

    If multiple experience entities are returned, selects the entity citing
    the appointment's title_block_id (preferring designation citations), and discards the others.
    """
    if len(body_output.experience) > 1:
        chosen = body_output.experience[0]
        for it in body_output.experience:
            if (
                title_block_id
                and it.designation
                and title_block_id in it.designation.source_block_ids
            ):
                chosen = it
                break
            elif title_block_id and title_block_id in it.source_block_ids:
                chosen = it
                break
        body_output.experience = [chosen]
    return body_output


def constrain_appointment_experience_provenance(
    body_output: BodySemanticOutput,
    allowed_block_ids: list[str],
) -> BodySemanticOutput:
    """Constrain top-level source_block_ids of experience items in an appointment unit
    to the authoritative deterministic appointment block IDs.

    Preserves individual field values and field source_block_ids unmodified.
    Preserves ordering and deduplicates top-level source_block_ids.
    """
    if not allowed_block_ids or not body_output.experience:
        return body_output

    allowed_set = set(allowed_block_ids)
    for exp in body_output.experience:
        if exp.source_block_ids:
            exp.source_block_ids = list(
                dict.fromkeys(bid for bid in exp.source_block_ids if bid in allowed_set)
            )

    return body_output


def isolate_unit_target_collections(
    body_output: BodySemanticOutput,
    canonical_target: str,
) -> BodySemanticOutput:
    """Retain only the collection corresponding to canonical_target, clearing all others.

    Mapping:
    - education -> education
    - experience -> experience
    - skills -> skills
    - achievements -> achievements
    - certifications -> certifications
    - languages -> languages
    - projects -> projects
    - summary -> summary
    - overview -> summary, skills, certifications, languages, achievements
    - entities -> experience, education, projects

    Metadata fields (document_archetype, block_classifications) are preserved.
    Summary is preserved if canonical_target in {"summary", "overview"}.
    If canonical_target is unknown/unrecognized, body_output is returned unchanged.
    """
    if canonical_target == "overview":
        return BodySemanticOutput(
            document_archetype=body_output.document_archetype,
            block_classifications=body_output.block_classifications,
            summary=body_output.summary,
            skills=body_output.skills,
            experience=[],
            education=[],
            projects=[],
            certifications=body_output.certifications,
            languages=body_output.languages,
            achievements=body_output.achievements,
        )
    if canonical_target == "entities":
        return BodySemanticOutput(
            document_archetype=body_output.document_archetype,
            block_classifications=body_output.block_classifications,
            summary=None,
            skills=[],
            experience=body_output.experience,
            education=body_output.education,
            projects=body_output.projects,
            certifications=[],
            languages=[],
            achievements=[],
        )

    recognized_single_targets = {
        "education",
        "experience",
        "skills",
        "achievements",
        "certifications",
        "languages",
        "projects",
        "summary",
    }
    if canonical_target not in recognized_single_targets:
        return body_output

    return BodySemanticOutput(
        document_archetype=body_output.document_archetype,
        block_classifications=body_output.block_classifications,
        summary=body_output.summary if canonical_target == "summary" else None,
        skills=body_output.skills if canonical_target == "skills" else [],
        experience=body_output.experience if canonical_target == "experience" else [],
        education=body_output.education if canonical_target == "education" else [],
        projects=body_output.projects if canonical_target == "projects" else [],
        certifications=body_output.certifications if canonical_target == "certifications" else [],
        languages=body_output.languages if canonical_target == "languages" else [],
        achievements=body_output.achievements if canonical_target == "achievements" else [],
    )



def plan_section_aware_body_passes(
    input_data: SemanticInput,
) -> list[SectionAwareExtractionUnit]:
    """Plan isolated extraction units for section-aware body extraction.

    For ACADEMIC_CV:
    - Multiple physical experience sections (such as RESEARCH EXPERIENCE and TEACHING EXPERIENCE)
      are preserved as separate sections and NEVER collapsed by group_sections_by_target()
      prior to appointment segmentation.
    - Each physical experience section is partitioned into deterministic appointment groups.
    - Each appointment group forms an isolated extraction unit requesting exactly one entity.
    - Non-experience supported sections (e.g. education, achievements) are grouped by canonical
      target to avoid duplicate requests.

    For other archetypes:
    - All supported sections are grouped by canonical target.
    """
    sections = partition_semantic_input_into_sections(input_data)
    supported = [s for s in sections if s.canonical_target != "unsupported"]
    if not supported:
        return []

    units: list[SectionAwareExtractionUnit] = []

    if input_data.archetype == DocumentArchetype.ACADEMIC_CV:
        # Group non-experience sections by canonical target, preserving first appearance
        non_exp_grouped: dict[str, SemanticSection] = {}
        for s in group_sections_by_target([sec for sec in supported if sec.canonical_target != "experience"]):
            non_exp_grouped[s.canonical_target] = s

        handled_targets: set[str] = set()
        for sec in supported:
            if sec.canonical_target == "experience":
                appt_groups = build_deterministic_appointment_groups(sec, input_data)
                if appt_groups:
                    for idx, grp in enumerate(appt_groups):
                        grp_input = filter_semantic_input_to_blocks(input_data, grp.block_ids, include_headers=False)
                        units.append(
                            SectionAwareExtractionUnit(
                                pass_name="body_appt_experience",
                                section_input=grp_input,
                                section_heading=sec.heading_text,
                                canonical_target="experience",
                                title_block_id=grp.title_block_id,
                                is_appointment=True,
                                appt_index=idx + 1,
                                total_appts=len(appt_groups),
                                appointment_block_ids=list(grp.block_ids),
                            )
                        )
                else:
                    sec_input = filter_semantic_input_to_blocks(input_data, sec.block_ids, include_headers=False)
                    units.append(
                        SectionAwareExtractionUnit(
                            pass_name="body_sec_experience",
                            section_input=sec_input,
                            section_heading=sec.heading_text,
                            canonical_target="experience",
                        )
                    )
            elif sec.canonical_target not in handled_targets:
                handled_targets.add(sec.canonical_target)
                target_sec = non_exp_grouped[sec.canonical_target]
                sec_input = filter_semantic_input_to_blocks(input_data, target_sec.block_ids, include_headers=False)
                units.append(
                    SectionAwareExtractionUnit(
                        pass_name=f"body_sec_{target_sec.canonical_target}",
                        section_input=sec_input,
                        section_heading=target_sec.heading_text,
                        canonical_target=target_sec.canonical_target,
                    )
                )
    else:
        grouped_supported = group_sections_by_target(supported)
        for sec in grouped_supported:
            sec_input = filter_semantic_input_to_blocks(input_data, sec.block_ids, include_headers=False)
            units.append(
                SectionAwareExtractionUnit(
                    pass_name=f"body_sec_{sec.canonical_target}",
                    section_input=sec_input,
                    section_heading=sec.heading_text,
                    canonical_target=sec.canonical_target,
                )
            )

    return units


EXPERIENCE_ROLE_COMPATIBILITY: dict[str, frozenset[str]] = {
    "designation": frozenset({"ENTRY_TITLE", "UNKNOWN"}),
    "company": frozenset({"ORGANIZATION", "UNKNOWN"}),
    "location": frozenset({"LOCATION", "UNKNOWN"}),
    "startDate": frozenset({"DATE", "UNKNOWN"}),
    "endDate": frozenset({"DATE", "UNKNOWN"}),
    "description": frozenset({"DESCRIPTION", "BULLET", "UNKNOWN"}),
}


def build_deterministic_experience_spans(
    semantic_input: SemanticInput,
) -> list[DeterministicEntitySpan]:
    """Deterministically partition experience sections in SemanticInput into discrete entity spans.

    An ENTRY_TITLE or ORGANIZATION begins a new entity span within an experience section.
    Trailing date/location/organization/description/bullet blocks are grouped with that entity
    until the next entity boundary or section boundary.
    """
    sections = partition_semantic_input_into_sections(semantic_input)
    blocks_by_id = {b.block_id: b for b in semantic_input.blocks}
    spans: list[DeterministicEntitySpan] = []

    global_entity_idx = 0
    for sec_idx, sec in enumerate(sections):
        if sec.canonical_target != "experience":
            continue

        # For ACADEMIC_CV, use authoritative appointment groups directly
        if semantic_input.archetype == DocumentArchetype.ACADEMIC_CV:
            appt_groups = build_deterministic_appointment_groups(sec, semantic_input)
            if appt_groups:
                for grp in appt_groups:
                    spans.append(
                        DeterministicEntitySpan(
                            section_index=sec_idx,
                            entity_index=global_entity_idx,
                            block_ids=list(grp.block_ids),
                            title_block_id=grp.title_block_id,
                        )
                    )
                    global_entity_idx += 1
                continue

        child_blocks = [
            blocks_by_id[bid]
            for bid in sec.block_ids
            if bid in blocks_by_id and bid != sec.heading_block_id
        ]
        if not child_blocks:
            continue

        # Filter out table header rows
        valid_blocks = [
            b for b in child_blocks
            if not (b.table_id is not None and (b.cell_role == "HEADER" or b.row_index == 0))
        ]
        if not valid_blocks:
            continue

        # Group table blocks by (table_id, row_index)
        table_rows: dict[tuple[str, int], list[SemanticBlockInput]] = {}
        non_table_blocks: list[SemanticBlockInput] = []
        for b in valid_blocks:
            if b.table_id is not None and b.row_index is not None:
                table_rows.setdefault((b.table_id, b.row_index), []).append(b)
            else:
                non_table_blocks.append(b)

        units: list[tuple[str, tuple[int, int, float], list[SemanticBlockInput]]] = []
        for (t_id, r_idx), r_blocks in table_rows.items():
            r_blocks.sort(key=lambda x: (x.column_index if x.column_index is not None else 0, x.reading_order, x.block_id))
            min_pos = (min(x.page for x in r_blocks), min(x.reading_order for x in r_blocks), min(x.bbox[1] for x in r_blocks))
            units.append(("table_row", min_pos, r_blocks))

        for b in non_table_blocks:
            pos = (b.page, b.reading_order, b.bbox[1])
            units.append(("non_table", pos, [b]))

        units.sort(key=lambda u: u[1])

        current_span_blocks: list[str] = []
        current_title_id: str | None = None
        current_org_id: str | None = None
        current_date_id: str | None = None
        last_role: str | None = None
        has_content = False

        for u_type, _, u_blocks in units:
            if u_type == "table_row":
                if current_span_blocks:
                    spans.append(
                        DeterministicEntitySpan(
                            section_index=sec_idx,
                            entity_index=global_entity_idx,
                            block_ids=list(current_span_blocks),
                            title_block_id=current_title_id,
                        )
                    )
                    global_entity_idx += 1
                    current_span_blocks = []
                    current_title_id = None
                    current_org_id = None
                    current_date_id = None
                    last_role = None
                    has_content = False

                row_title_id = next((b.block_id for b in u_blocks if b.suggested_role == "ENTRY_TITLE"), None)
                if row_title_id is None and u_blocks:
                    row_title_id = u_blocks[0].block_id

                spans.append(
                    DeterministicEntitySpan(
                        section_index=sec_idx,
                        entity_index=global_entity_idx,
                        block_ids=[b.block_id for b in u_blocks],
                        title_block_id=row_title_id,
                    )
                )
                global_entity_idx += 1
            else:
                b = u_blocks[0]
                is_new_boundary = False
                if b.suggested_role == "ENTRY_TITLE":
                    if current_title_id is not None or has_content:
                        is_new_boundary = True
                elif b.suggested_role == "ORGANIZATION":
                    if current_org_id is not None or has_content:
                        is_new_boundary = True
                elif b.suggested_role == "DATE":
                    if (
                        current_date_id is not None
                        and last_role != "DATE"
                        and (current_title_id is not None or current_org_id is not None or has_content)
                    ):
                        is_new_boundary = True

                if is_new_boundary:
                    if current_span_blocks:
                        spans.append(
                            DeterministicEntitySpan(
                                section_index=sec_idx,
                                entity_index=global_entity_idx,
                                block_ids=list(current_span_blocks),
                                title_block_id=current_title_id,
                            )
                        )
                        global_entity_idx += 1
                    current_span_blocks = [b.block_id]
                    current_title_id = b.block_id if b.suggested_role == "ENTRY_TITLE" else None
                    current_org_id = b.block_id if b.suggested_role == "ORGANIZATION" else None
                    current_date_id = b.block_id if b.suggested_role == "DATE" else None
                    has_content = False
                else:
                    current_span_blocks.append(b.block_id)
                    if current_title_id is None and b.suggested_role == "ENTRY_TITLE":
                        current_title_id = b.block_id
                    if current_org_id is None and b.suggested_role == "ORGANIZATION":
                        current_org_id = b.block_id
                    if current_date_id is None and b.suggested_role == "DATE":
                        current_date_id = b.block_id
                    if b.suggested_role in ("DESCRIPTION", "BULLET"):
                        has_content = True

                last_role = b.suggested_role

        if current_span_blocks:
            spans.append(
                DeterministicEntitySpan(
                    section_index=sec_idx,
                    entity_index=global_entity_idx,
                    block_ids=list(current_span_blocks),
                    title_block_id=current_title_id,
                )
            )
            global_entity_idx += 1

    return spans


def _assign_span_to_experience_item(
    exp: GroundedExperienceItem,
    spans: list[DeterministicEntitySpan],
    exp_idx: int,
    total_exp_count: int,
) -> DeterministicEntitySpan | None:
    """Deterministically map an extracted GroundedExperienceItem to its corresponding DeterministicEntitySpan."""
    if not spans:
        return None

    exp_cited_bids: set[str] = set(exp.source_block_ids)
    for f in (exp.company, exp.designation, exp.startDate, exp.endDate, exp.location, exp.description):
        if f and f.source_block_ids:
            exp_cited_bids.update(f.source_block_ids)
    for tech in exp.technologies:
        if tech.source_block_ids:
            exp_cited_bids.update(tech.source_block_ids)

    scored: list[tuple[int, DeterministicEntitySpan]] = []
    for span in spans:
        overlap = len(exp_cited_bids.intersection(span.block_ids))
        scored.append((overlap, span))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_overlap, best_span = scored[0]

    if best_overlap > 0:
        if len(scored) > 1 and scored[1][0] == best_overlap:
            if exp_idx < len(spans) and spans[exp_idx] in (best_span, scored[1][1]):
                return spans[exp_idx]
            return best_span
        return best_span

    if total_exp_count == len(spans) and exp_idx < len(spans):
        return spans[exp_idx]

    return None


def repair_grounded_provenance(
    output: TOutput,
    semantic_input: SemanticInput,
) -> tuple[TOutput, list[dict[str, Any]]]:
    """Deterministically repair provenance for experience fields where an exact, unambiguous
    compatible candidate block exists in the same entity span.

    Repair ONLY occurs when:
    1. The emitted value is already exactly grounded under _is_value_semantically_supported().
    2. The declared source block is incompatible with the field OR belongs to a different entity.
    3. A nearby candidate source block exists in the same deterministic entity span.
    4. The candidate has a structurally compatible role for that field.
    5. The candidate contains the exact emitted value under deterministic normalization rules.
    6. There is no ambiguity between multiple compatible candidate blocks.
    """
    repairs: list[dict[str, Any]] = []
    spans = build_deterministic_experience_spans(semantic_input)
    blocks_by_id = {b.block_id: b for b in semantic_input.blocks}

    # 1. Experience field provenance repair
    experience_list = getattr(output, "experience", [])
    total_exp = len(experience_list)

    for exp_idx, exp in enumerate(experience_list):
        assigned_span = _assign_span_to_experience_item(exp, spans, exp_idx, total_exp)
        if assigned_span is None:
            continue

        span_block_ids_set = set(assigned_span.block_ids)

        for field_name in ("designation", "company", "location", "startDate", "endDate"):
            field_obj: GroundedString | None = getattr(exp, field_name, None)
            if field_obj is None or not field_obj.value:
                continue

            val = field_obj.value.strip()
            current_bids = field_obj.source_block_ids
            compatible_roles = EXPERIENCE_ROLE_COMPATIBILITY.get(field_name, frozenset())

            # Check if current citation is valid, compatible, and within span
            current_text = " ".join(
                blocks_by_id[bid].text for bid in current_bids if bid in blocks_by_id
            )
            current_supported = (
                bool(current_bids)
                and _is_value_semantically_supported(val, current_text, is_single_block=(len(current_bids) == 1))
            )
            current_roles_compatible = (
                bool(current_bids)
                and all(
                    blocks_by_id.get(bid) is not None
                    and blocks_by_id[bid].suggested_role in compatible_roles
                    for bid in current_bids
                )
            )
            current_in_span = (
                bool(current_bids)
                and all(bid in span_block_ids_set for bid in current_bids)
            )

            # If all conditions hold, current citation is already fully valid
            if current_supported and current_roles_compatible and current_in_span:
                continue

            # Citation defect exists: search for a compatible candidate in assigned_span
            matching_candidates: list[SemanticBlockInput] = []
            for bid in assigned_span.block_ids:
                b = blocks_by_id.get(bid)
                if not b:
                    continue
                if b.table_id is not None:
                    continue
                if b.region_kind in ("header", "footer"):
                    continue
                if b.suggested_role in (
                    "SECTION_HEADING", "HEADER", "FOOTER", "TABLE_HEADER", "BOILERPLATE", "REFERENCE"
                ):
                    continue
                if b.suggested_role not in compatible_roles:
                    continue
                if _is_value_semantically_supported(val, b.text):
                    matching_candidates.append(b)

            # Repair ONLY if strictly unambiguous (exactly 1 candidate)
            if len(matching_candidates) == 1:
                cand = matching_candidates[0]
                orig_bids = list(current_bids)
                field_obj.source_block_ids = [cand.block_id]

                # Synchronize exp.source_block_ids: replace any old bid that was not in span
                new_exp_bids: list[str] = []
                replaced = False
                for eb in exp.source_block_ids:
                    if eb in orig_bids and eb not in span_block_ids_set:
                        if not replaced:
                            new_exp_bids.append(cand.block_id)
                            replaced = True
                    else:
                        new_exp_bids.append(eb)
                if cand.block_id not in new_exp_bids:
                    new_exp_bids.append(cand.block_id)
                exp.source_block_ids = new_exp_bids

                repairs.append({
                    "field": f"experience[{exp_idx}].{field_name}",
                    "value": val,
                    "original_source_block_ids": orig_bids,
                    "repaired_source_block_ids": [cand.block_id],
                    "expected_structural_role": sorted(list(compatible_roles)),
                    "reason": "incompatible_or_cross_entity_provenance",
                })
            elif not current_in_span and field_name == "location":
                # For an optional experience subfield such as location:
                # If the LLM cites blocks outside the assigned experience span, and no unique
                # grounded replacement exists in the assigned span, clear the ungrounded optional
                # field and remove foreign entity blocks from exp.source_block_ids.
                orig_bids = list(current_bids)
                setattr(exp, field_name, None)

                # Prune cross-entity bids from exp.source_block_ids
                other_span_bids = {
                    b_id for s in spans if s.entity_index != assigned_span.entity_index for b_id in s.block_ids
                }
                exp.source_block_ids = [
                    eb for eb in exp.source_block_ids
                    if eb not in orig_bids and eb not in other_span_bids
                ]

                repairs.append({
                    "field": f"experience[{exp_idx}].{field_name}",
                    "original_value": val,
                    "repaired_value": None,
                    "original_source_block_ids": orig_bids,
                    "repaired_source_block_ids": [],
                    "reason": "cross_entity_optional_subfield_cleared",
                })

    # 2. Block classifications normalization guard for experience items mislabeled as TABLE_HEADER
    exp_all_bids: set[str] = set()
    for exp in experience_list:
        exp_all_bids.update(exp.source_block_ids)
        for f in (exp.company, exp.designation, exp.startDate, exp.endDate, exp.location, exp.description):
            if f and f.source_block_ids:
                exp_all_bids.update(f.source_block_ids)
        for tech in exp.technologies:
            if tech.source_block_ids:
                exp_all_bids.update(tech.source_block_ids)

    for bc in getattr(output, "block_classifications", []):
        if bc.category == SemanticBlockCategory.TABLE_HEADER and bc.block_id in exp_all_bids:
            b = blocks_by_id.get(bc.block_id)
            if b and b.table_id is None and b.suggested_role in {
                "ENTRY_TITLE", "ORGANIZATION", "LOCATION", "DATE", "DESCRIPTION", "BULLET"
            }:
                bc.category = SemanticBlockCategory.EXPERIENCE
                repairs.append({
                    "field": f"block_classifications[{bc.block_id}]",
                    "original_category": "TABLE_HEADER",
                    "repaired_category": "EXPERIENCE",
                    "reason": "non_table_block_in_experience",
                })

    return output, repairs


_STRUCTURAL_LABEL_PREFIX_RE = re.compile(
    r"(?:^|(?<=[\s,;]))(?:vessel\s+(?:name|type)|ship\s+(?:name|type)|project\s+(?:name|title|description|scope)|company\s+name|type\s*/\s*gt|role|designation|title|description|scope|responsibilities|duties|vessel|ship|type|flag|imo|gt|grt|dwt|nrt)\s*:\s*",
    re.IGNORECASE,
)
_UNGROUNDED_METADATA_CLAUSE_RE = re.compile(r",?\s*(?:gt|grt|dwt|nrt)\s*:\s*\d+\b", re.IGNORECASE)


def _iter_grounded_string_fields(output: Any):
    """Yield tuples of (field_path, parent_object, field_name_or_index, GroundedString)."""
    # Personal
    if hasattr(output, "personal") and output.personal:
        for fname in ("name", "email", "phone", "location", "linkedin", "github", "portfolio"):
            gs = getattr(output.personal, fname, None)
            if gs is not None and isinstance(gs, GroundedString):
                yield f"personal.{fname}", output.personal, fname, gs

    # Experience
    for i, exp in enumerate(getattr(output, "experience", [])):
        for fname in ("company", "designation", "location", "startDate", "endDate", "description"):
            gs = getattr(exp, fname, None)
            if gs is not None and isinstance(gs, GroundedString):
                yield f"experience[{i}].{fname}", exp, fname, gs
        for j, tech in enumerate(getattr(exp, "technologies", [])):
            if isinstance(tech, GroundedString):
                yield f"experience[{i}].technologies[{j}]", exp.technologies, j, tech

    # Education
    for i, edu in enumerate(getattr(output, "education", [])):
        for fname in ("institution", "degree", "fieldOfStudy", "startDate", "endDate", "grade"):
            gs = getattr(edu, fname, None)
            if gs is not None and isinstance(gs, GroundedString):
                yield f"education[{i}].{fname}", edu, fname, gs

    # Projects
    for i, prj in enumerate(getattr(output, "projects", [])):
        for fname in ("name", "description", "startDate", "endDate"):
            gs = getattr(prj, fname, None)
            if gs is not None and isinstance(gs, GroundedString):
                yield f"projects[{i}].{fname}", prj, fname, gs
        for j, tech in enumerate(getattr(prj, "technologies", [])):
            if isinstance(tech, GroundedString):
                yield f"projects[{i}].technologies[{j}]", prj.technologies, j, tech

    # Collections
    for col_name in ("skills", "certifications", "languages", "achievements"):
        for i, item in enumerate(getattr(output, col_name, [])):
            if isinstance(item, GroundedString):
                yield f"{col_name}[{i}]", getattr(output, col_name), i, item

    # Summary
    if hasattr(output, "summary") and isinstance(output.summary, GroundedString):
        yield "summary", output, "summary", output.summary


def _normalize_structural_labels_in_output(output: Any, blocks_by_id: dict[str, Any], repairs: list[dict[str, Any]]) -> None:
    for context, parent, key, gs in _iter_grounded_string_fields(output):
        if not gs.value or not gs.source_block_ids:
            continue
        val = gs.value.strip()
        if not _STRUCTURAL_LABEL_PREFIX_RE.search(val):
            continue
        source_text = " ".join(blocks_by_id[bid].text for bid in gs.source_block_ids if bid in blocks_by_id)
        is_single = len(gs.source_block_ids) == 1
        if _is_value_semantically_supported(val, source_text, is_single_block=is_single) or (len(gs.source_block_ids) > 1 and _is_multiblock_text_semantically_supported(val, source_text)):
            continue

        val_clean = val
        m_gt = _UNGROUNDED_METADATA_CLAUSE_RE.search(val_clean)
        if m_gt:
            gt_digits = re.findall(r"\d+", m_gt.group(0))
            if gt_digits and not any(d in source_text for d in gt_digits):
                val_clean = _UNGROUNDED_METADATA_CLAUSE_RE.sub("", val_clean)

        val_clean = _STRUCTURAL_LABEL_PREFIX_RE.sub("", val_clean)
        val_clean = re.sub(r"\s*,\s*", ", ", val_clean).strip(" ,;")

        if val_clean and val_clean != val:
            if _is_value_semantically_supported(val_clean, source_text, is_single_block=is_single) or (len(gs.source_block_ids) > 1 and _is_multiblock_text_semantically_supported(val_clean, source_text)):
                repairs.append({
                    "field": context,
                    "original_value": val,
                    "repaired_value": val_clean,
                    "reason": "structural_label_normalization",
                })
                gs.value = val_clean


def repair_semantic_output_provenance(
    output: TOutput,
    semantic_input: SemanticInput,
) -> tuple[TOutput, list[dict[str, Any]]]:
    """Deterministically repair provenance and normalize structural labels across semantic output.

    Applies:
    1. Bullet citation remapping: resolves citations pointing to bullet symbols onto sibling text blocks.
    2. Structural label normalization pass 1: strips schema labels when text is already supported.
    3. Wrapped provenance completion: expands citations for wrapped table cells, table entries, and contiguous text lines.
    4. Non-contributing cited block pruning: drops extraneous cited blocks contributing 0 tokens to value.
    5. Structural label normalization pass 2: cleans any values unblocked by completed provenance.
    6. Deterministic experience span and role repair.
    """
    repairs: list[dict[str, Any]] = []
    blocks_by_id = {b.block_id: b for b in semantic_input.blocks}
    sorted_blocks = sorted(semantic_input.blocks, key=lambda b: (b.page, b.reading_order, b.block_id))

    # 1. Bullet Citation Remapping
    for context, parent, key, gs in _iter_grounded_string_fields(output):
        if not gs.value or not gs.source_block_ids:
            continue
        val = gs.value.strip()
        bullet_bids = [
            bid for bid in gs.source_block_ids
            if bid in blocks_by_id and (
                blocks_by_id[bid].suggested_role == "BULLET"
                or re.match(r"^[•·\-\*]$", blocks_by_id[bid].text.strip())
            )
        ]
        if not bullet_bids:
            continue

        for b_bullet_id in bullet_bids:
            b_bullet = blocks_by_id[b_bullet_id]
            if _is_value_semantically_supported(val, b_bullet.text):
                continue
            candidates = []
            for cand in blocks_by_id.values():
                if cand.page != b_bullet.page or cand.block_id == b_bullet_id:
                    continue
                if cand.suggested_role == "BULLET" or re.match(r"^[•·\-\*]$", cand.text.strip()):
                    continue
                y_dist = abs(cand.bbox[1] - b_bullet.bbox[1])
                ro_dist = abs(cand.reading_order - b_bullet.reading_order)
                if y_dist <= 15.0 and _is_value_semantically_supported(val, cand.text):
                    candidates.append((y_dist, cand))
                elif ro_dist <= 2 and _is_value_semantically_supported(val, cand.text):
                    candidates.append((100.0 + ro_dist, cand))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_cand = candidates[0][1]
                orig = list(gs.source_block_ids)
                new_bids = [best_cand.block_id if bid == b_bullet_id else bid for bid in gs.source_block_ids]
                deduped_bids = list(dict.fromkeys(new_bids))
                gs.source_block_ids = deduped_bids
                repairs.append({
                    "field": context,
                    "original_source_block_ids": orig,
                    "repaired_source_block_ids": deduped_bids,
                    "reason": "bullet_citation_remapped",
                })

    # 2. Structural Label Normalization Pass 1
    _normalize_structural_labels_in_output(output, blocks_by_id, repairs)

    # 3. Wrapped Provenance Completion (Table Cells, Table Entries & Sequential Layout Lines)
    for context, parent, key, gs in _iter_grounded_string_fields(output):
        if not gs.value or not gs.source_block_ids:
            continue
        val = gs.value.strip()
        curr_text = " ".join(blocks_by_id[bid].text for bid in gs.source_block_ids if bid in blocks_by_id)
        
        check_vals = [val]
        if _STRUCTURAL_LABEL_PREFIX_RE.search(val):
            val_no_prefix = _STRUCTURAL_LABEL_PREFIX_RE.sub("", val).strip(" ,;")
            if val_no_prefix and val_no_prefix != val:
                check_vals.append(val_no_prefix)

        def _is_supp(t: str, bids: list[str]) -> bool:
            for cv in check_vals:
                if _is_value_semantically_supported(cv, t, is_single_block=(len(bids) == 1)) or (len(bids) > 1 and _is_multiblock_text_semantically_supported(cv, t)):
                    return True
            return False

        if _is_supp(curr_text, gs.source_block_ids):
            continue

        valid_blocks = [blocks_by_id[bid] for bid in gs.source_block_ids if bid in blocks_by_id]
        if not valid_blocks:
            continue

        # Case A: Table cell contiguous slice completion / same cell remap
        if all(b.table_id is not None for b in valid_blocks):
            t_id = valid_blocks[0].table_id
            r_idx = valid_blocks[0].row_index
            c_idx = valid_blocks[0].column_index
            if all(b.table_id == t_id and b.row_index == r_idx and b.column_index == c_idx for b in valid_blocks):
                cell_blocks = sorted(
                    [b for b in sorted_blocks if b.table_id == t_id and b.row_index == r_idx and b.column_index == c_idx],
                    key=lambda x: (x.page, x.reading_order, x.bbox[1]),
                )
                matching_slices = []
                # First try slices that intersect cited blocks
                for start in range(len(cell_blocks)):
                    for end in range(start + 1, len(cell_blocks) + 1):
                        slice_blocks = cell_blocks[start:end]
                        if not any(b.block_id in gs.source_block_ids for b in slice_blocks):
                            continue
                        slice_text = " ".join(b.text for b in slice_blocks)
                        if _is_supp(slice_text, [b.block_id for b in slice_blocks]):
                            matching_slices.append((end - start, [b.block_id for b in slice_blocks]))
                if matching_slices:
                    matching_slices.sort(key=lambda x: x[0])
                    best_bids = matching_slices[0][1]
                    if best_bids != gs.source_block_ids:
                        repairs.append({
                            "field": context,
                            "original_source_block_ids": list(gs.source_block_ids),
                            "repaired_source_block_ids": list(best_bids),
                            "reason": "wrapped_table_cell_slice_completion",
                        })
                        gs.source_block_ids = list(best_bids)
                        continue
                else:
                    # Fallback: any slice in the same cell
                    for start in range(len(cell_blocks)):
                        for end in range(start + 1, len(cell_blocks) + 1):
                            slice_blocks = cell_blocks[start:end]
                            slice_text = " ".join(b.text for b in slice_blocks)
                            if _is_supp(slice_text, [b.block_id for b in slice_blocks]):
                                matching_slices.append((end - start, [b.block_id for b in slice_blocks]))
                    if matching_slices:
                        matching_slices.sort(key=lambda x: x[0])
                        best_bids = matching_slices[0][1]
                        if best_bids != gs.source_block_ids:
                            repairs.append({
                                "field": context,
                                "original_source_block_ids": list(gs.source_block_ids),
                                "repaired_source_block_ids": list(best_bids),
                                "reason": "table_cell_slice_remap",
                            })
                            gs.source_block_ids = list(best_bids)
                            continue

            # Case B: Table cell offset citation remap (e.g. single cited block pointing to adjacent row/col in same table)
            if len(gs.source_block_ids) == 1:
                b = valid_blocks[0]
                table_candidates = [
                    cand for cand in sorted_blocks
                    if cand.table_id == b.table_id
                    and cand.block_id != b.block_id
                    and abs((cand.row_index or 0) - (b.row_index or 0)) <= 1
                    and _is_supp(cand.text, [cand.block_id])
                ]
                if len(table_candidates) == 1:
                    orig = list(gs.source_block_ids)
                    gs.source_block_ids = [table_candidates[0].block_id]
                    repairs.append({
                        "field": context,
                        "original_source_block_ids": orig,
                        "repaired_source_block_ids": list(gs.source_block_ids),
                        "reason": "table_cell_offset_remap",
                    })
                    continue

            # Case B2: Table entry multi-row continuation (cited blocks in same table within 1-2 rows)
            if all(b.table_id == t_id for b in valid_blocks):
                min_r = min(b.row_index for b in valid_blocks if b.row_index is not None)
                max_r = max(b.row_index for b in valid_blocks if b.row_index is not None)
                if (max_r - min_r) <= 2:
                    table_entry_cands = [
                        cand for cand in sorted_blocks
                        if cand.table_id == t_id
                        and cand.block_id not in set(gs.source_block_ids)
                        and cand.row_index is not None
                        and min_r <= cand.row_index <= max(max_r, min_r + 1)
                        and cand.suggested_role not in ("SECTION_HEADING", "HEADER", "FOOTER", "TABLE_HEADER")
                    ]
                    test_bids = list(gs.source_block_ids)
                    for cand in table_entry_cands:
                        test_bids.append(cand.block_id)
                        test_text = " ".join(blocks_by_id[bid].text for bid in test_bids if bid in blocks_by_id)
                        if _is_supp(test_text, test_bids):
                            repairs.append({
                                "field": context,
                                "original_source_block_ids": list(gs.source_block_ids),
                                "repaired_source_block_ids": list(test_bids),
                                "reason": "table_entry_continuation",
                            })
                            gs.source_block_ids = list(test_bids)
                            break

        # Case C: Non-table reading order continuation in same region/entity
        elif all(b.table_id is None for b in valid_blocks):
            page = valid_blocks[0].page
            min_ro = min(b.reading_order for b in valid_blocks)
            max_ro = max(b.reading_order for b in valid_blocks)
            candidates = [
                cand for cand in sorted_blocks
                if cand.page == page
                and cand.table_id is None
                and cand.block_id not in set(gs.source_block_ids)
                and cand.suggested_role not in ("SECTION_HEADING", "HEADER", "FOOTER", "TABLE_HEADER")
                and (
                    (1 <= (cand.reading_order - max_ro) <= 3)
                    or (1 <= (min_ro - cand.reading_order) <= 2)
                    or (min_ro <= cand.reading_order <= max_ro)
                )
            ]
            candidates.sort(key=lambda c: (0 if min_ro <= c.reading_order <= max_ro else (abs(c.reading_order - max_ro) if c.reading_order > max_ro else 10 + abs(min_ro - c.reading_order))))
            test_bids = list(gs.source_block_ids)
            for cand in candidates:
                test_bids.append(cand.block_id)
                test_bids_sorted = sorted(test_bids, key=lambda bid: blocks_by_id[bid].reading_order if bid in blocks_by_id else 0)
                test_text = " ".join(blocks_by_id[bid].text for bid in test_bids_sorted if bid in blocks_by_id)
                if _is_supp(test_text, test_bids_sorted):
                    repairs.append({
                        "field": context,
                        "original_source_block_ids": list(gs.source_block_ids),
                        "repaired_source_block_ids": list(test_bids_sorted),
                        "reason": "wrapped_nontable_continuation",
                    })
                    gs.source_block_ids = list(test_bids_sorted)
                    break

    # 4. Non-Contributing Cited Block Pruning
    for context, parent, key, gs in _iter_grounded_string_fields(output):
        if not gs.value or len(gs.source_block_ids) <= 1:
            continue
        val = gs.value.strip()
        curr_text = " ".join(blocks_by_id[bid].text for bid in gs.source_block_ids if bid in blocks_by_id)
        
        check_vals = [val]
        if _STRUCTURAL_LABEL_PREFIX_RE.search(val):
            val_no_prefix = _STRUCTURAL_LABEL_PREFIX_RE.sub("", val).strip(" ,;")
            if val_no_prefix and val_no_prefix != val:
                check_vals.append(val_no_prefix)

        def _is_supp(t: str, bids: list[str]) -> bool:
            for cv in check_vals:
                if _is_value_semantically_supported(cv, t, is_single_block=(len(bids) == 1)) or (len(bids) > 1 and _is_multiblock_text_semantically_supported(cv, t)):
                    return True
            return False

        if _is_supp(curr_text, gs.source_block_ids):
            continue

        val_tokens = set()
        for cv in check_vals:
            val_tokens.update(t.lower() for t in re.findall(r"[A-Za-z0-9]+", cv))
        if not val_tokens:
            continue

        contrib_bids = [
            bid for bid in gs.source_block_ids
            if bid in blocks_by_id
            and not set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", blocks_by_id[bid].text)).isdisjoint(val_tokens)
        ]
        if 0 < len(contrib_bids) < len(gs.source_block_ids):
            contrib_text = " ".join(blocks_by_id[bid].text for bid in contrib_bids)
            if _is_supp(contrib_text, contrib_bids):
                repairs.append({
                    "field": context,
                    "original_source_block_ids": list(gs.source_block_ids),
                    "repaired_source_block_ids": list(contrib_bids),
                    "reason": "prune_non_contributing_blocks",
                })
                gs.source_block_ids = list(contrib_bids)
            else:
                last_contrib = max((blocks_by_id[bid] for bid in contrib_bids), key=lambda x: x.reading_order)
                continuation_candidates = [
                    cand for cand in sorted_blocks
                    if cand.page == last_contrib.page
                    and 1 <= (cand.reading_order - last_contrib.reading_order) <= 2
                    and cand.table_id == last_contrib.table_id
                    and cand.column_index == last_contrib.column_index
                    and cand.suggested_role not in ("SECTION_HEADING", "HEADER", "FOOTER", "TABLE_HEADER")
                ]
                test_bids = list(contrib_bids)
                for cand in continuation_candidates:
                    test_bids.append(cand.block_id)
                    test_text = " ".join(blocks_by_id[bid].text for bid in test_bids if bid in blocks_by_id)
                    if _is_supp(test_text, test_bids):
                        repairs.append({
                            "field": context,
                            "original_source_block_ids": list(gs.source_block_ids),
                            "repaired_source_block_ids": list(test_bids),
                            "reason": "prune_and_continue_blocks",
                        })
                        gs.source_block_ids = list(test_bids)
                        break

    # 5. Structural Label Normalization Pass 2
    _normalize_structural_labels_in_output(output, blocks_by_id, repairs)

    # 6. Deterministic Experience Span & Role Repair
    output, exp_repairs = repair_grounded_provenance(output, semantic_input)
    repairs.extend(exp_repairs)

    return output, repairs


# 4. Builder and Validation Invariants
# =====================================================================

_MARITIME_ARCHETYPE_SIGNALS = [
    re.compile(r"\bcdc\b", re.IGNORECASE),
    re.compile(r"\bstcw\b", re.IGNORECASE),
    re.compile(r"\bvessel\b", re.IGNORECASE),
    re.compile(r"\bsea\s+service\b", re.IGNORECASE),
    re.compile(r"\bseafarer\b", re.IGNORECASE),
    re.compile(r"\bdischarge\s+book\b", re.IGNORECASE),
    re.compile(r"\bgmdss\b", re.IGNORECASE),
    re.compile(r"\bdeck\s+cadet\b", re.IGNORECASE),
    re.compile(r"\bmaster\s+mariner\b", re.IGNORECASE),
    re.compile(r"\b(?:chief|second|2nd|third|3rd)\s+officer\b", re.IGNORECASE),
    re.compile(r"\bchief\s+engineer\b", re.IGNORECASE),
    re.compile(r"\bengine\s+room\b", re.IGNORECASE),
    re.compile(r"\bdwt\b", re.IGNORECASE),
    re.compile(r"\bgrt\b", re.IGNORECASE),
    re.compile(r"\bsign\s+on\b", re.IGNORECASE),
    re.compile(r"\bsign\s+off\b", re.IGNORECASE),
]

_FORM_ARCHETYPE_SIGNALS = [
    re.compile(r"\bapplication\s+form\b", re.IGNORECASE),
    re.compile(r"\bbio-?data\b", re.IGNORECASE),
    re.compile(r"\bseafarer\s+profile\b", re.IGNORECASE),
    re.compile(r"\bdg\s+shipping\b", re.IGNORECASE),
    re.compile(r"\bpersonal\s+data\b", re.IGNORECASE),
    re.compile(r"\bnext\s+of\s+kin\b", re.IGNORECASE),
]

_ACADEMIC_ARCHETYPE_SIGNALS = [
    re.compile(r"\bpeer-?reviewed\b", re.IGNORECASE),
    re.compile(r"\bjournal\s+publications\b", re.IGNORECASE),
    re.compile(r"\bdissertation\b", re.IGNORECASE),
    re.compile(r"\bconference\s+proceedings\b", re.IGNORECASE),
]


def classify_document_archetype(document: Document) -> DocumentArchetype:
    """Deterministically classify document archetype based on layout structure and domain indicators."""
    all_text = " ".join(
        line.text
        for page in document.pages
        for region in page.regions
        for line in region.lines
    )
    m_count = sum(1 for pat in _MARITIME_ARCHETYPE_SIGNALS if pat.search(all_text))
    f_count = sum(1 for pat in _FORM_ARCHETYPE_SIGNALS if pat.search(all_text))
    a_count = sum(1 for pat in _ACADEMIC_ARCHETYPE_SIGNALS if pat.search(all_text))

    cols_per_page = [
        len([r for r in page.regions if r.kind == "column"])
        for page in document.pages
    ]
    max_cols = max(cols_per_page) if cols_per_page else 0

    if f_count >= 1 and (m_count >= 2 or max_cols >= 3):
        return DocumentArchetype.STRUCTURED_FORM
    if m_count >= 2:
        if max_cols >= 3:
            return DocumentArchetype.MARITIME_TABULAR
        return DocumentArchetype.MARITIME_CV
    if a_count >= 2:
        return DocumentArchetype.ACADEMIC_CV
    return DocumentArchetype.STANDARD_CV


_CONTACT_TEXT_RE = re.compile(
    r"(?:@|https?://|www\.|linkedin\.com|github\.com|leetcode\.com|gitlab\.com|\+?\d[\d\s\-\(\)]{7,}\d|^\s*(?:e-?mail|phone|tel|mobile|cell|linkedin|github|portfolio|website)\b)",
    re.IGNORECASE,
)


def _is_false_unknown_table(ctx: Any) -> bool:
    """Detect whether a candidate table with UNKNOWN purpose is a false table.

    Rejects candidate tables that originate from:
    1. Letterhead / contact headers falsely detected as multi-column tables.
    2. Multi-column page layouts (columns of prose resumes) falsely detected as tables.
    3. Tables whose data rows contain top-level section headings or multi-line prose paragraphs.
    """
    from app.pipeline.stages.table_semantic_context import TablePurpose
    from app.pipeline.stages.structural_roles import _is_known_section_alias

    if ctx.purpose != TablePurpose.UNKNOWN:
        return False

    non_empty_headers = [c.header_text.strip() for c in ctx.columns if c.header_text.strip()]
    if non_empty_headers:
        contact_header_count = sum(1 for h in non_empty_headers if _CONTACT_TEXT_RE.search(h))
        if contact_header_count > 0 and contact_header_count >= len(non_empty_headers) * 0.5:
            return True

    for h in non_empty_headers:
        h_upper = h.upper()
        if _is_known_section_alias(h_upper):
            return True
        words = h_upper.split()
        if any(w in words for w in ("SUMMARY", "PROFILE", "EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "OBJECTIVE")):
            return True

    section_heading_cells = 0
    long_prose_cells = 0
    for row in ctx.rows:
        for cell in row:
            text = cell.text.strip()
            if not text:
                continue
            text_upper = text.upper()
            words = text.split()
            if _is_known_section_alias(text_upper) or (
                len(words) <= 3 and any(w in text_upper.split() for w in ("EXPERIENCE", "EDUCATION", "PROJECTS", "EMPLOYMENT HISTORY"))
            ):
                section_heading_cells += 1
            if len(words) > 18 or (len(words) > 12 and text.startswith(("•", "-", "–", "—", "*"))):
                long_prose_cells += 1

    if section_heading_cells >= 1:
        return True
    if long_prose_cells >= 2:
        return True

    return False


def build_semantic_input(
    document: Document,
    document_id: str = "doc-1",
    structural_blocks: list[StructuralBlock] | None = None,
    archetype: DocumentArchetype | None = None,
) -> SemanticInput:
    """Build a SemanticInput payload from layout Document IR.

    Populates suggested_role from existing StructuralBlock roles and determines document archetype.
    """
    if archetype is None:
        if (
            hasattr(document, "document_type")
            and document.document_type
            and document.document_type != "unknown"
        ):
            try:
                archetype = DocumentArchetype(document.document_type)
            except ValueError:
                archetype = classify_document_archetype(document)
        else:
            archetype = classify_document_archetype(document)

    if structural_blocks is None:
        structural_blocks = build_structural_blocks(document)

    role_by_line_id: dict[str, str] = {}
    for sb in structural_blocks:
        for lid in sb.line_ids:
            role_by_line_id[lid] = sb.role.value

    all_blocks: list[SemanticBlockInput] = []
    pages: list[SemanticPageMeta] = []

    block_counter = 0
    for page in document.pages:
        for region in page.regions:
            current_heading_block_id: str | None = None
            for line in region.lines:
                block_id = f"b_p{page.page_number}_{block_counter}"
                block_counter += 1
                suggested_role = role_by_line_id.get(line.line_id, "UNKNOWN")

                is_heading = False
                if suggested_role == "SECTION_HEADING":
                    is_heading = True
                elif suggested_role in ("UNKNOWN", "HEADING") and region.kind not in ("header", "footer"):
                    txt = line.text.strip()
                    words = txt.split()
                    if (
                        1 <= len(words) <= 6
                        and len(txt) <= 60
                        and not any(ch.isdigit() for ch in txt)
                        and not txt.endswith((".", ":", ";", ","))
                    ):
                        emphasis = bool(line.style.bold) or (line.style.font_size is not None and line.style.font_size >= 12.5) or (txt.isupper() and any(c.isalpha() for c in txt))
                        if emphasis and not ("@" in txt or re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", txt) or re.search(r"\b(?:inc|llc|ltd|corp|corporation)\b", txt, re.I)):
                            is_heading = True

                if is_heading:
                    current_heading_block_id = block_id
                    parent_h_id = None
                else:
                    parent_h_id = current_heading_block_id

                sblock = SemanticBlockInput(
                    block_id=block_id,
                    text=line.text.strip(),
                    page=page.page_number,
                    bbox=[line.bbox.x0, line.bbox.y0, line.bbox.x1, line.bbox.y1],
                    region_id=region.region_id,
                    region_kind=region.kind,
                    column_id=region.column_id,
                    reading_order=line.reading_order if line.reading_order is not None else block_counter,
                    is_bold=line.style.bold,
                    is_italic=getattr(line.style, "italic", None),
                    font_size=line.style.font_size,
                    suggested_role=suggested_role,
                    heading_candidate=is_heading if is_heading else None,
                    parent_heading_id=parent_h_id,
                    spans=[
                        {"text": s.text, "bbox": [s.bbox.x0, s.bbox.y0, s.bbox.x1, s.bbox.y1]}
                        for s in line.spans
                    ]
                    if hasattr(line, "spans") and line.spans
                    else [],
                )
                all_blocks.append(sblock)

        pages.append(
            SemanticPageMeta(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
            )
        )

    semantic_tables: list[Any] = []
    from app.pipeline.stages.table_binding import GeometricTableBinder
    from app.pipeline.stages.table_semantic_context import (
        TablePurpose,
        apply_table_semantics_to_blocks,
        build_table_semantic_contexts,
    )
    from app.pipeline.stages.structural_roles import _is_known_section_alias

    binder = GeometricTableBinder()
    tables = binder.detect_document_tables(all_blocks)
    if tables:
        candidate_contexts = build_table_semantic_contexts(tables, all_blocks)
        valid_contexts = []
        for ctx in candidate_contexts:
            # 1. Never accept a table whose header contains top-level resume section headings or paragraphs
            is_layout_header = False
            for col in ctx.columns:
                hdr = col.header_text.strip().upper()
                if any(_is_known_section_alias(t) for t in hdr.split()):
                    if any(w in hdr for w in ("SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "CONTACT", "PROJECTS")) and len(ctx.columns) == 2:
                        is_layout_header = True
                        break
                if len(hdr.split()) > 15:
                    is_layout_header = True
                    break
            if is_layout_header:
                continue

            # Reject contact-dominated or prose-dominated false tables with UNKNOWN purpose
            if _is_false_unknown_table(ctx):
                continue

            # 2. 3+ columns with multiple rows is strong tabular evidence
            if len(ctx.columns) >= 3 and len(ctx.rows) >= 2:
                valid_contexts.append(ctx)
                continue

            # 3. 2 columns: require conservative evidence (form table or high-confidence table purpose)
            if len(ctx.columns) == 2:
                # Date or lengthy prose in header indicates standard title + right-aligned date, not a genuine table
                has_date_header = any(re.search(r"\b(19|20)\d{2}\b", col.header_text) for col in ctx.columns)
                has_long_header = any(len(col.header_text.split()) > 7 for col in ctx.columns)
                if has_date_header or has_long_header:
                    continue

                if ctx.purpose in (
                    TablePurpose.PERSONAL_DATA,
                    TablePurpose.CONTACT_DETAILS,
                    TablePurpose.DOCUMENTS,
                    TablePurpose.SEA_SERVICE,
                    TablePurpose.EDUCATION,
                    TablePurpose.CERTIFICATION,
                    TablePurpose.COURSES_CERTIFICATIONS,
                ) and ctx.confidence >= 0.25:
                    valid_contexts.append(ctx)
                elif ctx.is_form_table and len(ctx.rows) >= 3:
                    col0_texts = [r[0].text for r in ctx.rows if len(r) > 0 and r[0].text.strip()]
                    if col0_texts and all(len(txt.split()) <= 6 for txt in col0_texts):
                        valid_contexts.append(ctx)

        if valid_contexts:
            valid_table_ids = {c.table_id for c in valid_contexts}
            valid_tables = [t for t in tables if t.table_id in valid_table_ids]
            all_blocks = binder.bind_document_tables(all_blocks, valid_tables)
            semantic_tables = valid_contexts
            all_blocks = apply_table_semantics_to_blocks(semantic_tables, all_blocks)

    return SemanticInput(
        document_id=document_id,
        page_count=len(document.pages),
        archetype=archetype,
        pages=pages,
        blocks=all_blocks,
        tables=semantic_tables,
    )


def _is_two_digit_date_supported(year: str, month: str | None, day: str | None, s_text: str) -> bool:
    """Conservative validation checking if a 4-digit canonical year is supported by an explicit 2-digit date in source.

    Permits only explicit date patterns (e.g. '28/Mar/21', '01/11/21', '28-03-21', 'Mar 21', '21-03-28').
    Strictly forbids treating arbitrary 2-digit numbers as years.
    """
    if len(year) != 4 or not year.isdigit():
        return False
    yy = year[2:]

    if month is not None:
        m_int = str(int(month))
        m_abbr = _MONTH_NAMES.get(month, "")
        m_full = _FULL_MONTH_NAMES.get(month, "")
        month_tokens = [re.escape(tok) for tok in [month, m_int, m_abbr, m_full] if tok]
        m_pat = "(?:" + "|".join(month_tokens) + ")"

        if day is not None:
            d_int = str(int(day))
            day_tokens = [re.escape(tok) for tok in [day, d_int] if tok]
            d_pat = "(?:" + "|".join(day_tokens) + ")"

            # Day-Month-Year (e.g. 28/Mar/21, 28-03-21, 28.Mar.21, 28 Mar 21, 28 Mar '21)
            p1 = rf"(?<!\d){d_pat}[/.\-\s]+{m_pat}[/.\-\s]+'?{yy}(?!\d)"
            # Month-Day-Year (e.g. Mar/28/21, March 28, 21, 03/28/21)
            p2 = rf"\b{m_pat}[/.\-\s]+{d_pat}[/.\-,\s]+'?{yy}(?!\d)"
            # Year-Month-Day (e.g. 21-03-28, 21/Mar/28)
            p3 = rf"(?<!\d){yy}[/.\-\s]+{m_pat}[/.\-\s]+{d_pat}(?!\d)"

            combined = rf"(?:{p1}|{p2}|{p3})"
            if re.search(combined, s_text, re.IGNORECASE):
                return True
        else:
            # Month-Year (e.g. Mar/21, Mar-21, Mar '21, 03/21)
            p1 = rf"\b{m_pat}[/.\-\s]+'?{yy}(?!\d)"
            # Year-Month (e.g. 21-03, 21/Mar)
            p2 = rf"(?<!\d){yy}[/.\-\s]+{m_pat}\b"
            combined = rf"(?:{p1}|{p2})"
            if re.search(combined, s_text, re.IGNORECASE):
                return True

    return False


def _is_value_semantically_supported(canonical_val: str, source_text: str, is_single_block: bool = True) -> bool:
    """Deterministic validation boundary checking if canonical value is supported by source evidence.

    Permits ONLY:
    1. Whitespace, case, and punctuation normalization (exact alphanumeric substring match).
    2. Phone digit normalization (canonical digits sequence is exact substring of source digits).
    3. ISO date normalization (year digits and month name/number explicitly present in source text,
       including explicit 2-digit dates matching the 4-digit canonical year).
    4. Current-status date markers ('Present' supported by 'Till Now', etc.).
    5. Single-block ordered subphrase grounding (at least 2 tokens, exact order preserved in single source block).

    Strictly forbids:
    - Multi-block token combination fabrication
    - Synonym replacement
    - Company renaming (e.g. 'Darya Shaan' -> 'Darya Shipping')
    - Title expansion (e.g. 'Senior Software Engineer' -> 'Principal Software Engineer')
    - Semantic enrichment or fuzzy matching
    """
    c_val = canonical_val.strip()
    s_text = source_text.strip()
    if not c_val or not s_text:
        return False

    norm_src = "".join(ch.lower() for ch in s_text if ch.isalnum())
    norm_val = "".join(ch.lower() for ch in c_val if ch.isalnum())

    # 1. Whitespace, case, and punctuation normalization: exact alphanumeric substring
    if norm_val in norm_src:
        return True

    # 2. ISO date normalization (e.g. "2018-07" from "July 2018", "2018" from "2018")
    date_match = _ISO_DATE_RE.match(c_val)
    if date_match:
        year = date_match.group(1)
        month = date_match.group(2)
        day = date_match.group(3)
        if year in s_text:
            if month is None:
                return True
            month_abbr = _MONTH_NAMES.get(month, "")
            month_supported = (month in s_text) or (bool(month_abbr) and month_abbr in s_text.lower())
            if not month_supported:
                return False
            if day is not None:
                day_stripped = str(int(day))
                return day in s_text or day_stripped in s_text
            return True
        elif _is_two_digit_date_supported(year, month, day, s_text):
            return True

    # 3. Phone digit normalization (e.g. "+919829519017" from "+91 98295 19017", "+16504981240" from "(650) 498-1240")
    val_clean_phone = "".join(ch for ch in c_val if ch.isdigit() or ch == "+")
    if _PHONE_CANONICAL_RE.match(val_clean_phone):
        val_digits = "".join(ch for ch in c_val if ch.isdigit())
        src_digits = "".join(ch for ch in s_text if ch.isdigit())
        if val_digits and val_digits in src_digits:
            return True
        if (
            len(src_digits) >= 7
            and 1 <= len(val_digits) - len(src_digits) <= 3
            and val_digits.endswith(src_digits)
        ):
            return True

    # 4. Current-status date marker normalization (e.g. "Present" supported by "Till Now", "ongoing", "current")
    if c_val.lower() in ("present", "current", "now"):
        if any(pat.search(s_text) for pat in ACCEPTED_CURRENT_MARKERS):
            return True

    # 5. Conservative single-block ordered subphrase grounding
    # (e.g. "Medicine & Public Health" grounded by
    # "Doctor of Medicine (M.D.) & Master of Public Health (M.P.H.)")
    if is_single_block:
        val_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", c_val)]
        src_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", s_text)]
        if len(val_tokens) >= 2 and (Counter(val_tokens) <= Counter(src_tokens)):
            it = iter(src_tokens)
            if all(tok in it for tok in val_tokens):
                return True

    return False


NAME_FORM_DESCRIPTOR_TOKENS: frozenset[str] = frozenset({
    "first",
    "firstname",
    "last",
    "lastname",
    "surname",
    "given",
    "givenname",
    "family",
    "familyname",
    "middle",
    "middlename",
    "full",
    "fullname",
    "name",
    "names",
})


def _is_name_form_descriptor_supported(canonical_name: str, source_text: str, is_multiblock: bool = False) -> bool:
    """Check if canonical human name tokens match source evidence tokens after filtering form descriptors.

    Requires exact multiset equality:
    - Source evidence MUST contain at least one recognized name form descriptor token (e.g. 'First name', 'Surname', 'Last name')
      OR source evidence is multi-block name where canonical tokens are an exact multiset permutation of source tokens
    - Token order may differ (e.g., 'Surname Alam First Name Akibul' -> 'Akibul Alam', 'Last name PARASHAR First name JOSH' -> 'JOSH PARASHAR')
    - Standard name descriptor labels are filtered from source tokens
    - Zero tokens may be added (no hallucinated middle names/words)
    - Zero non-descriptor tokens may be omitted
    - Strictly scoped to personal.name validation
    """
    val_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", canonical_name)]
    src_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", source_text)]

    if not val_tokens:
        return False

    has_descriptor = any(t in NAME_FORM_DESCRIPTOR_TOKENS for t in src_tokens)
    if not has_descriptor and not is_multiblock:
        return False

    # Filter recognized standard form descriptor tokens from source text
    filtered_src_tokens = [t for t in src_tokens if t not in NAME_FORM_DESCRIPTOR_TOKENS]

    # Must have at least 1 grounded token and exact multiset match
    if not filtered_src_tokens:
        return False

    return sorted(val_tokens) == sorted(filtered_src_tokens)


def _is_name_semantically_supported(canonical_name: str, source_text: str, is_multiblock: bool = False) -> bool:
    """Deterministic validation boundary specifically for personal.name.

    1. Checks generic deterministic support first (exact alphanumeric substring).
    2. Falls back to form-descriptor / multi-block token multiset matching specifically for human names.
    """
    if _is_value_semantically_supported(canonical_name, source_text):
        return True
    return _is_name_form_descriptor_supported(canonical_name, source_text, is_multiblock=is_multiblock)


def _is_multiblock_text_semantically_supported(canonical_val: str, source_text: str) -> bool:
    """Deterministic validation boundary for multi-block long text (e.g. descriptions, summaries).

    Permits deterministic sentence reconstruction / clause reordering across split/interleaved
    layout blocks (such as multi-column table cells) ONLY when:
    1. Canonical text tokens and source evidence tokens have exact multiset equality.
    2. Zero tokens are added (no hallucinated words, numbers, or facts).
    3. Zero tokens are omitted (no truncated evidence).
    """
    val_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", canonical_val)]
    src_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", source_text)]

    if not val_tokens or not src_tokens:
        return False

    return sorted(val_tokens) == sorted(src_tokens)


def validate_semantic_output(output: SemanticOutput, input_data: SemanticInput) -> list[str]:
    """Validate semantic invariants on LLM output to prevent systemic failure modes.

    Returns a list of violation messages. An empty list indicates a valid payload.
    """
    violations: list[str] = []
    known_blocks = {b.block_id: b for b in input_data.blocks}
    blocks_by_parent: dict[str, list[SemanticBlockInput]] = {}
    for b in input_data.blocks:
        if b.parent_block_id:
            blocks_by_parent.setdefault(b.parent_block_id, []).append(b)

    def _resolve_block_id(bid: str) -> bool:
        return bid in known_blocks or bid in blocks_by_parent

    def _get_block_text(bid: str) -> str:
        if bid in known_blocks:
            return known_blocks[bid].text
        if bid in blocks_by_parent:
            return " ".join(b.text for b in blocks_by_parent[bid])
        return ""

    def _get_blocks(bid: str) -> list[SemanticBlockInput]:
        if bid in known_blocks:
            return [known_blocks[bid]]
        return blocks_by_parent.get(bid, [])

    # Index explicit block classifications
    category_by_block_id: dict[str, SemanticBlockCategory] = {
        bc.block_id: bc.category for bc in output.block_classifications
    }

    # Helper to check block existence
    def _verify_block_ids(block_ids: list[str], context: str) -> None:
        if not block_ids:
            violations.append(f"MISSING_PROVENANCE in {context}: no source_block_ids provided")
            return
        for bid in block_ids:
            if not _resolve_block_id(bid):
                violations.append(f"UNKNOWN_BLOCK_ID in {context}: {bid!r}")

    # Helper for generic deterministic support boundary validation
    def _verify_grounded_string(gs: GroundedString | None, context: str) -> None:
        if gs is None:
            return
        _verify_block_ids(gs.source_block_ids, context)
        if gs.source_block_ids:
            source_text = " ".join(_get_block_text(bid) for bid in gs.source_block_ids if _resolve_block_id(bid))
            is_single = len(gs.source_block_ids) == 1
            if not _is_value_semantically_supported(gs.value, source_text, is_single_block=is_single):
                if len(gs.source_block_ids) > 1 and _is_multiblock_text_semantically_supported(gs.value, source_text):
                    return
                violations.append(f"UNSUPPORTED_CANONICAL_VALUE in {context}: {gs.value!r} not supported by {source_text!r}")

    # Helper specifically for long-text description/summary validation
    def _verify_grounded_text(gs: GroundedString | None, context: str) -> None:
        if gs is None:
            return
        _verify_block_ids(gs.source_block_ids, context)
        if gs.source_block_ids:
            source_text = " ".join(_get_block_text(bid) for bid in gs.source_block_ids if _resolve_block_id(bid))
            is_single = len(gs.source_block_ids) == 1
            if not _is_value_semantically_supported(gs.value, source_text, is_single_block=is_single):
                # Narrowly scoped fallback only for long-text fields with multiple source_block_ids
                if len(gs.source_block_ids) > 1 and _is_multiblock_text_semantically_supported(gs.value, source_text):
                    return
                violations.append(f"UNSUPPORTED_CANONICAL_VALUE in {context}: {gs.value!r} not supported by {source_text!r}")

    # Helper specifically for personal.name validation
    def _verify_grounded_name(gs: GroundedString | None, context: str) -> None:
        if gs is None:
            return
        _verify_block_ids(gs.source_block_ids, context)
        if gs.source_block_ids:
            source_text = " ".join(_get_block_text(bid) for bid in gs.source_block_ids if _resolve_block_id(bid))
            is_multiblock = len(gs.source_block_ids) > 1
            if not _is_name_semantically_supported(gs.value, source_text, is_multiblock=is_multiblock):
                violations.append(f"UNSUPPORTED_CANONICAL_VALUE in {context}: {gs.value!r} not supported by {source_text!r}")

    # Helper for grounded boolean validation
    def _verify_grounded_bool(gb: GroundedBool | None, context: str) -> None:
        if gb is None:
            return
        _verify_block_ids(gb.source_block_ids, context)
        if gb.source_block_ids:
            source_text = " ".join(_get_block_text(bid) for bid in gb.source_block_ids if _resolve_block_id(bid))
            if gb.value is True:
                if not any(pat.search(source_text) for pat in ACCEPTED_CURRENT_MARKERS):
                    violations.append(
                        f"UNSUPPORTED_CURRENT_STATUS in {context}: current=True not supported by source {source_text!r}"
                    )

    # 1. Personal Name Invariant: Never allow document titles or form labels
    if output.personal.name and output.personal.name.value:
        _verify_grounded_name(output.personal.name, "personal.name")
        val = output.personal.name.value.strip()
        for pat in INVALID_NAME_PATTERNS:
            if pat.search(val):
                violations.append(f"DOCUMENT_TITLE_AS_NAME: {val!r}")
                break

    # 2. Location Scope Invariant: Must originate from page 1 personal/contact/header context
    if output.personal.location and output.personal.location.value:
        _verify_grounded_string(output.personal.location, "personal.location")
        val = output.personal.location.value.strip()
        for pat in INVALID_LOCATION_PATTERNS:
            if pat.search(val):
                violations.append(f"SECTION_HEADER_AS_LOCATION: {val!r}")
                break

        # Collect source block IDs mapped to body collections to prevent cross-collection leakage
        body_mapped_block_ids: set[str] = set()
        for exp in output.experience:
            body_mapped_block_ids.update(exp.source_block_ids)
            if exp.company:
                body_mapped_block_ids.update(exp.company.source_block_ids)
            if exp.designation:
                body_mapped_block_ids.update(exp.designation.source_block_ids)
            if exp.startDate:
                body_mapped_block_ids.update(exp.startDate.source_block_ids)
            if exp.endDate:
                body_mapped_block_ids.update(exp.endDate.source_block_ids)
            if exp.location:
                body_mapped_block_ids.update(exp.location.source_block_ids)
            if exp.description:
                body_mapped_block_ids.update(exp.description.source_block_ids)
            for tech in exp.technologies:
                body_mapped_block_ids.update(tech.source_block_ids)
        for edu in output.education:
            body_mapped_block_ids.update(edu.source_block_ids)
            if edu.institution:
                body_mapped_block_ids.update(edu.institution.source_block_ids)
            if edu.degree:
                body_mapped_block_ids.update(edu.degree.source_block_ids)
            if edu.fieldOfStudy:
                body_mapped_block_ids.update(edu.fieldOfStudy.source_block_ids)
            if edu.startDate:
                body_mapped_block_ids.update(edu.startDate.source_block_ids)
            if edu.endDate:
                body_mapped_block_ids.update(edu.endDate.source_block_ids)
            if edu.grade:
                body_mapped_block_ids.update(edu.grade.source_block_ids)
        for prj in output.projects:
            body_mapped_block_ids.update(prj.source_block_ids)
            if prj.name:
                body_mapped_block_ids.update(prj.name.source_block_ids)
            if prj.description:
                body_mapped_block_ids.update(prj.description.source_block_ids)
            if prj.startDate:
                body_mapped_block_ids.update(prj.startDate.source_block_ids)
            if prj.endDate:
                body_mapped_block_ids.update(prj.endDate.source_block_ids)
            for tech in prj.technologies:
                body_mapped_block_ids.update(tech.source_block_ids)

        for bid in output.personal.location.source_block_ids:
            for b in _get_blocks(bid):
                if b.page != 1:
                    violations.append(f"LOCATION_OUTSIDE_HEADER_REGION: block {bid} is on page {b.page}")
                elif b.region_kind == "footer":
                    violations.append(f"LOCATION_OUTSIDE_HEADER_REGION: block {bid} belongs to footer region")
                elif bid in body_mapped_block_ids or (b.parent_block_id and b.parent_block_id in body_mapped_block_ids):
                    violations.append(f"LOCATION_OUTSIDE_HEADER_REGION: block {bid} is also mapped to body collections")
                elif category_by_block_id.get(bid) in EXCLUDED_LOCATION_CATEGORIES or (b.parent_block_id and category_by_block_id.get(b.parent_block_id) in EXCLUDED_LOCATION_CATEGORIES):
                    cat = category_by_block_id.get(bid) or (category_by_block_id.get(b.parent_block_id) if b.parent_block_id else None)
                    violations.append(
                        f"LOCATION_OUTSIDE_HEADER_REGION: block {bid} belongs to excluded category {cat.value if cat else ''}"
                    )
                elif b.suggested_role in EXCLUDED_LOCATION_ROLES:
                    violations.append(
                        f"LOCATION_OUTSIDE_HEADER_REGION: block {bid} has non-personal structural role {b.suggested_role}"
                    )

    # 3. Personal Contact Provenance
    if output.personal.email:
        _verify_grounded_string(output.personal.email, "personal.email")
    if output.personal.phone:
        _verify_grounded_string(output.personal.phone, "personal.phone")
    if output.personal.linkedin:
        _verify_grounded_string(output.personal.linkedin, "personal.linkedin")
    if output.personal.github:
        _verify_grounded_string(output.personal.github, "personal.github")
    if output.personal.portfolio:
        _verify_grounded_string(output.personal.portfolio, "personal.portfolio")

    # 4. Reference & Table Header Exclusion Guard for Experience
    for i, exp in enumerate(output.experience):
        all_exp_block_ids = list(exp.source_block_ids)
        if exp.company:
            _verify_grounded_string(exp.company, f"experience[{i}].company")
            all_exp_block_ids.extend(exp.company.source_block_ids)
        if exp.designation:
            _verify_grounded_string(exp.designation, f"experience[{i}].designation")
            all_exp_block_ids.extend(exp.designation.source_block_ids)
        if exp.startDate:
            _verify_grounded_string(exp.startDate, f"experience[{i}].startDate")
            all_exp_block_ids.extend(exp.startDate.source_block_ids)
        if exp.endDate:
            _verify_grounded_string(exp.endDate, f"experience[{i}].endDate")
            all_exp_block_ids.extend(exp.endDate.source_block_ids)
        if exp.current:
            _verify_grounded_bool(exp.current, f"experience[{i}].current")
            all_exp_block_ids.extend(exp.current.source_block_ids)
        if exp.location:
            _verify_grounded_string(exp.location, f"experience[{i}].location")
            all_exp_block_ids.extend(exp.location.source_block_ids)
        if exp.description:
            _verify_grounded_text(exp.description, f"experience[{i}].description")
            all_exp_block_ids.extend(exp.description.source_block_ids)
        for tech in exp.technologies:
            _verify_grounded_string(tech, f"experience[{i}].technologies")
            all_exp_block_ids.extend(tech.source_block_ids)

        comp = (exp.company.value if exp.company else "").strip()
        desig = (exp.designation.value if exp.designation else "").strip()
        for pat in TABLE_HEADER_PATTERNS:
            if pat.search(comp):
                violations.append(f"TABLE_HEADER_AS_COMPANY in experience[{i}]: {comp!r}")
            if pat.search(desig):
                violations.append(f"TABLE_HEADER_AS_DESIGNATION in experience[{i}]: {desig!r}")

        # Referee separation check: reject blocks classified as REFERENCE or BOILERPLATE
        for bid in all_exp_block_ids:
            cat = category_by_block_id.get(bid)
            if cat == SemanticBlockCategory.REFERENCE:
                violations.append(f"REFERENCE_IN_EXPERIENCE: block {bid} is classified as REFERENCE but mapped to experience[{i}]")
            elif cat in (SemanticBlockCategory.BOILERPLATE, SemanticBlockCategory.TABLE_HEADER):
                violations.append(f"EXCLUDED_CATEGORY_IN_EXPERIENCE: block {bid} is classified as {cat.value} but mapped to experience[{i}]")

    # 4b. Deterministic Entity Span & Cross-Entity Validation for Experience
    exp_spans = build_deterministic_experience_spans(input_data)
    if exp_spans and output.experience:
        total_exps = len(output.experience)
        for i, exp in enumerate(output.experience):
            assigned_span = _assign_span_to_experience_item(exp, exp_spans, i, total_exps)
            if assigned_span is not None:
                for fname, fval in [
                    ("company", exp.company),
                    ("designation", exp.designation),
                    ("startDate", exp.startDate),
                    ("endDate", exp.endDate),
                    ("location", exp.location),
                ]:
                    if fval and fval.source_block_ids:
                        for bid in fval.source_block_ids:
                            bids_to_check = {bid} | {b.block_id for b in blocks_by_parent.get(bid, [])}
                            for other_s in exp_spans:
                                if other_s.entity_index != assigned_span.entity_index and any(b_chk in other_s.block_ids for b_chk in bids_to_check):
                                    violations.append(
                                        f"CROSS_ENTITY_PROVENANCE in experience[{i}].{fname}: "
                                        f"block {bid!r} belongs to experience entity span {other_s.entity_index}"
                                    )
                for bid in exp.source_block_ids:
                    bids_to_check = {bid} | {b.block_id for b in blocks_by_parent.get(bid, [])}
                    for other_s in exp_spans:
                        if other_s.entity_index != assigned_span.entity_index and any(b_chk in other_s.block_ids for b_chk in bids_to_check):
                            if not any(b_chk in assigned_span.block_ids for b_chk in bids_to_check):
                                violations.append(
                                    f"CROSS_ENTITY_PROVENANCE in experience[{i}]: "
                                    f"block {bid!r} belongs to experience entity span {other_s.entity_index}"
                                )

    # 5. Table Header Exclusion Guard for Education
    for i, edu in enumerate(output.education):
        if edu.institution:
            _verify_grounded_string(edu.institution, f"education[{i}].institution")
        if edu.degree:
            _verify_grounded_string(edu.degree, f"education[{i}].degree")
        if edu.fieldOfStudy:
            _verify_grounded_string(edu.fieldOfStudy, f"education[{i}].fieldOfStudy")
        if edu.startDate:
            _verify_grounded_string(edu.startDate, f"education[{i}].startDate")
        if edu.endDate:
            _verify_grounded_string(edu.endDate, f"education[{i}].endDate")
        if edu.grade:
            _verify_grounded_string(edu.grade, f"education[{i}].grade")

        deg = (edu.degree.value if edu.degree else "").strip()
        inst = (edu.institution.value if edu.institution else "").strip()
        for pat in TABLE_HEADER_PATTERNS:
            if pat.search(deg):
                violations.append(f"TABLE_HEADER_AS_DEGREE in education[{i}]: {deg!r}")
            if pat.search(inst):
                violations.append(f"TABLE_HEADER_AS_INSTITUTION in education[{i}]: {inst!r}")

    # 6. Provenance for Projects
    for i, prj in enumerate(output.projects):
        if prj.name:
            _verify_grounded_string(prj.name, f"projects[{i}].name")
        if prj.description:
            _verify_grounded_text(prj.description, f"projects[{i}].description")
        if prj.startDate:
            _verify_grounded_string(prj.startDate, f"projects[{i}].startDate")
        if prj.endDate:
            _verify_grounded_string(prj.endDate, f"projects[{i}].endDate")
        if prj.current:
            _verify_grounded_bool(prj.current, f"projects[{i}].current")
        for tech in prj.technologies:
            _verify_grounded_string(tech, f"projects[{i}].technologies")

    # 7. Provenance for Summary, Skills, Certifications, Languages, Achievements
    if output.summary:
        _verify_grounded_text(output.summary, "summary")
    for i, s in enumerate(output.skills):
        _verify_grounded_string(s, f"skills[{i}]")
    for i, c in enumerate(output.certifications):
        _verify_grounded_string(c, f"certifications[{i}]")
    for i, l in enumerate(output.languages):
        _verify_grounded_string(l, f"languages[{i}]")
    for i, a in enumerate(output.achievements):
        _verify_grounded_string(a, f"achievements[{i}]")

    return violations


def semantic_output_to_resume(output: SemanticOutput, parser_version: str = "2.0.0") -> Resume:
    """Project validated SemanticOutput into the public Resume domain model."""
    personal = PersonalInfo(
        name=output.personal.name.value if output.personal.name else None,
        email=output.personal.email.value if output.personal.email else None,
        phone=output.personal.phone.value if output.personal.phone else None,
        location=output.personal.location.value if output.personal.location else None,
        linkedin=output.personal.linkedin.value if output.personal.linkedin else None,
        github=output.personal.github.value if output.personal.github else None,
        portfolio=output.personal.portfolio.value if output.personal.portfolio else None,
    )

    experience = [
        ExperienceItem(
            company=exp.company.value if exp.company else None,
            designation=exp.designation.value if exp.designation else None,
            startDate=exp.startDate.value if exp.startDate else None,
            endDate=exp.endDate.value if exp.endDate else None,
            current=exp.current.value if exp.current else None,
            location=exp.location.value if exp.location else None,
            description=exp.description.value if exp.description else None,
            skills=[t.value for t in exp.technologies],
        )
        for exp in output.experience
    ]

    education = [
        EducationItem(
            institution=edu.institution.value if edu.institution else None,
            degree=edu.degree.value if edu.degree else None,
            fieldOfStudy=edu.fieldOfStudy.value if edu.fieldOfStudy else None,
            startDate=edu.startDate.value if edu.startDate else None,
            endDate=edu.endDate.value if edu.endDate else None,
            grade=edu.grade.value if edu.grade else None,
        )
        for edu in output.education
    ]

    projects = [
        ProjectItem(
            name=prj.name.value if prj.name else None,
            description=prj.description.value if prj.description else None,
            technologies=[t.value for t in prj.technologies],
            startDate=prj.startDate.value if prj.startDate else None,
            endDate=prj.endDate.value if prj.endDate else None,
            current=prj.current.value if prj.current else None,
        )
        for prj in output.projects
    ]

    skills = [s.value for s in output.skills]
    certifications = [
        CertificationItem(name=c.value) for c in output.certifications
    ]
    languages = [l.value for l in output.languages]
    achievements = [a.value for a in output.achievements]

    return Resume(
        parserVersion=parser_version,
        personal=personal,
        summary=output.summary.value if output.summary else None,
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
        certifications=certifications,
        achievements=achievements,
        languages=languages,
        metadata={"archetype": output.document_archetype.value, "extractor": "semantic_llm"},
    )
