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

import re
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

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
from app.pipeline.stages.structural_roles import build_structural_blocks

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
    font_size: float | None = None
    suggested_role: str | None = None  # from existing StructuralRole

    # Generic table representation (populated when block is in a table structure)
    table_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    cell_role: str | None = None  # e.g. "HEADER", "DATA"


class SemanticPageMeta(BaseModel):
    """Page-level spatial metadata only (canonical blocks reside in top-level blocks list)."""

    page_number: int
    width: float | None = None
    height: float | None = None


class SemanticInput(BaseModel):
    """Non-redundant, canonical JSON-serializable input payload for LLM extraction."""

    document_id: str
    page_count: int
    pages: list[SemanticPageMeta] = Field(default_factory=list)
    blocks: list[SemanticBlockInput] = Field(default_factory=list)


# =====================================================================
# 3. SemanticOutput Contract (Constrained LLM Response Schema)
# =====================================================================


class GroundedString(BaseModel):
    """A field value retaining mandatory source provenance and supporting canonical normalization.

    - value: The canonical/normalized representation (e.g. '2018-07', '+919829519017', 'John Doe').
    - raw_value: Optional verbatim raw text from source blocks when it differs from canonical value.
    - source_block_ids: Mandatory list of referenced block IDs providing evidence.
    """

    value: str
    raw_value: str | None = None
    source_block_ids: list[str] = Field(default_factory=list)


class GroundedBool(BaseModel):
    """A boolean field with mandatory source provenance evidence (e.g. current employment)."""

    value: bool
    source_block_ids: list[str] = Field(default_factory=list)


class BlockClassification(BaseModel):
    """Explicit classification of a source block into semantic categories or exclusions."""

    block_id: str
    category: SemanticBlockCategory
    exclusion_reason: str | None = None


class GroundedPersonal(BaseModel):
    """Candidate personal identity with provenance grounding."""

    name: GroundedString | None = None
    email: GroundedString | None = None
    phone: GroundedString | None = None
    location: GroundedString | None = None
    linkedin: GroundedString | None = None
    github: GroundedString | None = None
    portfolio: GroundedString | None = None


class GroundedExperienceItem(BaseModel):
    """Single professional employment entry with complete provenance."""

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
    """Single academic credential with complete provenance."""

    institution: GroundedString | None = None
    degree: GroundedString | None = None
    fieldOfStudy: GroundedString | None = None
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    grade: GroundedString | None = None
    source_block_ids: list[str] = Field(default_factory=list)


class GroundedProjectItem(BaseModel):
    """Single project entry with complete provenance."""

    name: GroundedString | None = None
    description: GroundedString | None = None
    technologies: list[GroundedString] = Field(default_factory=list)
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    current: GroundedBool | None = None
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


# =====================================================================
# 4. Builder and Validation Invariants
# =====================================================================


def build_semantic_input(
    document: Document,
    document_id: str = "doc-1",
    structural_blocks: list[StructuralBlock] | None = None,
) -> SemanticInput:
    """Build a SemanticInput payload from layout Document IR.

    Populates suggested_role from existing StructuralBlock roles.
    """
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
            for line in region.lines:
                block_id = f"b_p{page.page_number}_{block_counter}"
                block_counter += 1
                suggested_role = role_by_line_id.get(line.line_id, "UNKNOWN")

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
                    font_size=line.style.font_size,
                    suggested_role=suggested_role,
                )
                all_blocks.append(sblock)

        pages.append(
            SemanticPageMeta(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
            )
        )

    return SemanticInput(
        document_id=document_id,
        page_count=len(document.pages),
        pages=pages,
        blocks=all_blocks,
    )


def _is_value_semantically_supported(canonical_val: str, source_text: str) -> bool:
    """Deterministic validation boundary checking if canonical value is supported by source evidence.

    Permits ONLY:
    1. Whitespace, case, and punctuation normalization (exact alphanumeric substring match).
    2. Phone digit normalization (canonical digits sequence is exact substring of source digits).
    3. ISO date normalization (year digits and month name/number explicitly present in source text).

    Strictly forbids:
    - Token subset combinations
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

    # 3. Phone digit normalization (e.g. "+919829519017" from "+91 98295 19017")
    if _PHONE_CANONICAL_RE.match(c_val):
        val_digits = "".join(ch for ch in c_val if ch.isdigit())
        src_digits = "".join(ch for ch in s_text if ch.isdigit())
        if val_digits and val_digits in src_digits:
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


def _is_name_form_descriptor_supported(canonical_name: str, source_text: str) -> bool:
    """Check if canonical human name tokens match source evidence tokens after filtering form descriptors.

    Requires exact multiset equality:
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

    # Filter recognized standard form descriptor tokens from source text
    filtered_src_tokens = [t for t in src_tokens if t not in NAME_FORM_DESCRIPTOR_TOKENS]

    # Must have at least 1 grounded token and exact multiset match
    if not filtered_src_tokens:
        return False

    return sorted(val_tokens) == sorted(filtered_src_tokens)


def _is_name_semantically_supported(canonical_name: str, source_text: str) -> bool:
    """Deterministic validation boundary specifically for personal.name.

    1. Checks generic deterministic support first (exact alphanumeric substring).
    2. Falls back to form-descriptor token multiset matching specifically for human names.
    """
    if _is_value_semantically_supported(canonical_name, source_text):
        return True
    return _is_name_form_descriptor_supported(canonical_name, source_text)


def validate_semantic_output(output: SemanticOutput, input_data: SemanticInput) -> list[str]:
    """Validate semantic invariants on LLM output to prevent systemic failure modes.

    Returns a list of violation messages. An empty list indicates a valid payload.
    """
    violations: list[str] = []
    known_blocks = {b.block_id: b for b in input_data.blocks}

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
            if bid not in known_blocks:
                violations.append(f"UNKNOWN_BLOCK_ID in {context}: {bid!r}")

    # Helper for generic deterministic support boundary validation
    def _verify_grounded_string(gs: GroundedString | None, context: str) -> None:
        if gs is None:
            return
        _verify_block_ids(gs.source_block_ids, context)
        if gs.source_block_ids:
            source_text = " ".join(known_blocks[bid].text for bid in gs.source_block_ids if bid in known_blocks)
            if not _is_value_semantically_supported(gs.value, source_text):
                violations.append(f"UNSUPPORTED_CANONICAL_VALUE in {context}: {gs.value!r} not supported by {source_text!r}")

    # Helper specifically for personal.name validation
    def _verify_grounded_name(gs: GroundedString | None, context: str) -> None:
        if gs is None:
            return
        _verify_block_ids(gs.source_block_ids, context)
        if gs.source_block_ids:
            source_text = " ".join(known_blocks[bid].text for bid in gs.source_block_ids if bid in known_blocks)
            if not _is_name_semantically_supported(gs.value, source_text):
                violations.append(f"UNSUPPORTED_CANONICAL_VALUE in {context}: {gs.value!r} not supported by {source_text!r}")

    # Helper for grounded boolean validation
    def _verify_grounded_bool(gb: GroundedBool | None, context: str) -> None:
        if gb is None:
            return
        _verify_block_ids(gb.source_block_ids, context)
        if gb.source_block_ids:
            source_text = " ".join(known_blocks[bid].text for bid in gb.source_block_ids if bid in known_blocks)
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

    # 2. Location Scope Invariant: Must originate from structural header region
    if output.personal.location and output.personal.location.value:
        _verify_grounded_string(output.personal.location, "personal.location")
        val = output.personal.location.value.strip()
        for pat in INVALID_LOCATION_PATTERNS:
            if pat.search(val):
                violations.append(f"SECTION_HEADER_AS_LOCATION: {val!r}")
                break
        for bid in output.personal.location.source_block_ids:
            b = known_blocks.get(bid)
            if b:
                if b.page != 1:
                    violations.append(f"LOCATION_OUTSIDE_HEADER_REGION: block {bid} is on page {b.page}")
                elif b.region_kind != "header":
                    violations.append(
                        f"LOCATION_OUTSIDE_HEADER_REGION: block {bid} belongs to region kind {b.region_kind!r}, expected 'header'"
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
            _verify_grounded_string(exp.description, f"experience[{i}].description")
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
            _verify_grounded_string(prj.description, f"projects[{i}].description")
        if prj.startDate:
            _verify_grounded_string(prj.startDate, f"projects[{i}].startDate")
        if prj.endDate:
            _verify_grounded_string(prj.endDate, f"projects[{i}].endDate")
        if prj.current:
            _verify_grounded_bool(prj.current, f"projects[{i}].current")
        for tech in prj.technologies:
            _verify_grounded_string(tech, f"projects[{i}].technologies")

    # 7. Provenance for Skills, Certifications, Languages, Achievements
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
