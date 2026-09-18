"""Phase 10W: Generic Table Semantic Interpretation.

Recovers high-level semantic meaning for tables, columns, and cells across
tabular grids (Style A) and form/key-value grids (Style B) without document-specific
hacks, fixed coordinates, or hardcoded column counts.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.domain.semantic_contract import SemanticBlockInput
from app.pipeline.stages.table_binding import GeometricCell, GeometricTable


class TablePurpose(str, Enum):
    """Semantic domain purpose of a structured table."""

    PERSONAL_DATA = "personal_data"
    CONTACT_DETAILS = "contact_details"
    EDUCATION = "education"
    EXPERIENCE = "experience"
    SEA_SERVICE = "sea_service"
    CERTIFICATION = "certification"
    COURSES_CERTIFICATIONS = "courses_certifications"
    CREDENTIAL = "credential"
    DOCUMENTS = "documents"
    SKILLS = "skills"
    OTHER = "other"
    UNKNOWN = "unknown"


class ColumnSemanticDescriptor(BaseModel):
    """Semantic role and metadata for a single table column."""

    model_config = ConfigDict(extra="ignore")

    column_index: int
    header_text: str = ""
    semantic_role: str = "unknown"
    source_block_ids: list[str] = Field(default_factory=list)


class CellSemanticDescriptor(BaseModel):
    """Semantic descriptor for an individual cell coordinate (row, column)."""

    model_config = ConfigDict(extra="ignore")

    row_index: int
    column_index: int
    text: str = ""
    cell_role: str = "DATA"  # "HEADER" | "DATA" | "LABEL" | "VALUE"
    semantic_role: str = "unknown"
    source_block_ids: list[str] = Field(default_factory=list)


class TableSemanticContext(BaseModel):
    """Interpreted semantic context for a deterministically bound table."""

    model_config = ConfigDict(extra="ignore")

    table_id: str
    page: int
    purpose: TablePurpose = TablePurpose.UNKNOWN
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    is_form_table: bool = False
    columns: list[ColumnSemanticDescriptor] = Field(default_factory=list)
    rows: list[list[CellSemanticDescriptor]] = Field(default_factory=list)
    physical_rows: list[list[CellSemanticDescriptor]] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)

    @property
    def logical_rows(self) -> list[list[CellSemanticDescriptor]]:
        """Convenience alias for the table's logical rows."""
        return self.rows

    def to_dict(self) -> dict[str, Any]:
        """Compact dictionary representation for prompt serialization."""
        return {
            "table_id": self.table_id,
            "page": self.page,
            "purpose": self.purpose.value,
            "confidence": round(self.confidence, 2),
            "is_form_table": self.is_form_table,
            "evidence": self.evidence,
            "columns": [
                {
                    "col_idx": c.column_index,
                    "header": c.header_text,
                    "semantic_role": c.semantic_role,
                }
                for c in self.columns
            ],
            "row_count": len(self.rows),
        }


# Lexical patterns for Table Purpose Inference
_PURPOSE_PATTERNS: dict[TablePurpose, dict[str, list[str]]] = {
    TablePurpose.SEA_SERVICE: {
        "strong": [
            r"\bvessels?\b",
            r"\bships?\s*name\b",
            r"\bname\s*of\s*vessels?\b",
            r"\bsea\s*service\b",
            r"\bsea\s*time\b",
            r"\bsea\s*experience\b",
            r"\bsailing\b",
            r"\bvsl\s*name\b",
            r"\bship\s*name\b",
        ],
        "medium": [
            r"\branks?\b",
            r"\bsign\s*[-–—]?\s*on\b",
            r"\bsign\s*[-–—]?\s*off\b",
            r"\bdwt\b",
            r"\bgrt\b",
            r"\bnrt\b",
            r"\bbhp\b",
            r"\bpropulsion\b",
            r"\bdeadweight\b",
            r"\bgross\s*tonnage\b",
        ],
        "context": [
            r"\bengines?\b",
            r"\bflag\b",
            r"\bowners?\b",
            r"\bduration\b",
            r"\bmonths\b",
        ],
    },
    TablePurpose.DOCUMENTS: {
        "strong": [
            r"\bpassports?\b",
            r"\bcdc\b",
            r"\bseaman’?s?\s*books?\b",
            r"\bindos\b",
            r"\bsid\s*numbers?\b",
            r"\btravel\s*documents?\b",
            r"\bnational\s*documents?\b",
            r"\bstatutory\s*documents?\b",
            r"\bdangerous\s*cargo\b",
            r"\bendorsements?\b",
            r"\bdocument\s*names?\b",
            r"\bdocuments?\b",
        ],
        "medium": [
            r"\bvisas?\b",
            r"\blicen[sc]es?\b",
            r"\bcertificates?\s*of\s*competency\b",
            r"\bcoc\b",
            r"\bblank\s*pages\b",
            r"\becnr\b",
            r"\bdocument\s*no\b",
            r"\bdocument\s*types?\b",
            r"\bdocuments?\b",
        ],
        "context": [
            r"\bissue\s*dates?\b",
            r"\bexpiry\s*dates?\b",
            r"\bvalid\s*until\b",
            r"\bplace\s*of\s*issue\b",
            r"\bissued\s*by\b",
        ],
    },
    TablePurpose.COURSES_CERTIFICATIONS: {
        "strong": [
            r"\bstcw\b",
            r"\bcourses?\s*&\s*certificates?\b",
            r"\btraining\s*courses?\b",
            r"\bcourse\s*names?\b",
            r"\bdetails\s*of\s*courses\b",
            r"\bsafety\s*courses?\b",
            r"\bvalue\s*added\b",
        ],
        "medium": [
            r"\bcourses?\b",
            r"\btraining\b",
            r"\bcertificate\s*no\b",
            r"\bcert\s*no\b",
            r"\bcop\b",
        ],
        "context": [
            r"\bvalidity\b",
            r"\bvalid\s*until\b",
            r"\bissued\s*by\b",
            r"\binstitutes?\b",
        ],
    },
    TablePurpose.EDUCATION: {
        "strong": [
            r"\bacademic\s*qualifications?\b",
            r"\beducational?\b",
            r"\bdegrees?\b",
            r"\bname\s*of\s*institutes?\b",
            r"\btypes?\s*of\s*degrees?\b",
        ],
        "medium": [
            r"\bqualifications?\b",
            r"\bcolleges?\b",
            r"\buniversity\b",
            r"\bschools?\b",
            r"\bpassing\s*years?\b",
            r"\byear\s*of\s*passing\b",
            r"\bboard\b",
        ],
        "context": [
            r"\bgrades?\b",
            r"\bmarks\b",
            r"\bpercentages?\b",
            r"\bcgpa\b",
            r"\bdivisions?\b",
        ],
    },
    TablePurpose.PERSONAL_DATA: {
        "strong": [
            r"\bpersonal\s*details\b",
            r"\bpersonal\s*data\b",
            r"\bseafarer\s*profile\b",
            r"\bparticulars\s*of\s*seafarer\b",
            r"\bbio[- ]?data\b",
            r"\bsurnames?\b",
            r"\bfirst\s*names?\b",
            r"\bmiddle\s*names?\b",
            r"\bdates?\s*of\s*birth\b",
            r"\bplaces?\s*of\s*birth\b",
            r"\bnext\s*of\s*kin\b",
        ],
        "medium": [
            r"\bnationality\b",
            r"\bmarital\s*status\b",
            r"\bblood\s*groups?\b",
            r"\bheight\b",
            r"\bweight\b",
            r"\bhair\s*colors?\b",
            r"\beye\s*colors?\b",
            r"\bcomplexion\b",
            r"\bboiler\s*suit\b",
            r"\bshoe\s*size\b",
        ],
        "context": [
            r"\baddress(?:es)?\b",
            r"\bpermanent\s*address\b",
            r"\bpresent\s*address\b",
            r"\bairports?\b",
            r"\bnearest\s*airport\b",
            r"\breligion\b",
            r"\bgender\b",
            r"\bcivil\s*status\b",
            r"\bpost\s*applied\s*for\b",
        ],
    },
}


def _find_surrounding_heading(
    table: GeometricTable,
    page_blocks: list[SemanticBlockInput],
) -> str:
    """Find the most proximate section heading above the table on the same page."""
    if not table.cells:
        return ""

    t_y0 = min(c.bbox[1] for c in table.cells)
    candidates = [
        b
        for b in page_blocks
        if b.page == table.page and b.bbox[3] <= t_y0 + 5.0 and t_y0 - b.bbox[3] < 120.0
    ]
    if not candidates:
        return ""

    # Sort by vertical distance ascending
    candidates.sort(key=lambda b: t_y0 - b.bbox[3])
    for cand in candidates:
        if cand.suggested_role == "SECTION_HEADING":
            return cand.text.lower()
        text_lower = cand.text.lower()
        if any(
            w in text_lower
            for w in (
                "service",
                "details",
                "experience",
                "education",
                "courses",
                "profile",
                "particulars",
                "documents",
                "qualification",
            )
        ):
            return text_lower

    return ""


def detect_is_form_table(table: GeometricTable) -> bool:
    """Detect if table follows Style B (key-value form structure) rather than tabular grid."""
    if not table.cells:
        return False

    colons_count = sum(1 for c in table.cells if ":" in c.text)
    if colons_count >= 3:
        return True
    if len(table.cells) >= 4 and (colons_count / len(table.cells)) >= 0.2:
        return True

    row0_cells = [c.text.strip().lower() for c in table.cells if c.row_index == 0]
    if any(
        re.search(
            r"\b(surname|date of birth|height|weight|hair color|address|post applied for)\b",
            txt,
        )
        for txt in row0_cells
    ):
        return True

    return False


def infer_table_purpose(
    table: GeometricTable,
    page_blocks: list[SemanticBlockInput],
    surrounding_heading: str | None = None,
) -> tuple[TablePurpose, float, list[str]]:
    """Determine table semantic purpose based on heading, header row, and cell evidence."""
    if not table.cells:
        return TablePurpose.UNKNOWN, 0.0, []

    if surrounding_heading is None:
        surrounding_heading = _find_surrounding_heading(table, page_blocks)
    else:
        surrounding_heading = surrounding_heading.lower()

    r0_cells = [c.text.lower() for c in table.cells if c.row_index == 0]
    r0_text = " ".join(r0_cells)

    is_form = detect_is_form_table(table)
    all_cells = [c.text.lower() for c in table.cells]
    all_text = " ".join(all_cells)

    scores: dict[TablePurpose, float] = {}
    evidence_map: dict[TablePurpose, list[str]] = {}

    for purpose, patterns in _PURPOSE_PATTERNS.items():
        score = 0.0
        ev: list[str] = []

        # 1. Surrounding heading evidence
        for pat in patterns["strong"]:
            m = re.search(pat, surrounding_heading, re.I)
            if m:
                score += 4.5
                ev.append(f"heading:{m.group(0)}")
        for pat in patterns["medium"]:
            m = re.search(pat, surrounding_heading, re.I)
            if m:
                score += 3.0
                ev.append(f"heading:{m.group(0)}")

        # 2. Row 0 (Column Header) evidence
        for pat in patterns["strong"]:
            m = re.search(pat, r0_text, re.I)
            if m:
                score += 4.0
                ev.append(f"r0:{m.group(0)}")
        for pat in patterns["medium"]:
            m = re.search(pat, r0_text, re.I)
            if m:
                score += 2.5
                ev.append(f"r0:{m.group(0)}")

        # 3. Cell evidence (especially for Form tables or content verification)
        for pat in patterns["strong"]:
            m = re.search(pat, all_text, re.I)
            if m and not any(m.group(0) in e for e in ev):
                score += 1.0
                ev.append(f"cell:{m.group(0)}")
        for pat in patterns["medium"]:
            m = re.search(pat, all_text, re.I)
            if m and not any(m.group(0) in e for e in ev):
                score += 0.5
                ev.append(f"cell:{m.group(0)}")

        # 4. Context markers
        for pat in patterns["context"]:
            if re.search(pat, r0_text, re.I) or (is_form and re.search(pat, all_text, re.I)):
                score += 0.5

        scores[purpose] = score
        evidence_map[purpose] = ev

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_purpose, best_score = sorted_scores[0]

    # Conservative threshold: must have at least 2.0 evidence score
    if best_score < 2.0:
        return TablePurpose.UNKNOWN, 0.0, []

    confidence = min(1.0, best_score / 10.0)
    return best_purpose, confidence, evidence_map[best_purpose]


def infer_column_semantics(
    table: GeometricTable,
    purpose: TablePurpose,
) -> list[ColumnSemanticDescriptor]:
    """Infer semantic roles for each column in a tabular grid."""
    descriptors: list[ColumnSemanticDescriptor] = []

    # Group header cells by column_index
    header_cells_by_col: dict[int, list[GeometricCell]] = {}
    for c in table.cells:
        if c.row_index == 0 or c.cell_role == "HEADER":
            header_cells_by_col.setdefault(c.column_index, []).append(c)

    for col_idx in range(table.num_columns):
        col_cells = header_cells_by_col.get(col_idx, [])
        header_text = " ".join(c.text for c in col_cells).strip()
        source_ids = [c.block_id for c in col_cells]

        role = "unknown"
        norm = header_text.lower()

        if purpose == TablePurpose.SEA_SERVICE:
            if re.search(r"\b(rank|capacity|position)\b", norm, re.I):
                role = "rank"
            elif re.search(
                r"\b(vessel\s*name|name\s*of\s*vessel|ship\s*name|ship'?s?\s*name|vsl\s*name|^vessel$|^ship$)\b",
                norm,
                re.I,
            ):
                role = "vessel_name"
            elif re.search(r"\b(owner|owners|company|manager|management|agency)\b", norm, re.I):
                role = "company"
            elif re.search(r"\b(type\s*of\s*vessel|vessel\s*type|ship\s*type|^type$)\b", norm, re.I):
                role = "vessel_type"
            elif re.search(r"\b(flag)\b", norm, re.I):
                role = "flag"
            elif re.search(r"\b(grt|gross\s*tonnage|gt)\b", norm, re.I):
                role = "grt"
            elif re.search(r"\b(dwt|deadweight)\b", norm, re.I):
                role = "dwt"
            elif re.search(r"\b(engine|bhp|kw|propulsion|power|make\s*&\s*model)\b", norm, re.I):
                role = "engine_type"
            elif re.search(r"\b(sign\s*[-–—]?\s*on|date\s*from|from|joined|commenced|embarked)\b", norm, re.I):
                role = "sign_on"
            elif re.search(r"\b(sign\s*[-–—]?\s*off|date\s*to|to|discharged|left|completed|disembarked)\b", norm, re.I):
                role = "sign_off"
            elif re.search(r"\b(duration|period|months|days|time\s*served|total\s*mm/dd)\b", norm, re.I):
                role = "duration"
            elif re.search(r"\b(s\.?\s*no|sr\.?\s*no|no\.?|#)\b", norm, re.I):
                role = "serial_no"

        elif purpose == TablePurpose.EDUCATION:
            if re.search(r"\b(degree|qualification|course|standard|examination|type\s*of\s*degree)\b", norm, re.I):
                role = "degree"
            elif re.search(r"\b(institute|institution|college|school|university|board|name\s*of\s*institute)\b", norm, re.I):
                role = "institution"
            elif re.search(r"\b(from|start|commenced)\b", norm, re.I):
                role = "start_date"
            elif re.search(r"\b(to|end|passing\s*year|year\s*of\s*passing|year)\b", norm, re.I):
                role = "end_date"
            elif re.search(r"\b(grade|marks|percentage|cgpa|division)\b", norm, re.I):
                role = "grade"
            elif re.search(r"\b(s\.?\s*no|sr\.?\s*no|no\.?)\b", norm, re.I):
                role = "serial_no"

        elif purpose == TablePurpose.DOCUMENTS:
            if re.search(r"\b(issued\s*by|issuing\s*authority|place\s*of\s*issue|place|country)\b", norm, re.I):
                role = "issuing_authority"
            elif re.search(r"\b(document|document\s*name|documents?|description)\b", norm, re.I):
                role = "document_name"
            elif re.search(r"\b(number|doc\s*no|certificate\s*no|cert\s*no)\b", norm, re.I):
                role = "document_number"
            elif re.search(r"\b(date\s*of\s*expiry|expiry\s*date|valid\s*until|valid\s*till|d\.?o\.?e)\b", norm, re.I):
                role = "expiry_date"
            elif re.search(r"\b(date\s*of\s*issue|issue\s*date|issued\s*on|issued\s*date)\b", norm, re.I):
                role = "issue_date"
            elif re.search(r"\b(s\.?\s*no|sr\.?\s*no|no\.?)\b", norm, re.I):
                role = "serial_no"

        elif purpose == TablePurpose.COURSES_CERTIFICATIONS:
            if re.search(r"\b(issued\s*by|issuing\s*authority|institute|place)\b", norm, re.I):
                role = "issuing_authority"
            elif re.search(r"\b(course|course\s*name|details\s*of\s*courses|training|subject)\b", norm, re.I):
                role = "course_name"
            elif re.search(r"\b(number|cert\s*no|certificate\s*no)\b", norm, re.I):
                role = "certificate_number"
            elif re.search(r"\b(date\s*of\s*expiry|expiry\s*date|valid\s*until|valid\s*till|d\.?o\.?e)\b", norm, re.I):
                role = "expiry_date"
            elif re.search(r"\b(date\s*of\s*issue|issue\s*date|issued\s*on|issued\s*date)\b", norm, re.I):
                role = "issue_date"
            elif re.search(r"\b(s\.?\s*no|sr\.?\s*no|no\.?)\b", norm, re.I):
                role = "serial_no"

        elif purpose == TablePurpose.PERSONAL_DATA:
            if re.search(r"\b(surname|first\s*name|middle\s*name|name)\b", norm, re.I):
                role = "name"
            elif re.search(r"\b(date\s*of\s*birth|dob)\b", norm, re.I):
                role = "date_of_birth"
            elif re.search(r"\b(place\s*of\s*birth|pob)\b", norm, re.I):
                role = "place_of_birth"
            elif re.search(r"\b(nationality|citizenship)\b", norm, re.I):
                role = "nationality"
            elif re.search(r"\b(marital\s*status)\b", norm, re.I):
                role = "marital_status"
            elif re.search(r"\b(blood\s*group)\b", norm, re.I):
                role = "blood_group"
            elif re.search(r"\b(height)\b", norm, re.I):
                role = "height"
            elif re.search(r"\b(weight)\b", norm, re.I):
                role = "weight"

        descriptors.append(
            ColumnSemanticDescriptor(
                column_index=col_idx,
                header_text=header_text,
                semantic_role=role,
                source_block_ids=source_ids,
            )
        )

    return descriptors


_CONTINUATION_LEADING_PATTERNS = re.compile(
    r"^\s*(?:[a-z]|(?:of|and|or|for|in|to|with|by|at|on|de)\b|\([A-Za-z0-9\s/+\-–—.]+\)|[&/+\-–—,])"
)

_CONTINUATION_TRAILING_PATTERNS = re.compile(
    r"(?:[-/&,:]$|\b(?:and|or|of|in|for|with|by|at|to)\s*$|\b(?:Department|Ministry|Directorate|Board|University|College|Institute|Authority|Academy|Association|Corporation|Limited|Ltd)\s*$)",
    re.IGNORECASE,
)

_SERIAL_NO_PATTERN = re.compile(r"^\s*(?:\d+|[ivxIVX]+|[A-Za-z])[\.\)]\s*$|^\s*#?\d+\s*$")
_TRAILING_PUNCT_WORD = re.compile(r"[/&,\-]\s*\w+\s*$")
_STANDALONE_ACRONYM = re.compile(r"^[A-Z]{2,}(?:\s*/\s*[A-Z]{2,})*$")
_RANK_PREFIX = re.compile(r"\b(?:1st|2nd|3rd|4th|Chief|Deck|Junior|Senior|Trainee)\b", re.IGNORECASE)
_RANK_SUFFIX = re.compile(r"^\s*(?:Officer|Cadet|Engineer|Master|Captain|Rating|Crew)\b", re.IGNORECASE)



def is_logical_row_continuation(
    prev_row: list[CellSemanticDescriptor],
    curr_row: list[CellSemanticDescriptor],
    columns: list[ColumnSemanticDescriptor],
    purpose: TablePurpose,
    is_form_table: bool = False,
) -> bool:
    """Determine whether curr_row is a continuation of prev_row based on multiple deterministic signals."""
    if is_form_table:
        return False

    prev_pop = [c for c in prev_row if c.text.strip()]
    curr_pop = [c for c in curr_row if c.text.strip()]
    if not prev_pop or not curr_pop:
        return False

    # Negative Guard 1: Serial number present
    for c in curr_pop:
        if c.semantic_role == "serial_no":
            return False
        if c.column_index == 0 and _SERIAL_NO_PATTERN.match(c.text):
            is_col_serial = (columns and columns[0].semantic_role == "serial_no")
            clean_num = re.sub(r"\D", "", c.text)
            if is_col_serial or (clean_num and int(clean_num) > 0 and len(c.text.strip()) <= 4):
                return False

    # Negative Guard 2: Full date interval in curr_row (both start/issue and end/expiry dates)
    has_issue = any(c.semantic_role in ("issue_date", "sign_on", "start_date") and c.text.strip() for c in curr_pop)
    has_expiry = any(c.semantic_role in ("expiry_date", "sign_off", "end_date") and c.text.strip() for c in curr_pop)
    if has_issue and has_expiry:
        return False

    # Negative Guard 3: Conflicting distinct certificate/document numbers
    prev_cert = next(
        (c.text.strip() for c in prev_row if c.semantic_role in ("certificate_number", "document_number") and c.text.strip()),
        None,
    )
    curr_cert = next(
        (c.text.strip() for c in curr_row if c.semantic_role in ("certificate_number", "document_number") and c.text.strip()),
        None,
    )
    if prev_cert and curr_cert and prev_cert != curr_cert:
        return False

    # Negative Guard 4: Both rows have distinct primary entity values or distinct dates in matching columns
    curr_has_dates = any(re.search(r"\b(?:\d{1,2}[/-]\w{3}[/-]\d{2,4}|\d{4})\b", c.text) for c in curr_pop)
    for c in curr_pop:
        k = c.column_index
        p_c = prev_row[k]
        p_txt = p_c.text.strip()
        c_txt = c.text.strip()
        if not p_txt or not c_txt:
            continue
        if c.semantic_role == "vessel_name" and curr_has_dates and not _CONTINUATION_TRAILING_PATTERNS.search(p_txt) and not _CONTINUATION_LEADING_PATTERNS.match(c_txt):
            return False
        if c.semantic_role in ("sign_off", "end_date", "expiry_date") and re.search(r"\d{4}", c_txt) and re.search(r"\d{4}", p_txt):
            return False

    # Positive Signal Check
    for c in curr_pop:
        k = c.column_index
        p_c = prev_row[k]
        p_text = p_c.text.strip()
        c_text = c.text.strip()

        # 1. Leading continuation markers in current cell (lowercase, preposition, parenthesized acronym)
        if _CONTINUATION_LEADING_PATTERNS.match(c_text):
            return True

        # 2. Trailing continuation markers in previous cell (punctuation, preposition, organization word)
        if p_text and _CONTINUATION_TRAILING_PATTERNS.search(p_text):
            return True

        # 3. Numeric continuation after trailing dash or slash (e.g. '27-04-' + '2030', 'Ship /' + '43679')
        if p_text and (p_text.endswith("-") or p_text.endswith("/")) and re.match(r"^\d+$", c_text):
            return True

        # 4. Preceding cell has conjunction/slash-word and current cell is a continuation fragment
        # (e.g. 'Survival Craft / Rescue' + 'Boat', 'Personal Survival & Social' + 'Responsibility (PSSR)')
        if (
            p_text
            and _TRAILING_PUNCT_WORD.search(p_text)
            and not re.search(r"[-/]\s*\d{4}\s*$", p_text)
            and not _STANDALONE_ACRONYM.match(c_text)
            and not re.search(r"\d{4}", c_text)
            and len(c_text.split()) <= 3
        ):
            return True

        # 5. Model or numeric wrapping where previous cell ends with slash/dash + digits (e.g. '/200')
        # and current row contains isolated wrapped digits (e.g. '0')
        if (
            p_text
            and re.search(r"[/–\-]\s*\d+\s*$", p_text)
            and re.fullmatch(r"\d{1,2}", c_text)
            and len(curr_pop) == 1
        ):
            return True

        # 6. Rank or designation continuation where previous cell ends with rank prefix (e.g. '3rd', '2nd', 'Deck')
        # and current cell starts with rank noun (e.g. 'Officer', 'Cadet', 'Engineer') without separate dates
        if (
            p_text
            and not curr_has_dates
            and _RANK_PREFIX.search(p_text)
            and _RANK_SUFFIX.match(c_text)
        ):
            return True

    return False


def merge_logical_table_rows(
    rows: list[list[CellSemanticDescriptor]],
    columns: list[ColumnSemanticDescriptor],
    purpose: TablePurpose,
    is_form_table: bool = False,
) -> list[list[CellSemanticDescriptor]]:
    """Merge physical continuation rows into logical records while preserving all source_block_ids."""
    if is_form_table or len(rows) <= 1:
        return rows

    # Row 0 is the table header (when not a form table)
    merged: list[list[CellSemanticDescriptor]] = [rows[0]]

    for r in rows[1:]:
        if len(merged) == 1:
            # First data row cannot continue the header row (row 0)
            merged.append(r)
            continue

        prev_r = merged[-1]
        if is_logical_row_continuation(prev_r, r, columns, purpose, is_form_table=is_form_table):
            new_r: list[CellSemanticDescriptor] = []
            for k in range(len(columns)):
                p_c = prev_r[k]
                c_c = r[k]
                p_txt = p_c.text.strip()
                c_txt = c_c.text.strip()

                if p_txt and c_txt:
                    if (p_txt.endswith("-") and not p_txt.endswith("--")) or (re.search(r"[/–\-]\d+$", p_txt) and c_txt.isdigit()):
                        combined_text = p_txt + c_txt
                    else:
                        combined_text = f"{p_txt} {c_txt}"
                elif p_txt:
                    combined_text = p_txt
                else:
                    combined_text = c_txt

                combined_ids = list(dict.fromkeys(p_c.source_block_ids + c_c.source_block_ids))
                sem_role = p_c.semantic_role if p_c.semantic_role != "unknown" else c_c.semantic_role

                new_r.append(
                    CellSemanticDescriptor(
                        row_index=p_c.row_index,
                        column_index=k,
                        text=combined_text,
                        cell_role=p_c.cell_role,
                        semantic_role=sem_role,
                        source_block_ids=combined_ids,
                    )
                )
            merged[-1] = new_r
        else:
            merged.append(r)

    # Re-index logical row indices deterministically
    reindexed: list[list[CellSemanticDescriptor]] = []
    for l_idx, r in enumerate(merged):
        reindexed_row = [
            CellSemanticDescriptor(
                row_index=l_idx,
                column_index=c.column_index,
                text=c.text,
                cell_role=c.cell_role,
                semantic_role=c.semantic_role,
                source_block_ids=c.source_block_ids,
            )
            for c in r
        ]
        reindexed.append(reindexed_row)

    return reindexed


def build_table_semantic_contexts(
    tables: list[GeometricTable],
    blocks: list[SemanticBlockInput],
) -> list[TableSemanticContext]:
    """Construct full TableSemanticContext objects for all bound tables in a document."""
    contexts: list[TableSemanticContext] = []

    # Map blocks by page
    by_page: dict[int, list[SemanticBlockInput]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    for table in tables:
        page_blocks = by_page.get(table.page, [])
        purpose, confidence, evidence = infer_table_purpose(table, page_blocks)
        is_form = detect_is_form_table(table)
        columns = infer_column_semantics(table, purpose)
        col_role_map = {c.column_index: c.semantic_role for c in columns}

        # Build CellSemanticDescriptors organized by row
        cells_by_row: dict[int, list[GeometricCell]] = {}
        for cell in table.cells:
            cells_by_row.setdefault(cell.row_index, []).append(cell)

        rows: list[list[CellSemanticDescriptor]] = []
        for r_idx in range(table.num_rows):
            row_cells = cells_by_row.get(r_idx, [])
            cells_by_col: dict[int, list[GeometricCell]] = {}
            for cell in row_cells:
                cells_by_col.setdefault(cell.column_index, []).append(cell)

            row_descriptors: list[CellSemanticDescriptor] = []
            for c_idx in range(table.num_columns):
                matched = cells_by_col.get(c_idx, [])
                text = " ".join(c.text for c in matched).strip()
                cell_role = "HEADER" if r_idx == 0 else "DATA"
                sem_role = col_role_map.get(c_idx, "unknown")
                source_ids = [c.block_id for c in matched]

                row_descriptors.append(
                    CellSemanticDescriptor(
                        row_index=r_idx,
                        column_index=c_idx,
                        text=text,
                        cell_role=cell_role,
                        semantic_role=sem_role,
                        source_block_ids=source_ids,
                    )
                )
            rows.append(row_descriptors)

        all_table_block_ids = sorted({c.block_id for c in table.cells})

        logical_rows = merge_logical_table_rows(
            rows,
            columns,
            purpose,
            is_form_table=is_form,
        )

        contexts.append(
            TableSemanticContext(
                table_id=table.table_id,
                page=table.page,
                purpose=purpose,
                confidence=confidence,
                evidence=evidence,
                is_form_table=is_form,
                columns=columns,
                rows=logical_rows,
                physical_rows=rows,
                source_block_ids=all_table_block_ids,
            )
        )

    return contexts


def apply_table_semantics_to_blocks(
    tables: list[TableSemanticContext],
    blocks: list[SemanticBlockInput],
) -> list[SemanticBlockInput]:
    """Populate table_purpose and column_semantic, and enforce role precedence over TECHNOLOGY."""
    if not tables:
        return blocks

    table_map = {t.table_id: t for t in tables}

    enriched_blocks: list[SemanticBlockInput] = []
    for b in blocks:
        if not b.table_id or b.table_id not in table_map:
            enriched_blocks.append(b)
            continue

        ctx = table_map[b.table_id]
        purpose_str = ctx.purpose.value
        col_semantic = None

        if b.column_index is not None:
            col_desc = next((c for c in ctx.columns if c.column_index == b.column_index), None)
            if col_desc:
                col_semantic = col_desc.semantic_role

        # Enforce structural role precedence
        new_role = b.suggested_role

        if b.cell_role == "HEADER":
            new_role = "TABLE_HEADER"
        elif new_role == "TECHNOLOGY":
            if ctx.purpose in (TablePurpose.PERSONAL_DATA, TablePurpose.CONTACT_DETAILS):
                new_role = "PERSONAL"
            else:
                new_role = "DESCRIPTION"

        enriched_blocks.append(
            b.model_copy(
                update={
                    "table_purpose": purpose_str,
                    "column_semantic": col_semantic,
                    "suggested_role": new_role,
                }
            )
        )

    return enriched_blocks
