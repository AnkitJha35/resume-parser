from __future__ import annotations

import json
import re
from pathlib import Path

from app.domain.document import Document, Line, Region
from app.domain.structural import StructuralBlock, StructuralRole
from app.extractors.date_parser import DateRangeParser
from app.pipeline.stages.semantic_paths import SemanticPath, build_semantic_paths


RESOURCE_DIR = Path(__file__).resolve().parents[2] / "resources"
SECTION_ALIASES_PATH = RESOURCE_DIR / "section_aliases.json"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[ \-/.]?)?(?:\(\d{2,4}\)|\d{2,4})[ \-/.]?\d{3,4}[ \-/.]?\d{3,4}"
)
_URL_RE = re.compile(r"(?:https?://|www\.|linkedin\.com/|github\.com/)", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[\u2022\u2023\u25E6\-\*\u00B7●\uf0b7]|\d+\.)\s+")
_BULLET_ONLY_RE = re.compile(r"^[\u2022\u2023\u25E6\-\*\u00B7●.\uf0b7]+$")

_ROLE_WORDS = {
    "assistant",
    "secretary",
    "coordinator",
    "manager",
    "analyst",
    "specialist",
    "developer",
    "engineer",
    "supervisor",
    "administrator",
    "associate",
    "consultant",
    "director",
    "executive",
    "clerk",
    "intern",
    "officer",
    "lead",
    "picker",
    "packer",
    "teacher",
    "professor",
    "physician",
    "investigator",
    "scientist",
    "researcher",
    "clinician",
    "fellow",
    "resident",
    "architect",
    "designer",
}

_ORG_SUFFIXES: tuple[str, ...] = (
    "inc",
    "llc",
    "ltd",
    "corp",
    "corporation",
    "company",
    "co",
    "pvt",
    "private",
)

_INSTITUTIONAL_NOUNS: tuple[str, ...] = (
    "institute",
    "institution",
    "foundation",
    "alliance",
    "forum",
    "laboratories",
    "laboratory",
    "labs",
    "association",
    "society",
    "center",
    "centre",
    "hospital",
    "clinic",
    "university",
    "college",
    "school",
    "academy",
    "agency",
    "firm",
    "group",
)

_ABBREVIATED_ORG_SUFFIXES: set[str] = {"inc", "corp", "co", "ltd", "pvt"}
_GEO_TOKENS = {
    "india",
    "usa",
    "united states",
    "uk",
    "england",
    "canada",
    "delhi",
    "noida",
    "mumbai",
    "pune",
    "bengaluru",
    "bangalore",
    "california",
    "new york",
    "ny",
    "chicago",
    "boston",
    "il",
    "ma",
    "pa",
}

_CREDENTIAL_RE = re.compile(
    r"\b(certif(?:icate|ication)|certified|license|credential|issued|expiry|expires)\b",
    re.IGNORECASE,
)
# Structural date shapes beyond DateRangeParser's token set (e.g. YYYY-MM).
_STRUCTURAL_DATE_RE = re.compile(
    r"^\(?\s*(?:(?:19|20)\d{2}(?:[/-]\d{1,2})?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2})"
    r"\s*[–—-]\s*"
    r"(?:Present|Current|Now|(?:19|20)\d{2}(?:[/-]\d{1,2})?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2})\s*\)?$",
    re.IGNORECASE,
)

_section_alias_normalized: set[str] | None = None


def _normalized_section_aliases() -> set[str]:
    global _section_alias_normalized
    if _section_alias_normalized is None:
        with SECTION_ALIASES_PATH.open("r", encoding="utf-8") as handle:
            aliases = json.load(handle)
        normalized: set[str] = set()
        for names in aliases.values():
            for alias in names:
                normalized.add(re.sub(r"[^a-z0-9]+", "", alias.lower()))
        _section_alias_normalized = normalized
    return _section_alias_normalized


_CANONICAL_SECTION_KEYWORDS: set[str] = {
    "summary",
    "profile",
    "overview",
    "objective",
    "experience",
    "employment",
    "career",
    "work",
    "education",
    "academics",
    "academic",
    "training",
    "postdoctoral",
    "skills",
    "expertise",
    "competencies",
    "projects",
    "coursework",
    "courses",
    "certifications",
    "licensure",
    "licenses",
    "credentials",
    "certificates",
    "honors",
    "awards",
    "achievements",
    "accomplishments",
    "leadership",
    "publications",
    "presentations",
    "research",
    "languages",
    "affiliations",
    "activities",
}


def _contains_section_keyword(text: str) -> bool:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)}
    return bool(tokens & _CANONICAL_SECTION_KEYWORDS)


def build_structural_blocks(document: Document) -> list[StructuralBlock]:
    """Build StructuralBlocks from a layout Document with path/region provenance.

    One Line → one StructuralBlock for Phase 1. Neighbor links are path-local.
    Roles are structural only; no EXPERIENCE/SKILLS/… assignment.
    """
    paths = build_semantic_paths(document)
    region_kinds = _region_kind_index(document)
    blocks: list[StructuralBlock] = []

    for path in paths:
        path_blocks: list[StructuralBlock] = []
        for order, line in enumerate(path.lines):
            path_blocks.append(
                _block_from_line(
                    line=line,
                    path=path,
                    region_kind=region_kinds.get(path.region_id, "unknown"),
                    reading_order=order if line.reading_order is None else line.reading_order,
                    role=StructuralRole.UNKNOWN,
                    role_score=0.0,
                    role_reasons=("pending",),
                )
            )
        assigned = _assign_roles_for_path(path_blocks)
        blocks.extend(_link_neighbors(assigned))

    return blocks


def classify_structural_role(
    text: str,
    *,
    font_size: float | None = None,
    bold: bool | None = None,
    region_kind: str = "unknown",
    previous_text: str | None = None,
    next_text: str | None = None,
    following_texts: tuple[str, ...] = (),
) -> tuple[StructuralRole, float, tuple[str, ...]]:
    """Classify a single unit into a structural role without semantic section labels."""
    value = (text or "").strip()
    if not value:
        return StructuralRole.UNKNOWN, 0.0, ("empty",)

    region_role = _region_kind_role(region_kind)
    if region_role is not None and region_role in {
        StructuralRole.HEADER,
        StructuralRole.FOOTER,
        StructuralRole.SIDEBAR,
        StructuralRole.TABLE_CELL,
    }:
        # Region kind is a strong structural prior for margin/sidebar/table content,
        # but contact-like lines still surface as CONTACT for downstream use.
        if _looks_like_contact(value):
            return StructuralRole.CONTACT, 0.95, ("contact_pattern", f"region:{region_kind}")
        return region_role, 0.9, (f"region:{region_kind}",)

    if _looks_like_contact(value):
        return StructuralRole.CONTACT, 0.95, ("contact_pattern",)

    if (
        DateRangeParser.parse(value) is not None
        or _parenthesized_date(value)
        or _STRUCTURAL_DATE_RE.match(value)
    ):
        return StructuralRole.DATE, 1.0, ("date_pattern",)

    if _BULLET_ONLY_RE.match(value) or _BULLET_RE.match(value):
        return StructuralRole.BULLET, 1.0, ("bullet_marker",)

    follow = tuple(t for t in ((next_text,) if next_text else ()) + following_texts if t and t.strip())

    if _is_known_section_alias(value):
        return StructuralRole.SECTION_HEADING, 1.0, ("known_section_alias",)

    if _looks_like_section_heading(value, font_size=font_size, bold=bold, following=follow):
        return StructuralRole.SECTION_HEADING, 0.8, ("section_boundary_geometry",)

    if _looks_like_entry_title(value, font_size=font_size, bold=bold, following=follow):
        return StructuralRole.ENTRY_TITLE, 0.8, ("entry_title_structure",)

    if _CREDENTIAL_RE.search(value):
        return StructuralRole.CREDENTIAL, 0.85, ("credential_pattern",)

    if _looks_like_organization(value):
        return StructuralRole.ORGANIZATION, 0.85, ("organization_pattern",)

    if _looks_like_location(value):
        return StructuralRole.LOCATION, 0.8, ("location_pattern",)

    if _is_wrapped_description(previous_text, value, font_size=font_size, bold=bold):
        return StructuralRole.DESCRIPTION, 0.75, ("wrapped_continuation",)

    if len(value) > 80:
        return StructuralRole.DESCRIPTION, 0.7, ("long_text",)
        return StructuralRole.ENTRY_TITLE, 0.8, ("entry_title_structure",)

    if _looks_like_technology(value, following=follow, previous_text=previous_text):
        return StructuralRole.TECHNOLOGY, 0.55, ("short_list_technology_candidate",)

    return StructuralRole.UNKNOWN, 0.0, ("no_strong_signal",)


def _assign_roles_for_path(blocks: list[StructuralBlock]) -> list[StructuralBlock]:
    texts = [block.text for block in blocks]
    assigned: list[StructuralBlock] = []
    for index, block in enumerate(blocks):
        previous_text = texts[index - 1] if index > 0 else None
        next_text = texts[index + 1] if index + 1 < len(texts) else None
        following = tuple(texts[index + 1 : index + 4])
        role, score, reasons = classify_structural_role(
            block.text,
            font_size=block.style.font_size,
            bold=block.style.bold,
            region_kind=block.region_kind,
            previous_text=previous_text,
            next_text=next_text,
            following_texts=following,
        )
        assigned.append(
            StructuralBlock(
                block_id=block.block_id,
                text=block.text,
                line_ids=block.line_ids,
                source_span_ids=block.source_span_ids,
                page_number=block.page_number,
                region_id=block.region_id,
                region_kind=block.region_kind,
                path_id=block.path_id,
                bbox=block.bbox,
                style=block.style,
                reading_order=block.reading_order,
                reconstruction_method=block.reconstruction_method,
                role=role,
                role_score=score,
                role_reasons=reasons,
            )
        )
    return assigned


def _link_neighbors(blocks: list[StructuralBlock]) -> list[StructuralBlock]:
    linked: list[StructuralBlock] = []
    for index, block in enumerate(blocks):
        linked.append(
            StructuralBlock(
                block_id=block.block_id,
                text=block.text,
                line_ids=block.line_ids,
                source_span_ids=block.source_span_ids,
                page_number=block.page_number,
                region_id=block.region_id,
                region_kind=block.region_kind,
                path_id=block.path_id,
                bbox=block.bbox,
                style=block.style,
                reading_order=block.reading_order,
                reconstruction_method=block.reconstruction_method,
                role=block.role,
                role_score=block.role_score,
                role_reasons=block.role_reasons,
                previous_block_id=blocks[index - 1].block_id if index > 0 else None,
                next_block_id=blocks[index + 1].block_id if index + 1 < len(blocks) else None,
            )
        )
    return linked


def _block_from_line(
    *,
    line: Line,
    path: SemanticPath,
    region_kind: str,
    reading_order: int,
    role: StructuralRole,
    role_score: float,
    role_reasons: tuple[str, ...],
) -> StructuralBlock:
    return StructuralBlock(
        block_id=f"{path.path_id}:{line.line_id}",
        text=line.text,
        line_ids=(line.line_id,),
        source_span_ids=tuple(line.source_span_ids),
        page_number=line.page_number,
        region_id=path.region_id,
        region_kind=region_kind,
        path_id=path.path_id,
        bbox=line.bbox,
        style=line.style,
        reading_order=reading_order,
        reconstruction_method=line.reconstruction_method,
        role=role,
        role_score=role_score,
        role_reasons=role_reasons,
    )


def _region_kind_index(document: Document) -> dict[str, str]:
    return {
        region.region_id: region.kind
        for page in document.pages
        for region in page.regions
    }


def _region_kind_role(region_kind: str) -> StructuralRole | None:
    mapping = {
        "header": StructuralRole.HEADER,
        "footer": StructuralRole.FOOTER,
        "sidebar": StructuralRole.SIDEBAR,
        "table": StructuralRole.TABLE_CELL,
        "table_cell": StructuralRole.TABLE_CELL,
    }
    return mapping.get((region_kind or "").lower())


def _looks_like_contact(text: str) -> bool:
    if _EMAIL_RE.search(text) or _URL_RE.search(text):
        return True
    match = _PHONE_RE.search(re.sub(r"\s", " ", text))
    if match and len(re.sub(r"[^0-9]", "", match.group(0))) >= 7:
        return True
    return False


def _parenthesized_date(text: str) -> bool:
    match = re.fullmatch(r"\(\s*(.+?)\s*\)", text.strip())
    return bool(match and DateRangeParser.parse(match.group(1)) is not None)


def _looks_like_organization(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    words = value.split()
    if not (1 <= len(words) <= 10):
        return False

    if _contains_role_word(value):
        return False

    last_word_clean = re.sub(r"[^A-Za-z0-9]", "", words[-1]).lower()

    if value.endswith((".", "!", "?", ";", ":")):
        if not (value.endswith(".") and last_word_clean in _ABBREVIATED_ORG_SUFFIXES):
            return False

    has_corp_suffix = last_word_clean in _ORG_SUFFIXES or any(
        re.sub(r"[^A-Za-z0-9]", "", w).lower() in {"inc", "llc", "ltd", "corp", "corporation"}
        for w in words
    )

    has_institutional_noun = last_word_clean in _INSTITUTIONAL_NOUNS

    has_academic_or_medical_org = any(
        re.sub(r"[^A-Za-z0-9]", "", w).lower() in {"university", "college", "institute", "hospital"}
        for w in words
    )

    if has_corp_suffix or has_institutional_noun or has_academic_or_medical_org:
        upper_like = value.upper() == value and any(ch.isalpha() for ch in value)
        content_words = [w for w in words if w.lower() not in {"and", "of", "the", "for", "&", "in", "at"}]
        all_capitalized = bool(content_words) and all(
            re.sub(r"^[^A-Za-z0-9]+", "", w)[:1].isupper() for w in content_words if re.sub(r"^[^A-Za-z0-9]+", "", w)
        )
        if upper_like or all_capitalized:
            return True

    if "&" in value and len(words) <= 6:
        if _is_known_section_alias(value) or _contains_section_keyword(value):
            return False
        content_words = [w for w in words if w not in {"&", "and", "of", "the"}]
        if content_words and all(w[:1].isupper() for w in content_words):
            return True

    return False


def _looks_like_location(text: str) -> bool:
    lowered = text.lower().strip()
    if lowered.startswith("location:"):
        return True
    if "," not in text:
        return False
    return any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in _GEO_TOKENS)


def _is_known_section_alias(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    return bool(normalized) and normalized in _normalized_section_aliases()


def _has_typography_emphasis(font_size: float | None, bold: bool | None) -> bool:
    # Body text is often ~11–12pt; section headings need stronger cues than a
    # bare 12pt threshold (bold, all-caps handled by callers, or larger type).
    return bool(bold) or (font_size is not None and font_size >= 13.0)


def _looks_like_section_heading(
    text: str,
    *,
    font_size: float | None,
    bold: bool | None,
    following: tuple[str, ...],
) -> bool:
    value = text.strip()
    words = value.split()
    if not value or not (1 <= len(words) <= 6) or len(value) > 60:
        return False
    if any(ch.isdigit() for ch in value):
        return False
    if value.endswith((".", ";", ",", ":", "!", "?")):
        return False
    if _looks_like_contact(value):
        return False
    if _contains_role_word(value):
        return False
    if _looks_like_organization(value):
        return False

    last_word_clean = re.sub(r"[^A-Za-z0-9]", "", words[-1]).lower()
    if last_word_clean in _INSTITUTIONAL_NOUNS:
        return False

    upper_like = value.upper() == value and any(ch.isalpha() for ch in value)
    if not _has_typography_emphasis(font_size, bold) and not upper_like:
        return False

    # Known section alias (single-word or multi-word)
    if _is_known_section_alias(value):
        return True

    # Compound section heading (joined by '&', 'and', or '/') containing section keywords
    # e.g. "HONORS & LEADERSHIP", "EDUCATION & TRAINING", "CLINICAL & PHARMACEUTICAL EXPERIENCE",
    #      "ACADEMIC & OPEN-SOURCE PROJECTS", "BOARD CERTIFICATIONS & LICENSURE"
    is_compound = bool(re.search(r"\b(?:&|and)\b|/", value, re.IGNORECASE))
    if is_compound and _contains_section_keyword(value):
        return True

    # Qualified section heading without coordinator:
    # Anchor keyword must appear as head noun (last word) or leading category keyword
    # e.g. "PROFESSIONAL EXPERIENCE", "TECHNICAL SKILLS", "ACADEMIC PROJECTS",
    #      "RELEVANT COURSEWORK", "ACADEMIC APPOINTMENTS", "PROJECT HIGHLIGHTS"
    first_word_clean = re.sub(r"[^A-Za-z0-9]", "", words[0]).lower()
    if last_word_clean in _CANONICAL_SECTION_KEYWORDS or first_word_clean in _CANONICAL_SECTION_KEYWORDS:
        return True

    # Section boundary geometry fallback: following content looks like a list / short competency cluster
    if _following_looks_like_entry_metadata(following):
        return False
    if not following:
        return False
    return _following_looks_like_section_body(following)


def _looks_like_entry_title(
    text: str,
    *,
    font_size: float | None,
    bold: bool | None,
    following: tuple[str, ...],
) -> bool:
    value = text.strip()
    words = value.split()
    if not value or len(words) > 6 or len(value) > 60:
        return False
    if any(ch.isdigit() for ch in value):
        return False
    if _is_known_section_alias(value):
        return False
    if value.endswith((".", ";", ":")):
        return False

    has_role = _contains_role_word(value)
    if not has_role and _contains_section_keyword(value):
        return False
    if not has_role and _looks_like_organization(value):
        return False
    title_case_or_upper = value.upper() == value or all(w[:1].isupper() for w in words if w and w[0].isalpha())
    typography = _has_typography_emphasis(font_size, bold) or title_case_or_upper
    if not typography and not has_role:
        return False

    if _following_looks_like_entry_metadata(following):
        return True
    if has_role and typography and len(words) <= 5:
        return True
    return False


def _following_looks_like_entry_metadata(following: tuple[str, ...]) -> bool:
    texts = [item.strip() for item in following if item and item.strip()]
    if not texts:
        return False
    has_date = any(
        DateRangeParser.parse(item) is not None
        or _STRUCTURAL_DATE_RE.match(item.strip())
        or re.search(r"\b(?:19|20)\d{2}\b", item)
        or _parenthesized_date(item)
        for item in texts
    )
    has_org_or_location = any(
        _looks_like_organization(item)
        or _looks_like_location(item)
        or "," in item
        or " / " in item
        for item in texts
    )
    return has_date and has_org_or_location


def _following_looks_like_section_body(following: tuple[str, ...]) -> bool:
    texts = [item.strip() for item in following if item and item.strip()]
    if not texts:
        return False
    if _following_looks_like_entry_metadata(texts):
        return False
    # Degree / school stacks are entry-like, not section bodies.
    if any(
        DateRangeParser.parse(item) is not None
        or _STRUCTURAL_DATE_RE.match(item)
        or re.search(r"\b(?:19|20)\d{2}\b", item)
        or re.search(r"\b(?:university|college|institute|academy|school)\b", item, re.I)
        or _looks_like_organization(item)
        for item in texts
    ):
        return False
    bulletish = sum(1 for item in texts if _BULLET_RE.match(item) or _BULLET_ONLY_RE.match(item) or len(item.split()) <= 4)
    short_lines = sum(1 for item in texts if len(item.split()) <= 5 and not DateRangeParser.parse(item))
    return bulletish >= 1 or short_lines >= 2


def _contains_role_word(text: str) -> bool:
    words = {token.lower() for token in re.findall(r"[A-Za-z]+", text)}
    return bool(words & _ROLE_WORDS)


def _is_wrapped_description(
    previous_text: str | None,
    current: str,
    *,
    font_size: float | None,
    bold: bool | None,
) -> bool:
    if not previous_text:
        return False
    previous = previous_text.strip()
    value = current.strip()
    if not previous or not value:
        return False
    if previous.endswith((".", "!", "?", ":", ";")):
        return False
    if _is_known_section_alias(value):
        return False
    if value.upper() == value and any(ch.isalpha() for ch in value) and len(value.split()) <= 6:
        return False
    if not (value[:1].islower() or len(previous.split()) >= 5):
        return False
    if DateRangeParser.parse(value) is not None or _looks_like_organization(value):
        return False
    return True


def _looks_like_technology(
    text: str,
    *,
    following: tuple[str, ...],
    previous_text: str | None,
) -> bool:
    value = text.strip()
    words = value.split()
    if not (1 <= len(words) <= 3) or len(value) > 40:
        return False
    if any(ch.isdigit() for ch in value):
        return False
    if _contains_role_word(value) or _is_known_section_alias(value):
        return False
    # Only in list-like neighborhoods to avoid labeling arbitrary short nouns.
    neighbors = [item for item in ((previous_text,) if previous_text else ()) + following if item]
    short_neighbors = sum(1 for item in neighbors if 1 <= len(item.split()) <= 3)
    return short_neighbors >= 1
