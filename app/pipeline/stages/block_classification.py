from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from app.pipeline.stages.normalization import TextNormalizer

RESOURCE_DIR = Path(__file__).resolve().parents[2] / "resources"
JOB_TITLES_PATH = RESOURCE_DIR / "job_titles.json"
DEGREES_PATH = RESOURCE_DIR / "degrees.json"
SECTION_ALIASES_PATH = RESOURCE_DIR / "section_aliases.json"


@dataclass
class ClassifiedBlock:
    original: Any
    label: str
    score: float
    reasons: List[str]


def _load_json(path: Path):
    try:
        import json

        with path.open("r", encoding="utf-8") as h:
            return json.load(h)
    except Exception:
        return {}


# Try to use existing DateRangeParser if available, else fallback to regex
try:
    from app.extractors.date_parser import DateRangeParser

    def _is_date(text: str) -> bool:
        return DateRangeParser.parse(text) is not None

except Exception:
    DATE_RE = re.compile(r"\b\d{4}\b|\b\d{1,2}[/-]\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", re.I)

    def _is_date(text: str) -> bool:
        return bool(DATE_RE.search(text))


# Load resources once
_JOB_TITLES = None
_DEGREES = None


def _ensure_resources():
    global _JOB_TITLES, _DEGREES
    if _JOB_TITLES is None:
        jt = _load_json(JOB_TITLES_PATH)
        # job_titles.json may be a list
        if isinstance(jt, dict):
            _JOB_TITLES = [k.lower() for k in jt.keys()]
        elif isinstance(jt, list):
            _JOB_TITLES = [t.lower() for t in jt]
        else:
            _JOB_TITLES = []
    if _DEGREES is None:
        dg = _load_json(DEGREES_PATH)
        if isinstance(dg, dict):
            _DEGREES = [k.lower() for k in dg.keys()]
        elif isinstance(dg, list):
            _DEGREES = [d.lower() for d in dg]
        else:
            _DEGREES = []
    # load section aliases
    if not hasattr(_ensure_resources, "_sections"):
        try:
            sec = _load_json(SECTION_ALIASES_PATH)
            _ensure_resources._sections = {alias.lower() for names in sec.values() for alias in names} if isinstance(sec, dict) else set()
        except Exception:
            _ensure_resources._sections = set()



def classify_block(block: Any) -> ClassifiedBlock:
    """Conservative classification of a TextBlock-like object.

    The `block` only needs attributes: `text`, `x0`, `y0`, `x1`, `y1`, `page_number`,
    `font_size`, `bold`.
    """
    _ensure_resources()
    text = getattr(block, "text", "") or ""
    text_stripped = TextNormalizer.normalize_text(text).strip()
    reasons: List[str] = []

    # Empty
    if not text_stripped:
        return ClassifiedBlock(original=block, label="UNKNOWN", score=0.0, reasons=["empty"])

    # DATE: use parser or regex
    if _is_date(text_stripped):
        reasons.append("date_pattern")
        return ClassifiedBlock(original=block, label="DATE", score=1.0, reasons=reasons)

    parenthesized_date = re.fullmatch(r"\(\s*(.+?)\s*\)", text_stripped)
    if parenthesized_date and _is_date(parenthesized_date.group(1)):
        reasons.append("parenthesized_date_pattern")
        return ClassifiedBlock(original=block, label="DATE", score=1.0, reasons=reasons)

    lowered = text_stripped.lower()

    # BULLET: require explicit bullet characters or numeric list markers
    if re.match(r"^\s*(?:[\u2022\u2023\u25E6\-\*\u00B7]|\d+\.)\s+", text_stripped):
        reasons.append("bullet_marker")
        return ClassifiedBlock(original=block, label="BULLET", score=1.0, reasons=reasons)

    # DEGREE
    for deg in _DEGREES:
        if re.match(rf"^{re.escape(deg)}(?:$|\s+(?:in|of)\b|[:(/.-])", lowered):
            reasons.append("degree_dict")
            return ClassifiedBlock(original=block, label="DEGREE", score=1.0, reasons=reasons)

    if re.search(r"\bdegree\b", lowered) and "/" in text_stripped and len(text_stripped) <= 60:
        reasons.append("degree_placeholder_pattern")
        return ClassifiedBlock(original=block, label="DEGREE", score=0.8, reasons=reasons)

    # SECTION HEADER
    if text_stripped.strip().lower() in getattr(_ensure_resources, "_sections", set()):
        reasons.append("section_header")
        return ClassifiedBlock(original=block, label="SECTION_HEADER", score=1.0, reasons=reasons)

    # JOB_TITLE: exact dictionary match
    if lowered in _JOB_TITLES:
        reasons.append("job_title_dict_exact")
        return ClassifiedBlock(original=block, label="JOB_TITLE", score=1.0, reasons=reasons)

    # JOB_TITLE: heuristic - contains known title words AND typographic emphasis
    title_keywords = ["engineer", "developer", "manager", "director", "intern", "sde", "software"]
    if any(k in lowered for k in title_keywords):
        # require some typographic cue or short length to be conservative
        if getattr(block, "bold", False) or (getattr(block, "font_size", None) and getattr(block, "font_size") >= 11):
            reasons.append("title_keyword+typography")
            return ClassifiedBlock(original=block, label="JOB_TITLE", score=0.7, reasons=reasons)

    # COMPANY: look for company suffixes but require more evidence
    company_suffixes = ["inc", "llc", "ltd", "corp", "corporation", "company", "pvt.", "pvt", "private"]
    has_suffix = any(re.search(rf"\b{re.escape(s)}\b", lowered) for s in company_suffixes)
    # require either multiple capitalized words or punctuation like '!' or '&'
    capitals = sum(1 for w in text_stripped.split() if w[:1].isupper())
    if has_suffix and (capitals >= 2 or '!' in text_stripped or '&' in text_stripped):
        reasons.append("company_suffix+capitalization")
        score = 0.9
        if capitals <= 1:
            score = 0.7
            reasons.append("weak_capitalization")
        return ClassifiedBlock(original=block, label="COMPANY", score=score, reasons=reasons)

    # Conservative company detection: do not classify based solely on capitalization/punctuation
    # However, treat clearly branded names with exclamation/punct as likely companies
    if '!' in text_stripped and capitals >= 2:
        reasons.append("branding_exclamation+capitalization")
        return ClassifiedBlock(original=block, label="COMPANY", score=0.7, reasons=reasons)

    # INSTITUTION
    if re.search(r"\b(university|college|institute|academy|school|polytechnic)\b", lowered):
        reasons.append("institution_keyword")
        return ClassifiedBlock(original=block, label="INSTITUTION", score=0.9, reasons=reasons)

    # LOCATION: require explicit label or geographic tokens
    geo_tokens = {"india", "usa", "united states", "uk", "england", "canada", "delhi", "noida", "kerala", "mumbai", "pune", "bengaluru", "bangalore", "california", "new york", "ny"}
    if "location:" in lowered or "based in" in lowered:
        reasons.append("location_label")
        return ClassifiedBlock(original=block, label="LOCATION", score=1.0, reasons=reasons)
    if "," in text_stripped:
        if any(re.search(rf"\b{re.escape(tok)}\b", lowered) for tok in geo_tokens):
            reasons.append("geo_token+comma")
            return ClassifiedBlock(original=block, label="LOCATION", score=0.9, reasons=reasons)


    # PROJECT_TITLE: avoid forcing; remain UNKNOWN for now (grouping will resolve)

    # DESCRIPTION: long paragraph
    if len(text_stripped) > 80:
        reasons.append("long_text")
        return ClassifiedBlock(original=block, label="DESCRIPTION", score=0.6, reasons=reasons)

    # Fallback: UNKNOWN (conservative)
    return ClassifiedBlock(original=block, label="UNKNOWN", score=0.0, reasons=["no_strong_signal"])


def classify_blocks(blocks: Iterable[Any]) -> List[ClassifiedBlock]:
    return [classify_block(b) for b in blocks]
