from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, List
from app.pipeline.stages.candidate_grouping import CandidateGroup

from app.pipeline.stages.confidence import ConfidenceScorer
from app.extractors.date_parser import DateRangeParser
from app.pipeline.stages.normalization import TextNormalizer

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
DEGREES_PATH = RESOURCE_DIR / "degrees.json"


class EducationExtractor:
    def __init__(self) -> None:
        self._degree_map = self._load_degree_map()
        self._degree_pattern = self._build_degree_pattern()
        self._date_parser = DateRangeParser()
        self._confidence = ConfidenceScorer()

    def _load_degree_map(self) -> dict[str, str]:
        with open(DEGREES_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_degree_pattern(self) -> re.Pattern[str]:
        prefixes = sorted(self._degree_map.keys(), key=lambda alias: -len(alias))
        escaped = [re.escape(prefix) for prefix in prefixes]
        generic_prefixes = [
            r"bachelor of [a-z &()]+",
            r"master of [a-z &()]+",
        ]
        return re.compile(
            rf"^(?:{'|'.join(escaped + generic_prefixes)})(?:\s+(?:in\b|of\b)|\s*\(|\s*\||\s*$)",
            re.IGNORECASE,
        )

    def _extract_from_lines(self, lines: list[str]) -> list[EducationEntry]:
        """Original line-oriented extraction logic extracted to a helper."""
        education_entries: list[EducationEntry] = []
        current_entry: dict[str, Any] | None = None

        for line in lines:
            normalized = TextNormalizer.normalize_text(line).strip()
            if not normalized:
                continue

            if self._is_degree_line(normalized):
                if current_entry:
                    education_entries.append(self._build_entry(current_entry))
                degree_value, institution, field_of_study = self._extract_degree_line(normalized)
                current_entry = {
                    "degree": degree_value,
                    "institution": institution,
                    "fieldOfStudy": field_of_study,
                    "startDate": None,
                    "endDate": None,
                    "grade": None,
                }
                continue

            if current_entry is None:
                continue

            single_date = self._parse_parenthesized_date(normalized)
            if single_date:
                current_entry["startDate"] = single_date
                continue

            if self._contains_date_range(normalized):
                date_range = self._date_parser.parse(normalized)
                if date_range:
                    current_entry["startDate"] = date_range.startDate
                    current_entry["endDate"] = date_range.endDate or "Present"
                continue

            if self._looks_like_institution(normalized) and current_entry["institution"] is None:
                current_entry["institution"] = normalized
                continue

            if self._looks_like_grade(normalized) and current_entry["grade"] is None:
                current_entry["grade"] = normalized
                continue

            if current_entry["fieldOfStudy"] is None and self._looks_like_field_of_study(normalized):
                current_entry["fieldOfStudy"] = normalized
                continue

            if current_entry["institution"] is None and self._is_fallback_institution_candidate(normalized, current_entry):
                current_entry["institution"] = self._normalize_institution(normalized)
                continue

        if current_entry:
            education_entries.append(self._build_entry(current_entry))

        return education_entries

    def extract(self, lines: Optional[list[str]] = None, groups: Optional[list[CandidateGroup]] = None) -> list[EducationEntry]:
        """Accept either raw lines or candidate groups. For groups, extract per-group."""
        results: list[EducationEntry] = []
        if groups:
            # Prefer a group-aware extraction path that uses classification labels
            # as hints when the original line-oriented matcher fails to find a degree.
            return self._extract_from_groups(groups)

        if not lines:
            return []

        return self._extract_from_lines(lines)

    def _extract_from_groups(self, groups: list[CandidateGroup]) -> list[EducationEntry]:
        results: list[EducationEntry] = []
        for group in groups:
            entry: dict[str, Any] = {
                "degree": None,
                "institution": None,
                "fieldOfStudy": None,
                "startDate": None,
                "endDate": None,
                "grade": None,
            }

            # Process blocks in visual order and use labels as hints
            for cb in group.blocks:
                text = (cb.original.text or "").strip()
                if not text:
                    continue

                normalized_text = TextNormalizer.normalize_text(text)

                # Date
                date_range = self._date_parser.parse(normalized_text)
                if date_range and entry["startDate"] is None:
                    entry["startDate"] = date_range.startDate
                    entry["endDate"] = date_range.endDate or "Present"
                    continue

                single_date = self._parse_parenthesized_date(normalized_text)
                if single_date and entry["startDate"] is None:
                    entry["startDate"] = single_date
                    continue

                lbl = str(getattr(cb, "label", "")).upper()
                if lbl == "DEGREE" and entry["degree"] is None:
                    trailing_year = re.fullmatch(r"(?P<degree>.+?)\s*[-–—]\s*(?P<year>\d{4})", text)
                    if trailing_year:
                        self._apply_degree_text(entry, trailing_year.group("degree").strip())
                        entry["startDate"] = trailing_year.group("year")
                        entry["_preserve_degree_text"] = True
                    else:
                        self._apply_degree_text(entry, text)
                    continue
                if lbl == "INSTITUTION" and entry["institution"] is None:
                    # Pipe lines may embed DEGREE (FIELD) | INSTITUTION in one block.
                    if "|" in text and entry["degree"] is None and self._is_degree_line(text.split("|", 1)[0].strip()):
                        self._apply_degree_text(entry, text)
                    else:
                        entry["institution"] = self._split_institution_location(text)[0]
                    continue

                if entry["grade"] is None and self._looks_like_grade(text):
                    entry["grade"] = text
                    continue

                # Unknown label: use existing line heuristics
                if lbl == "UNKNOWN":
                    if entry["degree"] is None and self._is_degree_line(text):
                        self._apply_degree_text(entry, text)
                        continue
                    if entry["institution"] is None and self._looks_like_institution(text):
                        entry["institution"] = text
                        continue

            # Fallback: if no degree but institution present, set degree to first non-date block text
            if entry["degree"] is None:
                for cb in group.blocks:
                    t = (cb.original.text or "").strip()
                    if t and not self._contains_date_range(t):
                        # Try to split compact 'degree institution' lines
                        deg, inst, field = self._extract_degree_line(t)
                        if inst is not None or field is not None:
                            entry["degree"] = deg
                            if inst is not None:
                                entry["institution"] = inst
                            if field is not None:
                                entry["fieldOfStudy"] = field
                        else:
                            entry["degree"] = t
                        break

            # Post-process: if degree contains both degree+institution on one
            # line, try to split it into degree and institution
            if entry["degree"] and entry["institution"] is None:
                deg_text = entry["degree"]
                deg, inst, field = self._extract_degree_line(deg_text)
                if inst is not None:
                    entry["degree"] = deg
                    entry["institution"] = inst
                    if field and not entry.get("fieldOfStudy"):
                        entry["fieldOfStudy"] = field

            # Peel parenthetical specialization from degree before normalization.
            if entry["degree"] and not entry.get("fieldOfStudy"):
                deg, _inst, field = self._extract_degree_line(entry["degree"])
                if field:
                    entry["degree"] = deg
                    entry["fieldOfStudy"] = field

            # Fallback: if institution is still None in candidate group, check
            # remaining unconsumed blocks for an institution/location line.
            if entry["institution"] is None:
                for cb in group.blocks:
                    t = (cb.original.text or "").strip()
                    if self._is_fallback_institution_candidate(t, entry):
                        entry["institution"] = self._normalize_institution(t)
                        break

            results.append(self._build_entry(entry))

        return results

    def _apply_degree_text(self, entry: dict[str, Any], text: str) -> None:
        """Assign degree (+ optional field/institution) before canonicalization."""
        deg, inst, field = self._extract_degree_line(text)
        entry["degree"] = deg
        if field and not entry.get("fieldOfStudy"):
            entry["fieldOfStudy"] = field
        if inst and not entry.get("institution"):
            entry["institution"] = inst

    def _is_degree_line(self, text: str) -> bool:
        return bool(self._degree_pattern.match(text))

    def _extract_degree_line(self, text: str) -> tuple[str, str | None, str | None]:
        institution = None
        if "|" in text:
            degree_text, institution_text = (part.strip() for part in text.split("|", 1))
            institution = self._normalize_institution(institution_text)
        else:
            degree_text = text

        field_of_study = None
        paren = re.fullmatch(
            r"(?P<degree>.+?)\s*\((?P<field>[^)]+)\)\s*",
            degree_text.strip(),
        )
        if paren and self._is_parenthetical_specialization(paren.group("field")):
            degree_text = paren.group("degree").strip()
            field_of_study = paren.group("field").strip()

        # Compact degree+institution on one line (e.g. "M.C.A NIT Calicut").
        # Do not treat a lone parenthetical remainder as an institution.
        if institution is None and "|" not in text:
            m = re.match(
                r"^(?P<deg>[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)+)\s+(?P<inst>.+)$",
                degree_text,
            )
            if not m:
                m = re.match(r"^(?P<deg>[A-Z]{2,6})\s+(?P<inst>.+)$", degree_text)

            if m:
                inst_candidate = m.group("inst").strip()
                if re.fullmatch(r"\([^)]*\)", inst_candidate):
                    pass
                elif not re.match(r"^(in|of)\b", inst_candidate, re.I):
                    degree_text = m.group("deg").strip()
                    institution = self._normalize_institution(inst_candidate)

        if field_of_study is None:
            match = re.match(r"^(?P<degree>.+?)\s+in\s+(?P<field>.+)$", degree_text, re.I)
            if match:
                degree_text = match.group("degree").strip()
                field_of_study = match.group("field").strip()

        return degree_text, institution, field_of_study

    def _is_parenthetical_specialization(self, text: str) -> bool:
        """True when parentheses look like a field/specialization, not honors/date/school."""
        value = (text or "").strip()
        if len(value) < 2 or not re.search(r"[A-Za-z]", value):
            return False
        if re.fullmatch(r"hons\.?|honou?rs\.?", value, re.IGNORECASE):
            return False
        if re.fullmatch(r"general", value, re.IGNORECASE):
            return False
        if re.search(r"\b(?:hons\.?|honou?rs)\b", value, re.IGNORECASE) and len(value.split()) <= 2:
            return False
        if re.fullmatch(r"[\d\s./\-–—]+", value):
            return False
        if re.search(r"\b(?:19|20)\d{2}\b", value) and re.search(r"[-–—/]", value):
            return False
        if re.fullmatch(r"(?:19|20)\d{2}", value):
            return False
        if self._contains_date_range(value):
            return False
        if self._looks_like_institution(value):
            return False
        return True

    def _contains_date_range(self, text: str) -> bool:
        return self._date_parser.parse(TextNormalizer.normalize_text(text)) is not None

    def _parse_parenthesized_date(self, text: str) -> str | None:
        normalized = TextNormalizer.normalize_text(text)
        match = re.fullmatch(r"\(\s*(?P<date>[A-Za-z]+\s+\d{4}|\d{4})\s*\)", normalized)
        if not match:
            return None
        return DateRangeParser._parse_date_token(match.group("date"))

    def _looks_like_institution(self, text: str) -> bool:
        return bool(re.search(r"\b(university|college|institute|school|academy|polytechnic)\b", text, re.I))

    def _looks_like_grade(self, text: str) -> bool:
        return bool(re.search(r"\b(grade|cgpa|gpa|percentage|distinction|honors|honours|marks|cum\s+laude|graduated?)\b", text, re.I))

    def _normalize_institution(self, text: str) -> str:
        text = re.sub(r"\s*,\s*", ", ", text.strip())
        return " ".join(text.split())

    def _split_institution_location(self, text: str) -> tuple[str, str | None]:
        match = re.fullmatch(
            r"(?P<institution>.+?)\s*[–—-]\s*(?P<location>[A-Za-z][A-Za-z .'-]*(?:,\s*[A-Za-z][A-Za-z .'-]*)+)",
            text.strip(),
        )
        if not match:
            return text, None
        return self._normalize_institution(match.group("institution")), self._normalize_institution(match.group("location"))

    def _looks_like_field_of_study(self, text: str) -> bool:
        return bool(re.search(r"\b(computer science|information technology|electronics|mechanical|civil|business administration|commerce|finance|mathematics|physics|data science|machine learning)\b", text, re.I))

    def _is_fallback_institution_candidate(self, text: str, entry: dict[str, Any]) -> bool:
        value = text.strip()
        if not value or len(value.split()) > 8 or len(value) > 60:
            return False
        if self._contains_date_range(value) or self._parse_parenthesized_date(value):
            return False
        if self._looks_like_grade(value) or self._is_degree_line(value):
            return False
        if self._looks_like_field_of_study(value):
            return False
        if value.startswith(("•", "-", "*", "\u2022", "\u25e6")):
            return False
        if "@" in value or "http://" in value or "https://" in value:
            return False
        raw_degree = entry.get("degree") or ""
        if raw_degree and (value.lower() == raw_degree.lower() or value.lower() in raw_degree.lower()):
            return False
        if entry.get("grade") and value.lower() == entry["grade"].lower():
            return False
        if entry.get("fieldOfStudy") and value.lower() == entry["fieldOfStudy"].lower():
            return False
        has_location_sep = bool(re.search(r"[,–—-]", value))
        has_alpha = bool(re.search(r"[A-Za-z]", value))
        return has_alpha and (has_location_sep or self._looks_like_institution(value))

    def _build_entry(self, entry: dict[str, Any]) -> dict[str, object]:
        # Preserve compact/acroynmic degree tokens (e.g., 'M.C.A', 'B.SC-IT') when
        # they were split from an institution and no fieldOfStudy is present.
        raw_degree = entry.get("degree")
        if entry.get("_preserve_degree_text") and raw_degree:
            degree_value = raw_degree.strip()
        elif raw_degree and entry.get("fieldOfStudy") is None and re.search(r"[.\-]", raw_degree):
            degree_value = raw_degree.strip()
        else:
            degree_value = self._normalize_degree(entry["degree"])
        return {
            "institution": entry["institution"],
            "degree": degree_value,
            "fieldOfStudy": entry["fieldOfStudy"],
            "startDate": entry["startDate"],
            "endDate": entry["endDate"],
            "grade": entry["grade"],
            "confidence": self._confidence.section_extraction(),
        }

    def _normalize_degree(self, text: str | None) -> str | None:
        if text is None:
            return None
        lowered = text.lower().strip()
        for alias, canonical in self._degree_map.items():
            if alias in lowered:
                return canonical
        return text
