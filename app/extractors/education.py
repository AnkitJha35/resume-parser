from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, List
from app.pipeline.stages.candidate_grouping import CandidateGroup

from app.pipeline.stages.confidence import ConfidenceScorer
from app.extractors.date_parser import DateRangeParser

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
            normalized = line.strip()
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

                # Date
                date_range = self._date_parser.parse(text)
                if date_range and entry["startDate"] is None:
                    entry["startDate"] = date_range.startDate
                    entry["endDate"] = date_range.endDate or "Present"
                    continue

                lbl = str(getattr(cb, "label", "")).upper()
                if lbl == "DEGREE" and entry["degree"] is None:
                    trailing_year = re.fullmatch(r"(?P<degree>.+?)\s*[-–—]\s*(?P<year>\d{4})", text)
                    if trailing_year:
                        entry["degree"] = trailing_year.group("degree").strip()
                        entry["startDate"] = trailing_year.group("year")
                        entry["_preserve_degree_text"] = True
                    else:
                        entry["degree"] = text
                    continue
                if lbl == "INSTITUTION" and entry["institution"] is None:
                    entry["institution"] = text
                    continue

                # Unknown label: use existing line heuristics
                if lbl == "UNKNOWN":
                    if entry["degree"] is None and self._is_degree_line(text):
                        entry["degree"] = text
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
                        if inst is not None:
                            entry["degree"] = deg
                            entry["institution"] = inst
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

            results.append(self._build_entry(entry))

        return results

    def _is_degree_line(self, text: str) -> bool:
        return bool(self._degree_pattern.match(text))

    def _extract_degree_line(self, text: str) -> tuple[str, str | None, str | None]:
        institution = None
        if "|" in text:
            degree_text, institution_text = (part.strip() for part in text.split("|", 1))
            institution = self._normalize_institution(institution_text)
        else:
            degree_text = text
            # Heuristic: handle common compact formats like "M.C.A NIT Calicut" or
            # "B.SC-IT Magadh University" where degree and institution appear
            # on the same line separated by whitespace. Only split when the
            # leading token looks like an acronym/degree (contains dots or
            # hyphens) or is an all-caps short token to avoid false positives.
            m = re.match(r"^(?P<deg>[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)+)\s+(?P<inst>.+)$", text)
            if not m:
                # also match short all-caps tokens (e.g., "MCA NIT Calicut")
                m2 = re.match(r"^(?P<deg>[A-Z]{2,6})\s+(?P<inst>.+)$", text)
                if m2:
                    m = m2

            if m:
                inst_candidate = m.group("inst").strip()
                # Avoid splitting when the remainder is a field-of-study like
                # 'in Electronics' or starts with 'of ...', which should be
                # interpreted as degree + field, not degree + institution.
                if not re.match(r"^(in|of)\b", inst_candidate, re.I):
                    degree_text = m.group("deg").strip()
                    institution = self._normalize_institution(inst_candidate)

        field_of_study = None
        match = re.match(r"^(?P<degree>.+?)\s+in\s+(?P<field>.+)$", degree_text, re.I)
        if match:
            degree_text = match.group("degree").strip()
            field_of_study = match.group("field").strip()

        return degree_text, institution, field_of_study

    def _contains_date_range(self, text: str) -> bool:
        return self._date_parser.parse(text) is not None

    def _looks_like_institution(self, text: str) -> bool:
        return bool(re.search(r"\b(university|college|institute|school|academy|polytechnic)\b", text, re.I))

    def _looks_like_grade(self, text: str) -> bool:
        return bool(re.search(r"\b(grade|cgpa|gpa|percentage|distinction|honors|honours|marks)\b", text, re.I))

    def _normalize_institution(self, text: str) -> str:
        text = re.sub(r"\s*,\s*", ", ", text.strip())
        return " ".join(text.split())

    def _looks_like_field_of_study(self, text: str) -> bool:
        return bool(re.search(r"\b(computer science|information technology|electronics|mechanical|civil|business administration|commerce|finance|mathematics|physics|data science|machine learning)\b", text, re.I))

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
