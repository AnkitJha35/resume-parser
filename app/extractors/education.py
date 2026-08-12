from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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

    def extract(self, lines: list[str]) -> list[EducationEntry]:
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
