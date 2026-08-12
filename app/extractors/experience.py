from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from app.pipeline.stages.confidence import ConfidenceScorer
from app.extractors.date_parser import DateRangeParser
from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.text_extraction import TextBlock

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
JOB_TITLES_PATH = RESOURCE_DIR / "job_titles.json"


def _load_job_titles() -> list[str]:
    with JOB_TITLES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ExperienceExtractor:
    def __init__(self) -> None:
        self.job_titles = [title.lower() for title in _load_job_titles()]
        self.skills_extractor = SkillsExtractor()
        self._confidence = ConfidenceScorer()

    def extract(self, blocks: Iterable[TextBlock]) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        block_list = list(blocks)
        current_entry: dict[str, object] | None = None

        for block in block_list:
            text = block.text.strip()
            if not text:
                continue

            date_range = DateRangeParser.parse(text)
            if date_range:
                if current_entry:
                    entries.append(current_entry)
                current_entry = self._new_entry(date_range)
                continue

            if current_entry is None:
                continue

            if self._is_company_line(text):
                current_entry["company"] = text
                continue

            if self._is_job_title(text):
                current_entry["designation"] = text
                continue

            if self._is_location_line(text):
                current_entry["location"] = text
                continue

            description = current_entry.get("description", "")
            current_entry["description"] = "\n".join(filter(None, [description, text])).strip()

        if current_entry:
            entries.append(current_entry)

        for entry in entries:
            entry["skills"] = [skill["value"] for skill in self.skills_extractor.extract([TextBlock(text=entry.get("description", ""), page_number=1, x0=0, y0=0, x1=0, y1=0)], section_name=None)]
            entry["confidence"] = self._confidence.section_extraction()

        return entries

    def _new_entry(self, date_range: object) -> dict[str, object]:
        return {
            "company": None,
            "designation": None,
            "location": None,
            "startDate": date_range.startDate,
            "endDate": date_range.endDate,
            "current": date_range.current,
            "description": None,
            "skills": [],
            "confidence": self._confidence.section_extraction(),
        }

    def _is_company_line(self, text: str) -> bool:
        return any(keyword in text.lower() for keyword in ["inc", "llc", "ltd", "corp", "company", "technologies"])

    def _is_job_title(self, text: str) -> bool:
        normalized = text.lower()
        return any(normalized == title for title in self.job_titles)

    def _is_location_line(self, text: str) -> bool:
        return "," in text and any(char.isalpha() for char in text)
