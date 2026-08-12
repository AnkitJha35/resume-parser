from __future__ import annotations

import re
from typing import Iterable

from app.pipeline.stages.confidence import ConfidenceScorer
from app.extractors.date_parser import DateRangeParser
from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.text_extraction import TextBlock

URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)


class ProjectExtractor:
    def __init__(self) -> None:
        self.skills_extractor = SkillsExtractor()
        self.date_parser = DateRangeParser()
        self._confidence = ConfidenceScorer()

    def extract(self, blocks: Iterable[TextBlock]) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        block_list = list(blocks)
        current_entry: dict[str, object] | None = None

        for block in block_list:
            text = block.text.strip()
            if not text:
                continue

            if self._is_date_range(text):
                if current_entry:
                    entries.append(current_entry)
                date_range = self.date_parser.parse(text)
                current_entry = self._new_entry()
                if date_range:
                    current_entry["startDate"] = date_range.startDate
                    current_entry["endDate"] = date_range.endDate or "Present"
                continue

            if self._is_url(text):
                if current_entry is None:
                    current_entry = self._new_entry()
                current_entry["url"] = text
                continue

            if current_entry is None:
                current_entry = self._new_entry()
                current_entry["name"] = text
                continue

            if current_entry["name"] is None:
                current_entry["name"] = text
                continue

            description = current_entry.get("description", "")
            current_entry["description"] = "\n".join(filter(None, [description, text])).strip()

        if current_entry:
            entries.append(current_entry)

        for entry in entries:
            description_text = entry.get("description", "")
            skills = self.skills_extractor.extract(
                [TextBlock(text=description_text, page_number=1, x0=0, y0=0, x1=0, y1=0)],
                section_name="PROJECTS",
            )
            entry["technologies"] = [skill["value"] for skill in skills]
            entry["confidence"] = self._confidence.section_extraction()

        return entries

    def _new_entry(self) -> dict[str, object]:
        return {
            "name": None,
            "description": None,
            "technologies": [],
            "startDate": None,
            "endDate": None,
            "url": None,
            "confidence": self._confidence.section_extraction(),
        }

    def _is_date_range(self, text: str) -> bool:
        return self.date_parser.parse(text) is not None

    def _is_url(self, text: str) -> bool:
        return bool(URL_PATTERN.search(text))
