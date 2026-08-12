from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from app.pipeline.stages.confidence import ConfidenceScorer
from app.pipeline.stages.text_extraction import TextBlock

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
SKILLS_PATH = RESOURCE_DIR / "skills.json"


def _load_skills() -> dict[str, str]:
    with SKILLS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class SkillsExtractor:
    def __init__(self) -> None:
        self.skills = _load_skills()
        self.pattern = re.compile(r"\b(?:" + r"|".join(re.escape(alias) for alias in sorted(self.skills.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)
        self._confidence = ConfidenceScorer()

    def extract(self, blocks: Iterable[TextBlock], section_name: str | None = None) -> list[dict[str, object]]:
        found: dict[str, dict[str, object]] = {}
        text = "\n".join(block.text for block in blocks)
        for match in self.pattern.finditer(text):
            alias = match.group(0).strip().lower()
            canonical = self.skills.get(alias)
            if canonical is None:
                continue
            if canonical not in found:
                found[canonical] = {
                    "value": canonical,
                    "confidence": self._confidence.skill(),
                    "source": "skills_section" if section_name == "SKILLS" else "full_text",
                }
        return list(found.values())
