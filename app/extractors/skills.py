from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from app.pipeline.stages.confidence import ConfidenceScorer
from app.pipeline.stages.text_extraction import TextBlock

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
SKILLS_PATH = RESOURCE_DIR / "skills.json"
EXTRACTOR_ALIASES_PATH = RESOURCE_DIR / "skills_extractor_aliases.json"

_FA_PREFIX = re.compile(r"\\fa[A-Za-z]+\s*:?\s*")
_GOOGLE_SHEETS_DOCS = re.compile(r"\bgoogle\s+sheets\s*&\s*docs\b", re.IGNORECASE)

# Soft/behavioral canonicals that are valid on SKILLS lists but too permissive
# when scanning ordinary experience-description prose. Experience-description
# mode suppresses these; top-level SKILLS extraction is unchanged.
_EXPERIENCE_DESCRIPTION_SOFT_SKILLS = frozenset(
    {
        "Communication",
        "Teamwork",
        "Willingness to Learn",
        "Collaboration",
        "Quick Learner",
        "Adaptability",
        "Problem Solving",
        "Time Management",
        "Complex Problem Solver",
        "Resourceful Problem Solver",
    }
)

_VALID_EXTRACT_MODES = frozenset({None, "experience_description"})


def _load_json_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _load_skills() -> dict[str, str]:
    """Merge shared skills.json with extractor-only aliases.

    Extractor-only aliases live in a separate file so SemanticResources /
    section inference keep using the unchanged shared vocabulary.
    """
    merged = dict(_load_json_dict(SKILLS_PATH))
    merged.update(_load_json_dict(EXTRACTOR_ALIASES_PATH))
    return merged


def _normalize_skills_text(text: str) -> str:
    """Normalize decorative prefixes and a few compound tool phrases for matching."""
    normalized = _FA_PREFIX.sub(" ", text)
    normalized = _GOOGLE_SHEETS_DOCS.sub("Google Sheets, Google Docs", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


class SkillsExtractor:
    def __init__(self) -> None:
        self.skills = _load_skills()
        self.pattern = re.compile(
            r"\b(?:"
            + r"|".join(re.escape(alias) for alias in sorted(self.skills.keys(), key=len, reverse=True))
            + r")\b",
            re.IGNORECASE,
        )
        self._confidence = ConfidenceScorer()

    def extract(
        self,
        blocks: Iterable[TextBlock],
        section_name: str | None = None,
        *,
        mode: str | None = None,
    ) -> list[dict[str, object]]:
        """Extract skills from text blocks.

        mode:
            None — default matching (used by top-level SKILLS and other callers).
            "experience_description" — keep tech/tool/domain matches from prose;
            suppress generic soft-skill canonicals that over-fire in narrative text.
        """
        if mode not in _VALID_EXTRACT_MODES:
            raise ValueError(f"Unsupported SkillsExtractor mode: {mode!r}")

        found: dict[str, dict[str, object]] = {}
        # Defensive: handle blocks whose .text may be None
        text = "\n".join((block.text or "") for block in blocks)
        # For matching only, collapse runs of whitespace (including newlines)
        # into single spaces so multi-word skills split across physical PDF
        # line breaks still match their canonical aliases. Do not modify
        # the original TextBlock objects.
        normalized_text = _normalize_skills_text(text)
        suppress_soft = mode == "experience_description"

        for match in self.pattern.finditer(normalized_text):
            alias = match.group(0).strip().lower()
            canonical = self.skills.get(alias)
            if canonical is None:
                continue
            if suppress_soft and canonical in _EXPERIENCE_DESCRIPTION_SOFT_SKILLS:
                continue
            if canonical not in found:
                if section_name == "SKILLS":
                    source = "skills_section"
                elif mode == "experience_description":
                    source = "experience_description"
                else:
                    source = "full_text"
                found[canonical] = {
                    "value": canonical,
                    "confidence": self._confidence.skill(),
                    "source": source,
                }
        return list(found.values())
