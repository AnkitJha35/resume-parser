from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from app.pipeline.stages.text_extraction import TextBlock


RESOURCE_DIR = Path(__file__).resolve().parents[2] / "resources"
SECTION_ALIASES_PATH = RESOURCE_DIR / "section_aliases.json"
DEGREES_PATH = RESOURCE_DIR / "degrees.json"

SECTION_NAMES = [
    "SUMMARY",
    "EXPERIENCE",
    "EDUCATION",
    "SKILLS",
    "PROJECTS",
    "CERTIFICATIONS",
    "ACHIEVEMENTS",
    "LANGUAGES",
]


def _load_section_aliases() -> dict[str, list[str]]:
    with SECTION_ALIASES_PATH.open("r", encoding="utf-8") as handle:
        aliases = json.load(handle)
    return {name: [alias.lower() for alias in names] for name, names in aliases.items()}


def _load_degree_prefixes() -> list[str]:
    with DEGREES_PATH.open("r", encoding="utf-8") as handle:
        degrees = json.load(handle)
    return [re.escape(degree) for degree in degrees.keys()]


class SectionDetector:
    LINE_Y_TOLERANCE = 1.0

    def __init__(self) -> None:
        self.aliases = _load_section_aliases()
        self._header_patterns = {
            section: [re.compile(rf"^\s*{re.escape(alias)}\s*$", re.IGNORECASE) for alias in section_aliases]
            for section, section_aliases in self.aliases.items()
        }
        degree_prefixes = _load_degree_prefixes()
        self._degree_prefix_pattern = re.compile(
            rf"^(?:{'|'.join(degree_prefixes)})",
            re.IGNORECASE,
        )
        self._education_header_pattern = re.compile(
            r"^(.+?)\s*\|\s*(.+)$",
            re.IGNORECASE,
        )

    def detect(self, blocks: Iterable[TextBlock]) -> dict[str, list[TextBlock]]:
        sections: dict[str, list[TextBlock]] = {section: [] for section in SECTION_NAMES}
        current_section = None
        sorted_blocks = sorted(blocks, key=lambda block: (block.page_number, block.y0, block.x0))

        index = 0
        while index < len(sorted_blocks):
            block = sorted_blocks[index]
            block_text = block.text.strip()
            matched_section = self._find_section_header(block_text)
            line_blocks: list[TextBlock] = []

            if matched_section is None:
                line_blocks = self._collect_same_line_blocks(sorted_blocks, index)
                line_text = self._line_text(line_blocks)
                matched_section = self._find_section_header(line_text)
                if matched_section is None and current_section is None and self._looks_like_education_header(line_text):
                    matched_section = "EDUCATION"
                    sections[matched_section].extend(line_blocks)

            if matched_section:
                current_section = matched_section
                index += len(line_blocks) if line_blocks else 1
                continue

            if current_section is not None:
                sections[current_section].append(block)

            index += 1

        return sections

    def _collect_same_line_blocks(self, blocks: list[TextBlock], start_index: int) -> list[TextBlock]:
        reference = blocks[start_index]
        same_line = [reference]

        for block in blocks[start_index + 1 :]:
            if block.page_number != reference.page_number:
                break
            if abs(block.y0 - reference.y0) > self.LINE_Y_TOLERANCE:
                break
            same_line.append(block)

        return same_line

    def _line_text(self, blocks: list[TextBlock]) -> str:
        return " ".join(block.text.strip() for block in blocks if block.text.strip()).strip()

    def _find_section_header(self, text: str) -> str | None:
        for section, patterns in self._header_patterns.items():
            for pattern in patterns:
                if pattern.match(text):
                    return section
        return None

    def _looks_like_education_header(self, text: str) -> bool:
        match = self._education_header_pattern.match(text)
        if not match:
            return False

        degree_part = match.group(1).strip()
        return bool(self._degree_prefix_pattern.match(degree_part))
