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
        self._normalized_aliases = {
            section: [re.sub(r"[^a-z0-9]+", "", alias.lower()) for alias in section_aliases]
            for section, section_aliases in self.aliases.items()
        }
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
        blocks = list(blocks)

        index = 0
        while index < len(blocks):
            block = blocks[index]
            block_text = block.text.strip()
            matched_section = self._find_section_header(block_text)
            line_blocks: list[TextBlock] = []
            inferred_transition = False
            if matched_section is None:
                line_blocks = self._collect_same_line_blocks(blocks, index)
                line_text = self._line_text(line_blocks)
                matched_section = self._find_section_header(line_text)
                if matched_section is None and current_section is None and self._looks_like_education_header(line_text):
                    matched_section = "EDUCATION"
                    sections[matched_section].extend(line_blocks)

            if matched_section is None and current_section is not None:
                matched_section = self._infer_section_transition(current_section, block, sections)
                inferred_transition = matched_section is not None

            if matched_section:
                current_section = matched_section
                if inferred_transition:
                    sections[matched_section].append(block)
                elif matched_section == "SUMMARY" and self._is_summary_header(block_text):
                    sections[matched_section].append(block)
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
        if text is None:
            return None

        stripped = text.strip()
        if not stripped:
            return None

        normalized_text = re.sub(r"[^a-z0-9]+", "", stripped.lower())

        if re.search(r"\bobjective\b", stripped, re.IGNORECASE):
            return "SUMMARY"

        for section, aliases in self._normalized_aliases.items():
            for alias in aliases:
                if alias == normalized_text:
                    return section

        for section, patterns in self._header_patterns.items():
            for pattern in patterns:
                if pattern.match(stripped):
                    return section
                if pattern.match(re.sub(r"\s+", " ", stripped)):
                    return section
        return None

    def _is_summary_header(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        return bool(re.search(r"(?:^|\s)(?:resume\s+)?objective(?:\s|$)", normalized, re.IGNORECASE))

    def _looks_like_skill_text(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False

        lowered = normalized.lower()
        if re.search(r"\b(?:manager|developer|engineer|director|analyst|assistant|secretary|teacher|professor|intern)\b", lowered):
            return False
        if len(normalized) > 60:
            return False
        words = normalized.split()
        if len(words) > 6:
            return False
        return True

    def _looks_like_education_content(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        lowered = normalized.lower()
        if re.search(r"(?:degree|major|university|college|school|academy|institute|location\s+\d{4})", lowered):
            return True
        if re.search(r"(?:b\.a|b\.sc|btech|b\.tech|m\.a|m\.sc|mba|phd|bachelor|master)", lowered):
            return True
        return "educat" in lowered or "academ" in lowered

    def _looks_like_certification_content(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        lowered = normalized.lower()
        return "certificat" in lowered or "credential" in lowered or "licens" in lowered

    def _infer_section_transition(self, current_section: str, block: TextBlock, sections: dict[str, list[TextBlock]]) -> str | None:
        text = (block.text or "").strip()
        if not text:
            return None

        if current_section == "SKILLS":
            from app.extractors.date_parser import DateRangeParser

            if DateRangeParser.parse(text) is not None and self._is_aligned_education_date(block, sections["EDUCATION"]):
                return "EDUCATION"
            if self._looks_like_education_content(text):
                return "EDUCATION"
            if self._looks_like_certification_content(text):
                return "CERTIFICATIONS"

        if current_section == "EDUCATION":
            if self._looks_like_certification_content(text):
                return "CERTIFICATIONS"
            # Right-column date ranges like '2024 – 2026' and '2020 – 2023' are
            # valid education content and must not be reclassified as skills.
            from app.extractors.date_parser import DateRangeParser

            if DateRangeParser.parse(text) is not None:
                return None
            if block.x0 > 200 and self._looks_like_skill_text(text):
                return "SKILLS"

        if current_section == "CERTIFICATIONS":
            if block.x0 > 200 and self._looks_like_skill_text(text):
                return "SKILLS"

        if current_section == "ACHIEVEMENTS" and re.search(r"\bobjective\b", text, re.IGNORECASE):
            return "SUMMARY"

        return None

    def _is_aligned_education_date(self, block: TextBlock, education_blocks: list[TextBlock]) -> bool:
        return any(
            candidate.page_number == block.page_number
            and abs(candidate.y0 - block.y0) <= self.LINE_Y_TOLERANCE
            and block.x0 > candidate.x1
            for candidate in education_blocks
        )

    def _looks_like_education_header(self, text: str) -> bool:
        match = self._education_header_pattern.match(text)
        if not match:
            return False

        degree_part = match.group(1).strip()
        return bool(self._degree_prefix_pattern.match(degree_part))
