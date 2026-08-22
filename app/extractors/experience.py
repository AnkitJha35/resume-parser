from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

from app.pipeline.stages.confidence import ConfidenceScorer
from app.extractors.date_parser import DateRangeParser
from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.text_extraction import TextBlock
from app.pipeline.stages.block_classification import ClassifiedBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup

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

    def extract(self, blocks: Optional[Iterable[TextBlock]] = None, groups: Optional[list[CandidateGroup]] = None) -> list[dict[str, object]]:
        # If groups are provided, use the group-based extraction path
        if groups is not None and len(groups) > 0:
            return self._extract_from_groups(groups)

        # If no groups and no blocks provided, nothing to do
        if blocks is None:
            return []

        # Otherwise use the existing block-based path
        entries: list[dict[str, object]] = []
        block_list = list(blocks)
        current_entry: dict[str, object] | None = None
        buffer: list[str] = []

        for block in block_list:
            text = block.text.strip()
            if not text:
                continue

            date_range = DateRangeParser.parse(text)

            if current_entry is None:
                # buffer lines until we see a signal that starts an entry
                buffer.append(text)

                signal = False
                if date_range or self._is_job_title(text) or self._is_company_line(text):
                    signal = True

                if not signal:
                    continue

                # Start a new entry using any available date_range (may be None)
                current_entry = self._new_entry(date_range)

                # Attach buffered lines to the new entry, using heuristics
                for b in buffer:
                    # If this buffered line contains a date, set date fields and do not
                    # treat the same line as resume body copy.
                    dr = DateRangeParser.parse(b)
                    if dr:
                        if current_entry.get("startDate") is None:
                            current_entry["startDate"] = dr.startDate
                            current_entry["endDate"] = dr.endDate
                            current_entry["current"] = dr.current
                        continue

                    if self._is_job_title(b) and not current_entry.get("designation"):
                        current_entry["designation"] = b
                        continue

                    if self._is_company_line(b) and not current_entry.get("company"):
                        current_entry["company"] = b
                        continue

                    if self._is_location_line(b) and not current_entry.get("location"):
                        current_entry["location"] = self._normalize_location(b)
                        continue

                    # Otherwise treat as description
                    description = current_entry.get("description", "") or ""
                    current_entry["description"] = "\n".join(filter(None, [description, b])).strip()

                buffer.clear()
                continue

            # current_entry exists
            if date_range:
                # If there's an open entry that has identity (designation or company)
                # but no startDate yet, treat this DATE as completing that entry.
                if current_entry and current_entry.get("startDate") is None and (
                    current_entry.get("designation") or current_entry.get("company")
                ):
                    current_entry["startDate"] = date_range.startDate
                    current_entry["endDate"] = date_range.endDate
                    current_entry["current"] = date_range.current
                    continue

                # Otherwise close current and start new
                if current_entry:
                    entries.append(current_entry)
                current_entry = self._new_entry(date_range)
                continue

            if self._is_company_line(text) and not current_entry.get("company") and not current_entry.get("description"):
                current_entry["company"] = text
                continue

            if self._is_job_title(text):
                # If the open entry already has a date, designation, company and description,
                # treat a following job-title as the start of a new entry.
                if (
                    current_entry.get("startDate")
                    and current_entry.get("designation")
                    and current_entry.get("company")
                    and current_entry.get("description")
                ):
                    entries.append(current_entry)
                    current_entry = self._new_entry(None)

                # If the open entry has a date and a designation but no company, the new
                # job-title likely belongs to the next entry — start a new one.
                elif current_entry.get("startDate") and not current_entry.get("company") and current_entry.get("designation"):
                    entries.append(current_entry)
                    current_entry = self._new_entry(None)

                # Only treat a standalone title as metadata before description content begins.
                if not current_entry.get("description"):
                    current_entry["designation"] = text
                continue

            if self._is_location_line(text) and not current_entry.get("location") and not current_entry.get("description"):
                current_entry["location"] = self._normalize_location(text)
                continue

            # Structural fallback: if we already have a designation but no company
            # or description yet, the next undecided short line is likely the company.
            if (
                current_entry.get("designation")
                and not current_entry.get("company")
                and not current_entry.get("description")
            ):
                current_entry["company"] = text
                continue

            description = current_entry.get("description", "") or ""
            current_entry["description"] = "\n".join(filter(None, [description, text])).strip()

        if current_entry:
            entries.append(current_entry)

        for entry in entries:
            # Clean description for presentation and for skills extraction
            desc_text = entry.get("description") or ""
            cleaned = self._clean_experience_description(desc_text)
            entry["description"] = cleaned
            entry["skills"] = [skill["value"] for skill in self.skills_extractor.extract([TextBlock(text=cleaned, page_number=1, x0=0, y0=0, x1=0, y1=0)], section_name=None)]
            entry["confidence"] = self._confidence.section_extraction()

        return entries

    def _extract_from_groups(self, groups: list[CandidateGroup]) -> list[dict[str, object]]:
        """Extract experience entries from CandidateGroups.

        Each group is processed independently. Labels are used as hints but not
        as absolute positional requirements. Existing heuristics are applied to
        identify designation, company, location, and description.
        """
        entries: list[dict[str, object]] = []

        for group in groups:
            entry = self._new_entry(None)
            description_parts: list[str] = []

            # Extract blocks and their labels from the group
            blocks_in_group = [(cb.original, cb.label) for cb in group.blocks]

            # First pass: identify date, designation, company, location
            for block, label in blocks_in_group:
                text = block.text.strip() if hasattr(block, "text") else ""
                if not text:
                    continue

                # Try to parse date
                date_range = DateRangeParser.parse(text)
                if date_range and entry.get("startDate") is None:
                    entry["startDate"] = date_range.startDate
                    entry["endDate"] = date_range.endDate
                    entry["current"] = date_range.current
                    continue

                # Use label as a hint (case-insensitive)
                lbl = str(label).upper() if label is not None else ""
                if lbl == "JOB_TITLE" and not entry.get("designation"):
                    entry["designation"] = text
                    continue

                if lbl == "COMPANY" and not entry.get("company"):
                    entry["company"] = text
                    continue

                if lbl == "LOCATION" and not entry.get("location"):
                    entry["location"] = self._normalize_location(text)
                    continue

                # If label is not decisive, use heuristics
                if lbl == "UNKNOWN":
                    if not entry.get("designation") and self._is_job_title(text):
                        entry["designation"] = text
                        continue

                    if not entry.get("company") and self._is_company_line(text):
                        entry["company"] = text
                        continue

                    if not entry.get("location") and self._is_location_line(text):
                        entry["location"] = text
                        continue

                # Treat as description
                description_parts.append(text)

            # Combine description parts
            if description_parts:
                entry["description"] = "\n".join(description_parts).strip()

            # Extract skills from description
            if entry.get("description"):
                # Clean description before skills extraction and presentation
                cleaned = self._clean_experience_description(entry.get("description", ""))
                entry["description"] = cleaned
                entry["skills"] = [skill["value"] for skill in self.skills_extractor.extract(
                    [TextBlock(text=cleaned, page_number=1, x0=0, y0=0, x1=0, y1=0)],
                    section_name=None
                )]
            else:
                entry["skills"] = []

            entry["confidence"] = self._confidence.section_extraction()
            entries.append(entry)

        return entries

    def _new_entry(self, date_range: object | None) -> dict[str, object]:
        if date_range:
            start = date_range.startDate
            end = date_range.endDate
            current = date_range.current
        else:
            start = None
            end = None
            current = False

        return {
            "company": None,
            "designation": None,
            "location": None,
            "startDate": start,
            "endDate": end,
            "current": current,
            "description": None,
            "skills": [],
            "confidence": self._confidence.section_extraction(),
        }

    def _is_company_line(self, text: str) -> bool:
        normalized = text.lower().strip()
        if not normalized:
            return False
        keywords = ["inc", "llc", "ltd", "corp", "company", "technologies"]
        for keyword in keywords:
            pattern = rf"(?<![a-z]){re.escape(keyword)}(?![a-z])"
            if re.search(pattern, normalized):
                return True
        return False

    def _is_job_title(self, text: str) -> bool:
        normalized = text.lower()
        return any(normalized == title for title in self.job_titles)

    def _normalize_location(self, text: str) -> str:
        value = text.strip()
        if value.lower().startswith("location:"):
            value = value.split(":", 1)[1].strip()
        return value.strip()

    def _is_location_line(self, text: str) -> bool:
        return "," in text and any(char.isalpha() for char in text)

    def _is_bullet_marker(self, s: str) -> bool:
        if not s:
            return False
        t = s.strip()
        if t in ("•", "\u2022", "\u2023", "\u25E6", "-", "*", "●", "\u00B7"):
            return True
        if len(t) <= 3 and not any(ch.isalnum() for ch in t):
            return True
        if re.match(r"^[\-\*]\s+", s.strip()):
            return True
        return False

    def _clean_experience_description(self, text: str) -> str:
        """Clean experience description while preserving real bullet boundaries."""
        if not text:
            return text

        cleaned_lines: list[str] = []
        for raw_line in text.replace('\r', '').split('\n'):
            line = raw_line.strip()
            if not line:
                continue
            if self._is_bullet_marker(line) and line in ("•", "\u2022", "\u2023", "\u25E6", "●", "\u00B7"):
                continue
            cleaned_lines.append(line)

        bullet_lines: list[str] = []
        current_bullet: list[str] = []

        def flush_current() -> None:
            nonlocal current_bullet
            if current_bullet:
                bullet_lines.append("• " + " ".join(part.strip() for part in current_bullet if part and part.strip()))
                current_bullet = []

        for line in cleaned_lines:
            if re.match(r"^[\-\*•\u2022\u2023\u25E6\u25CF\u00B7]\s*.*$", line):
                flush_current()
                current_bullet = [line.lstrip(" -•\u2022\u2023\u25E6\u25CF\u00B7").strip()]
                continue

            if current_bullet:
                current_bullet.append(line)
            else:
                bullet_lines.append(line)

        flush_current()

        out = "\n".join(bullet_lines)
        out = re.sub(r'([,;:])([^\s])', r'\1 \2', out)
        out = re.sub(r' {2,}', ' ', out)
        return out.strip()
