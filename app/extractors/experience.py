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
            entry["skills"] = [
                skill["value"]
                for skill in self.skills_extractor.extract(
                    [TextBlock(text=cleaned, page_number=1, x0=0, y0=0, x1=0, y1=0)],
                    mode="experience_description",
                )
            ]
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

            blocks_in_group = [(cb.original, cb.label) for cb in group.blocks]

            for block, label in blocks_in_group:
                text = block.text.strip() if hasattr(block, "text") else ""
                if not text:
                    continue

                parenthesized_date = self._parse_parenthesized_date_range(text)
                if parenthesized_date and entry.get("startDate") is None:
                    entry["startDate"] = parenthesized_date.startDate
                    entry["endDate"] = parenthesized_date.endDate
                    entry["current"] = parenthesized_date.current
                    continue

                combined_title_date = self._parse_combined_title_date(text)
                if combined_title_date and entry.get("designation") is None and entry.get("startDate") is None:
                    entry["designation"] = self._canonicalize_title(combined_title_date["title"])
                    entry["startDate"] = combined_title_date["startDate"]
                    entry["endDate"] = combined_title_date["endDate"]
                    entry["current"] = combined_title_date["current"]
                    continue

                if not entry.get("designation") and self._looks_like_title_header(text):
                    entry["designation"] = self._canonicalize_title(text)
                    continue

                date_range = DateRangeParser.parse(text)
                if date_range and entry.get("startDate") is None:
                    entry["startDate"] = date_range.startDate
                    entry["endDate"] = date_range.endDate
                    entry["current"] = date_range.current
                    continue

                combined_header = self._parse_combined_company_location_date(text)
                if combined_header and entry.get("company") is None and entry.get("location") is None and entry.get("startDate") is None:
                    entry["company"] = combined_header["company"]
                    entry["location"] = combined_header["location"]
                    entry["startDate"] = combined_header["startDate"]
                    entry["endDate"] = combined_header["endDate"]
                    entry["current"] = combined_header["current"]
                    continue

                lbl = str(label).upper() if label is not None else ""
                split_company_location = self._parse_company_location(text)
                if (
                    split_company_location
                    and entry.get("designation")
                    and entry.get("startDate") is not None
                    and not entry.get("company")
                    and not entry.get("location")
                ):
                    entry["company"] = split_company_location["company"]
                    entry["location"] = split_company_location["location"]
                    continue

                if lbl == "JOB_TITLE" and not entry.get("designation"):
                    entry["designation"] = self._canonicalize_title(text)
                    continue

                if lbl == "COMPANY" and not entry.get("company"):
                    entry["company"] = text
                    continue

                if (
                    lbl == "LOCATION"
                    and split_company_location
                    and entry.get("designation")
                    and entry.get("startDate") is not None
                    and entry.get("endDate") is not None
                    and not entry.get("company")
                    and not entry.get("location")
                ):
                    entry["company"] = split_company_location["company"]
                    entry["location"] = split_company_location["location"]
                    continue

                if lbl == "LOCATION" and not entry.get("location"):
                    entry["location"] = self._normalize_location(text)
                    continue

                if lbl == "UNKNOWN":
                    if not entry.get("company") and self._is_company_line(text):
                        entry["company"] = text
                        continue

                    if not entry.get("location") and self._is_location_line(text):
                        entry["location"] = text
                        continue

                description_parts.append(text)

            if entry.get("designation") is None:
                for block, _ in blocks_in_group:
                    text = block.text.strip() if hasattr(block, "text") else ""
                    if text and self._looks_like_title_header(text):
                        entry["designation"] = self._canonicalize_title(text)
                        break

            if entry.get("description") is None and description_parts:
                entry["description"] = "\n".join(description_parts).strip()

            if entry.get("description"):
                cleaned = self._clean_experience_description(entry.get("description", ""))
                entry["description"] = cleaned
                entry["skills"] = [
                    skill["value"]
                    for skill in self.skills_extractor.extract(
                        [TextBlock(text=cleaned, page_number=1, x0=0, y0=0, x1=0, y1=0)],
                        mode="experience_description",
                    )
                ]
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

    def _parse_combined_company_location_date(self, text: str) -> dict[str, object] | None:
        if not text or "/" not in text:
            return None

        left, right = text.split("/", 1)
        right = right.strip()
        date_range = DateRangeParser.parse(right)
        if date_range is None:
            return None

        left = left.strip()
        if "," not in left:
            return None

        left_parts = [part.strip() for part in left.split(",") if part.strip()]
        if len(left_parts) < 2:
            return None

        company = left_parts[0]
        location = ", ".join(left_parts[1:])
        if not company or not location:
            return None

        return {
            "company": company,
            "location": location,
            "startDate": date_range.startDate,
            "endDate": date_range.endDate,
            "current": date_range.current,
        }

    def _parse_combined_title_date(self, text: str) -> dict[str, str | bool] | None:
        match = re.fullmatch(
            r"(?P<title>.+?)\s+-\s+(?P<start>(?:\d{1,2}[/-]\d{4}|[A-Za-z]+\s+\d{4}|\d{4}))\s+to\s+"
            r"(?P<end>(?:\d{1,2}[/-]\d{4}|[A-Za-z]+\s+\d{4}|\d{4}|Present|Current|Now))",
            text.strip(),
            re.IGNORECASE,
        )
        if not match:
            return None

        date_range = DateRangeParser.parse(f"{match.group('start')} - {match.group('end')}")
        if date_range is None:
            return None

        return {
            "title": match.group("title").strip(),
            "startDate": date_range.startDate,
            "endDate": date_range.endDate,
            "current": date_range.current,
        }

    def _parse_parenthesized_date_range(self, text: str):
        match = re.fullmatch(r"\(\s*(.+?)\s*\)", text.strip())
        if not match:
            return None
        return DateRangeParser.parse(match.group(1))

    def _parse_company_location(self, text: str) -> dict[str, str] | None:
        match = re.fullmatch(
            r"(?P<company>[A-Za-z][A-Za-z .&'-]*?)(?:\s+,\s+|\s+[–—]\s+)(?P<location>[A-Za-z][A-Za-z .&'-]*(?:,\s*[A-Za-z][A-Za-z .&'-]*)*)",
            text.strip(),
        )
        if not match:
            return None

        company = match.group("company").strip()
        location = match.group("location").strip()
        if not company or not location:
            return None

        return {"company": company, "location": location}

    def _looks_like_title_header(self, text: str) -> bool:
        value = re.sub(r"\s+", " ", text or "").strip()
        if not value or len(value) > 60:
            return False
        if re.search(r"\b(?:inc|llc|ltd|corp|corporation|company)\b", value, re.I):
            return False
        words = value.split()
        if len(words) > 5:
            return False
        if any(ch.isdigit() for ch in value):
            return False

        lower_words = [word.lower() for word in words]
        role_words = {
            "assistant",
            "secretary",
            "coordinator",
            "manager",
            "analyst",
            "specialist",
            "developer",
            "engineer",
            "supervisor",
            "administrator",
            "associate",
            "consultant",
            "director",
            "executive",
            "clerk",
            "intern",
            "officer",
            "lead",
            "teacher",
            "professor",
        }
        if any(word in role_words for word in lower_words):
            return True
        if len(words) <= 3 and all(word.isupper() or word[:1].isupper() for word in words):
            return True
        return False

    def _canonicalize_title(self, text: str) -> str:
        value = re.sub(r"\s+", " ", text or "").strip()
        if not value:
            return value

        normalized = []
        for token in value.split():
            if token.isdigit():
                normalized.append(token)
            elif re.fullmatch(r"[A-Z]{2,3}", token):
                normalized.append(token)
            elif token.lower() in {"and", "of", "for", "to", "in", "on", "with"}:
                normalized.append(token.lower())
            else:
                normalized.append(token.capitalize())
        return " ".join(normalized)

    def _is_company_line(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        lower = normalized.lower()
        if re.search(r"\b(?:inc|llc|ltd|corp|corporation|co\.|company)\b", lower):
            return bool(re.search(r"[A-Z]", normalized)) and len(normalized.split()) <= 8
        if "&" in normalized and len(normalized.split()) <= 6 and any(ch.isupper() for ch in normalized):
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
        normalized = text.strip()
        if not normalized or "/" in normalized or "," not in normalized:
            return False
        if normalized.lower().startswith("location:"):
            location = normalized.split(":", 1)[1].strip()
            return bool(re.fullmatch(r"[A-Za-z][A-Za-z .&'-]*(?:,\s*[A-Za-z][A-Za-z .&'-]*)+", location)) and len(location.split()) <= 8
        if bool(re.fullmatch(r"[A-Za-z][A-Za-z .&'-]*,\s*[A-Za-z][A-Za-z .&'-]*", normalized)):
            return len(normalized.split()) <= 8
        return False

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
            if self._is_bullet_marker(line) and not any(ch.isalnum() for ch in line):
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
