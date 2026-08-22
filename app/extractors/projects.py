from __future__ import annotations

import re
from typing import Iterable, Optional, List

from app.pipeline.stages.confidence import ConfidenceScorer
from app.extractors.date_parser import DateRangeParser
from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.text_extraction import TextBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup

URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)


class ProjectExtractor:
    def __init__(self) -> None:
        self.skills_extractor = SkillsExtractor()
        self.date_parser = DateRangeParser()
        self._confidence = ConfidenceScorer()

    def extract(self, blocks: Optional[Iterable[TextBlock]] = None, groups: Optional[List[CandidateGroup]] = None) -> list[dict[str, object]]:
        """Accept either raw blocks (backwards compatible) or CandidateGroups.

        When `groups` is provided, extract per-group by reusing the existing
        block-oriented extraction logic on the group's original `TextBlock`s.
        """
        # Group-aware path: delegate to group-aware extractor
        if groups:
            return self._extract_from_groups(groups)

        # Backwards-compatible block-oriented path
        if blocks is None:
            return []

        return self._extract_from_block_list(list(blocks))

    def _extract_from_block_list(self, block_list: list[TextBlock]) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
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
            # Ensure description_text is a string (default to empty string if None)
            description_text = entry.get("description") or ""
            # Clean up PDF layout artifacts and join broken lines for final
            # description representation without modifying original TextBlock
            description_text = self._clean_description_text(description_text)
            # update the stored description with cleaned text for presentation
            entry["description"] = description_text
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

    def _extract_from_groups(self, groups: List[CandidateGroup]) -> list[dict[str, object]]:
        """Extract one project record per CandidateGroup.

        Rules:
        - Find the first DATE block in the group; parse it for start/end dates.
        - The nearest meaningful non-date block immediately before the DATE is
          treated as the project title.
        - All blocks after the DATE belong to the project's description.
        - If no DATE is found, fall back to block-list extraction for that group.
        """
        results: list[dict[str, object]] = []

        def _is_bullet_marker(s: str) -> bool:
            if not s:
                return False
            t = s.strip()
            if t in ("•", "\u2022", "\u2023", "\u25E6", "-", "*", "●", "\u00B7"):
                return True
            if len(t) <= 3 and not any(ch.isalnum() for ch in t):
                return True
            return False

        for group in groups:
            blocks = group.blocks

            # locate first DATE block
            date_idx = next((i for i, cb in enumerate(blocks) if getattr(cb, "label", None) == "DATE"), None)

            if date_idx is None:
                # fallback: use existing block-oriented logic
                block_list = [cb.original for cb in blocks]
                results.extend(self._extract_from_block_list(block_list))
                continue

            # find title: nearest meaningful non-date block before date_idx
            title = None
            for j in range(date_idx - 1, -1, -1):
                tb = blocks[j].original
                ttext = (getattr(tb, "text", "") or "").strip()
                if not ttext:
                    continue
                # skip pure bullets
                if _is_bullet_marker(ttext):
                    continue
                # prefer non-DATE labels
                if getattr(blocks[j], "label", None) != "DATE":
                    title = ttext
                    break

            # parse date
            date_text = (getattr(blocks[date_idx].original, "text", "") or "").strip()
            date_range = self.date_parser.parse(date_text)

            entry = self._new_entry()
            if title:
                entry["name"] = title
            if date_range:
                entry["startDate"] = date_range.startDate
                entry["endDate"] = date_range.endDate or "Present"

            # description: all blocks after the DATE
            desc_parts: list[str] = []
            for k in range(date_idx + 1, len(blocks)):
                t = (getattr(blocks[k].original, "text", "") or "").strip()
                if t:
                    desc_parts.append(t)

            if desc_parts:
                entry["description"] = "\n".join(desc_parts)
            else:
                entry["description"] = None

            # technologies extraction
            description_text = entry.get("description") or ""
            # Clean description for presentation and for downstream skills
            # extraction (without mutating original blocks)
            cleaned_description = self._clean_description_text(description_text)
            # update stored description for presentation
            entry["description"] = cleaned_description
            skills = self.skills_extractor.extract(
                [TextBlock(text=cleaned_description, page_number=1, x0=0, y0=0, x1=0, y1=0)],
                section_name="PROJECTS",
            )
            entry["technologies"] = [skill["value"] for skill in skills]
            entry["confidence"] = self._confidence.section_extraction()

            results.append(entry)

        return results

    def _is_date_range(self, text: str) -> bool:
        return self.date_parser.parse(text) is not None

    def _is_url(self, text: str) -> bool:
        return bool(URL_PATTERN.search(text))

    def _is_bullet_only(self, s: str) -> bool:
        if not s:
            return False
        t = s.strip()
        if t in ("•", "\u2022", "\u2023", "\u25E6", "-", "*", "●", "\u00B7"):
            return True
        # very short markers of punctuation only
        if len(t) <= 3 and not any(ch.isalnum() for ch in t):
            return True
        return False

    def _clean_description_text(self, text: str) -> str:
        """Return a cleaned, presentation-ready description string.

        Rules:
        - Remove lines that are bullet-only artifacts.
        - Collapse sequences of non-empty lines into paragraphs by joining
          with a single space when the previous line does not end with
          sentence-ending punctuation. Preserve paragraph breaks on blank
          lines.
        - Ensure punctuation such as commas/semicolons are followed by a
          single space.
        """
        if not text:
            return text

        # Normalize CRLF and stray carriage returns
        lines = [ln for ln in text.replace('\r', '').split('\n')]

        cleaned_lines: list[str] = []
        for ln in lines:
            if self._is_bullet_only(ln):
                # drop bullet-only artifact
                continue
            cleaned_lines.append(ln)

        paragraphs: list[str] = []
        buf: list[str] = []

        def flush_buf():
            if not buf:
                return
            # join buffer lines: if a line ends with sentence-ending
            # punctuation, keep as paragraph boundary; otherwise join with space
            joined = buf[0].strip()
            for additional in buf[1:]:
                prev = joined.rstrip()
                if prev.endswith(('.', '!', '?', ';', ':')):
                    joined = joined + '\n' + additional.strip()
                else:
                    # join with space to repair PDF line breaks
                    joined = joined + ' ' + additional.strip()
            paragraphs.append(joined.strip())

        for ln in cleaned_lines:
            if not ln.strip():
                # blank line -> paragraph break
                flush_buf()
                buf = []
                continue
            buf.append(ln)

        flush_buf()

        # Re-join paragraphs with double newlines to preserve separations
        out = '\n\n'.join(paragraphs)

        # Ensure punctuation followed by no space gets a space (e.g., 'Kubernetes,and')
        out = re.sub(r'([,;:])([^\s])', r'\1 \2', out)

        # Collapse multiple spaces
        out = re.sub(r' {2,}', ' ', out)

        return out
