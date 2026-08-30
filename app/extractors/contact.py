from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from app.pipeline.stages.confidence import ConfidenceScorer
from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.text_extraction import TextBlock


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(
    r"(\+?\d{1,3}[ \-/.]?)?(?:\(\d{2,4}\)|\d{2,4})[ \-/.]?\d{3,4}[ \-/.]?\d{3,4}"
)
LINKEDIN_REGEX = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[A-Za-z0-9_\-/]+", re.IGNORECASE)
LINKEDIN_LABEL_HANDLE_REGEX = re.compile(
    r"^\s*(?:linkedin(?:\s+profile)?|linked\s*in)\s*[:\-]\s*(?P<handle>.+?)\s*$",
    re.IGNORECASE,
)
LINKEDIN_HANDLE_TOKEN_REGEX = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,98}[A-Za-z0-9])?$")
_INVALID_LINKEDIN_HANDLES = frozenset(
    {"n/a", "na", "none", "-", "--", "nil", "null", "tbd", "todo"}
)
GITHUB_REGEX = re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_\-/]+", re.IGNORECASE)
URL_REGEX = re.compile(r"https?://[^\s]+", re.IGNORECASE)
LOCATION_PATTERN = re.compile(r"[A-Za-z ]+(?:,\s*[A-Za-z ]+)+")


class ContactExtractor:
    @staticmethod
    def extract(blocks: Iterable[TextBlock]) -> dict[str, dict[str, object]]:
        ordered_blocks = sorted(blocks, key=lambda block: (block.page_number, block.y0))
        text_lines = [block.text.strip() for block in ordered_blocks if block.text.strip()]

        scorer = ConfidenceScorer()
        email = ContactExtractor._find_email(text_lines)
        phone = ContactExtractor._find_phone(text_lines)
        linkedin = ContactExtractor._find_linkedin(text_lines)
        github = ContactExtractor._find_github(text_lines)
        portfolio = ContactExtractor._find_portfolio(text_lines, linkedin, github)
        name = ContactExtractor._find_name(ordered_blocks, email, phone, linkedin, github)
        location = ContactExtractor._find_location(text_lines, name, email, phone, linkedin, github)

        return {
            "name": ContactExtractor._field(name, scorer.header_name(), "header_name" if name else "not_found"),
            "email": ContactExtractor._field(email, scorer.email_regex(), "email_regex" if email else "not_found"),
            "phone": ContactExtractor._field(phone, scorer.phone_regex(), "phone_regex" if phone else "not_found"),
            "location": ContactExtractor._field(location, scorer.location_heuristic(), "location_heuristic" if location else "not_found"),
            "linkedin": ContactExtractor._field(linkedin, scorer.url(), "linkedin_url" if linkedin else "not_found"),
            "github": ContactExtractor._field(github, scorer.url(), "github_url" if github else "not_found"),
            "portfolio": ContactExtractor._field(portfolio, scorer.url(), "portfolio_url" if portfolio else "not_found"),
        }

    @staticmethod
    def _field(value: str | None, confidence: float, source: str) -> dict[str, object]:
        return {"value": value, "confidence": confidence if value else 0.0, "source": source}

    @staticmethod
    def _find_email(lines: list[str]) -> str | None:
        for line in lines:
            match = EMAIL_REGEX.search(line)
            if match:
                return match.group(0).strip()
        return None

    @staticmethod
    def _find_phone(lines: list[str]) -> str | None:
        for line in lines:
            # Always attempt to extract a phone number from the line even if
            # the line also contains an email or URL (some headers place email
            # and phone on the same physical line).
            normalized_line = re.sub(r"\s", " ", line)
            match = PHONE_REGEX.search(normalized_line)
            if match:
                phone = match.group(0).strip()
                if len(re.sub(r"[^0-9]", "", phone)) >= 7:
                    return phone
        return None

    @staticmethod
    def _find_linkedin(lines: list[str]) -> str | None:
        for line in lines:
            normalized_line = unicodedata.normalize("NFKC", line)
            match = LINKEDIN_REGEX.search(normalized_line)
            if match:
                return match.group(0).strip()
            handle_url = ContactExtractor._linkedin_from_labeled_handle(normalized_line)
            if handle_url:
                return handle_url
        return None

    @staticmethod
    def _linkedin_from_labeled_handle(line: str) -> str | None:
        """Accept labeled bare handles; never invent LinkedIn from unlabeled text."""
        match = LINKEDIN_LABEL_HANDLE_REGEX.match(line or "")
        if not match:
            return None
        handle = (match.group("handle") or "").strip().strip("/")
        if not handle:
            return None
        lowered = handle.lower()
        if lowered in _INVALID_LINKEDIN_HANDLES:
            return None
        if "@" in handle or "://" in handle or "/" in handle or "." in handle:
            # Emails, foreign URLs, and path-like values are not bare handles.
            # LinkedIn URLs are already handled by LINKEDIN_REGEX on the full line.
            return None
        if not LINKEDIN_HANDLE_TOKEN_REGEX.fullmatch(handle):
            return None
        return f"https://www.linkedin.com/in/{handle}"

    @staticmethod
    def _find_github(lines: list[str]) -> str | None:
        for line in lines:
            match = GITHUB_REGEX.search(line)
            if match:
                return match.group(0).strip()
        return None

    @staticmethod
    def _find_portfolio(lines: list[str], linkedin: str | None, github: str | None) -> str | None:
        for line in lines:
            for match in URL_REGEX.findall(line):
                url = match.strip()
                if linkedin and linkedin in url:
                    continue
                if github and github in url:
                    continue
                return url
        return None

    @staticmethod
    def _find_name(blocks: list[TextBlock], email: str | None, phone: str | None, linkedin: str | None, github: str | None) -> str | None:
        candidates: list[tuple[float, str]] = []
        spaced_name = ContactExtractor._find_letter_spaced_name(blocks[:8])
        if spaced_name and ContactExtractor._looks_like_name(spaced_name):
            first_spaced = next(
                block for block in blocks[:8] if ContactExtractor._is_letter_spaced_name((block.text or "").strip())
            )
            candidates.append((
                max(0.0, 100.0 - blocks.index(first_spaced) * 5.0)
                + min(float(getattr(first_spaced, "font_size", None) or 0.0), 40.0) * 2.0
                + 5.0,
                spaced_name,
            ))
        for index, block in enumerate(blocks[:12]):
            text = (block.text or "").strip()
            if not text or EMAIL_REGEX.search(text) or PHONE_REGEX.search(text) or LINKEDIN_REGEX.search(text) or GITHUB_REGEX.search(text):
                continue

            normalized = ContactExtractor._normalize_spaced_name(text)
            candidate = normalized or text
            if not ContactExtractor._looks_like_name(candidate):
                continue

            score = max(0.0, 100.0 - index * 5.0)
            score += min(float(getattr(block, "font_size", None) or 0.0), 40.0) * 2.0
            if getattr(block, "bold", False):
                score += 10.0
            if normalized:
                score += 5.0
            candidates.append((score, candidate))

        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _find_letter_spaced_name(blocks: list[TextBlock]) -> str | None:
        words: list[str] = []
        for block in blocks:
            text = (block.text or "").strip()
            fragments = re.split(r"\s{2,}", text) if text else []
            for fragment in fragments:
                if ContactExtractor._is_letter_spaced_name(fragment):
                    letters = "".join(ch for ch in fragment if ch.isalpha())
                    if letters:
                        words.append(letters)

        if not words:
            return None
        return " ".join(words)

    @staticmethod
    def _normalize_spaced_name(text: str) -> str | None:
        if not ContactExtractor._is_letter_spaced_name(text):
            return None
        return "".join(token for token in text.split() if token.isalpha())

    @staticmethod
    def _is_letter_spaced_name(text: str) -> bool:
        tokens = text.split()
        if len(tokens) < 2:
            return False
        return all(len(token) == 1 and token.isalpha() for token in tokens)

    @staticmethod
    def _looks_like_name(text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if re.search(r"\b(assistant|manager|developer|engineer|specialist|analyst|coordinator|secretary|supervisor|director|consultant|executive|officer|associate|lead|intern|clerk|administrator|representative)\b", text, re.IGNORECASE):
            return False
        words = text.split()
        if not (1 <= len(words) <= 4):
            return False
        structural_words = {"career", "objective", "experience", "skills", "university", "location", "administrative"}
        if any(word.lower() in structural_words for word in words):
            return False
        name_token = r"[^\W\d_]+(?:[-'][^\W\d_]+)*"
        return all(
            word[0].isupper() for word in words if word
        ) and all(re.fullmatch(name_token, word, re.UNICODE) for word in words)

    @staticmethod
    def _find_location(lines: list[str], name: str | None, email: str | None, phone: str | None, linkedin: str | None, github: str | None) -> str | None:
        section_detector = SectionDetector()
        skills_extractor = SkillsExtractor()
        candidates: list[tuple[float, str]] = []

        for line in lines:
            m = re.search(r"LOCATION\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if m:
                value = m.group(1).strip()
                if value and not section_detector._find_section_header(value):
                    return value

        for index, line in enumerate(lines[:-1]):
            street = line.strip()
            city = lines[index + 1].strip()
            if (
                re.match(r"^\d+\s+[A-Za-z][A-Za-z .'-]*\b(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way)\b", street, re.IGNORECASE)
                and "," in city
                and (re.search(r"\b\d{5}(?:-\d{4})?\b", city) or LOCATION_PATTERN.fullmatch(city))
            ):
                return f"{street}, {city}"

        role_words = {
            "assistant", "manager", "developer", "engineer", "specialist", "analyst",
            "coordinator", "secretary", "supervisor", "director", "consultant",
            "executive", "officer", "associate", "lead", "intern", "clerk",
            "administrator", "representative",
        }
        skill_words = {
            "skills", "experience", "summary", "profile", "education", "certification",
            "languages", "management", "communication", "leadership", "excel",
        }

        for index, line in enumerate(lines):
            text = line.strip()
            lowered = text.lower()
            if not text or section_detector._find_section_header(text):
                continue
            if EMAIL_REGEX.search(text) or PHONE_REGEX.search(text) or LINKEDIN_REGEX.search(text) or GITHUB_REGEX.search(text) or URL_REGEX.search(text):
                continue
            if name and lowered == name.strip().lower():
                continue
            words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
            if not words or len(words) > 8 or any(word.lower() in role_words for word in words):
                continue
            if any(word.lower() in skill_words for word in words):
                continue
            if len(words) == 1 and skills_extractor.extract([TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)], section_name="SKILLS"):
                continue
            if text.endswith((".", ";", ":")) or len(words) > 5:
                continue
            score = 0.0
            if len(words) == 1 and text[:1].isupper() and len(text) >= 4:
                score += 5.0
            if "," in text and LOCATION_PATTERN.fullmatch(text):
                score += 5.0
            if any(char.isdigit() for char in text) and "," in text:
                score += 4.0
            if len(words) <= 3:
                score += 1.0
            score += max(0.0, 2.0 - index * 0.1)
            if score >= 5.0:
                candidates.append((score, text))

        return max(candidates, key=lambda item: item[0])[1] if candidates else None
