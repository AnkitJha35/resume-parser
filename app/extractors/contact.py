from __future__ import annotations

import re
from typing import Iterable

from app.pipeline.stages.confidence import ConfidenceScorer
from app.pipeline.stages.text_extraction import TextBlock


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(
    r"(\+?\d{1,3}[ \-/.]?)?(?:\(\d{2,4}\)|\d{2,4})[ \-/.]?\d{3,4}[ \-/.]?\d{3,4}"
)
LINKEDIN_REGEX = re.compile(r"https?://(?:www\.)?linkedin\.com/[A-Za-z0-9_\-/]+", re.IGNORECASE)
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
        location = ContactExtractor._find_location(text_lines, email, phone, linkedin, github)

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
            if EMAIL_REGEX.search(line) or LINKEDIN_REGEX.search(line) or GITHUB_REGEX.search(line):
                continue
            match = PHONE_REGEX.search(line)
            if match:
                phone = match.group(0).strip()
                if len(re.sub(r"[^0-9]", "", phone)) >= 7:
                    return phone
        return None

    @staticmethod
    def _find_linkedin(lines: list[str]) -> str | None:
        for line in lines:
            match = LINKEDIN_REGEX.search(line)
            if match:
                return match.group(0).strip()
        return None

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
        for block in blocks[:5]:
            text = block.text.strip()
            if not text or EMAIL_REGEX.search(text) or PHONE_REGEX.search(text) or LINKEDIN_REGEX.search(text) or GITHUB_REGEX.search(text):
                continue
            if ContactExtractor._looks_like_name(text):
                return text

        for block in blocks[:3]:
            text = block.text.strip()
            if text and not EMAIL_REGEX.search(text) and not PHONE_REGEX.search(text) and not LINKEDIN_REGEX.search(text) and not GITHUB_REGEX.search(text):
                return text

        return None

    @staticmethod
    def _looks_like_name(text: str) -> bool:
        words = text.split()
        if not (1 < len(words) <= 4):
            return False
        return all(word[0].isupper() for word in words if word)

    @staticmethod
    def _find_location(lines: list[str], email: str | None, phone: str | None, linkedin: str | None, github: str | None) -> str | None:
        for line in lines:
            if EMAIL_REGEX.search(line) or PHONE_REGEX.search(line) or LINKEDIN_REGEX.search(line) or GITHUB_REGEX.search(line):
                continue
            if URL_REGEX.search(line):
                continue
            if LOCATION_PATTERN.search(line) and not any(char.isdigit() for char in line):
                return line.strip()
        return None
