from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from app.pipeline.stages.normalization import TextNormalizer


MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

CURRENT_TERMS = {"present", "current", "now"}

DATE_RANGE_PATTERN = re.compile(r"^\s*(?P<start>.+?)\s*[–—-]\s*(?P<end>.+?)\s*$", re.IGNORECASE)
SINGLE_DATE_PATTERN = re.compile(r"^\s*(?P<date>.+?)\s*$")


class DateRange(BaseModel):
    startDate: str
    endDate: str | None = None
    current: bool = False


class DateRangeParser:
    @staticmethod
    def parse(date_range_text: str) -> Optional[DateRange]:
        if not date_range_text or not date_range_text.strip():
            return None

        # Remove invisible Unicode characters commonly introduced by PDF extraction.
        date_range_text = (
            date_range_text.replace("\u200b", "")
            .replace("\u200c", "")
            .replace("\u200d", "")
            .replace("\ufeff", "")
            .strip()
        )

        # Strip optional leading label like "Date:" or "Date :" (case-insensitive)
        date_range_text = re.sub(
            r"^\s*date\s*:?\s*",
            "",
            date_range_text,
            flags=re.IGNORECASE,
        )

        match = DATE_RANGE_PATTERN.match(date_range_text)
        if not match:
            return None

        start_text = match.group("start").strip()
        end_text = match.group("end").strip()

        start_date = DateRangeParser._parse_date_token(start_text)
        if start_date is None:
            return None

        end_date = None
        current = False

        if DateRangeParser._is_current(end_text):
            current = True
        else:
            end_date = DateRangeParser._parse_date_token(end_text)
            if end_date is None:
                return None

        return DateRange(
            startDate=start_date,
            endDate=end_date,
            current=current,
        )

    @staticmethod
    def _is_current(token: str) -> bool:
        return token.strip().lower() in CURRENT_TERMS

    @staticmethod
    def _parse_date_token(token: str) -> Optional[str]:
        token = token.strip()
        if not token:
            return None

        # Month name + year
        month_year_match = re.match(r"^(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})$", token)
        if month_year_match:
            month = month_year_match.group("month").lower()
            year = month_year_match.group("year")
            month_number = MONTHS.get(month)
            if month_number:
                return f"{year}-{month_number}"
            return None

        # Numeric month/year (MM/YYYY or M/YYYY)
        numeric_month_match = re.match(r"^(?P<month>\d{1,2})[/-](?P<year>\d{4})$", token)
        if numeric_month_match:
            month = int(numeric_month_match.group("month"))
            year = numeric_month_match.group("year")
            if 1 <= month <= 12:
                return f"{year}-{month:02d}"
            return None

        # Year only
        year_match = re.match(r"^(?P<year>\d{4})$", token)
        if year_match:
            return year_match.group("year")

        return None
