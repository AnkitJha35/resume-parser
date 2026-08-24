from __future__ import annotations

import re
from typing import Iterable

from app.pipeline.stages.confidence import ConfidenceScorer
from app.extractors.date_parser import DateRangeParser
from app.pipeline.stages.text_extraction import TextBlock

URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
CREDENTIAL_ID_PATTERN = re.compile(r"\b(?:credential id|certification id|cert id|id)\b[:\-]?\s*(\S+)", re.IGNORECASE)
ISSUE_DATE_PATTERN = re.compile(r"\b(?:issued on|issue date|issued)\b[:\-]?\s*(.+)", re.IGNORECASE)
EXPIRY_DATE_PATTERN = re.compile(r"\b(?:valid until|valid through|valid thru|expires|expiry date|expiration date)\b[:\-]?\s*(.+)", re.IGNORECASE)


class CertificationExtractor:
    def __init__(self) -> None:
        self.date_parser = DateRangeParser()
        self._confidence = ConfidenceScorer()

    def extract(self, blocks: Iterable[TextBlock]) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        current_entry = self._new_entry()

        for block in blocks:
            text = block.text.strip()
            if not text:
                if self._has_content(current_entry):
                    entries.append(current_entry)
                    current_entry = self._new_entry()
                continue

            if self._is_url(text):
                current_entry["credentialUrl"] = text
                continue

            credential_id = self._extract_credential_id(text)
            if credential_id:
                current_entry["credentialId"] = credential_id
                continue

            issue_date = self._extract_issue_date(text)
            if issue_date:
                current_entry["issueDate"] = issue_date
                continue

            expiry_date = self._extract_expiry_date(text)
            if expiry_date:
                current_entry["expiryDate"] = expiry_date
                continue

            if text.lower().startswith("programming languages:"):
                continue

            if current_entry["name"] is None:
                current_entry["name"] = text
                continue

            if current_entry["issuingOrganization"] is None:
                current_entry["issuingOrganization"] = text
                continue

            description = current_entry.get("description", "")
            current_entry["description"] = "\n".join(filter(None, [description, text])).strip()

        if self._has_content(current_entry):
            entries.append(current_entry)

        return entries

    def _new_entry(self) -> dict[str, object]:
        return {
            "name": None,
            "issuingOrganization": None,
            "issueDate": None,
            "expiryDate": None,
            "credentialId": None,
            "credentialUrl": None,
            "description": None,
            "confidence": self._confidence.section_extraction(),
        }

    def _has_content(self, entry: dict[str, object]) -> bool:
        return any(entry[key] for key in ["name", "issuingOrganization", "issueDate", "expiryDate", "credentialId", "credentialUrl", "description"])

    def _is_url(self, text: str) -> bool:
        return bool(URL_PATTERN.search(text))

    def _extract_credential_id(self, text: str) -> str | None:
        match = CREDENTIAL_ID_PATTERN.search(text)
        return match.group(1).strip() if match else None

    def _extract_issue_date(self, text: str) -> str | None:
        match = ISSUE_DATE_PATTERN.search(text)
        if match:
            return self._parse_date_token(match.group(1).strip())
        return None

    def _extract_expiry_date(self, text: str) -> str | None:
        match = EXPIRY_DATE_PATTERN.search(text)
        if match:
            return self._parse_date_token(match.group(1).strip())
        return None

    def _parse_date_token(self, token: str) -> str | None:
        if not token:
            return None

        range_result = self.date_parser.parse(token)
        if range_result and range_result.startDate is not None and range_result.endDate is not None:
            return range_result.endDate

        return DateRangeParser._parse_date_token(token)
