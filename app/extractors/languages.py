"""Minimal LANGUAGES section cleanup at Resume assembly."""

from __future__ import annotations

import re
from typing import Iterable

from app.pipeline.stages.text_extraction import TextBlock

_NUMERIC_ONLY_RE = re.compile(r"^\d+(?:\.\d+)?$")
# Require whitespace (and optional separator) before a trailing rating so
# tokens like "Python3" are not stripped merely for ending in a digit.
_TRAILING_RATING_RE = re.compile(
    r"^(?P<name>.+?)\s+(?:[-–—:]\s*)?(?P<rating>\d{1,2}(?:\.\d+)?)$"
)


def clean_language_value(text: str | None) -> str | None:
    """Return a language name, or None for empty / numeric-only rating tokens."""
    value = (text or "").strip()
    if not value:
        return None
    if _NUMERIC_ONLY_RE.fullmatch(value):
        return None
    trailing = _TRAILING_RATING_RE.fullmatch(value)
    if trailing:
        name = trailing.group("name").strip(" \t-–—:")
        if name and not _NUMERIC_ONLY_RE.fullmatch(name):
            return name
    return value


def clean_language_texts(texts: Iterable[str | None]) -> list[str]:
    cleaned: list[str] = []
    for text in texts:
        value = clean_language_value(text)
        if value:
            cleaned.append(value)
    return cleaned


def clean_language_blocks(blocks: Iterable[TextBlock]) -> list[str]:
    return clean_language_texts(getattr(block, "text", None) for block in blocks)
