from __future__ import annotations

import re
from typing import Iterable

from app.pipeline.stages.text_extraction import TextBlock


_LIGATURE_REPLACEMENTS = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

_CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]+")
_WHITESPACE_REGEX = re.compile(r"[ \t]+")


class TextNormalizer:
    _WHITESPACE_REGEX = _WHITESPACE_REGEX
    @staticmethod
    def normalize_blocks(blocks: Iterable[TextBlock]) -> list[TextBlock]:
        return [TextNormalizer._normalize_block(block) for block in blocks]

    @staticmethod
    def _normalize_block(block: TextBlock) -> TextBlock:
        normalized_text = TextNormalizer.normalize_text(block.text)
        return TextBlock(
            text=normalized_text,
            page_number=block.page_number,
            x0=block.x0,
            y0=block.y0,
            x1=block.x1,
            y1=block.y1,
            font_size=block.font_size,
            bold=block.bold,
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        if text is None:
            return ""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = TextNormalizer._replace_control_chars(normalized)
        normalized = TextNormalizer._replace_ligatures(normalized)

        lines = normalized.split("\n")
        normalized_lines = [TextNormalizer._WHITESPACE_REGEX.sub(" ", line).strip() for line in lines]

        return "\n".join(line for line in normalized_lines if line)

    @staticmethod
    def _replace_control_chars(text: str) -> str:
        return _CONTROL_CHARS_REGEX.sub(" ", text)

    @staticmethod
    def _replace_ligatures(text: str) -> str:
        for old, new in _LIGATURE_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text
        normalized_text = TextNormalizer.normalize_text(block.text)
        return TextBlock(
            text=normalized_text,
            page_number=block.page_number,
            x0=block.x0,
            y0=block.y0,
            x1=block.x1,
            y1=block.y1,
            font_size=block.font_size,
            bold=block.bold,
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        if text is None:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = TextNormalizer._replace_control_chars(text)
        text = TextNormalizer._replace_ligatures(text)

        lines = text.split("\n")
        normalized_lines = []
        for line in lines:
            clean = TextNormalizer._WHITESPACE_REGEX.sub(" ", line).strip()
            normalized_lines.append(clean)

        return "\n".join(normalized_lines).strip()

    @staticmethod
    def _replace_control_chars(text: str) -> str:
        return _CONTROL_CHARS_REGEX.sub(" ", text)

    @staticmethod
    def _replace_ligatures(text: str) -> str:
        for old, new in _LIGATURE_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text
