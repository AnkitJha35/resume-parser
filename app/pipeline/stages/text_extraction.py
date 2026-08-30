from __future__ import annotations

import unicodedata

import fitz
from dataclasses import dataclass


# Invisible formatting noise from PDF writers; not resume content.
_ZWSP = "\u200b"


@dataclass
class TextBlock:
    text: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float | None = None
    bold: bool | None = None


class PDFExtractor:
    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """Normalize compatibility characters (e.g. ligatures) and strip U+200B.

        NFKC expands PDF ligatures (ﬁ/ﬂ/ﬀ/…) without fixture-specific maps.
        ZWSP is removed afterward without inserting spaces (Phase 6I).
        """
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        return text.replace(_ZWSP, "")

    @staticmethod
    def keep_extracted_fragment(text: str) -> bool:
        """False when cleanup leaves only empty/whitespace content."""
        return bool(PDFExtractor.clean_extracted_text(text).strip())

    @staticmethod
    def extract(raw_pdf_bytes: bytes) -> list[TextBlock]:
        blocks: list[TextBlock] = []

        with fitz.open(stream=raw_pdf_bytes, filetype="pdf") as document:
            for page_index, page in enumerate(document, start=1):
                page_blocks = PDFExtractor._extract_page_blocks(page, page_index)
                blocks.extend(page_blocks)

        return blocks

    @staticmethod
    def _extract_page_blocks(page: fitz.Page, page_number: int) -> list[TextBlock]:
        text_blocks: list[TextBlock] = []
        blocks = page.get_text("dict").get("blocks", [])

        for block in blocks:
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                spans = sorted(spans, key=lambda span: span.get("bbox", [0.0, 0.0, 0.0, 0.0])[0])
                kept_spans: list[dict] = []
                text_parts: list[str] = []
                for span in spans:
                    cleaned = PDFExtractor.clean_extracted_text(span.get("text", "")).strip()
                    if not cleaned:
                        continue
                    kept_spans.append(span)
                    text_parts.append(cleaned)
                if not text_parts:
                    # ZWSP-only / empty fragments must not enter reconstruction.
                    continue

                text = " ".join(text_parts)
                x0 = min(span.get("bbox", [0.0, 0.0, 0.0, 0.0])[0] for span in kept_spans)
                y0 = min(span.get("bbox", [0.0, 0.0, 0.0, 0.0])[1] for span in kept_spans)
                x1 = max(span.get("bbox", [0.0, 0.0, 0.0, 0.0])[2] for span in kept_spans)
                y1 = max(span.get("bbox", [0.0, 0.0, 0.0, 0.0])[3] for span in kept_spans)

                first_span = kept_spans[0]
                font_size = first_span.get("size")
                font_flags = first_span.get("flags", 0)
                bold = bool(font_flags & 2)

                text_blocks.append(
                    TextBlock(
                        text=text,
                        page_number=page_number,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        font_size=font_size,
                        bold=bold,
                    )
                )

        return text_blocks
