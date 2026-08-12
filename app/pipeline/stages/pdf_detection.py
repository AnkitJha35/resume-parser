from __future__ import annotations

import fitz


class PDFDetectionResult:
    def __init__(self, is_pdf: bool, has_text: bool, page_count: int) -> None:
        self.is_pdf = is_pdf
        self.has_text = has_text
        self.page_count = page_count


class PDFDetector:
    @staticmethod
    def detect(raw_pdf_bytes: bytes) -> PDFDetectionResult:
        try:
            with fitz.open(stream=raw_pdf_bytes, filetype="pdf") as document:
                page_count = document.page_count
                has_text = any(page.get_text().strip() for page in document)
                return PDFDetectionResult(is_pdf=True, has_text=has_text, page_count=page_count)
        except Exception:
            return PDFDetectionResult(is_pdf=False, has_text=False, page_count=0)
