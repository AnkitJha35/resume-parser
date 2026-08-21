from __future__ import annotations

from typing import Any

from app.domain.resume import Resume
from app.pipeline.stages.pdf_detection import PDFDetectionResult
from app.pipeline.stages.text_extraction import TextBlock


class PipelineContext:
    def __init__(self, raw_pdf_bytes: bytes) -> None:
        self.raw_pdf_bytes = raw_pdf_bytes
        self.pdf_detection: PDFDetectionResult | None = None
        self.text_blocks: list[TextBlock] = []
        self.ordered_blocks: list[TextBlock] = []
        self.normalized_blocks: list[TextBlock] = []
        self.sections: dict[str, list[TextBlock]] = {}
        self.partial_result: dict[str, Any] = {}
        self.resume: Resume | None = None
        self.error: dict[str, Any] | None = None
