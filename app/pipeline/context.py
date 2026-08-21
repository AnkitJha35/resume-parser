from __future__ import annotations

from typing import Any

from app.domain.resume import Resume
from app.pipeline.stages.pdf_detection import PDFDetectionResult
from app.pipeline.stages.text_extraction import TextBlock
from app.pipeline.stages.block_classification import ClassifiedBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup


class PipelineContext:
    def __init__(self, raw_pdf_bytes: bytes) -> None:
        self.raw_pdf_bytes = raw_pdf_bytes
        self.pdf_detection: PDFDetectionResult | None = None
        self.text_blocks: list[TextBlock] = []
        self.ordered_blocks: list[TextBlock] = []
        self.normalized_blocks: list[TextBlock] = []
        self.sections: dict[str, list[TextBlock]] = {}
        self.classified_sections: dict[str, list[ClassifiedBlock]] = {}
        self.candidate_groups: dict[str, list[CandidateGroup]] = {}
        self.partial_result: dict[str, Any] = {}
        self.resume: Resume | None = None
        self.error: dict[str, Any] | None = None
