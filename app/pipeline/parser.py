from __future__ import annotations

from typing import Any

from app.domain.resume import Resume, validate_resume
from app.extractors.certifications import CertificationExtractor
from app.extractors.contact import ContactExtractor
from app.extractors.education import EducationExtractor
from app.extractors.experience import ExperienceExtractor
from app.extractors.projects import ProjectExtractor
from app.extractors.skills import SkillsExtractor
from app.pipeline.context import PipelineContext
from app.pipeline.stages.confidence import ConfidenceScorer
from app.pipeline.stages.pdf_detection import PDFDetector
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.normalization import TextNormalizer
from app.pipeline.stages.text_extraction import PDFExtractor
from app.pipeline.stages.reading_order import ReadingOrder
from app.pipeline.stages.block_classification import classify_block
from app.pipeline.stages.candidate_grouping import group_candidates


class PipelineError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class ResumeParser:
    def __init__(self) -> None:
        self.pdf_detector = PDFDetector()
        self.section_detector = SectionDetector()
        self.confidence_scorer = ConfidenceScorer()
        self.contact_extractor = ContactExtractor()
        self.skills_extractor = SkillsExtractor()
        self.experience_extractor = ExperienceExtractor()
        self.education_extractor = EducationExtractor()
        self.project_extractor = ProjectExtractor()
        self.certification_extractor = CertificationExtractor()

    def parse(self, raw_pdf_bytes: bytes) -> Resume:
        context = PipelineContext(raw_pdf_bytes)

        context.pdf_detection = self.pdf_detector.detect(raw_pdf_bytes)
        if not context.pdf_detection.is_pdf:
            raise PipelineError("INVALID_PDF", "The file is not a valid PDF.")

        if not context.pdf_detection.has_text:
            raise PipelineError("PDF_EXTRACTION_FAILED", "Unable to extract meaningful text.")

        context.text_blocks = PDFExtractor.extract(raw_pdf_bytes)
        # Apply column-aware reading order before normalization
        context.ordered_blocks = ReadingOrder.reorder(context.text_blocks)
        context.normalized_blocks = TextNormalizer.normalize_blocks(context.ordered_blocks)
        context.sections = self.section_detector.detect(context.normalized_blocks)

        # Classify blocks in each section
        for section_name, blocks in context.sections.items():
            context.classified_sections[section_name] = [classify_block(b) for b in blocks]

        # Group the classified blocks
        for section_name, classified_blocks in context.classified_sections.items():
            context.candidate_groups[section_name] = group_candidates(classified_blocks, section_name)

        context.partial_result["parserVersion"] = "1.0.0"
        context.partial_result["personal"] = self.contact_extractor.extract(context.sections.get("SUMMARY", []) or context.normalized_blocks)
        context.partial_result["skills"] = [skill["value"] for skill in self.skills_extractor.extract(context.sections.get("SKILLS", []) or context.normalized_blocks, section_name="SKILLS")]
        context.partial_result["experience"] = self.experience_extractor.extract(
            context.sections.get("EXPERIENCE", []),
            groups=context.candidate_groups.get("EXPERIENCE")
        )
        context.partial_result["education"] = self.education_extractor.extract([block.text for block in context.sections.get("EDUCATION", []) or []])
        context.partial_result["projects"] = self.project_extractor.extract(context.sections.get("PROJECTS", []) or [])
        context.partial_result["certifications"] = self.certification_extractor.extract(context.sections.get("CERTIFICATIONS", []) or [])
        context.partial_result["achievements"] = [block.text for block in context.sections.get("ACHIEVEMENTS", []) or []]
        context.partial_result["languages"] = [block.text for block in context.sections.get("LANGUAGES", []) or []]
        context.partial_result["metadata"] = {
            "pageCount": context.pdf_detection.page_count,
            "ocrUsed": False,
        }

        context.resume = validate_resume(context.partial_result)
        return context.resume

    def _build_resume(self, partial_result: dict[str, Any]) -> Resume:
        return Resume(
            parserVersion="1.0.0",
            personal=partial_result["personal"],
            summary=None,
            skills=partial_result["skills"],
            experience=partial_result["experience"],
            education=partial_result["education"],
            projects=partial_result["projects"],
            certifications=partial_result["certifications"],
            achievements=partial_result["achievements"],
            languages=partial_result["languages"],
            metadata={}
        )
