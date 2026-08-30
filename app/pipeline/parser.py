from __future__ import annotations

from typing import Any

from app.domain.resume import Resume, validate_resume
from app.extractors.certifications import CertificationExtractor
from app.extractors.contact import ContactExtractor
from app.extractors.education import EducationExtractor
from app.extractors.experience import ExperienceExtractor
from app.extractors.languages import clean_language_blocks
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
from app.pipeline.stages.sections import SECTION_NAMES
from app.domain.document import document_from_text_blocks
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.semantic_compat import (
    semantic_header_to_text_blocks,
    semantic_sections_to_text_blocks,
)
from app.pipeline.stages.semantic_paths import detect_region_aware_sections
from app.pipeline.stages.structural_roles import build_structural_blocks
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.candidate_entries import build_candidate_entries


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
        # Extract header blocks (blocks before the first recognized section heading)
        header_blocks = self._header_blocks(context.normalized_blocks)
        context.partial_result["personal"] = self.contact_extractor.extract(header_blocks)
        context.partial_result["skills"] = [skill["value"] for skill in self.skills_extractor.extract(context.sections.get("SKILLS", []) or context.normalized_blocks, section_name="SKILLS")]
        context.partial_result["experience"] = self.experience_extractor.extract(
            context.sections.get("EXPERIENCE", []),
            groups=context.candidate_groups.get("EXPERIENCE")
        )
        context.partial_result["education"] = self.education_extractor.extract(
            [block.text for block in context.sections.get("EDUCATION", []) or []],
            groups=context.candidate_groups.get("EDUCATION"),
        )
        context.partial_result["projects"] = self.project_extractor.extract(
            context.sections.get("PROJECTS", []) or [],
            groups=context.candidate_groups.get("PROJECTS"),
        )
        # Summary: join non-empty blocks from the SUMMARY section into a single paragraph
        summary_blocks = context.sections.get("SUMMARY", []) or []
        summary_text = " ".join(b.text.strip() for b in summary_blocks if (b.text or "").strip())
        context.partial_result["summary"] = summary_text or None
        context.partial_result["certifications"] = self.certification_extractor.extract(context.sections.get("CERTIFICATIONS", []) or [])
        context.partial_result["achievements"] = [block.text for block in context.sections.get("ACHIEVEMENTS", []) or []]
        context.partial_result["languages"] = clean_language_blocks(context.sections.get("LANGUAGES", []) or [])
        context.partial_result["metadata"] = {
            "pageCount": context.pdf_detection.page_count,
            "ocrUsed": False,
        }

        context.resume = validate_resume(context.partial_result)
        return context.resume

    def parse_with_layout_pipeline(self, raw_pdf_bytes: bytes) -> Resume:
        """Run semantic extraction from the opt-in layout-aware compatibility path."""
        detection = self.pdf_detector.detect(raw_pdf_bytes)
        if not detection.is_pdf:
            raise PipelineError("INVALID_PDF", "The file is not a valid PDF.")
        if not detection.has_text:
            raise PipelineError("PDF_EXTRACTION_FAILED", "Unable to extract meaningful text.")

        physical_document = document_from_text_blocks(PDFExtractor.extract(raw_pdf_bytes))
        reconstructed_document = reconstruct_document(physical_document)
        layout_document = interpret_layout(reconstructed_document)
        # Phase 1–3 seams: StructuralBlocks → CandidateSections → CandidateEntries.
        # Resume assembly still uses detect_region_aware_sections until equivalence
        # is proven for multi-path continuation and extractor inputs.
        structural_blocks = build_structural_blocks(layout_document)
        candidate_sections = build_candidate_sections(structural_blocks, self.section_detector)
        build_candidate_entries(candidate_sections)
        semantic_document = detect_region_aware_sections(layout_document, self.section_detector)
        sections = semantic_sections_to_text_blocks(semantic_document)
        classified_sections = {
            section: [classify_block(block) for block in blocks]
            for section, blocks in sections.items()
        }
        candidate_groups = {
            section: group_candidates(classified, section)
            for section, classified in classified_sections.items()
            if section != "UNASSIGNED"
        }

        header_blocks = semantic_header_to_text_blocks(semantic_document)
        partial_result: dict[str, Any] = {
            "parserVersion": "1.0.0",
            "personal": self.contact_extractor.extract(header_blocks),
            "skills": [
                skill["value"]
                for skill in self.skills_extractor.extract(
                    sections.get("SKILLS", []), section_name="SKILLS"
                )
            ],
            "experience": self.experience_extractor.extract(
                sections.get("EXPERIENCE", []), groups=candidate_groups.get("EXPERIENCE")
            ),
            "education": self.education_extractor.extract(
                [block.text for block in sections.get("EDUCATION", [])],
                groups=candidate_groups.get("EDUCATION"),
            ),
            "projects": self.project_extractor.extract(
                sections.get("PROJECTS", []), groups=candidate_groups.get("PROJECTS")
            ),
            "summary": " ".join(block.text.strip() for block in sections.get("SUMMARY", [])) or None,
            "certifications": self.certification_extractor.extract(sections.get("CERTIFICATIONS", [])),
            "achievements": [block.text for block in sections.get("ACHIEVEMENTS", [])],
            "languages": clean_language_blocks(sections.get("LANGUAGES", [])),
            "metadata": {"pageCount": detection.page_count, "ocrUsed": False},
        }
        return validate_resume(partial_result)

    def _build_resume(self, partial_result: dict[str, Any]) -> Resume:
        return Resume(
            parserVersion="1.0.0",
            personal=partial_result["personal"],
            summary=partial_result.get("summary"),
            skills=partial_result["skills"],
            experience=partial_result["experience"],
            education=partial_result["education"],
            projects=partial_result["projects"],
            certifications=partial_result["certifications"],
            achievements=partial_result["achievements"],
            languages=partial_result["languages"],
            metadata={}
        )

    def _header_blocks(self, normalized_blocks: list) -> list:
        """Return the sequence of blocks before the first recognized section header.

        Uses SECTION_NAMES to detect headings (exact match of trimmed uppercase text).
        """
        blocks = list(normalized_blocks or [])
        for idx, b in enumerate(blocks):
            text = (getattr(b, "text", "") or "").strip()
            if not text:
                continue
            if text.strip().upper() in SECTION_NAMES:
                return blocks[:idx]
        return blocks
