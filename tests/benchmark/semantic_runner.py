"""Semantic benchmark runner for evaluating real resume PDFs with LLM extraction."""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.domain.resume import Resume
from app.extractors.semantic_extractor import (
    SemanticExtractionError,
    SemanticExtractor,
    SemanticValidationError,
)
from app.domain.document import document_from_text_blocks
from app.pipeline.stages.text_extraction import PDFExtractor
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.semantic_pipeline import parse_document_semantically
from tests.benchmark.expectations import evaluate_status
from tests.benchmark.metadata import BENCHMARK_FIXTURES, DocumentArchetype, ResumeMetadata

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class SemanticParseResult:
    filename: str
    archetype: str
    target_domain: str
    semantic_success: bool
    status: str  # PASS, VALIDATION_FAILED, EXTRACTION_FAILED, PARSER_EXCEPTION
    failure_type: str | None = None  # None, VALIDATION_ERROR, EXTRACTION_ERROR, PARSER_EXCEPTION
    error_message: str | None = None
    validation_violations: list[str] = field(default_factory=list)
    passed_validation: bool = False
    elapsed_seconds: float = 0.0

    # Token/cost instrumentation (None if provider doesn't report)
    usage: dict[str, int | None] | None = None

    # Extracted core fields
    personal: dict[str, Any] = field(default_factory=dict)
    skills_count: int = 0
    skills_sample: list[str] = field(default_factory=list)
    experience_count: int = 0
    experience_items: list[dict[str, Any]] = field(default_factory=list)
    education_count: int = 0
    education_items: list[dict[str, Any]] = field(default_factory=list)
    projects_count: int = 0
    projects_items: list[dict[str, Any]] = field(default_factory=list)
    certifications_count: int = 0
    languages_count: int = 0
    achievements_count: int = 0

    # Diagnostics & Notes
    diagnostics: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticBenchmarkSummary:
    timestamp: str
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    representation: str = "candidate_b_compact"
    total_cases: int = 0
    successful_cases: int = 0
    extraction_failures: int = 0
    validation_failures: int = 0
    parser_exceptions: int = 0
    grounded_semantic_outputs: int = 0
    outputs_rejected_by_validation: int = 0
    validation_pass_rate_pct: float = 0.0
    total_elapsed_seconds: float = 0.0
    total_tokens: int | None = None
    archetype_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    results: list[SemanticParseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticBenchmarkRunner:
    """Executes semantic LLM extraction across benchmark resumes and captures grounded metrics."""

    def __init__(
        self,
        extractor: SemanticExtractor | None = None,
        fixtures_dir: Path | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        if extractor is None:
            from app.extractors.providers.ollama import OllamaSemanticExtractor
            self.extractor = OllamaSemanticExtractor()
            self.provider_name = provider_name or "ollama"
            self.model_name = model_name or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        else:
            self.extractor = extractor
            if provider_name:
                self.provider_name = provider_name
            else:
                cname = type(extractor).__name__.lower()
                if "ollama" in cname:
                    self.provider_name = "ollama"
                elif "gemini" in cname:
                    self.provider_name = "gemini"
                elif "mock" in cname:
                    self.provider_name = "mock"
                else:
                    self.provider_name = "custom"

            if model_name:
                self.model_name = model_name
            else:
                self.model_name = (
                    getattr(extractor, "_explicit_model", None)
                    or getattr(extractor, "model", None)
                    or "default"
                )

    def discover_fixtures(self, fixture_name: str | None = None) -> list[Path]:
        """Discover existing PDF benchmark fixtures in the fixtures directory.

        If fixture_name is provided, validates that it is a registered benchmark fixture
        and returns only that fixture path.

        Raises:
            ValueError: If fixture_name is not registered in BENCHMARK_FIXTURES or not found on disk.
        """
        if fixture_name:
            if fixture_name not in BENCHMARK_FIXTURES:
                registered = ", ".join(sorted(BENCHMARK_FIXTURES.keys()))
                raise ValueError(
                    f"Unknown benchmark fixture '{fixture_name}'. "
                    f"Available registered fixtures:\n  {registered}"
                )
            path = self.fixtures_dir / fixture_name
            if not path.exists():
                raise ValueError(
                    f"Fixture file '{fixture_name}' is registered but does not exist at '{path}'."
                )
            return [path]

        found: list[Path] = []
        for filename in BENCHMARK_FIXTURES:
            path = self.fixtures_dir / filename
            if path.exists():
                found.append(path)
        return sorted(found, key=lambda p: p.name)

    def run_single(self, pdf_path: Path) -> SemanticParseResult:
        """Parse a single PDF resume through the semantic extraction pipeline."""
        filename = pdf_path.name
        metadata = BENCHMARK_FIXTURES.get(
            filename,
            ResumeMetadata(
                filename=filename,
                archetype=DocumentArchetype.STANDARD_CV,
                candidate_name="",
                target_domain="Unknown",
                page_count_estimate=1,
                has_tables=False,
                notes="",
            ),
        )

        start_t = time.perf_counter()
        try:
            raw_bytes = pdf_path.read_bytes()
            # 1. Document IR extraction
            physical_doc = document_from_text_blocks(PDFExtractor.extract(raw_bytes))
            reconstructed_doc = reconstruct_document(physical_doc)
            layout_doc = interpret_layout(reconstructed_doc)

            # 2. Semantic execution through trust boundary
            resume = parse_document_semantically(
                layout_doc,
                self.extractor,
                document_id=filename,
            )
            elapsed = time.perf_counter() - start_t

            # Capture usage if provider tracked it
            usage = getattr(self.extractor, "last_usage_metadata", None)

            status, diagnostics = evaluate_status(resume, filename)

            personal_dict = {
                "name": resume.personal.name,
                "email": resume.personal.email,
                "phone": resume.personal.phone,
                "location": resume.personal.location,
                "linkedin": resume.personal.linkedin,
                "github": resume.personal.github,
            }

            exp_items = [
                {
                    "company": exp.company,
                    "designation": exp.designation,
                    "startDate": exp.startDate,
                    "endDate": exp.endDate,
                    "current": exp.current,
                    "location": exp.location,
                }
                for exp in resume.experience
            ]

            edu_items = [
                {
                    "institution": edu.institution,
                    "degree": edu.degree,
                    "fieldOfStudy": edu.fieldOfStudy,
                    "startDate": edu.startDate,
                    "endDate": edu.endDate,
                    "grade": edu.grade,
                }
                for edu in resume.education
            ]

            proj_items = [
                {
                    "name": prj.name,
                    "startDate": prj.startDate,
                    "endDate": prj.endDate,
                    "current": prj.current,
                    "technologies": prj.technologies or [],
                }
                for prj in resume.projects
            ]

            return SemanticParseResult(
                filename=filename,
                archetype=metadata.archetype.value,
                target_domain=metadata.target_domain,
                semantic_success=True,
                status=status,
                passed_validation=True,
                elapsed_seconds=round(elapsed, 4),
                usage=usage,
                personal=personal_dict,
                skills_count=len(resume.skills),
                skills_sample=resume.skills[:10],
                experience_count=len(resume.experience),
                experience_items=exp_items,
                education_count=len(resume.education),
                education_items=edu_items,
                projects_count=len(resume.projects),
                projects_items=proj_items,
                certifications_count=len(resume.certifications),
                languages_count=len(resume.languages),
                achievements_count=len(resume.achievements),
                diagnostics=diagnostics,
                notes=metadata.notes,
            )

        except SemanticValidationError as exc:
            elapsed = time.perf_counter() - start_t
            usage = getattr(self.extractor, "last_usage_metadata", None)
            return SemanticParseResult(
                filename=filename,
                archetype=metadata.archetype.value,
                target_domain=metadata.target_domain,
                semantic_success=False,
                status="VALIDATION_FAILED",
                failure_type="VALIDATION_ERROR",
                passed_validation=False,
                elapsed_seconds=round(elapsed, 4),
                usage=usage,
                error_message=str(exc),
                validation_violations=list(getattr(exc, "violations", [])),
                diagnostics=[f"VALIDATION_VIOLATION: {v}" for v in getattr(exc, "violations", [])],
                notes=f"Semantic output failed validation invariants: {exc}",
            )

        except SemanticExtractionError as exc:
            elapsed = time.perf_counter() - start_t
            usage = getattr(self.extractor, "last_usage_metadata", None)
            return SemanticParseResult(
                filename=filename,
                archetype=metadata.archetype.value,
                target_domain=metadata.target_domain,
                semantic_success=False,
                status="EXTRACTION_FAILED",
                failure_type="EXTRACTION_ERROR",
                passed_validation=False,
                elapsed_seconds=round(elapsed, 4),
                usage=usage,
                error_message=str(exc),
                diagnostics=[f"EXTRACTION_ERROR: {exc}"],
                notes=f"Extraction failed: {exc}",
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start_t
            return SemanticParseResult(
                filename=filename,
                archetype=metadata.archetype.value,
                target_domain=metadata.target_domain,
                semantic_success=False,
                status="PARSER_EXCEPTION",
                failure_type="PARSER_EXCEPTION",
                passed_validation=False,
                elapsed_seconds=round(elapsed, 4),
                error_message=str(exc),
                diagnostics=[f"PARSER_EXCEPTION: {exc}"],
                notes=f"Unhandled exception: {exc}",
            )

    def run_all(self, pdf_paths: list[Path] | None = None) -> SemanticBenchmarkSummary:
        """Run semantic extraction across all discovered fixtures."""
        paths = pdf_paths or self.discover_fixtures()
        start_t = time.perf_counter()

        results: list[SemanticParseResult] = []
        for path in paths:
            result = self.run_single(path)
            results.append(result)

        total_elapsed = time.perf_counter() - start_t
        successful = sum(1 for r in results if r.semantic_success)
        val_fails = sum(1 for r in results if r.failure_type == "VALIDATION_ERROR")
        ext_fails = sum(1 for r in results if r.failure_type == "EXTRACTION_ERROR")
        exc_fails = sum(1 for r in results if r.failure_type == "PARSER_EXCEPTION")

        # Grounded semantic outputs formed (successful + validation failed)
        grounded_count = sum(1 for r in results if r.passed_validation or r.failure_type == "VALIDATION_ERROR")

        pass_rate = (successful / len(results) * 100) if results else 0.0

        # Sum total tokens if available
        total_tokens: int | None = None
        for r in results:
            if r.usage and r.usage.get("total_tokens") is not None:
                total_tokens = (total_tokens or 0) + int(r.usage["total_tokens"])

        # Compute breakdown by archetype
        breakdown: dict[str, dict[str, int]] = {}
        for r in results:
            arch = r.archetype
            if arch not in breakdown:
                breakdown[arch] = {"total": 0, "SUCCESS": 0, "VALIDATION_FAILED": 0, "EXTRACTION_FAILED": 0, "EXCEPTION": 0}
            breakdown[arch]["total"] += 1
            if r.semantic_success:
                breakdown[arch]["SUCCESS"] += 1
            elif r.failure_type == "VALIDATION_ERROR":
                breakdown[arch]["VALIDATION_FAILED"] += 1
            elif r.failure_type == "EXTRACTION_ERROR":
                breakdown[arch]["EXTRACTION_FAILED"] += 1
            else:
                breakdown[arch]["EXCEPTION"] += 1

        representation_name = "candidate_b_compact"
        if getattr(self.extractor, "_compact", None) is False:
            representation_name = "full_reference"

        return SemanticBenchmarkSummary(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            provider=self.provider_name,
            model=self.model_name,
            representation=representation_name,
            total_cases=len(results),
            successful_cases=successful,
            extraction_failures=ext_fails,
            validation_failures=val_fails,
            parser_exceptions=exc_fails,
            grounded_semantic_outputs=grounded_count,
            outputs_rejected_by_validation=val_fails,
            validation_pass_rate_pct=round(pass_rate, 2),
            total_elapsed_seconds=round(total_elapsed, 4),
            total_tokens=total_tokens,
            archetype_breakdown=breakdown,
            results=results,
        )
