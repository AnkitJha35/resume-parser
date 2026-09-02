"""Benchmark runner for executing real resume evaluations."""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.domain.resume import Resume
from app.pipeline.parser import ResumeParser
from tests.benchmark.expectations import evaluate_status
from tests.benchmark.metadata import BENCHMARK_FIXTURES, DocumentArchetype, ResumeMetadata

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class ResumeParseResult:
    filename: str
    archetype: str
    target_domain: str
    parser_success: bool
    status: str  # PASS, PARTIAL, FAIL
    elapsed_seconds: float
    error_message: str | None = None
    error_traceback: str | None = None

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

    # Diagnostics & Notes
    diagnostics: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BenchmarkRunSummary:
    total_resumes: int = 0
    success_count: int = 0
    error_count: int = 0
    pass_count: int = 0
    partial_count: int = 0
    fail_count: int = 0
    total_elapsed_seconds: float = 0.0
    archetype_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    results: list[ResumeParseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Executes layout-pipeline parsing across benchmark resumes and captures metrics."""

    def __init__(self, fixtures_dir: Path | None = None, parser: ResumeParser | None = None):
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        self.parser = parser or ResumeParser()

    def discover_fixtures(self) -> list[Path]:
        """Discover existing PDF benchmark fixtures in the fixtures directory."""
        found: list[Path] = []
        for filename in BENCHMARK_FIXTURES:
            path = self.fixtures_dir / filename
            if path.exists():
                found.append(path)
        return sorted(found, key=lambda p: p.name)

    def run_single(self, pdf_path: Path) -> ResumeParseResult:
        """Parse a single PDF resume and produce structured metrics."""
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
            resume: Resume = self.parser.parse_with_layout_pipeline(raw_bytes)
            elapsed = time.perf_counter() - start_t

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

            return ResumeParseResult(
                filename=filename,
                archetype=metadata.archetype.value,
                target_domain=metadata.target_domain,
                parser_success=True,
                status=status,
                elapsed_seconds=round(elapsed, 4),
                personal=personal_dict,
                skills_count=len(resume.skills),
                skills_sample=resume.skills[:10],
                experience_count=len(resume.experience),
                experience_items=exp_items,
                education_count=len(resume.education),
                education_items=edu_items,
                projects_count=len(resume.projects),
                projects_items=proj_items,
                diagnostics=diagnostics,
                notes=metadata.notes,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start_t
            return ResumeParseResult(
                filename=filename,
                archetype=metadata.archetype.value,
                target_domain=metadata.target_domain,
                parser_success=False,
                status="FAIL",
                elapsed_seconds=round(elapsed, 4),
                error_message=str(exc),
                error_traceback=traceback.format_exc(),
                diagnostics=[f"PARSER_EXCEPTION: {exc}"],
                notes=f"Failed with exception: {exc}",
            )

    def run_all(self, pdf_paths: list[Path] | None = None) -> BenchmarkRunSummary:
        """Run the full benchmark suite across all discovered fixtures."""
        paths = pdf_paths or self.discover_fixtures()
        start_t = time.perf_counter()

        results: list[ResumeParseResult] = []
        for path in paths:
            result = self.run_single(path)
            results.append(result)

        total_elapsed = time.perf_counter() - start_t

        summary = BenchmarkRunSummary(
            total_resumes=len(results),
            success_count=sum(1 for r in results if r.parser_success),
            error_count=sum(1 for r in results if not r.parser_success),
            pass_count=sum(1 for r in results if r.status == "PASS"),
            partial_count=sum(1 for r in results if r.status == "PARTIAL"),
            fail_count=sum(1 for r in results if r.status == "FAIL"),
            total_elapsed_seconds=round(total_elapsed, 4),
            results=results,
        )

        # Compute breakdown by archetype
        breakdown: dict[str, dict[str, int]] = {}
        for r in results:
            arch = r.archetype
            if arch not in breakdown:
                breakdown[arch] = {"total": 0, "PASS": 0, "PARTIAL": 0, "FAIL": 0}
            breakdown[arch]["total"] += 1
            breakdown[arch][r.status] = breakdown[arch].get(r.status, 0) + 1
        summary.archetype_breakdown = breakdown

        return summary
