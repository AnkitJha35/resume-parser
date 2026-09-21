"""API v1 endpoints for resume parsing, diagnostics, fixtures, and benchmarks."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.v1.models import (
    BenchmarkRequest,
    CandidateResumeListResponse,
    ConfigResponse,
    DiagnosticBlockItem,
    DiagnosticPageItem,
    DiagnosticRegionItem,
    DiagnosticResponse,
    DiagnosticTableItem,
    FixtureItem,
    HealthResponse,
    ParseMetadata,
    ParseResponse,
    ParseStatus,
    ProvenanceRecord,
    ResumeListResponse,
    ResumeSnapshotResponse,
)
from app.core.config import Settings
from app.domain.document import document_from_text_blocks
from app.domain.resume import Resume
from app.domain.semantic_contract import (
    GroundedBool,
    GroundedString,
    SemanticInput,
    SemanticOutput,
    build_semantic_input,
    semantic_output_to_resume,
)
from app.extractors.factory import get_semantic_extractor
from app.extractors.semantic_extractor import (
    SemanticCompletenessError,
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticExtractor,
    SemanticRateLimitError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
    SemanticValidationError,
)
from app.infrastructure.database.connection import get_connection
from app.infrastructure.database.repositories import (
    CandidateRepository,
    ResumeProvenanceRepository,
    ResumeSnapshotRepository,
)
from app.pipeline.parser import PipelineError, ResumeParser
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor

try:
    from tests.benchmark.generalization import GENERALIZATION_FIXTURES_DIR, GeneralizationCorpus
    from tests.benchmark.metadata import BENCHMARK_FIXTURES
    from tests.benchmark.semantic_runner import FIXTURES_DIR, SemanticBenchmarkRunner
    _TESTS_AVAILABLE = True
except ImportError:
    _TESTS_AVAILABLE = False
    GENERALIZATION_FIXTURES_DIR = None
    GeneralizationCorpus = None
    BENCHMARK_FIXTURES = {}
    FIXTURES_DIR = None
    SemanticBenchmarkRunner = None

logger = logging.getLogger(__name__)

router = APIRouter()


MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# Dependency hook for injecting mock or alternative extractors during tests
_extractor_override: Callable[[], SemanticExtractor] | None = None


def set_extractor_override(extractor_factory: Callable[[], SemanticExtractor] | None) -> None:
    """Set an extractor override function for testing purposes."""
    global _extractor_override
    _extractor_override = extractor_factory


def _resolve_extractor(settings: Settings) -> SemanticExtractor:
    if _extractor_override is not None:
        return _extractor_override()
    return get_semantic_extractor(settings)


class DiagnosticExtractorWrapper:
    """Non-invasive wrapper that captures semantic input and output for provenance inspection."""

    def __init__(self, inner: SemanticExtractor) -> None:
        self._inner = inner
        self.last_semantic_input: SemanticInput | None = None
        self.last_semantic_output: SemanticOutput | None = None

    @property
    def last_usage_metadata(self) -> dict[str, Any] | None:
        return getattr(self._inner, "last_usage_metadata", None)

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        self.last_semantic_input = input_data
        output = self._inner.extract(input_data)
        self.last_semantic_output = output
        return output


def _collect_provenance_records(
    output: SemanticOutput | None,
    semantic_input: SemanticInput | None,
) -> list[ProvenanceRecord]:
    """Deterministically map grounded output entities to source block provenance."""
    if not output or not semantic_input:
        return []

    block_map = {b.block_id: b for b in semantic_input.blocks}
    records: list[ProvenanceRecord] = []

    def _add_record(field_name: str, item: Any, confidence: float | None = None) -> None:
        if item is None:
            return
        if isinstance(item, GroundedString):
            source_ids = item.source_block_ids or []
            val = item.value
            raw_val = item.raw_value
        elif isinstance(item, GroundedBool):
            source_ids = item.source_block_ids or []
            val = item.value
            raw_val = str(item.value)
        else:
            return

        matching_blocks = [block_map[sid] for sid in source_ids if sid in block_map]
        pages = sorted(list({getattr(b, "page_number", getattr(b, "page", 1)) for b in matching_blocks}))
        texts = [b.text for b in matching_blocks]

        records.append(
            ProvenanceRecord(
                canonicalField=field_name,
                extractedValue=val,
                rawValue=raw_val,
                sourceBlockIds=source_ids,
                sourceTexts=texts,
                pageNumbers=pages,
                confidence=confidence,
            )
        )

    # 1. Personal fields
    if output.personal:
        _add_record("personal.name", output.personal.name)
        _add_record("personal.email", output.personal.email)
        _add_record("personal.phone", output.personal.phone)
        _add_record("personal.location", output.personal.location)
        _add_record("personal.linkedin", output.personal.linkedin)
        _add_record("personal.github", output.personal.github)
        _add_record("personal.portfolio", output.personal.portfolio)

    # 2. Summary
    _add_record("summary", output.summary)

    # 3. Skills
    for i, skill in enumerate(output.skills):
        _add_record(f"skills[{i}]", skill)

    # 4. Experience
    for i, exp in enumerate(output.experience):
        _add_record(f"experience[{i}].company", exp.company)
        _add_record(f"experience[{i}].designation", exp.designation)
        _add_record(f"experience[{i}].startDate", exp.startDate)
        _add_record(f"experience[{i}].endDate", exp.endDate)
        _add_record(f"experience[{i}].current", exp.current)
        _add_record(f"experience[{i}].location", exp.location)
        _add_record(f"experience[{i}].description", exp.description)
        for j, tech in enumerate(exp.technologies):
            _add_record(f"experience[{i}].technologies[{j}]", tech)

    # 5. Education
    for i, edu in enumerate(output.education):
        _add_record(f"education[{i}].institution", edu.institution)
        _add_record(f"education[{i}].degree", edu.degree)
        _add_record(f"education[{i}].fieldOfStudy", edu.fieldOfStudy)
        _add_record(f"education[{i}].startDate", edu.startDate)
        _add_record(f"education[{i}].endDate", edu.endDate)
        _add_record(f"education[{i}].grade", edu.grade)

    # 6. Projects
    for i, prj in enumerate(output.projects):
        _add_record(f"projects[{i}].name", prj.name)
        _add_record(f"projects[{i}].description", prj.description)
        _add_record(f"projects[{i}].startDate", prj.startDate)
        _add_record(f"projects[{i}].endDate", prj.endDate)
        _add_record(f"projects[{i}].current", prj.current)
        for j, tech in enumerate(prj.technologies):
            _add_record(f"projects[{i}].technologies[{j}]", tech)

    # 7. Certifications, languages, achievements
    for i, cert in enumerate(output.certifications):
        _add_record(f"certifications[{i}]", cert)
    for i, lang in enumerate(output.languages):
        _add_record(f"languages[{i}]", lang)
    for i, ach in enumerate(output.achievements):
        _add_record(f"achievements[{i}]", ach)

    return records


def _evaluate_parse_status(resume: Resume) -> ParseStatus:
    """Determine whether a successfully validated resume is complete (SUCCESS) or partial (PARTIAL).

    A resume is considered complete (SUCCESS) when it contains candidate identity (name),
    actionable contact details (email or phone), and professional/educational history (experience or education).
    If any of these foundational pillars is absent, it is classified as PARTIAL.
    """
    has_name = bool(resume.personal and resume.personal.name and resume.personal.name.strip())
    has_contact = bool(resume.personal and (resume.personal.email or resume.personal.phone))
    has_history = bool(resume.experience or resume.education)

    if has_name and has_contact and has_history:
        return ParseStatus.SUCCESS
    return ParseStatus.PARTIAL


def _run_parse_pipeline(
    pdf_bytes: bytes,
    filename: str,
    settings: Settings,
) -> ParseResponse:
    """Run PDF bytes through existing semantic parser pipeline."""
    raw_extractor = _resolve_extractor(settings)
    wrapped = DiagnosticExtractorWrapper(raw_extractor)
    parser = ResumeParser()

    start_time = time.perf_counter()
    try:
        resume = parser.parse_with_semantic_pipeline(
            pdf_bytes,
            wrapped,
            document_id=filename,
        )
    except SemanticValidationError as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.warning("Parse validation failed for %s with %d violations: %s", filename, len(exc.violations), exc)
        if wrapped.last_semantic_output is not None:
            try:
                resume = semantic_output_to_resume(wrapped.last_semantic_output)
                meta = wrapped.last_usage_metadata or {}
                page_count = len(wrapped.last_semantic_input.pages) if wrapped.last_semantic_input else 1
                provenance_records = _collect_provenance_records(
                    wrapped.last_semantic_output,
                    wrapped.last_semantic_input,
                )
                metadata = ParseMetadata(
                    filename=filename,
                    pageCount=page_count,
                    ocrUsed=False,
                    provider=meta.get("provider", settings.semantic_provider),
                    model=meta.get("model", "unknown"),
                    representation=meta.get("representation", "candidate_b_compact"),
                    latencyMs=meta.get("latency_ms", elapsed_ms),
                    requestCount=1,
                )
                archetype_str = "unknown"
                if hasattr(wrapped.last_semantic_output, "document_archetype"):
                    archetype_str = getattr(
                        wrapped.last_semantic_output.document_archetype,
                        "value",
                        str(wrapped.last_semantic_output.document_archetype),
                    )
                diagnostics = {
                    "provenance": [r.model_dump() for r in provenance_records],
                    "archetype": archetype_str,
                    "violations": exc.violations,
                    "blockClassifications": [
                        bc.model_dump() for bc in getattr(wrapped.last_semantic_output, "block_classifications", [])
                    ],
                    "totalBlocks": len(wrapped.last_semantic_input.blocks) if wrapped.last_semantic_input else 0,
                    "tokenUsage": {
                        "promptTokens": meta.get("prompt_tokens"),
                        "outputTokens": meta.get("output_tokens"),
                        "totalTokens": meta.get("total_tokens"),
                    },
                }
                return ParseResponse(
                    success=False,
                    status=ParseStatus.VALIDATION_FAILED,
                    resume=resume.model_dump(),
                    violations=exc.violations,
                    metadata=metadata,
                    diagnostics=diagnostics,
                )
            except Exception as conv_err:
                logger.warning("Failed to project semantic output to resume after validation failure: %s", conv_err)

        raise HTTPException(
            status_code=422,
            detail={
                "error": "SEMANTIC_VALIDATION_ERROR",
                "status": ParseStatus.VALIDATION_FAILED.value,
                "message": str(exc),
                "violations": exc.violations,
                "elapsedMs": elapsed_ms,
            },
        )
    except SemanticTimeoutError as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.warning("Semantic provider timeout for %s: %s", filename, exc)
        raise HTTPException(
            status_code=504,
            detail={
                "error": "GATEWAY_TIMEOUT",
                "status": ParseStatus.ERROR.value,
                "message": str(exc),
                "elapsedMs": elapsed_ms,
            },
        )
    except SemanticRateLimitError as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.warning("Semantic provider rate limit for %s: %s", filename, exc)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "status": ParseStatus.ERROR.value,
                "message": str(exc),
                "retryAfter": exc.retry_after,
                "elapsedMs": elapsed_ms,
            },
        )
    except SemanticConfigurationError as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.error("Semantic configuration error for %s: %s", filename, exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "CONFIGURATION_ERROR",
                "status": ParseStatus.ERROR.value,
                "message": str(exc),
                "elapsedMs": elapsed_ms,
            },
        )
    except SemanticCompletenessError as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.warning("Semantic completeness error for %s: %s", filename, exc)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "SEMANTIC_COMPLETENESS_ERROR",
                "status": ParseStatus.EXTRACTION_FAILED.value,
                "message": str(exc),
                "reason": exc.reason,
                "elapsedMs": elapsed_ms,
            },
        )
    except SemanticExtractionError as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.error("Semantic extraction failure for %s: %s", filename, exc)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "SEMANTIC_EXTRACTION_ERROR",
                "status": ParseStatus.EXTRACTION_FAILED.value,
                "message": str(exc),
                "elapsedMs": elapsed_ms,
            },
        )
    except PipelineError as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.warning("Pipeline error for %s: %s (%s)", filename, exc.message, exc.code)
        raise HTTPException(
            status_code=400,
            detail={
                "error": exc.code,
                "status": ParseStatus.ERROR.value,
                "message": exc.message,
                "elapsedMs": elapsed_ms,
            },
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.exception("Unexpected error parsing %s", filename)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PARSER_INTERNAL_ERROR",
                "status": ParseStatus.ERROR.value,
                "message": f"Failed to parse resume: {exc}",
                "elapsedMs": elapsed_ms,
            },
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    meta = wrapped.last_usage_metadata or {}
    page_count = len(wrapped.last_semantic_input.pages) if wrapped.last_semantic_input else 1

    provenance_records = _collect_provenance_records(
        wrapped.last_semantic_output,
        wrapped.last_semantic_input,
    )

    metadata = ParseMetadata(
        filename=filename,
        pageCount=page_count,
        ocrUsed=False,
        provider=meta.get("provider", settings.semantic_provider),
        model=meta.get("model", "unknown"),
        representation=meta.get("representation", "candidate_b_compact"),
        latencyMs=meta.get("latency_ms", elapsed_ms),
        requestCount=1,
    )

    archetype_str = "unknown"
    if wrapped.last_semantic_output and hasattr(wrapped.last_semantic_output, "document_archetype"):
        archetype_str = getattr(wrapped.last_semantic_output.document_archetype, "value", str(wrapped.last_semantic_output.document_archetype))

    diagnostics = {
        "provenance": [r.model_dump() for r in provenance_records],
        "archetype": archetype_str,
        "blockClassifications": [
            bc.model_dump() for bc in getattr(wrapped.last_semantic_output, "block_classifications", [])
        ],
        "totalBlocks": len(wrapped.last_semantic_input.blocks) if wrapped.last_semantic_input else 0,
        "tokenUsage": {
            "promptTokens": meta.get("prompt_tokens"),
            "outputTokens": meta.get("output_tokens"),
            "totalTokens": meta.get("total_tokens"),
        },
    }

    status = _evaluate_parse_status(resume)
    return ParseResponse(
        success=True,
        status=status,
        resume=resume.model_dump(),
        violations=[],
        metadata=metadata,
        diagnostics=diagnostics,
    )


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return health status and versions."""
    settings = Settings()
    return HealthResponse(
        status="ok",
        parser_version=settings.parser_version,
        application_version="1.0.0",
    )


@router.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    """Return active parser and provider configuration without exposing secrets."""
    settings = Settings()
    provider = settings.semantic_provider.lower()

    if provider == "gemini":
        model = settings.gemini_model
    elif provider == "nvidia":
        model = settings.nvidia_model
    elif provider == "openrouter":
        model = settings.openrouter_model
    elif provider == "ollama":
        model = settings.ollama_model
    else:
        model = "unknown"

    ocr_available = False
    ocr_engine = "none"
    try:
        import fitz
        ocr_available = True
        ocr_engine = f"PyMuPDF-{fitz.__version__}"
    except ImportError:
        pass

    return ConfigResponse(
        provider=provider,
        model=model,
        representation="candidate_b_compact",
        parser_version=settings.parser_version,
        ocr_available=ocr_available,
        ocr_engine=ocr_engine,
        two_pass=getattr(settings, f"{provider}_two_pass", False),
        fallback_enabled=settings.semantic_fallback_enabled,
    )


@router.post("/parse", response_model=ParseResponse)
async def parse_resume(file: UploadFile = File(...)) -> ParseResponse:
    """Upload and parse a resume PDF using the existing semantic parser pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_FILE_TYPE",
                "status": ParseStatus.ERROR.value,
                "message": f"Expected a PDF file, but received: '{file.filename}'",
            },
        )

    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "PAYLOAD_TOO_LARGE",
                "status": ParseStatus.ERROR.value,
                "message": f"File size ({len(pdf_bytes)} bytes) exceeds max limit of {MAX_UPLOAD_SIZE} bytes (10MB)",
            },
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_PDF_HEADER",
                "status": ParseStatus.ERROR.value,
                "message": "File header does not match valid PDF magic bytes (%PDF-).",
            },
        )

    settings = Settings()
    return await asyncio.to_thread(_run_parse_pipeline, pdf_bytes, file.filename, settings)


def _build_diagnostic_ir(pdf_bytes: bytes, filename: str) -> tuple[Any, SemanticInput]:
    """Execute CPU-bound PDF text extraction, layout reconstruction, and input building."""
    raw_text_blocks = PDFExtractor.extract(pdf_bytes)
    physical_doc = document_from_text_blocks(raw_text_blocks)
    reconstructed_doc = reconstruct_document(physical_doc)
    layout_doc = interpret_layout(reconstructed_doc)
    semantic_input = build_semantic_input(layout_doc, document_id=filename)
    return layout_doc, semantic_input


@router.post("/parse/diagnostic", response_model=DiagnosticResponse)
async def parse_diagnostic(
    file: UploadFile = File(...),
    max_blocks: int = Query(500, ge=10, le=5000, description="Maximum number of blocks to return"),
    max_text_length: int = Query(200, ge=20, le=2000, description="Maximum characters per block text"),
) -> DiagnosticResponse:
    """Run structural/layout pipeline without LLM calls and return bounded diagnostic IR."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_FILE_TYPE",
                "message": f"Expected a PDF file, but received: '{file.filename}'",
            },
        )

    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "PAYLOAD_TOO_LARGE",
                "message": f"File size exceeds max limit of {MAX_UPLOAD_SIZE} bytes",
            },
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_PDF_HEADER",
                "message": "File header does not match valid PDF magic bytes (%PDF-).",
            },
        )

    # Execute structural pipeline stages off the asyncio event loop
    try:
        layout_doc, semantic_input = await asyncio.to_thread(_build_diagnostic_ir, pdf_bytes, file.filename)
    except Exception as exc:
        logger.exception("Structural extraction failed for %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "STRUCTURAL_EXTRACTION_ERROR",
                "message": f"Failed during layout/structural extraction: {exc}",
            },
        )

    # Build bounded diagnostic structures
    pages: list[DiagnosticPageItem] = []
    regions: list[DiagnosticRegionItem] = []
    for page in layout_doc.pages:
        p_width = round(page.width, 2) if getattr(page, "width", None) is not None else 0.0
        p_height = round(page.height, 2) if getattr(page, "height", None) is not None else 0.0
        pages.append(
            DiagnosticPageItem(
                page_number=page.page_number,
                width=p_width,
                height=p_height,
                region_count=len(page.regions),
            )
        )
        for r in page.regions:
            r_box = [0.0, 0.0, 0.0, 0.0]
            if getattr(r, "bbox", None) is not None:
                r_box = [
                    round(getattr(r.bbox, "x0", 0.0), 1),
                    round(getattr(r.bbox, "y0", 0.0), 1),
                    round(getattr(r.bbox, "x1", 0.0), 1),
                    round(getattr(r.bbox, "y1", 0.0), 1),
                ]
            regions.append(
                DiagnosticRegionItem(
                    page_number=page.page_number,
                    region_id=r.region_id,
                    bbox=r_box,
                    reading_order=getattr(r, "reading_order", 0),
                    region_type=getattr(r, "region_type", "unknown"),
                    line_count=len(r.lines),
                )
            )

    all_blocks = semantic_input.blocks
    total_blocks = len(all_blocks)
    truncated = total_blocks > max_blocks
    sliced_blocks = all_blocks[:max_blocks]

    diagnostic_blocks: list[DiagnosticBlockItem] = []
    for b in sliced_blocks:
        txt = b.text.strip()
        if len(txt) > max_text_length:
            txt = txt[:max_text_length] + "..."

        b_box = [0.0, 0.0, 0.0, 0.0]
        if getattr(b, "bbox", None) is not None:
            b_box = [
                round(getattr(b.bbox, "x0", 0.0), 1),
                round(getattr(b.bbox, "y0", 0.0), 1),
                round(getattr(b.bbox, "x1", 0.0), 1),
                round(getattr(b.bbox, "y1", 0.0), 1),
            ]

        diagnostic_blocks.append(
            DiagnosticBlockItem(
                block_id=b.block_id,
                page_number=getattr(b, "page_number", getattr(b, "page", 1)),
                reading_order=b.reading_order,
                text=txt,
                bbox=b_box,
                suggested_role=b.suggested_role,
                table_id=b.table_id,
                row_index=b.row_index,
                column_index=b.column_index,
                is_continuation=getattr(b, "is_continuation", False),
                section_hint=getattr(b, "section_hint", None),
            )
        )

    tables: list[DiagnosticTableItem] = []
    if hasattr(semantic_input, "tables") and semantic_input.tables:
        for t in semantic_input.tables:
            tables.append(
                DiagnosticTableItem(
                    table_id=getattr(t, "table_id", "tbl"),
                    page_number=getattr(t, "page_number", 1),
                    row_count=len(getattr(t, "rows", [])),
                    column_count=len(getattr(t, "columns", [])),
                    headers=getattr(t, "headers", []),
                )
            )

    archetype_val = "unknown"
    if hasattr(semantic_input, "archetype") and semantic_input.archetype:
        archetype_val = getattr(semantic_input.archetype, "value", str(semantic_input.archetype))

    return DiagnosticResponse(
        page_count=len(layout_doc.pages),
        block_count=total_blocks,
        archetype=archetype_val,
        pages=pages,
        regions=regions,
        blocks=diagnostic_blocks,
        tables=tables,
        truncated=truncated,
    )


@router.get("/fixtures", response_model=list[FixtureItem])
def list_fixtures() -> list[FixtureItem]:
    """Enumerate regression_12 and generalization fixtures from existing constants/manifests."""
    items: list[FixtureItem] = []

    # 1. Regression 12 fixtures
    for filename, meta in BENCHMARK_FIXTURES.items():
        arch_val = meta.archetype.value if hasattr(meta.archetype, "value") else str(meta.archetype)
        items.append(
            FixtureItem(
                id=f"reg12_{filename}",
                suite="regression_12",
                filename=filename,
                candidate_name=meta.candidate_name,
                target_domain=meta.target_domain,
                archetype=arch_val,
                page_count_estimate=meta.page_count_estimate,
                has_tables=meta.has_tables,
                notes=meta.notes,
            )
        )

    # 2. Generalization fixtures
    try:
        corpus = GeneralizationCorpus.load_manifest()
        for fix in corpus.fixtures:
            arch_val = fix.archetype.value if hasattr(fix.archetype, "value") else str(fix.archetype)
            items.append(
                FixtureItem(
                    id=f"gen_{fix.id}",
                    suite="generalization",
                    filename=fix.filename,
                    candidate_name=None,
                    target_domain=fix.domain,
                    archetype=arch_val,
                    page_count_estimate=fix.page_count,
                    has_tables=fix.has_tables,
                    notes=f"Layout: {fix.layout.value}. Source: {fix.source}",
                )
            )
    except Exception as exc:
        logger.debug("Could not load generalization corpus: %s", exc)

    return items


def _find_fixture_path(fixture_id: str) -> Path | None:
    """Resolve fixture file path from fixture_id or filename."""
    # Check regression_12
    for filename in BENCHMARK_FIXTURES:
        if fixture_id in (filename, f"reg12_{filename}"):
            path = FIXTURES_DIR / filename
            if path.exists():
                return path

    # Check generalization
    try:
        corpus = GeneralizationCorpus.load_manifest()
        for fix in corpus.fixtures:
            if fixture_id in (fix.id, fix.filename, f"gen_{fix.id}"):
                path = GENERALIZATION_FIXTURES_DIR / fix.filename
                if path.exists():
                    return path
    except Exception:
        pass

    # Fallback direct search in FIXTURES_DIR
    direct_path = FIXTURES_DIR / fixture_id
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    return None


@router.post("/fixtures/{fixture_id}/parse", response_model=ParseResponse)
def parse_fixture(fixture_id: str) -> ParseResponse:
    """Resolve an existing benchmark fixture and run the standard parse path."""
    fixture_path = _find_fixture_path(fixture_id)
    if not fixture_path or not fixture_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "FIXTURE_NOT_FOUND",
                "message": f"Fixture '{fixture_id}' could not be resolved in benchmark directories.",
            },
        )

    pdf_bytes = fixture_path.read_bytes()
    settings = Settings()
    return _run_parse_pipeline(pdf_bytes, fixture_path.name, settings)


@router.post("/benchmark")
def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    """Execute existing benchmark runner/evaluation on specified suite."""
    settings = Settings()
    provider_name = request.provider or settings.semantic_provider
    extractor = _resolve_extractor(settings)

    runner = SemanticBenchmarkRunner(
        extractor=extractor,
        provider_name=provider_name,
        model_name=request.model,
        representation=request.representation,
        suite_id=request.suite,
    )

    summary = runner.run_all()
    return summary.to_dict()


# =====================================================================
# Product v1 Resume Read & Search Endpoints
# =====================================================================


@router.get("/resumes", response_model=ResumeListResponse)
def list_resumes(
    query: str | None = Query(None, description="Search candidate name, summary, company, designation"),
    skills: list[str] | None = Query(None, description="Match skills (ALL semantics, case-insensitive)"),
    status: ParseStatus | None = Query(None, description="Exact parse status filter"),
    company: str | None = Query(None, description="Filter experience company (case-insensitive)"),
    location: str | None = Query(None, description="Filter location (case-insensitive)"),
    created_after: datetime | None = Query(None, description="Inclusive start datetime"),
    created_before: datetime | None = Query(None, description="Inclusive end datetime"),
    candidate_id: str | None = Query(None, description="Filter by candidate ID"),
    is_latest: bool | None = Query(None, description="Filter by latest snapshot flag"),
    limit: int = Query(20, ge=1, le=100, description="Page size (1-100)"),
    offset: int = Query(0, ge=0, description="Page offset (>= 0)"),
) -> ResumeListResponse:
    """Paginated search and list of active resume snapshots using PostgreSQL-native search."""
    flattened_skills: list[str] = []
    if skills:
        for s in skills:
            for part in s.split(","):
                part_clean = part.strip()
                if part_clean and part_clean not in flattened_skills:
                    flattened_skills.append(part_clean)

    with get_connection() as conn:
        rows, total = ResumeSnapshotRepository.search(
            conn=conn,
            query=query,
            skills=flattened_skills if flattened_skills else None,
            status=status.value if status else None,
            company=company,
            location=location,
            created_after=created_after,
            created_before=created_before,
            candidate_id=candidate_id,
            is_latest=is_latest,
            limit=limit,
            offset=offset,
        )

        items = [
            ResumeSnapshotResponse(
                resume_id=r["resume_id"],
                document_id=r["document_id"],
                candidate_id=r["candidate_id"],
                status=ParseStatus(r["parse_status"]),
                success=r["success"],
                is_latest=r["is_latest"],
                created_at=r["created_at"],
                resume=r["resume_data"],
                metadata=r["metadata"] or {},
                violations=r["violations"] or [],
                provenance=None,
                candidate_name=r["candidate_name"],
                candidate_email=r["candidate_email"],
                candidate_location=r["candidate_location"],
                skills=r["skills"] or [],
            )
            for r in rows
        ]
        has_more = (offset + len(items)) < total
        return ResumeListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
            has_more=has_more,
        )


@router.get("/resumes/{resume_id}", response_model=ResumeSnapshotResponse)
def get_resume_snapshot(
    resume_id: str,
    include_provenance: bool = Query(False, description="Whether to include granular provenance records"),
) -> ResumeSnapshotResponse:
    """Fetch an active resume snapshot by resume_id."""
    with get_connection() as conn:
        snapshot = ResumeSnapshotRepository.get_active_by_id(conn, resume_id)
        if not snapshot:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESUME_NOT_FOUND",
                    "message": f"Resume snapshot '{resume_id}' not found or has been deleted.",
                },
            )

        provenance_data = None
        if include_provenance:
            prov_row = ResumeProvenanceRepository.get_by_resume_id(conn, resume_id)
            provenance_data = prov_row["records"] if prov_row else []

        return ResumeSnapshotResponse(
            resume_id=snapshot["resume_id"],
            document_id=snapshot["document_id"],
            candidate_id=snapshot["candidate_id"],
            status=ParseStatus(snapshot["parse_status"]),
            success=snapshot["success"],
            is_latest=snapshot["is_latest"],
            created_at=snapshot["created_at"],
            resume=snapshot["resume_data"],
            metadata=snapshot["metadata"] or {},
            violations=snapshot["violations"] or [],
            provenance=provenance_data,
            candidate_name=snapshot["candidate_name"],
            candidate_email=snapshot["candidate_email"],
            candidate_location=snapshot["candidate_location"],
            skills=snapshot["skills"] or [],
        )


@router.get("/candidates/{candidate_id}/resumes", response_model=CandidateResumeListResponse)
def list_candidate_resumes(
    candidate_id: str,
    is_latest: bool | None = Query(None, description="Filter by latest snapshot flag"),
    status: ParseStatus | None = Query(None, description="Exact parse status filter"),
    limit: int = Query(20, ge=1, le=100, description="Page size (1-100)"),
    offset: int = Query(0, ge=0, description="Page offset (>= 0)"),
) -> CandidateResumeListResponse:
    """Return paginated snapshots for the specified candidate."""
    with get_connection() as conn:
        candidate = CandidateRepository.get_by_id(conn, candidate_id)
        if not candidate:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CANDIDATE_NOT_FOUND",
                    "message": f"Candidate '{candidate_id}' not found or has been deleted.",
                },
            )

        rows, total = ResumeSnapshotRepository.list_by_candidate(
            conn=conn,
            candidate_id=candidate_id,
            is_latest=is_latest,
            status=status.value if status else None,
            limit=limit,
            offset=offset,
        )

        items = [
            ResumeSnapshotResponse(
                resume_id=r["resume_id"],
                document_id=r["document_id"],
                candidate_id=r["candidate_id"],
                status=ParseStatus(r["parse_status"]),
                success=r["success"],
                is_latest=r["is_latest"],
                created_at=r["created_at"],
                resume=r["resume_data"],
                metadata=r["metadata"] or {},
                violations=r["violations"] or [],
                provenance=None,
                candidate_name=r["candidate_name"],
                candidate_email=r["candidate_email"],
                candidate_location=r["candidate_location"],
                skills=r["skills"] or [],
            )
            for r in rows
        ]
        has_more = (offset + len(items)) < total
        return CandidateResumeListResponse(
            candidate_id=candidate_id,
            items=items,
            total=total,
            limit=limit,
            offset=offset,
            has_more=has_more,
        )

