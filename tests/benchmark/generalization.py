"""Generalization Benchmark Corpus Infrastructure.

Provides metadata models, schema validation, automatic discovery, and distribution
summarization for evaluating resume parsing across unseen, diverse resume corpora.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any

from tests.benchmark.metadata import DocumentArchetype

GENERALIZATION_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "generalization"


class BenchmarkSuiteId(str, Enum):
    REGRESSION_12 = "regression_12"
    GENERALIZATION = "generalization"


class GeneralizationLayout(str, Enum):
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    MULTI_COLUMN = "multi_column"
    TABULAR = "tabular"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class CareerLevel(str, Enum):
    STUDENT = "student"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"


class GeneralizationFailureCategory(str, Enum):
    PERSONAL_FIELD_ASSIGNMENT = "PERSONAL_FIELD_ASSIGNMENT"
    SECTION_BOUNDARY = "SECTION_BOUNDARY"
    EXPERIENCE_ENTITY_LINKING = "EXPERIENCE_ENTITY_LINKING"
    EDUCATION_ENTITY_LINKING = "EDUCATION_ENTITY_LINKING"
    PROJECT_BOUNDARY = "PROJECT_BOUNDARY"
    DATE_ASSOCIATION = "DATE_ASSOCIATION"
    TABLE_INTERPRETATION = "TABLE_INTERPRETATION"
    COLUMN_ORDER = "COLUMN_ORDER"
    MULTI_PAGE_CONTINUATION = "MULTI_PAGE_CONTINUATION"
    OCR_NOISE = "OCR_NOISE"
    PROVENANCE = "PROVENANCE"
    SEMANTIC_INTERPRETATION = "SEMANTIC_INTERPRETATION"
    OTHER = "OTHER"


class FailureOrigin(str, Enum):
    ALREADY_KNOWN = "already_known"
    NEW_FAILURE_CLASS = "new_failure_class"
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"


@dataclass
class FailureClassification:
    category: GeneralizationFailureCategory
    origin: FailureOrigin
    reason: str
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "origin": self.origin.value,
            "reason": self.reason,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class GeneralizationFixtureMetadata:
    """Metadata for a single generalization resume fixture."""
    fixture_id: str
    filename: str  # Path relative to generalization corpus root
    archetype: DocumentArchetype
    layout: GeneralizationLayout | str = GeneralizationLayout.UNKNOWN
    domain: str = "general"
    career_level: CareerLevel | str = CareerLevel.UNKNOWN
    ocr_required: bool = False
    table_heavy: bool = False
    page_count_estimate: int | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["archetype"] = self.archetype.value if hasattr(self.archetype, "value") else str(self.archetype)
        d["layout"] = self.layout.value if hasattr(self.layout, "value") else str(self.layout)
        d["career_level"] = self.career_level.value if hasattr(self.career_level, "value") else str(self.career_level)
        return d


@dataclass
class GeneralizationCorpusSummary:
    """Statistical summary of fixture distribution in the generalization corpus."""
    suite_id: str = BenchmarkSuiteId.GENERALIZATION.value
    total_fixtures: int = 0
    layout_distribution: dict[str, int] = field(default_factory=dict)
    domain_distribution: dict[str, int] = field(default_factory=dict)
    career_level_distribution: dict[str, int] = field(default_factory=dict)
    archetype_distribution: dict[str, int] = field(default_factory=dict)
    ocr_required_count: int = 0
    table_heavy_count: int = 0
    fixture_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_markdown(self) -> str:
        lines = [
            f"# Generalization Corpus Summary (`{self.suite_id}`)",
            "",
            f"- **Total Fixtures:** {self.total_fixtures}",
            f"- **OCR Required:** {self.ocr_required_count}",
            f"- **Table Heavy:** {self.table_heavy_count}",
            "",
            "### Layout Distribution",
            "",
            "| Layout | Count |",
            "|---|---|",
        ]
        for layout, count in sorted(self.layout_distribution.items()):
            lines.append(f"| `{layout}` | {count} |")

        lines.extend([
            "",
            "### Archetype Distribution",
            "",
            "| Archetype | Count |",
            "|---|---|",
        ])
        for arch, count in sorted(self.archetype_distribution.items()):
            lines.append(f"| `{arch}` | {count} |")

        lines.extend([
            "",
            "### Domain Distribution",
            "",
            "| Domain | Count |",
            "|---|---|",
        ])
        for domain, count in sorted(self.domain_distribution.items()):
            lines.append(f"| `{domain}` | {count} |")

        lines.extend([
            "",
            "### Career Level Distribution",
            "",
            "| Career Level | Count |",
            "|---|---|",
        ])
        for level, count in sorted(self.career_level_distribution.items()):
            lines.append(f"| `{level}` | {count} |")

        return "\n".join(lines)


def validate_generalization_metadata(
    meta: GeneralizationFixtureMetadata,
    corpus_dir: Path,
) -> list[str]:
    """Validate a single fixture metadata record against constraints and filesystem reality.

    Returns a list of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # 1. Identifier checks
    if not meta.fixture_id or not isinstance(meta.fixture_id, str) or not meta.fixture_id.strip():
        errors.append("fixture_id must be a non-empty string.")

    # 2. Path safety and traversal prevention
    if not meta.filename or not isinstance(meta.filename, str) or not meta.filename.strip():
        errors.append("filename must be a non-empty string.")
    else:
        fn = meta.filename.strip()
        if os.path.isabs(fn):
            errors.append(f"filename '{fn}' must be relative to corpus root, not absolute.")
        elif ".." in Path(fn).parts:
            errors.append(f"Path traversal detected in filename: '{fn}'.")
        else:
            resolved_corpus = corpus_dir.resolve()
            target_path = (corpus_dir / fn).resolve()
            try:
                target_path.relative_to(resolved_corpus)
            except ValueError:
                errors.append(f"Target file '{fn}' resolves outside corpus root '{corpus_dir}'.")

            # 3. File existence and extension
            if not target_path.exists():
                errors.append(f"Referenced PDF file does not exist: '{target_path}'.")
            elif not target_path.is_file():
                errors.append(f"Referenced path is not a file: '{target_path}'.")
            elif target_path.suffix.lower() != ".pdf":
                errors.append(f"Referenced file '{fn}' must have .pdf extension.")

    # 4. Archetype validation
    arch_val = meta.archetype.value if hasattr(meta.archetype, "value") else str(meta.archetype)
    try:
        DocumentArchetype(str(arch_val).lower())
    except ValueError:
        valid = [a.value for a in DocumentArchetype]
        errors.append(f"Invalid archetype '{meta.archetype}'. Must be one of {valid}.")

    # 5. Boolean field types
    if not isinstance(meta.ocr_required, bool):
        errors.append("ocr_required must be a boolean.")
    if not isinstance(meta.table_heavy, bool):
        errors.append("table_heavy must be a boolean.")

    return errors


def validate_corpus_collection(
    fixtures: list[GeneralizationFixtureMetadata] | dict[str, GeneralizationFixtureMetadata],
    corpus_dir: Path,
) -> list[str]:
    """Validate an entire collection of generalization fixtures for uniqueness and correctness."""
    errors: list[str] = []
    fixture_list = list(fixtures.values()) if isinstance(fixtures, dict) else list(fixtures)

    seen_ids: set[str] = set()
    seen_files: set[str] = set()

    for idx, item in enumerate(fixture_list):
        if not isinstance(item, GeneralizationFixtureMetadata):
            errors.append(f"Item at index {idx} is not a GeneralizationFixtureMetadata instance.")
            continue

        if item.fixture_id in seen_ids:
            errors.append(f"Duplicate fixture_id '{item.fixture_id}' found in corpus.")
        seen_ids.add(item.fixture_id)

        norm_file = str(Path(item.filename))
        if norm_file in seen_files:
            errors.append(f"Duplicate filename '{item.filename}' referenced by multiple fixtures.")
        seen_files.add(norm_file)

        item_errors = validate_generalization_metadata(item, corpus_dir)
        errors.extend(item_errors)

    return errors


class GeneralizationCorpus:
    """Manages discovery, registration, validation, and querying of generalization fixtures."""

    def __init__(
        self,
        corpus_dir: Path | None = None,
        manifest_file: Path | None = None,
    ) -> None:
        self.corpus_dir = (corpus_dir or GENERALIZATION_FIXTURES_DIR).resolve()
        self.manifest_file = manifest_file
        self._fixtures: dict[str, GeneralizationFixtureMetadata] = {}

        if manifest_file and manifest_file.exists():
            self.load_from_manifest(manifest_file)

    @property
    def fixtures(self) -> dict[str, GeneralizationFixtureMetadata]:
        return dict(self._fixtures)

    def register_fixture(self, meta: GeneralizationFixtureMetadata) -> None:
        """Register a single fixture into the corpus."""
        errors = validate_generalization_metadata(meta, self.corpus_dir)
        if errors:
            raise ValueError(f"Failed to register fixture '{meta.fixture_id}':\n" + "\n".join(errors))
        if meta.fixture_id in self._fixtures:
            raise ValueError(f"Fixture ID '{meta.fixture_id}' is already registered in corpus.")
        self._fixtures[meta.fixture_id] = meta

    def discover_pdf_files(self) -> list[Path]:
        """Discover all PDF files currently present in the generalization corpus directory.

        Sorted deterministically by relative path name.
        """
        if not self.corpus_dir.exists():
            return []
        found: list[Path] = []
        for root, _, files in os.walk(self.corpus_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    p = Path(root) / f
                    found.append(p)
        return sorted(found, key=lambda p: str(p.relative_to(self.corpus_dir)))

    def load_from_manifest(self, manifest_path: Path) -> None:
        """Load fixture metadata from a JSON manifest file."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: '{manifest_path}'.")
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else data.get("fixtures", [])
        parsed_fixtures: list[GeneralizationFixtureMetadata] = []

        for entry in items:
            raw_arch = entry.get("archetype", "unknown")
            try:
                arch = DocumentArchetype(raw_arch.lower())
            except ValueError:
                arch = DocumentArchetype.UNKNOWN

            raw_layout = entry.get("layout", "unknown")
            try:
                layout = GeneralizationLayout(raw_layout.lower())
            except ValueError:
                layout = GeneralizationLayout.UNKNOWN

            raw_level = entry.get("career_level", "unknown")
            try:
                level = CareerLevel(raw_level.lower())
            except ValueError:
                level = CareerLevel.UNKNOWN

            meta = GeneralizationFixtureMetadata(
                fixture_id=entry["fixture_id"],
                filename=entry["filename"],
                archetype=arch,
                layout=layout,
                domain=entry.get("domain", "general"),
                career_level=level,
                ocr_required=bool(entry.get("ocr_required", False)),
                table_heavy=bool(entry.get("table_heavy", False)),
                page_count_estimate=entry.get("page_count_estimate"),
                notes=entry.get("notes", ""),
                extra=entry.get("extra", {}),
            )
            parsed_fixtures.append(meta)

        validation_errors = validate_corpus_collection(parsed_fixtures, self.corpus_dir)
        if validation_errors:
            raise ValueError(
                f"Corpus manifest validation failed for '{manifest_path}':\n"
                + "\n".join(f"• {e}" for e in validation_errors)
            )

        self._fixtures = {m.fixture_id: m for m in parsed_fixtures}

    def get_summary(self) -> GeneralizationCorpusSummary:
        """Generate distribution and statistical breakdown of the corpus."""
        summary = GeneralizationCorpusSummary(
            suite_id=BenchmarkSuiteId.GENERALIZATION.value,
            total_fixtures=len(self._fixtures),
            fixture_ids=sorted(self._fixtures.keys()),
        )

        for meta in self._fixtures.values():
            arch_val = meta.archetype.value if hasattr(meta.archetype, "value") else str(meta.archetype)
            summary.archetype_distribution[arch_val] = summary.archetype_distribution.get(arch_val, 0) + 1

            layout_val = meta.layout.value if hasattr(meta.layout, "value") else str(meta.layout)
            summary.layout_distribution[layout_val] = summary.layout_distribution.get(layout_val, 0) + 1

            domain_val = str(meta.domain).lower()
            summary.domain_distribution[domain_val] = summary.domain_distribution.get(domain_val, 0) + 1

            level_val = meta.career_level.value if hasattr(meta.career_level, "value") else str(meta.career_level)
            summary.career_level_distribution[level_val] = summary.career_level_distribution.get(level_val, 0) + 1

            if meta.ocr_required:
                summary.ocr_required_count += 1
            if meta.table_heavy:
                summary.table_heavy_count += 1

        return summary


def classify_generalization_failure(
    result_dict: dict[str, Any],
    metadata: GeneralizationFixtureMetadata | None = None,
) -> FailureClassification | None:
    """Classify a non-PASS parse result into the standardized generalization failure taxonomy."""
    status = str(result_dict.get("status", "PASS")).upper()
    if status == "PASS":
        return None

    violations = result_dict.get("validation_violations") or []
    diagnostics = result_dict.get("diagnostics") or []
    err_msg = str(result_dict.get("error_message") or "")
    failure_type = str(result_dict.get("failure_type") or "")

    # 0. Completeness failure on evidence-rich document
    if "COMPLETENESS" in failure_type or any("COMPLETENESS_ERROR" in d for d in diagnostics):
        return FailureClassification(
            category=GeneralizationFailureCategory.SEMANTIC_INTERPRETATION,
            origin=FailureOrigin.NEW_FAILURE_CLASS,
            reason=f"Body completeness failure on evidence-rich document: {err_msg[:60]}",
            diagnostics=diagnostics or [err_msg],
        )

    # 1. Provenance Invariant Violations
    if violations or "VALIDATION" in failure_type:
        return FailureClassification(
            category=GeneralizationFailureCategory.PROVENANCE,
            origin=FailureOrigin.ALREADY_KNOWN,
            reason=f"Validation provenance violation: {violations[:2]}",
            diagnostics=violations,
        )

    # 2. Personal field assignments / Header name anomalies
    if any("NAME_IS_FORM_OR_DOC_TITLE" in d or "NAME_SPLIT" in d or "SECTION_HEADER_IN_LOCATION" in d for d in diagnostics):
        return FailureClassification(
            category=GeneralizationFailureCategory.PERSONAL_FIELD_ASSIGNMENT,
            origin=FailureOrigin.ALREADY_KNOWN,
            reason="Form title or label incorrectly assigned to personal fields",
            diagnostics=diagnostics,
        )

    # 3. Table interpretation anomalies
    if any("TABLE_HEADER_IN_EXPERIENCE" in d or "TABLE_METADATA_IN_EDUCATION" in d or "EXTREME_EDUCATION" in d for d in diagnostics):
        return FailureClassification(
            category=GeneralizationFailureCategory.TABLE_INTERPRETATION,
            origin=FailureOrigin.ALREADY_KNOWN,
            reason="Table headers or metadata contaminating experience/education records",
            diagnostics=diagnostics,
        )

    # 4. Experience entity linking
    if any("EXCESSIVE_EXPERIENCE_COUNT" in d for d in diagnostics):
        return FailureClassification(
            category=GeneralizationFailureCategory.EXPERIENCE_ENTITY_LINKING,
            origin=FailureOrigin.ALREADY_KNOWN,
            reason="Reference contacts or excess non-job items linked as experience",
            diagnostics=diagnostics,
        )

    # 5. Project boundaries
    if any("PROJECTS_WITHOUT_NAMES" in d for d in diagnostics):
        return FailureClassification(
            category=GeneralizationFailureCategory.PROJECT_BOUNDARY,
            origin=FailureOrigin.ALREADY_KNOWN,
            reason="Projects identified without names due to boundary detection failure",
            diagnostics=diagnostics,
        )

    # 6. OCR noise
    if metadata and metadata.ocr_required:
        return FailureClassification(
            category=GeneralizationFailureCategory.OCR_NOISE,
            origin=FailureOrigin.NEW_FAILURE_CLASS,
            reason="OCR-induced character or bounding-box fragmentation",
            diagnostics=diagnostics,
        )

    # 7. Multi-page continuation
    if any("CONTINUATION" in d for d in diagnostics):
        return FailureClassification(
            category=GeneralizationFailureCategory.MULTI_PAGE_CONTINUATION,
            origin=FailureOrigin.ALREADY_KNOWN,
            reason="Multi-page section continuation loss",
            diagnostics=diagnostics,
        )

    # 8. Unhandled exceptions / Provider / Transport
    if "EXTRACTION" in failure_type or "PARSER_EXCEPTION" in failure_type:
        return FailureClassification(
            category=GeneralizationFailureCategory.SEMANTIC_INTERPRETATION,
            origin=FailureOrigin.REQUIRES_MANUAL_REVIEW,
            reason=f"Provider or parser exception: {err_msg[:60]}",
            diagnostics=diagnostics or [err_msg],
        )

    return FailureClassification(
        category=GeneralizationFailureCategory.OTHER,
        origin=FailureOrigin.REQUIRES_MANUAL_REVIEW,
        reason=f"Unclassified failure with diagnostics: {diagnostics[:2]}",
        diagnostics=diagnostics,
    )


@dataclass
class GeneralizationAggregateReport:
    """Multi-dimensional aggregation report for generalization corpus evaluation."""
    suite_id: str = BenchmarkSuiteId.GENERALIZATION.value
    total_resumes: int = 0
    pass_count: int = 0
    partial_count: int = 0
    fail_count: int = 0
    error_count: int = 0

    pass_rate_pct: float = 0.0
    partial_rate_pct: float = 0.0
    fail_rate_pct: float = 0.0
    error_rate_pct: float = 0.0

    validation_failure_count: int = 0
    provider_failure_count: int = 0
    fallback_count: int = 0

    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_resume: float = 0.0
    avg_latency_seconds: float = 0.0

    # Multi-dimensional breakdowns
    results_by_archetype: dict[str, dict[str, int]] = field(default_factory=dict)
    results_by_layout: dict[str, dict[str, int]] = field(default_factory=dict)
    results_by_domain: dict[str, dict[str, int]] = field(default_factory=dict)
    results_by_career_level: dict[str, dict[str, int]] = field(default_factory=dict)
    results_by_ocr_required: dict[str, dict[str, int]] = field(default_factory=dict)
    results_by_table_heavy: dict[str, dict[str, int]] = field(default_factory=dict)

    # Failure taxonomy breakdown
    failures_by_category: dict[str, int] = field(default_factory=dict)
    failures_by_origin: dict[str, int] = field(default_factory=dict)

    failure_classifications: list[dict[str, Any]] = field(default_factory=list)
    fixture_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def export_json(self, path: Path) -> dict[str, Any]:
        data = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data


def build_generalization_aggregate_report(
    results: list[Any],
    corpus: GeneralizationCorpus | None = None,
) -> GeneralizationAggregateReport:
    """Build multi-dimensional aggregation report from benchmark results and corpus metadata."""
    report = GeneralizationAggregateReport()
    if not results:
        return report

    report.total_resumes = len(results)
    corpus_by_id = corpus.fixtures if corpus else {}
    corpus_by_fn = {Path(m.filename).name: m for m in corpus_by_id.values()} if corpus else {}

    total_latency = 0.0
    tokens_reported_count = 0

    for item in results:
        r_dict = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        fn = r_dict.get("filename", "")
        fid = r_dict.get("fixture_id", "")
        meta = corpus_by_id.get(fid) or corpus_by_fn.get(fn) or corpus_by_id.get(fn)

        status = str(r_dict.get("status", "FAIL")).upper()
        arch_obj = getattr(meta, "archetype", r_dict.get("archetype", "unknown"))
        arch = arch_obj.value if hasattr(arch_obj, "value") else str(arch_obj).lower()

        layout_obj = getattr(meta, "layout", "unknown")
        layout = layout_obj.value if hasattr(layout_obj, "value") else str(layout_obj).lower()

        domain = str(getattr(meta, "domain", r_dict.get("target_domain", "general"))).lower()

        level_obj = getattr(meta, "career_level", "unknown")
        level = level_obj.value if hasattr(level_obj, "value") else str(level_obj).lower()

        ocr_req = "ocr_required" if getattr(meta, "ocr_required", False) else "digital_pdf"
        table_heavy = "table_heavy" if getattr(meta, "table_heavy", False) else "standard_layout"

        # Count status
        if status == "PASS":
            report.pass_count += 1
        elif status == "PARTIAL":
            report.partial_count += 1
        elif status == "ERROR" or not r_dict.get("semantic_success", True):
            report.error_count += 1
            report.provider_failure_count += 1
        else:
            report.fail_count += 1

        if r_dict.get("validation_violations"):
            report.validation_failure_count += 1

        # Timing
        elapsed = float(r_dict.get("elapsed_seconds") or 0.0)
        total_latency += elapsed

        # Tokens
        usage = r_dict.get("usage") or {}
        if isinstance(usage, dict):
            p_tok = usage.get("prompt_tokens") or usage.get("prompt_token_count")
            o_tok = usage.get("output_tokens") or usage.get("candidates_token_count")
            t_tok = usage.get("total_tokens") or usage.get("total_token_count")
            if p_tok is not None:
                report.total_prompt_tokens += int(p_tok)
            if o_tok is not None:
                report.total_output_tokens += int(o_tok)
            if t_tok is not None:
                report.total_tokens += int(t_tok)
                tokens_reported_count += 1
            elif p_tok is not None and o_tok is not None:
                tot = int(p_tok) + int(o_tok)
                report.total_tokens += tot
                tokens_reported_count += 1

            if usage.get("fallback_invoked"):
                report.fallback_count += 1

        # Multi-dimensional aggregations
        def _update_dim(dim_dict: dict[str, dict[str, int]], key: str, st: str) -> None:
            if key not in dim_dict:
                dim_dict[key] = {"total": 0, "PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
            dim_dict[key]["total"] += 1
            dim_dict[key][st] = dim_dict[key].get(st, 0) + 1

        _update_dim(report.results_by_archetype, arch, status)
        _update_dim(report.results_by_layout, layout, status)
        _update_dim(report.results_by_domain, domain, status)
        _update_dim(report.results_by_career_level, level, status)
        _update_dim(report.results_by_ocr_required, ocr_req, status)
        _update_dim(report.results_by_table_heavy, table_heavy, status)

        # Failure classification
        if status != "PASS":
            classification = classify_generalization_failure(r_dict, meta)
            if classification:
                cat_name = classification.category.value
                orig_name = classification.origin.value
                report.failures_by_category[cat_name] = report.failures_by_category.get(cat_name, 0) + 1
                report.failures_by_origin[orig_name] = report.failures_by_origin.get(orig_name, 0) + 1
                report.failure_classifications.append({
                    "filename": fn,
                    "fixture_id": getattr(meta, "fixture_id", fn),
                    "status": status,
                    "classification": classification.to_dict(),
                })

        # Save sanitized machine-readable record
        sanitized_record = {
            "fixture_id": getattr(meta, "fixture_id", fn),
            "filename": fn,
            "suite_id": BenchmarkSuiteId.GENERALIZATION.value,
            "status": status,
            "archetype": arch,
            "layout": layout,
            "domain": domain,
            "career_level": level,
            "elapsed_seconds": round(elapsed, 4),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens") or usage.get("prompt_token_count"),
                "output_tokens": usage.get("output_tokens") or usage.get("candidates_token_count"),
                "total_tokens": usage.get("total_tokens") or usage.get("total_token_count"),
                "fallback_invoked": bool(usage.get("fallback_invoked", False)),
            } if isinstance(usage, dict) else None,
            "validation_violations_count": len(r_dict.get("validation_violations") or []),
            "sanitized_violation_codes": [v.split(":")[0] for v in (r_dict.get("validation_violations") or [])],
            "structural_counts": {
                "skills": r_dict.get("skills_count", 0),
                "experience": r_dict.get("experience_count", 0),
                "education": r_dict.get("education_count", 0),
                "projects": r_dict.get("projects_count", 0),
                "certifications": r_dict.get("certifications_count", 0),
                "languages": r_dict.get("languages_count", 0),
                "achievements": r_dict.get("achievements_count", 0),
            },
        }
        report.fixture_results.append(sanitized_record)

    report.pass_rate_pct = round((report.pass_count / report.total_resumes) * 100.0, 2)
    report.partial_rate_pct = round((report.partial_count / report.total_resumes) * 100.0, 2)
    report.fail_rate_pct = round((report.fail_count / report.total_resumes) * 100.0, 2)
    report.error_rate_pct = round((report.error_count / report.total_resumes) * 100.0, 2)

    report.avg_latency_seconds = round(total_latency / report.total_resumes, 2)
    if tokens_reported_count > 0:
        report.avg_tokens_per_resume = round(report.total_tokens / tokens_reported_count, 1)

    return report
