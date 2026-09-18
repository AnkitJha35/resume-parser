"""Phase 10A: Tests for Generalization Benchmark Corpus Infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.domain.resume import Resume
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedPersonal,
    GroundedString,
    SemanticInput,
    SemanticOutput,
)
from app.extractors.semantic_extractor import MockSemanticExtractor
from tests.benchmark.generalization import (
    GENERALIZATION_FIXTURES_DIR,
    BenchmarkSuiteId,
    CareerLevel,
    FailureClassification,
    FailureOrigin,
    GeneralizationCorpus,
    GeneralizationFailureCategory,
    GeneralizationFixtureMetadata,
    GeneralizationLayout,
    classify_generalization_failure,
    validate_corpus_collection,
    validate_generalization_metadata,
)
from tests.benchmark.metadata import BENCHMARK_FIXTURES
from tests.benchmark.semantic_runner import SemanticBenchmarkRunner


def _create_dummy_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid PDF header + object structure
    path.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF")
    return path


def test_empty_generalization_corpus_discovery_and_summary(tmp_path: Path):
    """An empty corpus directory discovers zero files and produces valid zero-count summary."""
    corpus = GeneralizationCorpus(corpus_dir=tmp_path)
    assert corpus.discover_pdf_files() == []

    summary = corpus.get_summary()
    assert summary.total_fixtures == 0
    assert summary.ocr_required_count == 0
    assert summary.table_heavy_count == 0
    assert summary.suite_id == BenchmarkSuiteId.GENERALIZATION.value


def test_valid_generalization_corpus_registration_and_distribution(tmp_path: Path):
    """Valid fixtures can be registered, discovered, and summarized by distribution."""
    f1 = _create_dummy_pdf(tmp_path / "engineering" / "senior_backend.pdf")
    f2 = _create_dummy_pdf(tmp_path / "finance" / "analyst_tabular.pdf")

    corpus = GeneralizationCorpus(corpus_dir=tmp_path)

    meta1 = GeneralizationFixtureMetadata(
        fixture_id="gen_001",
        filename="engineering/senior_backend.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        layout=GeneralizationLayout.TWO_COLUMN,
        domain="tech",
        career_level=CareerLevel.SENIOR,
        ocr_required=False,
        table_heavy=False,
    )
    meta2 = GeneralizationFixtureMetadata(
        fixture_id="gen_002",
        filename="finance/analyst_tabular.pdf",
        archetype=DocumentArchetype.MARITIME_TABULAR,
        layout=GeneralizationLayout.TABULAR,
        domain="finance",
        career_level=CareerLevel.MID,
        ocr_required=True,
        table_heavy=True,
    )

    corpus.register_fixture(meta1)
    corpus.register_fixture(meta2)

    assert len(corpus.fixtures) == 2
    assert corpus.discover_pdf_files() == [f2, f1] or corpus.discover_pdf_files() == [f1, f2]

    summary = corpus.get_summary()
    assert summary.total_fixtures == 2
    assert summary.ocr_required_count == 1
    assert summary.table_heavy_count == 1
    assert summary.layout_distribution["two_column"] == 1
    assert summary.layout_distribution["tabular"] == 1
    assert summary.domain_distribution["tech"] == 1
    assert summary.domain_distribution["finance"] == 1
    assert summary.career_level_distribution["senior"] == 1
    assert summary.career_level_distribution["mid"] == 1

    md = summary.format_markdown()
    assert "# Generalization Corpus Summary (`generalization`)" in md
    assert "**Total Fixtures:** 2" in md


def test_duplicate_fixture_ids_rejected(tmp_path: Path):
    """Duplicate fixture_id raises a ValueError during registration and collection validation."""
    _create_dummy_pdf(tmp_path / "resume_a.pdf")
    _create_dummy_pdf(tmp_path / "resume_b.pdf")

    corpus = GeneralizationCorpus(corpus_dir=tmp_path)
    meta1 = GeneralizationFixtureMetadata(
        fixture_id="dup_id",
        filename="resume_a.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
    )
    meta2 = GeneralizationFixtureMetadata(
        fixture_id="dup_id",
        filename="resume_b.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
    )

    corpus.register_fixture(meta1)
    with pytest.raises(ValueError, match="already registered"):
        corpus.register_fixture(meta2)

    errors = validate_corpus_collection([meta1, meta2], tmp_path)
    assert any("Duplicate fixture_id 'dup_id'" in e for e in errors)


def test_missing_referenced_pdf_file_fails_validation(tmp_path: Path):
    """Referencing a non-existent PDF file produces a validation error."""
    meta = GeneralizationFixtureMetadata(
        fixture_id="missing_file",
        filename="does_not_exist.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
    )
    errors = validate_generalization_metadata(meta, tmp_path)
    assert any("does not exist" in e for e in errors)


def test_invalid_metadata_values_fail_validation(tmp_path: Path):
    """Invalid types or empty identifiers fail validation."""
    pdf_path = _create_dummy_pdf(tmp_path / "valid.pdf")

    # Empty fixture_id
    meta_empty_id = GeneralizationFixtureMetadata(
        fixture_id="",
        filename="valid.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
    )
    errors = validate_generalization_metadata(meta_empty_id, tmp_path)
    assert any("fixture_id must be a non-empty string" in e for e in errors)


def test_unsupported_archetype_fails_validation(tmp_path: Path):
    """Unknown / unsupported archetype string fails validation."""
    _create_dummy_pdf(tmp_path / "sample.pdf")
    meta = GeneralizationFixtureMetadata(
        fixture_id="gen_arch",
        filename="sample.pdf",
        archetype="completely_invalid_archetype",  # type: ignore
    )
    errors = validate_generalization_metadata(meta, tmp_path)
    assert any("Invalid archetype" in e for e in errors)


def test_path_traversal_and_out_of_root_prevention(tmp_path: Path):
    """Attempts to escape corpus directory via '..' or absolute paths are blocked."""
    outside_file = _create_dummy_pdf(tmp_path.parent / "secret.pdf")

    # 1. Relative traversal
    meta_traversal = GeneralizationFixtureMetadata(
        fixture_id="trav_1",
        filename="../secret.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
    )
    errors = validate_generalization_metadata(meta_traversal, tmp_path)
    assert any("Path traversal" in e for e in errors)

    # 2. Absolute path outside root
    meta_abs = GeneralizationFixtureMetadata(
        fixture_id="trav_2",
        filename=str(outside_file.resolve()),
        archetype=DocumentArchetype.STANDARD_CV,
    )
    errors_abs = validate_generalization_metadata(meta_abs, tmp_path)
    assert any("must be relative to corpus root" in e for e in errors_abs)


def test_deterministic_ordering_of_discovered_fixtures(tmp_path: Path):
    """Discovered PDF fixtures are always returned in deterministic alphabetical order."""
    _create_dummy_pdf(tmp_path / "z_resume.pdf")
    _create_dummy_pdf(tmp_path / "a_resume.pdf")
    _create_dummy_pdf(tmp_path / "m_resume.pdf")

    corpus = GeneralizationCorpus(corpus_dir=tmp_path)
    discovered = corpus.discover_pdf_files()

    names = [p.name for p in discovered]
    assert names == ["a_resume.pdf", "m_resume.pdf", "z_resume.pdf"]


def test_corpus_manifest_loading(tmp_path: Path):
    """Manifest JSON can define multiple fixtures and load into GeneralizationCorpus."""
    _create_dummy_pdf(tmp_path / "tech_lead.pdf")
    _create_dummy_pdf(tmp_path / "nautical_officer.pdf")

    manifest = {
        "fixtures": [
            {
                "fixture_id": "gen_101",
                "filename": "tech_lead.pdf",
                "archetype": "standard_cv",
                "layout": "single_column",
                "domain": "engineering",
                "career_level": "senior",
                "ocr_required": False,
                "table_heavy": False,
            },
            {
                "fixture_id": "gen_102",
                "filename": "nautical_officer.pdf",
                "archetype": "maritime_cv",
                "layout": "tabular",
                "domain": "maritime",
                "career_level": "mid",
                "ocr_required": False,
                "table_heavy": True,
            },
        ]
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    corpus = GeneralizationCorpus(corpus_dir=tmp_path, manifest_file=manifest_file)
    assert len(corpus.fixtures) == 2
    assert "gen_101" in corpus.fixtures
    assert "gen_102" in corpus.fixtures


def test_suite_separation_between_regression_12_and_generalization():
    """Verify regression_12 suite fixtures and generalization suite remain completely separated."""
    # 1. Regression suite contains the canonical 12 fixtures
    assert len(BENCHMARK_FIXTURES) == 12
    assert "AditCV_SOL.pdf" in BENCHMARK_FIXTURES
    assert "swe_experienced_resume.pdf" in BENCHMARK_FIXTURES

    # 2. Generalization directory exists and is distinct
    assert GENERALIZATION_FIXTURES_DIR.name == "generalization"
    assert GENERALIZATION_FIXTURES_DIR != Path(BENCHMARK_FIXTURES["AditCV_SOL.pdf"].filename).parent


def test_real_generalization_fixtures_manifest_and_discovery():
    """Verify real generalization fixture directory contains exactly 10 valid registered PDF fixtures."""
    manifest_path = GENERALIZATION_FIXTURES_DIR / "manifest.json"
    assert manifest_path.exists(), "Generalization manifest.json does not exist"

    corpus = GeneralizationCorpus(corpus_dir=GENERALIZATION_FIXTURES_DIR, manifest_file=manifest_path)
    assert len(corpus.fixtures) == 10

    discovered = corpus.discover_pdf_files()
    assert len(discovered) == 10
    discovered_names = [p.name for p in discovered]
    assert discovered_names == [
        "academic_research_postdoc_cv.pdf",
        "dense_technical_infrastructure_engineer.pdf",
        "entry_level_fresher_swe.pdf",
        "executive_vp_engineering.pdf",
        "hybrid_multicolumn_product_designer.pdf",
        "long_academic_tenured_professor_cv.pdf",
        "modern_two_column_product_manager.pdf",
        "project_heavy_fullstack_dev.pdf",
        "scanned_clinical_specialist.pdf",
        "table_heavy_consulting_projects.pdf",
    ]

    summary = corpus.get_summary()
    assert summary.total_fixtures == 10
    assert summary.ocr_required_count == 1
    assert summary.table_heavy_count == 1
    assert "two_column" in summary.layout_distribution
    assert "single_column" in summary.layout_distribution
    assert "hybrid" in summary.layout_distribution
    assert "tabular" in summary.layout_distribution
    assert "academic_cv" in summary.archetype_distribution
    assert "standard_cv" in summary.archetype_distribution
    assert "executive" in summary.career_level_distribution
    assert "entry" in summary.career_level_distribution
    assert "mid" in summary.career_level_distribution
    assert "senior" in summary.career_level_distribution


def test_semantic_benchmark_runner_executes_on_generalization_corpus(tmp_path: Path):
    """SemanticBenchmarkRunner seamlessly runs with generalization suite using injected MockSemanticExtractor."""
    f1 = _create_dummy_pdf(tmp_path / "gen_candidate_1.pdf")

    # Mock extractor producing valid SemanticOutput
    class StubExtractor:
        last_usage_metadata = {
            "provider": "mock",
            "model": "stub-v1",
            "prompt_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
        }

        def extract(self, input_data: SemanticInput) -> SemanticOutput:
            if input_data.blocks:
                b0 = input_data.blocks[0]
                return SemanticOutput(
                    document_archetype=DocumentArchetype.STANDARD_CV,
                    personal=GroundedPersonal(
                        name=GroundedString(value=b0.text, source_block_ids=[b0.block_id])
                    ),
                )
            return SemanticOutput(document_archetype=DocumentArchetype.STANDARD_CV)

    meta = GeneralizationFixtureMetadata(
        fixture_id="gen_001",
        filename="gen_candidate_1.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        domain="biotech",
    )

    runner = SemanticBenchmarkRunner(
        extractor=StubExtractor(),
        fixtures_dir=tmp_path,
        suite_id=BenchmarkSuiteId.GENERALIZATION.value,
        metadata_registry={"gen_candidate_1.pdf": meta},
    )

    discovered = runner.discover_fixtures()
    assert discovered == [f1]

    summary = runner.run_all()
    assert summary.suite_id == "generalization"
    assert summary.total_cases == 1
    assert summary.successful_cases == 1
    assert summary.results[0].filename == "gen_candidate_1.pdf"
    assert summary.results[0].target_domain == "biotech"


def test_failure_taxonomy_classification():
    """Failures are correctly categorized and mapped to known or new failure classes."""
    # 1. Provenance violation -> PROVENANCE / ALREADY_KNOWN
    res_prov = {"status": "FAIL", "validation_violations": ["UNKNOWN_BLOCK_ID: block b_99 not in document"]}
    cls_prov = classify_generalization_failure(res_prov)
    assert cls_prov is not None
    assert cls_prov.category == GeneralizationFailureCategory.PROVENANCE
    assert cls_prov.origin == FailureOrigin.ALREADY_KNOWN

    # 2. Form doc title -> PERSONAL_FIELD_ASSIGNMENT / ALREADY_KNOWN
    res_name = {"status": "FAIL", "diagnostics": ["NAME_IS_FORM_OR_DOC_TITLE: 'APPLICATION FORM'"]}
    cls_name = classify_generalization_failure(res_name)
    assert cls_name is not None
    assert cls_name.category == GeneralizationFailureCategory.PERSONAL_FIELD_ASSIGNMENT
    assert cls_name.origin == FailureOrigin.ALREADY_KNOWN

    # 3. Table header in experience -> TABLE_INTERPRETATION / ALREADY_KNOWN
    res_tbl = {"status": "FAIL", "diagnostics": ["TABLE_HEADER_IN_EXPERIENCE: s.no"]}
    cls_tbl = classify_generalization_failure(res_tbl)
    assert cls_tbl is not None
    assert cls_tbl.category == GeneralizationFailureCategory.TABLE_INTERPRETATION
    assert cls_tbl.origin == FailureOrigin.ALREADY_KNOWN

    # 4. OCR noise -> OCR_NOISE / NEW_FAILURE_CLASS
    meta_ocr = GeneralizationFixtureMetadata(
        fixture_id="ocr_1",
        filename="scanned.pdf",
        archetype=DocumentArchetype.UNKNOWN,
        ocr_required=True,
    )
    res_ocr = {"status": "FAIL", "diagnostics": []}
    cls_ocr = classify_generalization_failure(res_ocr, meta_ocr)
    assert cls_ocr is not None
    assert cls_ocr.category == GeneralizationFailureCategory.OCR_NOISE
    assert cls_ocr.origin == FailureOrigin.NEW_FAILURE_CLASS

    # 5. Extraction / unhandled exception -> SEMANTIC_INTERPRETATION / REQUIRES_MANUAL_REVIEW
    res_exc = {"status": "ERROR", "failure_type": "EXTRACTION_ERROR", "error_message": "LLM timeout"}
    cls_exc = classify_generalization_failure(res_exc)
    assert cls_exc is not None
    assert cls_exc.category == GeneralizationFailureCategory.SEMANTIC_INTERPRETATION
    assert cls_exc.origin == FailureOrigin.REQUIRES_MANUAL_REVIEW


def test_generalization_aggregate_report_and_sanitization(tmp_path: Path):
    """Aggregate report computes multi-dimensional breakdowns and sanitizes outputs without PII."""
    meta1 = GeneralizationFixtureMetadata(
        fixture_id="gen_01",
        filename="eng_1.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        layout=GeneralizationLayout.TWO_COLUMN,
        domain="tech",
        career_level=CareerLevel.SENIOR,
    )
    meta2 = GeneralizationFixtureMetadata(
        fixture_id="gen_02",
        filename="mar_1.pdf",
        archetype=DocumentArchetype.MARITIME_TABULAR,
        layout=GeneralizationLayout.TABULAR,
        domain="maritime",
        career_level=CareerLevel.MID,
        table_heavy=True,
    )

    corpus = GeneralizationCorpus(corpus_dir=tmp_path)
    _create_dummy_pdf(tmp_path / "eng_1.pdf")
    _create_dummy_pdf(tmp_path / "mar_1.pdf")
    corpus.register_fixture(meta1)
    corpus.register_fixture(meta2)

    results = [
        {
            "filename": "eng_1.pdf",
            "status": "PASS",
            "semantic_success": True,
            "elapsed_seconds": 1.5,
            "usage": {"prompt_tokens": 2000, "output_tokens": 300, "total_tokens": 2300, "fallback_invoked": False},
            "skills_count": 12,
            "experience_count": 3,
        },
        {
            "filename": "mar_1.pdf",
            "status": "FAIL",
            "semantic_success": True,
            "elapsed_seconds": 2.5,
            "validation_violations": ["UNKNOWN_BLOCK_ID: block 4 not found"],
            "diagnostics": ["TABLE_HEADER_IN_EXPERIENCE: rank"],
            "usage": {"prompt_tokens": 4000, "output_tokens": 500, "total_tokens": 4500, "fallback_invoked": True},
            "skills_count": 5,
            "experience_count": 8,
        },
    ]

    from tests.benchmark.generalization import build_generalization_aggregate_report

    report = build_generalization_aggregate_report(results, corpus)

    assert report.total_resumes == 2
    assert report.pass_count == 1
    assert report.fail_count == 1
    assert report.pass_rate_pct == 50.0
    assert report.fail_rate_pct == 50.0
    assert report.fallback_count == 1
    assert report.avg_latency_seconds == 2.0
    assert report.total_tokens == 6800
    assert report.avg_tokens_per_resume == 3400.0

    # Multi-dimensional breakdowns
    assert report.results_by_layout["two_column"]["PASS"] == 1
    assert report.results_by_layout["tabular"]["FAIL"] == 1
    assert report.results_by_domain["tech"]["PASS"] == 1
    assert report.results_by_domain["maritime"]["FAIL"] == 1
    assert report.results_by_career_level["senior"]["PASS"] == 1
    assert report.results_by_career_level["mid"]["FAIL"] == 1

    # Failure taxonomy
    assert report.failures_by_category[GeneralizationFailureCategory.PROVENANCE.value] == 1
    assert report.failures_by_origin[FailureOrigin.ALREADY_KNOWN.value] == 1

    # Machine-readable sanitized JSON export
    json_path = tmp_path / "generalization_report.json"
    exported = report.export_json(json_path)

    assert json_path.exists()
    assert exported["total_resumes"] == 2
    assert len(exported["fixture_results"]) == 2

    # Zero PII assertions
    json_str = json.dumps(exported)
    assert "password" not in json_str
    assert "AIzaSy" not in json_str
    assert "Bearer " not in json_str


def test_generalization_runner_3_fixture_two_pass_telemetry(tmp_path: Path):
    """Verify that SemanticBenchmarkRunner executes a 3-fixture generalization suite in two-pass mode with full telemetry."""
    from tests.benchmark.generalization import build_generalization_aggregate_report

    f1 = _create_dummy_pdf(tmp_path / "eng_lead.pdf")
    f2 = _create_dummy_pdf(tmp_path / "deck_cadet.pdf")
    f3 = _create_dummy_pdf(tmp_path / "researcher.pdf")

    corpus = GeneralizationCorpus(corpus_dir=tmp_path)
    meta1 = GeneralizationFixtureMetadata(
        fixture_id="gen_01",
        filename="eng_lead.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        layout=GeneralizationLayout.SINGLE_COLUMN,
        domain="engineering",
        career_level=CareerLevel.SENIOR,
    )
    meta2 = GeneralizationFixtureMetadata(
        fixture_id="gen_02",
        filename="deck_cadet.pdf",
        archetype=DocumentArchetype.MARITIME_CV,
        layout=GeneralizationLayout.TABULAR,
        domain="maritime",
        career_level=CareerLevel.ENTRY,
        table_heavy=True,
    )
    meta3 = GeneralizationFixtureMetadata(
        fixture_id="gen_03",
        filename="researcher.pdf",
        archetype=DocumentArchetype.ACADEMIC_CV,
        layout=GeneralizationLayout.TWO_COLUMN,
        domain="academia",
        career_level=CareerLevel.STUDENT,
    )
    corpus.register_fixture(meta1)
    corpus.register_fixture(meta2)
    corpus.register_fixture(meta3)

    class TwoPassStubExtractor:
        _explicit_two_pass = True
        _explicit_model = "gemini-3.5-flash-lite"
        last_usage_metadata = {
            "provider": "gemini",
            "model": "gemini-3.5-flash-lite",
            "representation": "two_pass_candidate_b",
            "two_pass": True,
            "prompt_tokens": 1500,
            "output_tokens": 350,
            "total_tokens": 1850,
            "latency_ms": 1200.0,
            "retry_count": 0,
            "fallback_invoked": False,
        }

        def extract(self, input_data: SemanticInput) -> SemanticOutput:
            if input_data.blocks:
                b0 = input_data.blocks[0]
                return SemanticOutput(
                    document_archetype=input_data.archetype,
                    personal=GroundedPersonal(
                        name=GroundedString(value=b0.text, source_block_ids=[b0.block_id])
                    ),
                )
            return SemanticOutput(document_archetype=input_data.archetype)

    runner = SemanticBenchmarkRunner(
        extractor=TwoPassStubExtractor(),
        fixtures_dir=tmp_path,
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        suite_id=BenchmarkSuiteId.GENERALIZATION.value,
        metadata_registry=corpus.fixtures,
    )

    discovered = runner.discover_fixtures()
    assert len(discovered) == 3
    assert discovered == [f2, f1, f3] or set(discovered) == {f1, f2, f3}

    summary = runner.run_all()
    assert summary.suite_id == "generalization"
    assert summary.total_cases == 3
    assert summary.successful_cases == 3
    assert summary.extraction_mode == "two_pass"
    assert summary.pass_count == 2

    # Check that individual telemetry was populated on each result
    for r in summary.results:
        assert r.provider == "gemini"
        assert r.model == "gemini-3.5-flash-lite"
        assert r.extraction_mode == "two_pass"
        assert r.pass_count == 2
        assert r.usage is not None
        assert r.usage["total_tokens"] == 1850
        assert r.elapsed_seconds >= 0.0

    # Multi-dimensional aggregate report
    report = build_generalization_aggregate_report(summary.results, corpus)
    assert report.total_resumes == 3
    assert report.pass_count == 3
    assert report.pass_rate_pct == 100.0
    assert report.total_tokens == 1850 * 3
    assert len(report.results_by_archetype) == 3
    assert len(report.results_by_domain) == 3
    assert len(report.results_by_layout) == 3
    assert len(report.results_by_career_level) == 3
