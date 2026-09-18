"""Comparison helper for contrasting deterministic layout baseline and semantic LLM benchmark results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tests.benchmark.metadata import BENCHMARK_FIXTURES
from tests.benchmark.runner import BenchmarkRunSummary, ResumeParseResult
from tests.benchmark.semantic_runner import SemanticBenchmarkSummary, SemanticParseResult


@dataclass
class FieldDelta:
    field_name: str
    deterministic_val: Any
    semantic_val: Any
    change: str  # "improved", "regressed", "unchanged"


@dataclass
class ResumeComparison:
    filename: str
    archetype: str
    deterministic_status: str
    semantic_status: str
    field_deltas: list[FieldDelta] = field(default_factory=list)
    anomalies_reduced: list[str] = field(default_factory=list)
    anomalies_introduced: list[str] = field(default_factory=list)
    comparison_summary: str = ""


@dataclass
class BenchmarkComparisonReport:
    total_compared: int = 0
    deterministic_successes: int = 0
    semantic_successes: int = 0
    deterministic_succeeds_semantic_fails: list[str] = field(default_factory=list)
    semantic_succeeds_deterministic_fails: list[str] = field(default_factory=list)
    both_succeed: list[str] = field(default_factory=list)
    both_fail: list[str] = field(default_factory=list)
    total_anomalies_reduced: int = 0
    total_anomalies_introduced: int = 0
    resume_comparisons: list[ResumeComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_benchmarks(
    deterministic_summary: BenchmarkRunSummary,
    semantic_summary: SemanticBenchmarkSummary,
) -> BenchmarkComparisonReport:
    """Compare deterministic baseline results against semantic LLM results across identical fixtures."""
    det_map: dict[str, ResumeParseResult] = {r.filename: r for r in deterministic_summary.results}
    sem_map: dict[str, SemanticParseResult] = {r.filename: r for r in semantic_summary.results}

    all_filenames = sorted(set(det_map.keys()).union(sem_map.keys()))

    report = BenchmarkComparisonReport(total_compared=len(all_filenames))

    for fn in all_filenames:
        det = det_map.get(fn)
        sem = sem_map.get(fn)

        if not det or not sem:
            continue

        meta = BENCHMARK_FIXTURES.get(fn)
        expected_name = meta.candidate_name if meta else None

        # Status comparison
        det_ok = det.parser_success and det.status != "FAIL"
        sem_ok = sem.semantic_success

        if det_ok:
            report.deterministic_successes += 1
        if sem_ok:
            report.semantic_successes += 1

        if det_ok and not sem_ok:
            report.deterministic_succeeds_semantic_fails.append(fn)
        elif sem_ok and not det_ok:
            report.semantic_succeeds_deterministic_fails.append(fn)
        elif det_ok and sem_ok:
            report.both_succeed.append(fn)
        else:
            report.both_fail.append(fn)

        # Compare fields
        field_deltas: list[FieldDelta] = []

        # 1. Name comparison
        det_name = det.personal.get("name") if det.personal else None
        sem_name = sem.personal.get("name") if sem.personal else None
        if expected_name:
            if det_name != expected_name and sem_name == expected_name:
                name_change = "improved"
            elif det_name == expected_name and sem_name != expected_name:
                name_change = "regressed"
            else:
                name_change = "unchanged"
        else:
            name_change = "unchanged" if det_name == sem_name else ("improved" if (sem_name and not det_name) else "regressed")

        field_deltas.append(FieldDelta("name", det_name, sem_name, name_change))

        # 2. Contact details (email, phone, location)
        for contact_field in ("email", "phone", "location"):
            d_val = det.personal.get(contact_field) if det.personal else None
            s_val = sem.personal.get(contact_field) if sem.personal else None
            if not d_val and s_val:
                c_change = "improved"
            elif d_val and not s_val:
                c_change = "regressed"
            else:
                c_change = "unchanged"
            field_deltas.append(FieldDelta(contact_field, d_val, s_val, c_change))

        # 3. Counts: skills, experience, education, projects
        for count_field in ("skills", "experience", "education", "projects"):
            d_count = getattr(det, f"{count_field}_count", 0)
            s_count = getattr(sem, f"{count_field}_count", 0)
            c_change = "unchanged" if d_count == s_count else ("improved" if (s_count > 0 and d_count == 0) else "regressed" if (d_count > 0 and s_count == 0) else "unchanged")
            field_deltas.append(FieldDelta(f"{count_field}_count", d_count, s_count, c_change))

        # Anomaly diagnostics comparison
        det_diags = set(det.diagnostics)
        sem_diags = set(sem.diagnostics)

        anomalies_reduced = sorted(det_diags - sem_diags)
        anomalies_introduced = sorted(sem_diags - det_diags)

        report.total_anomalies_reduced += len(anomalies_reduced)
        report.total_anomalies_introduced += len(anomalies_introduced)

        summary_note = (
            f"det={det.status}, sem={sem.status}; "
            f"{len(anomalies_reduced)} anomalies reduced, {len(anomalies_introduced)} introduced"
        )

        res_comp = ResumeComparison(
            filename=fn,
            archetype=sem.archetype or det.archetype,
            deterministic_status=det.status,
            semantic_status=sem.status,
            field_deltas=field_deltas,
            anomalies_reduced=anomalies_reduced,
            anomalies_introduced=anomalies_introduced,
            comparison_summary=summary_note,
        )
        report.resume_comparisons.append(res_comp)

    return report


def format_comparison_summary(report: BenchmarkComparisonReport) -> str:
    """Format the comparison report into a clear console summary."""
    lines = [
        "Benchmark Comparison: Deterministic Baseline vs. Semantic LLM",
        "=" * 60,
        f"Total fixtures compared: {report.total_compared}",
        f"Deterministic successes: {report.deterministic_successes}/{report.total_compared}",
        f"Semantic successes:      {report.semantic_successes}/{report.total_compared}",
        f"Both succeed:            {len(report.both_succeed)}",
        f"Both fail:               {len(report.both_fail)}",
        f"Deterministic succeeds, semantic fails: {len(report.deterministic_succeeds_semantic_fails)} {report.deterministic_succeeds_semantic_fails}",
        f"Semantic succeeds, deterministic fails: {len(report.semantic_succeeds_deterministic_fails)} {report.semantic_succeeds_deterministic_fails}",
        f"Total anomaly counts reduced:    {report.total_anomalies_reduced}",
        f"Total anomaly counts introduced: {report.total_anomalies_introduced}",
        "",
        "Per-Resume Delta Breakdown:",
        "-" * 60,
    ]
    for rc in report.resume_comparisons:
        lines.append(f"- {rc.filename} [{rc.archetype}]: det={rc.deterministic_status} -> sem={rc.semantic_status}")
        if rc.anomalies_reduced:
            lines.append(f"    Reduced: {', '.join(rc.anomalies_reduced)}")
        if rc.anomalies_introduced:
            lines.append(f"    Introduced: {', '.join(rc.anomalies_introduced)}")

    return "\n".join(lines)


def _percentile(values: list[float], pct: float) -> float:
    """Calculate percentile from a list of float/int numbers."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(round((len(sorted_v) - 1) * pct))
    return float(sorted_v[idx])


@dataclass
class SemanticResumeDelta:
    filename: str
    archetype: str
    baseline_status: str
    candidate_status: str
    classification: str  # "newly_fixed", "regression", "unchanged", "manual_review"
    baseline_violations_count: int = 0
    candidate_violations_count: int = 0
    baseline_latency: float = 0.0
    candidate_latency: float = 0.0
    baseline_tokens: int = 0
    candidate_tokens: int = 0
    anomalies_reduced: list[str] = field(default_factory=list)
    anomalies_introduced: list[str] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticComparisonReport:
    total_compared: int = 0

    baseline_provider: str = "gemini"
    baseline_model: str = "gemini-3.5-flash-lite"
    baseline_representation: str = "single_pass_candidate_b"

    candidate_provider: str = "gemini"
    candidate_model: str = "gemini-3.5-flash-lite"
    candidate_representation: str = "two_pass_candidate_b"

    # Outcome counts
    baseline_pass_count: int = 0
    baseline_partial_count: int = 0
    baseline_fail_count: int = 0
    baseline_error_count: int = 0
    baseline_pass_rate_pct: float = 0.0

    candidate_pass_count: int = 0
    candidate_partial_count: int = 0
    candidate_fail_count: int = 0
    candidate_error_count: int = 0
    candidate_pass_rate_pct: float = 0.0

    # Invariants & Violations
    baseline_validation_failures: int = 0
    candidate_validation_failures: int = 0
    baseline_severe_anomalies: int = 0
    candidate_severe_anomalies: int = 0
    baseline_hallucinated_ids: int = 0
    candidate_hallucinated_ids: int = 0
    baseline_unsupported_values: int = 0
    candidate_unsupported_values: int = 0

    # Latency (seconds)
    baseline_avg_latency: float = 0.0
    baseline_p50_latency: float = 0.0
    baseline_p95_latency: float = 0.0
    candidate_avg_latency: float = 0.0
    candidate_p50_latency: float = 0.0
    candidate_p95_latency: float = 0.0

    # Total Tokens
    baseline_avg_tokens: float = 0.0
    baseline_p50_tokens: float = 0.0
    baseline_p95_tokens: float = 0.0
    candidate_avg_tokens: float = 0.0
    candidate_p50_tokens: float = 0.0
    candidate_p95_tokens: float = 0.0

    # Resume Categorization
    newly_fixed: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    manual_review: list[str] = field(default_factory=list)

    resume_deltas: list[SemanticResumeDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_results_list(data: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Helper to normalize SemanticBenchmarkSummary, dict, list, or Path to list of dicts."""
    import json
    from pathlib import Path

    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark file not found: '{p}'")
        data = json.loads(p.read_text(encoding="utf-8"))

    if hasattr(data, "to_dict"):
        data = data.to_dict()

    if isinstance(data, dict):
        meta = {
            "provider": data.get("provider", "unknown"),
            "model": data.get("model", "unknown"),
            "representation": data.get("representation", "unknown"),
        }
        res_list = data.get("results", [])
        return meta, [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in res_list]

    if isinstance(data, list):
        meta = {
            "provider": "unknown",
            "model": "unknown",
            "representation": "unknown",
        }
        return meta, [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in data]

    raise TypeError(f"Unsupported benchmark run data type: {type(data)}")


def compare_semantic_runs(
    baseline_data: Any,
    candidate_data: Any,
) -> SemanticComparisonReport:
    """Compare a baseline semantic benchmark run against a candidate semantic benchmark run."""
    base_meta, base_results = _extract_results_list(baseline_data)
    cand_meta, cand_results = _extract_results_list(candidate_data)

    base_map = {r.get("filename"): r for r in base_results if r.get("filename")}
    cand_map = {r.get("filename"): r for r in cand_results if r.get("filename")}

    all_filenames = sorted(set(base_map.keys()).union(cand_map.keys()))

    report = SemanticComparisonReport(
        total_compared=len(all_filenames),
        baseline_provider=base_meta.get("provider", "gemini"),
        baseline_model=base_meta.get("model", "gemini-3.5-flash-lite"),
        baseline_representation=base_meta.get("representation", "single_pass_candidate_b"),
        candidate_provider=cand_meta.get("provider", "gemini"),
        candidate_model=cand_meta.get("model", "gemini-3.5-flash-lite"),
        candidate_representation=cand_meta.get("representation", "two_pass_candidate_b"),
    )

    base_latencies: list[float] = []
    cand_latencies: list[float] = []
    base_tokens: list[float] = []
    cand_tokens: list[float] = []

    for fn in all_filenames:
        b = base_map.get(fn)
        c = cand_map.get(fn)

        b_stat = str(b.get("status", "UNKNOWN")).upper() if b else "MISSING"
        c_stat = str(c.get("status", "UNKNOWN")).upper() if c else "MISSING"

        # Baseline tallies
        if b:
            if b_stat == "PASS":
                report.baseline_pass_count += 1
            elif b_stat == "PARTIAL":
                report.baseline_partial_count += 1
            elif b_stat in ("ERROR", "PARSER_EXCEPTION", "EXTRACTION_FAILED"):
                report.baseline_error_count += 1
            else:
                report.baseline_fail_count += 1

            b_viols = b.get("validation_violations") or []
            if b_viols:
                report.baseline_validation_failures += 1
            report.baseline_hallucinated_ids += len(b.get("hallucinated_block_ids") or [v for v in b_viols if "UNKNOWN_BLOCK_ID" in v])
            report.baseline_unsupported_values += len(b.get("unsupported_canonical_values") or [v for v in b_viols if "UNSUPPORTED_CANONICAL_VALUE" in v])

            b_diags = b.get("diagnostics") or []
            severe_markers = ["NAME_IS_FORM_OR_DOC_TITLE", "SECTION_HEADER_IN_LOCATION", "NAME_SPLIT_INTO_LOCATION", "EXTREME_EDUCATION_COUNT_SKEW", "EXCESSIVE_EXPERIENCE_COUNT", "TABLE_HEADER_IN_EXPERIENCE"]
            report.baseline_severe_anomalies += sum(1 for d in b_diags if any(m in d for m in severe_markers))

            b_lat = float(b.get("elapsed_seconds") or 0.0)
            base_latencies.append(b_lat)

            b_usage = b.get("usage") or {}
            b_tok = b_usage.get("total_tokens") or (int(b_usage.get("prompt_tokens", 0) or 0) + int(b_usage.get("output_tokens", 0) or 0))
            if b_tok:
                base_tokens.append(float(b_tok))
        else:
            b_lat = 0.0
            b_tok = 0
            b_viols = []
            b_diags = []

        # Candidate tallies
        if c:
            if c_stat == "PASS":
                report.candidate_pass_count += 1
            elif c_stat == "PARTIAL":
                report.candidate_partial_count += 1
            elif c_stat in ("ERROR", "PARSER_EXCEPTION", "EXTRACTION_FAILED"):
                report.candidate_error_count += 1
            else:
                report.candidate_fail_count += 1

            c_viols = c.get("validation_violations") or []
            if c_viols:
                report.candidate_validation_failures += 1
            report.candidate_hallucinated_ids += len(c.get("hallucinated_block_ids") or [v for v in c_viols if "UNKNOWN_BLOCK_ID" in v])
            report.candidate_unsupported_values += len(c.get("unsupported_canonical_values") or [v for v in c_viols if "UNSUPPORTED_CANONICAL_VALUE" in v])

            c_diags = c.get("diagnostics") or []
            severe_markers = ["NAME_IS_FORM_OR_DOC_TITLE", "SECTION_HEADER_IN_LOCATION", "NAME_SPLIT_INTO_LOCATION", "EXTREME_EDUCATION_COUNT_SKEW", "EXCESSIVE_EXPERIENCE_COUNT", "TABLE_HEADER_IN_EXPERIENCE"]
            report.candidate_severe_anomalies += sum(1 for d in c_diags if any(m in d for m in severe_markers))

            c_lat = float(c.get("elapsed_seconds") or 0.0)
            cand_latencies.append(c_lat)

            c_usage = c.get("usage") or {}
            c_tok = c_usage.get("total_tokens") or (int(c_usage.get("prompt_tokens", 0) or 0) + int(c_usage.get("output_tokens", 0) or 0))
            if c_tok:
                cand_tokens.append(float(c_tok))
        else:
            c_lat = 0.0
            c_tok = 0
            c_viols = []
            c_diags = []

        # Diagnostics diff
        anomalies_reduced = sorted(set(b_diags) - set(c_diags))
        anomalies_introduced = sorted(set(c_diags) - set(b_diags))

        # Classification logic
        arch = (c.get("archetype") if c else b.get("archetype")) or "unknown"
        if not b:
            classification = "newly_fixed" if c_stat == "PASS" else "manual_review"
        elif not c:
            classification = "regression"
        elif c_stat in ("FAIL", "ERROR") and b_stat in ("PASS", "PARTIAL"):
            classification = "regression"
        elif len(c_viols) > len(b_viols) or len(anomalies_introduced) > len(anomalies_reduced):
            classification = "regression"
        elif c_stat == "PASS" and b_stat != "PASS":
            classification = "newly_fixed"
        elif len(b_viols) > 0 and len(c_viols) == 0:
            classification = "newly_fixed"
        elif len(anomalies_reduced) > 0 and len(anomalies_introduced) == 0:
            classification = "newly_fixed"
        elif c_stat == "PARTIAL" or bool(c_diags):
            classification = "manual_review"
        else:
            classification = "unchanged"

        if classification == "newly_fixed":
            report.newly_fixed.append(fn)
        elif classification == "regression":
            report.regressions.append(fn)
        elif classification == "manual_review":
            report.manual_review.append(fn)
        else:
            report.unchanged.append(fn)

        delta = SemanticResumeDelta(
            filename=fn,
            archetype=arch,
            baseline_status=b_stat,
            candidate_status=c_stat,
            classification=classification,
            baseline_violations_count=len(b_viols),
            candidate_violations_count=len(c_viols),
            baseline_latency=round(b_lat, 2),
            candidate_latency=round(c_lat, 2),
            baseline_tokens=int(b_tok),
            candidate_tokens=int(c_tok),
            anomalies_reduced=anomalies_reduced,
            anomalies_introduced=anomalies_introduced,
            details=f"b_stat={b_stat}, c_stat={c_stat}, reduced={len(anomalies_reduced)}, intro={len(anomalies_introduced)}",
        )
        report.resume_deltas.append(delta)

    # Calculate aggregate metrics
    tot = report.total_compared or 1
    report.baseline_pass_rate_pct = round((report.baseline_pass_count / tot) * 100.0, 1)
    report.candidate_pass_rate_pct = round((report.candidate_pass_count / tot) * 100.0, 1)

    if base_latencies:
        report.baseline_avg_latency = round(sum(base_latencies) / len(base_latencies), 2)
        report.baseline_p50_latency = round(_percentile(base_latencies, 0.50), 2)
        report.baseline_p95_latency = round(_percentile(base_latencies, 0.95), 2)

    if cand_latencies:
        report.candidate_avg_latency = round(sum(cand_latencies) / len(cand_latencies), 2)
        report.candidate_p50_latency = round(_percentile(cand_latencies, 0.50), 2)
        report.candidate_p95_latency = round(_percentile(cand_latencies, 0.95), 2)

    if base_tokens:
        report.baseline_avg_tokens = round(sum(base_tokens) / len(base_tokens), 1)
        report.baseline_p50_tokens = round(_percentile(base_tokens, 0.50), 1)
        report.baseline_p95_tokens = round(_percentile(base_tokens, 0.95), 1)

    if cand_tokens:
        report.candidate_avg_tokens = round(sum(cand_tokens) / len(cand_tokens), 1)
        report.candidate_p50_tokens = round(_percentile(cand_tokens, 0.50), 1)
        report.candidate_p95_tokens = round(_percentile(cand_tokens, 0.95), 1)

    return report


def format_semantic_comparison_markdown(report: SemanticComparisonReport) -> str:
    """Format the semantic comparison report into clean, sanitized Markdown."""
    lines = [
        "# Semantic Benchmark Comparison Report",
        "",
        f"- **Baseline:** `{report.baseline_provider}` / `{report.baseline_model}` (`{report.baseline_representation}`)",
        f"- **Candidate:** `{report.candidate_provider}` / `{report.candidate_model}` (`{report.candidate_representation}`)",
        f"- **Total Fixtures Evaluated:** {report.total_compared}",
        "",
        "## 1. Aggregate Outcome Comparison",
        "",
        "| Metric | Baseline (Single-Pass) | Candidate (Two-Pass) | Delta |",
        "|---|---|---|---|",
        f"| **PASS Count** | {report.baseline_pass_count} | {report.candidate_pass_count} | {report.candidate_pass_count - report.baseline_pass_count:+d} |",
        f"| **PARTIAL Count** | {report.baseline_partial_count} | {report.candidate_partial_count} | {report.candidate_partial_count - report.baseline_partial_count:+d} |",
        f"| **FAIL Count** | {report.baseline_fail_count} | {report.candidate_fail_count} | {report.candidate_fail_count - report.baseline_fail_count:+d} |",
        f"| **ERROR Count** | {report.baseline_error_count} | {report.candidate_error_count} | {report.candidate_error_count - report.baseline_error_count:+d} |",
        f"| **PASS Rate** | {report.baseline_pass_rate_pct:.1f}% | {report.candidate_pass_rate_pct:.1f}% | {report.candidate_pass_rate_pct - report.baseline_pass_rate_pct:+.1f}% |",
        f"| **Validation Failures** | {report.baseline_validation_failures} | {report.candidate_validation_failures} | {report.candidate_validation_failures - report.baseline_validation_failures:+d} |",
        f"| **Severe Anomalies** | {report.baseline_severe_anomalies} | {report.candidate_severe_anomalies} | {report.candidate_severe_anomalies - report.baseline_severe_anomalies:+d} |",
        f"| **Hallucinated Block IDs** | {report.baseline_hallucinated_ids} | {report.candidate_hallucinated_ids} | {report.candidate_hallucinated_ids - report.baseline_hallucinated_ids:+d} |",
        f"| **Unsupported Canonical Values** | {report.baseline_unsupported_values} | {report.candidate_unsupported_values} | {report.candidate_unsupported_values - report.baseline_unsupported_values:+d} |",
        f"| **Average Latency** | {report.baseline_avg_latency:.2f}s | {report.candidate_avg_latency:.2f}s | {report.candidate_avg_latency - report.baseline_avg_latency:+.2f}s |",
        f"| **p50 / p95 Latency** | {report.baseline_p50_latency:.2f}s / {report.baseline_p95_latency:.2f}s | {report.candidate_p50_latency:.2f}s / {report.candidate_p95_latency:.2f}s | — |",
        f"| **Average Total Tokens** | {report.baseline_avg_tokens:.1f} | {report.candidate_avg_tokens:.1f} | {report.candidate_avg_tokens - report.baseline_avg_tokens:+.1f} |",
        f"| **p50 / p95 Tokens** | {report.baseline_p50_tokens:.0f} / {report.baseline_p95_tokens:.0f} | {report.candidate_p50_tokens:.0f} / {report.candidate_p95_tokens:.0f} | — |",
        "",
        "## 2. Qualitative Classification Breakdown",
        "",
        f"- **Newly Fixed Resumes ({len(report.newly_fixed)}):** {', '.join(f'`{f}`' for f in report.newly_fixed) if report.newly_fixed else '_None_'}",
        f"- **Regressions ({len(report.regressions)}):** {', '.join(f'`{f}`' for f in report.regressions) if report.regressions else '_None_'}",
        f"- **Unchanged Resumes ({len(report.unchanged)}):** {', '.join(f'`{f}`' for f in report.unchanged) if report.unchanged else '_None_'}",
        f"- **Requiring Manual Review ({len(report.manual_review)}):** {', '.join(f'`{f}`' for f in report.manual_review) if report.manual_review else '_None_'}",
        "",
        "## 3. Per-Resume Delta Matrix",
        "",
        "| Resume | Archetype | Baseline Status | Candidate Status | Classification | Latency (Base -> Cand) | Tokens (Base -> Cand) | Anomalies Delta |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for d in report.resume_deltas:
        lat_str = f"{d.baseline_latency:.2f}s -> {d.candidate_latency:.2f}s"
        tok_str = f"{d.baseline_tokens} -> {d.candidate_tokens}"
        anom_str = f"-{len(d.anomalies_reduced)} / +{len(d.anomalies_introduced)}"
        lines.append(
            f"| `{d.filename}` | `{d.archetype}` | {d.baseline_status} | **{d.candidate_status}** | `{d.classification}` | {lat_str} | {tok_str} | {anom_str} |"
        )

    return "\n".join(lines)


def format_semantic_comparison_terminal(report: SemanticComparisonReport) -> str:
    """Format the semantic comparison report into a clear console summary."""
    lines = [
        "",
        "Semantic Benchmark Comparison: Single-Pass Baseline vs. Two-Pass Candidate",
        "=" * 72,
        f"Total fixtures compared: {report.total_compared}",
        f"PASS Rate:               {report.baseline_pass_rate_pct:.1f}% (Baseline) -> {report.candidate_pass_rate_pct:.1f}% (Candidate)",
        f"PASS / PARTIAL / FAIL:   {report.baseline_pass_count}/{report.baseline_partial_count}/{report.baseline_fail_count} -> {report.candidate_pass_count}/{report.candidate_partial_count}/{report.candidate_fail_count}",
        f"Validation Failures:     {report.baseline_validation_failures} -> {report.candidate_validation_failures}",
        f"Severe Anomalies:        {report.baseline_severe_anomalies} -> {report.candidate_severe_anomalies}",
        f"Hallucinated Block IDs:  {report.baseline_hallucinated_ids} -> {report.candidate_hallucinated_ids}",
        f"Avg Latency (p50/p95):   {report.baseline_avg_latency:.2f}s ({report.baseline_p50_latency:.2f}s/{report.baseline_p95_latency:.2f}s) -> {report.candidate_avg_latency:.2f}s ({report.candidate_p50_latency:.2f}s/{report.candidate_p95_latency:.2f}s)",
        f"Avg Tokens (p50/p95):    {report.baseline_avg_tokens:.0f} ({report.baseline_p50_tokens:.0f}/{report.baseline_p95_tokens:.0f}) -> {report.candidate_avg_tokens:.0f} ({report.candidate_p50_tokens:.0f}/{report.candidate_p95_tokens:.0f})",
        "",
        "Classification Breakdown:",
        f"  Newly Fixed ({len(report.newly_fixed)}): {', '.join(report.newly_fixed) if report.newly_fixed else 'None'}",
        f"  Regressions ({len(report.regressions)}): {', '.join(report.regressions) if report.regressions else 'None'}",
        f"  Unchanged ({len(report.unchanged)}):   {', '.join(report.unchanged) if report.unchanged else 'None'}",
        f"  Manual Review ({len(report.manual_review)}): {', '.join(report.manual_review) if report.manual_review else 'None'}",
        "",
        "Per-Resume Delta Breakdown:",
        "-" * 72,
    ]
    for d in report.resume_deltas:
        lines.append(f"  {d.filename:<42} : {d.baseline_status:<8} -> {d.candidate_status:<8} [{d.classification}] ({d.baseline_latency:.2f}s -> {d.candidate_latency:.2f}s)")
        if d.anomalies_reduced:
            lines.append(f"    - Reduced: {', '.join(d.anomalies_reduced)}")
        if d.anomalies_introduced:
            lines.append(f"    + Introduced: {', '.join(d.anomalies_introduced)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for comparing two semantic benchmark results."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compare semantic benchmark runs.")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline JSON report")
    parser.add_argument("--candidate", type=str, required=True, help="Path to candidate JSON report")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    args = parser.parse_args(argv)

    try:
        report = compare_semantic_runs(args.baseline, args.candidate)
        if args.markdown:
            print(format_semantic_comparison_markdown(report))
        else:
            print(format_semantic_comparison_terminal(report))
        return 0
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
