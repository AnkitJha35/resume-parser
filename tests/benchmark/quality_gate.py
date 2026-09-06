"""Production Semantic Quality Gate for resume parser benchmark evaluation.

Provides deterministic quality-gate calculations, archetype breakdowns, regression tracking,
and machine/human-readable report generators for CI/release qualification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BenchmarkOutcome(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    ERROR = "ERROR"


class FailureCategory(str, Enum):
    NONE = "NONE"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    JSON_DECODE_ERROR = "JSON_DECODE_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ANOMALY_ERROR = "ANOMALY_ERROR"
    EXPECTATION_MISMATCH = "EXPECTATION_MISMATCH"
    PIPELINE_ERROR = "PIPELINE_ERROR"


@dataclass
class QualityGateThresholds:
    """Configurable quality gate thresholds based on verified empirical benchmark evidence."""
    min_pass_rate_pct: float = 80.0
    max_partial_rate_pct: float = 20.0
    max_fail_rate_pct: float = 0.0
    max_errors_allowed: int = 0
    max_validation_failures: int = 0
    max_avg_latency_seconds: float = 20.0
    max_avg_tokens: int = 15000


@dataclass
class QualityGateSummary:
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
    provider_error_count: int = 0
    fallback_count: int = 0
    body_recovery_count: int = 0

    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_resume: float = 0.0
    avg_latency_seconds: float = 0.0

    archetype_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    regression_checks: dict[str, bool] = field(default_factory=dict)

    passed_gate: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


KNOWN_REGRESSION_KEYS = [
    ("no_doc_title_as_name", "NAME_IS_FORM_OR_DOC_TITLE"),
    ("no_section_header_in_location", "SECTION_HEADER_IN_LOCATION"),
    ("no_name_split_in_location", "NAME_SPLIT_INTO_LOCATION"),
    ("no_table_header_in_experience", "TABLE_HEADER_IN_EXPERIENCE"),
    ("no_table_metadata_in_education", "TABLE_METADATA_IN_EDUCATION"),
    ("no_excessive_experience_count", "EXCESSIVE_EXPERIENCE_COUNT"),
    ("no_extreme_education_skew", "EXTREME_EDUCATION_COUNT_SKEW"),
    ("no_nameless_projects", "PROJECTS_WITHOUT_NAMES"),
    ("no_hallucinated_block_ids", "UNKNOWN_BLOCK_ID"),
    ("no_unsupported_canonical_values", "UNSUPPORTED_CANONICAL_VALUE"),
]


def evaluate_quality_gate(
    results: list[Any],
    thresholds: QualityGateThresholds | None = None,
) -> QualityGateSummary:
    """Evaluate a collection of benchmark parse results against quality-gate thresholds."""
    th = thresholds or QualityGateThresholds()
    summary = QualityGateSummary()

    if not results:
        summary.passed_gate = False
        summary.failure_reasons.append("No benchmark results provided to evaluate.")
        return summary

    summary.total_resumes = len(results)

    total_latency = 0.0
    tokens_reported_count = 0
    all_diagnostics: list[str] = []
    all_violations: list[str] = []

    # Initialize archetype counters
    known_archetypes = [
        "standard_cv",
        "maritime_cv",
        "maritime_tabular",
        "structured_form",
        "academic_cv",
        "unknown",
    ]
    for arch in known_archetypes:
        summary.archetype_breakdown[arch] = {
            "total": 0,
            "PASS": 0,
            "PARTIAL": 0,
            "FAIL": 0,
            "ERROR": 0,
        }

    raw_result_dicts: list[dict[str, Any]] = []

    for item in results:
        # Standardize item to dict
        r_dict = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        raw_result_dicts.append(r_dict)

        status = str(r_dict.get("status", "FAIL")).upper()
        arch = str(r_dict.get("archetype", "unknown")).lower()
        if arch not in summary.archetype_breakdown:
            summary.archetype_breakdown[arch] = {
                "total": 0,
                "PASS": 0,
                "PARTIAL": 0,
                "FAIL": 0,
                "ERROR": 0,
            }

        summary.archetype_breakdown[arch]["total"] += 1

        # Classify status
        if status == BenchmarkOutcome.PASS.value:
            summary.pass_count += 1
            summary.archetype_breakdown[arch]["PASS"] += 1
        elif status == BenchmarkOutcome.PARTIAL.value:
            summary.partial_count += 1
            summary.archetype_breakdown[arch]["PARTIAL"] += 1
        elif status == BenchmarkOutcome.ERROR.value or not r_dict.get("semantic_success", True):
            summary.error_count += 1
            summary.provider_error_count += 1
            summary.archetype_breakdown[arch]["ERROR"] += 1
        else:
            summary.fail_count += 1
            summary.archetype_breakdown[arch]["FAIL"] += 1

        # Validation violations
        violations = r_dict.get("validation_violations") or []
        if violations:
            summary.validation_failure_count += 1
            all_violations.extend(violations)

        # Diagnostics & Anomalies
        diagnostics = r_dict.get("diagnostics") or []
        all_diagnostics.extend(diagnostics)

        # Timing
        elapsed = float(r_dict.get("elapsed_seconds") or 0.0)
        total_latency += elapsed

        # Usage / Token instrumentation
        usage = r_dict.get("usage") or {}
        if isinstance(usage, dict):
            p_tok = usage.get("prompt_tokens") or usage.get("prompt_token_count")
            o_tok = usage.get("output_tokens") or usage.get("candidates_token_count")
            t_tok = usage.get("total_tokens") or usage.get("total_token_count")

            if p_tok is not None:
                summary.total_prompt_tokens += int(p_tok)
            if o_tok is not None:
                summary.total_output_tokens += int(o_tok)
            if t_tok is not None:
                summary.total_tokens += int(t_tok)
                tokens_reported_count += 1
            elif p_tok is not None and o_tok is not None:
                tot = int(p_tok) + int(o_tok)
                summary.total_tokens += tot
                tokens_reported_count += 1

            if usage.get("fallback_invoked"):
                summary.fallback_count += 1

        if r_dict.get("body_recovery_invoked") or (isinstance(usage, dict) and usage.get("body_recovery_invoked")):
            summary.body_recovery_count += 1

    # Aggregate metric calculations
    summary.pass_rate_pct = round((summary.pass_count / summary.total_resumes) * 100.0, 2)
    summary.partial_rate_pct = round((summary.partial_count / summary.total_resumes) * 100.0, 2)
    summary.fail_rate_pct = round((summary.fail_count / summary.total_resumes) * 100.0, 2)
    summary.error_rate_pct = round((summary.error_count / summary.total_resumes) * 100.0, 2)

    summary.avg_latency_seconds = round(total_latency / summary.total_resumes, 2)
    if tokens_reported_count > 0:
        summary.avg_tokens_per_resume = round(summary.total_tokens / tokens_reported_count, 1)

    # Evaluate Regression Checks
    for check_key, marker in KNOWN_REGRESSION_KEYS:
        found_in_diag = any(marker in d for d in all_diagnostics)
        found_in_viol = any(marker in v for v in all_violations)
        summary.regression_checks[check_key] = not (found_in_diag or found_in_viol)

    # Evaluate Quality Gate Thresholds
    failure_reasons: list[str] = []

    if summary.error_count > th.max_errors_allowed:
        failure_reasons.append(
            f"Errors exceed threshold: {summary.error_count} > {th.max_errors_allowed}"
        )

    if summary.validation_failure_count > th.max_validation_failures:
        failure_reasons.append(
            f"Validation failures exceed threshold: {summary.validation_failure_count} > {th.max_validation_failures}"
        )

    if summary.pass_rate_pct < th.min_pass_rate_pct:
        failure_reasons.append(
            f"Pass rate below threshold: {summary.pass_rate_pct:.1f}% < {th.min_pass_rate_pct:.1f}%"
        )

    if summary.fail_rate_pct > th.max_fail_rate_pct:
        failure_reasons.append(
            f"Fail rate exceeds threshold: {summary.fail_rate_pct:.1f}% > {th.max_fail_rate_pct:.1f}%"
        )

    if summary.avg_latency_seconds > th.max_avg_latency_seconds:
        failure_reasons.append(
            f"Average latency exceeds threshold: {summary.avg_latency_seconds:.2f}s > {th.max_avg_latency_seconds:.2f}s"
        )

    if summary.avg_tokens_per_resume > th.max_avg_tokens:
        failure_reasons.append(
            f"Average token usage exceeds threshold: {summary.avg_tokens_per_resume:.1f} > {th.max_avg_tokens}"
        )

    # Severe regression check failures trigger gate failure
    severe_failed_checks = [k for k, passed in summary.regression_checks.items() if not passed]
    if severe_failed_checks:
        failure_reasons.append(
            f"Historical regression signatures detected: {severe_failed_checks}"
        )

    summary.failure_reasons = failure_reasons
    summary.passed_gate = len(failure_reasons) == 0
    summary.results = raw_result_dicts
    return summary


def format_quality_gate_markdown(summary: QualityGateSummary) -> str:
    """Generate human-readable Markdown report with zero PII."""
    decision_badge = "✅ **GATE PASSED**" if summary.passed_gate else "❌ **GATE FAILED**"
    lines = [
        "# Production Semantic Quality Gate Report",
        "",
        f"### Status: {decision_badge}",
        "",
        "## 1. Aggregate Quality Gate Summary",
        "",
        f"- **Total Resumes Evaluated:** {summary.total_resumes}",
        f"- **PASS:** {summary.pass_count} ({summary.pass_rate_pct:.1f}%)",
        f"- **PARTIAL:** {summary.partial_count} ({summary.partial_rate_pct:.1f}%)",
        f"- **FAIL:** {summary.fail_count} ({summary.fail_rate_pct:.1f}%)",
        f"- **ERROR:** {summary.error_count} ({summary.error_rate_pct:.1f}%)",
        f"- **Validation Violations:** {summary.validation_failure_count}",
        f"- **Fallbacks Invoked:** {summary.fallback_count}",
        f"- **Body Recoveries Invoked:** {summary.body_recovery_count}",
        f"- **Total Tokens:** {summary.total_tokens} (Prompt: {summary.total_prompt_tokens}, Output: {summary.total_output_tokens})",
        f"- **Average Tokens / Resume:** {summary.avg_tokens_per_resume:.1f}",
        f"- **Average Extraction Latency:** {summary.avg_latency_seconds:.2f}s",
        "",
    ]

    if summary.failure_reasons:
        lines.extend([
            "### Quality Gate Rejection Reasons:",
            *[f"- ⚠️ {reason}" for reason in summary.failure_reasons],
            "",
        ])

    lines.extend([
        "## 2. Archetype Breakdown",
        "",
        "| Archetype | Total | PASS | PARTIAL | FAIL | ERROR | Pass Rate |",
        "|---|---|---|---|---|---|---|",
    ])

    for arch, counts in summary.archetype_breakdown.items():
        tot = counts["total"]
        if tot == 0:
            continue
        p = counts.get("PASS", 0)
        part = counts.get("PARTIAL", 0)
        f = counts.get("FAIL", 0)
        err = counts.get("ERROR", 0)
        rate = round((p / tot) * 100.0, 1)
        lines.append(f"| `{arch}` | {tot} | {p} | {part} | {f} | {err} | {rate:.1f}% |")

    lines.extend([
        "",
        "## 3. Historical Regression Protections",
        "",
        "| Regression Rule | Status | Description |",
        "|---|---|---|",
    ])

    descriptions = {
        "no_doc_title_as_name": "Document or form title used as personal name",
        "no_section_header_in_location": "Section header leaking into location field",
        "no_name_split_in_location": "Name token split into location field",
        "no_table_header_in_experience": "Table headers contaminating experience records",
        "no_table_metadata_in_education": "Table metadata contaminating education records",
        "no_excessive_experience_count": "References or excess lines parsed as jobs",
        "no_extreme_education_skew": "Extreme row explosion in education table cells",
        "no_nameless_projects": "Projects missing project titles",
        "no_hallucinated_block_ids": "Ungrounded / hallucinated block IDs in output",
        "no_unsupported_canonical_values": "Unsupported canonical strings outside block evidence",
    }

    for check_key, passed in summary.regression_checks.items():
        status_sym = "✅ PASS" if passed else "❌ FAIL"
        desc = descriptions.get(check_key, check_key)
        lines.append(f"| `{check_key}` | {status_sym} | {desc} |")

    lines.extend([
        "",
        "## 4. Per-Resume Execution Summary",
        "",
        "| Resume | Archetype | Status | Tokens (P/O/Tot) | Latency | Recovery | Fallback | Violations | Counts (Sk/Ex/Ed/Pr) | Notes / Diagnostics |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])

    for r in summary.results:
        fn = r.get("filename", "unknown")
        arch = r.get("archetype", "unknown")
        st = r.get("status", "UNKNOWN")
        lat = f"{float(r.get('elapsed_seconds', 0.0)):.2f}s"
        usage = r.get("usage") or {}
        p_tok = usage.get("prompt_tokens") or 0
        o_tok = usage.get("output_tokens") or 0
        t_tok = usage.get("total_tokens") or (p_tok + o_tok)
        tok_str = f"{p_tok}/{o_tok}/{t_tok}"
        fb = "Yes" if (usage.get("fallback_invoked") or r.get("fallback_invoked")) else "No"
        rec = "Yes" if (usage.get("body_recovery_invoked") or r.get("body_recovery_invoked")) else "No"
        viol_count = len(r.get("validation_violations") or [])
        counts_str = f"{r.get('skills_count', 0)}/{r.get('experience_count', 0)}/{r.get('education_count', 0)}/{r.get('projects_count', 0)}"
        diag = "<br>".join(r.get("diagnostics") or []) or "_Clean_"

        lines.append(
            f"| `{fn}` | `{arch}` | **{st}** | {tok_str} | {lat} | {rec} | {fb} | {viol_count} | {counts_str} | {diag} |"
        )

    return "\n".join(lines)


def export_quality_gate_json(summary: QualityGateSummary, path: Path | None = None) -> dict[str, Any]:
    """Export machine-readable JSON output suitable for CI and release review."""
    data = summary.to_dict()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return data


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for evaluating Quality Gate on benchmark result files."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate Production Semantic Quality Gate.")
    parser.add_argument("--results", type=str, required=True, help="Path to benchmark JSON results file")
    parser.add_argument("--output-json", type=str, default=None, help="Path to export quality gate JSON summary")
    parser.add_argument("--markdown", action="store_true", default=True, help="Print markdown report")
    args = parser.parse_args(argv)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: '{results_path}'", file=sys.stderr)
        return 1

    try:
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "results" in data:
            results_list = data["results"]
        elif isinstance(data, list):
            results_list = data
        else:
            print("ERROR: Unrecognized benchmark results structure.", file=sys.stderr)
            return 1

        summary = evaluate_quality_gate(results_list)

        if args.markdown:
            print(format_quality_gate_markdown(summary))

        if args.output_json:
            export_quality_gate_json(summary, Path(args.output_json))
            print(f"\n[QUALITY GATE] Summary saved to: {args.output_json}")

        return 0 if summary.passed_gate else 2

    except Exception as err:
        print(f"ERROR evaluating quality gate: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
