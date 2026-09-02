"""CLI entrypoint to execute the real resume benchmark and output results."""

from __future__ import annotations

import sys
from pathlib import Path

from tests.benchmark.report import format_terminal_table, save_benchmark_report
from tests.benchmark.runner import BenchmarkRunner

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def main():
    print("Executing Phase 8-1 Real Resume Benchmark across 12 fixtures...\n")
    runner = BenchmarkRunner()
    summary = runner.run_all()

    # Print terminal table
    print(format_terminal_table(summary))
    print("\n" + "=" * 80)
    print(
        f"Benchmark Complete: {summary.total_resumes} evaluated | "
        f"{summary.pass_count} PASS | {summary.partial_count} PARTIAL | {summary.fail_count} FAIL | "
        f"{summary.error_count} ERRORS in {summary.total_elapsed_seconds:.2f}s"
    )
    print("=" * 80 + "\n")

    # Save reports
    json_p, md_p = save_benchmark_report(summary, REPORTS_DIR)
    print(f"Machine-readable report saved to: {json_p}")
    print(f"Human-readable Markdown report saved to: {md_p}")


if __name__ == "__main__":
    main()
