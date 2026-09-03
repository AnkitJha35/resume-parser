"""Opt-in CLI runner for executing real resume PDFs against Gemini semantic extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from app.core.config import Settings
from app.extractors.providers.gemini import GeminiSemanticExtractor
from tests.benchmark.compare import compare_benchmarks, format_comparison_summary
from tests.benchmark.runner import BenchmarkRunner
from tests.benchmark.semantic_runner import SemanticBenchmarkRunner, SemanticBenchmarkSummary


def format_semantic_terminal_summary(summary: SemanticBenchmarkSummary) -> str:
    """Format the semantic benchmark results into the specified human-readable console summary."""
    lines = [
        "",
        "Semantic Benchmark",
        "==================",
        "",
        f"Provider: {summary.provider}",
        f"Model: {summary.model}",
        "",
        f"Cases: {summary.total_cases}",
        f"Successful: {summary.successful_cases}",
        f"Extraction failures: {summary.extraction_failures}",
        f"Validation failures: {summary.validation_failures}",
        f"Validation pass rate: {summary.validation_pass_rate_pct}%",
        f"Total elapsed: {summary.total_elapsed_seconds:.2f}s",
    ]
    if summary.total_tokens is not None:
        lines.append(f"Total tokens recorded: {summary.total_tokens}")

    lines.extend([
        "",
        "By archetype:",
    ])
    for arch, counts in summary.archetype_breakdown.items():
        succ = counts.get("SUCCESS", 0)
        tot = counts.get("total", 0)
        pct = (succ / tot * 100) if tot else 0.0
        lines.append(f"  {arch:<20} : {succ}/{tot} passed ({pct:.1f}%) [val_fail={counts.get('VALIDATION_FAILED', 0)}, ext_fail={counts.get('EXTRACTION_FAILED', 0)}]")

    lines.extend([
        "",
        "Per resume:",
    ])
    for r in summary.results:
        diag = f" ({', '.join(r.diagnostics[:2])})" if r.diagnostics else ""
        lines.append(f"  {r.filename:<40} : {r.status}{diag}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run opt-in live semantic benchmark on real resumes.")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "gemini"],
        default="ollama",
        help="Semantic LLM provider (default: ollama)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. qwen2.5-coder:7b or gemini-3.5-flash-lite)")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for Ollama (default: http://localhost:11434)")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for Ollama (default: 8)")
    parser.add_argument(
        "--fixture",
        type=str,
        default=None,
        help="Run only a specific registered fixture PDF (e.g. 'AditCV_SOL.pdf')",
    )
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save JSON results")
    parser.add_argument("--compare", action="store_true", help="Also run deterministic baseline and print comparison")
    args = parser.parse_args(argv)

    provider = args.provider.lower()

    if provider == "ollama":
        from app.extractors.providers.ollama import OllamaSemanticExtractor

        base_url = args.base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = args.model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        timeout = args.timeout or float(os.environ.get("OLLAMA_TIMEOUT", "120.0"))
        num_threads = args.threads or (int(os.environ.get("OLLAMA_NUM_THREADS")) if os.environ.get("OLLAMA_NUM_THREADS") else None)

        # Pre-flight availability check
        available, error_msg = OllamaSemanticExtractor.check_availability(base_url, model)
        if not available:
            print(f"ERROR: {error_msg}", file=sys.stderr)
            return 1

        print(f"[LIVE BENCHMARK] Executing semantic benchmark via Ollama at '{base_url}' using model '{model}'...")
        print(f"[LIVE BENCHMARK] Request timeout: {timeout}s. CPU threads: {num_threads or 8}.")

        extractor = OllamaSemanticExtractor(
            base_url=base_url,
            model=model,
            timeout=timeout,
            num_threads=num_threads,
        )

    elif provider == "gemini":
        from app.extractors.providers.gemini import GeminiSemanticExtractor

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            try:
                settings = Settings()
                api_key = getattr(settings, "gemini_api_key", None)
            except Exception:
                pass

        if not api_key:
            print(
                "ERROR: GEMINI_API_KEY is not configured.\n"
                "Live Gemini benchmark requires setting GEMINI_API_KEY environment variable.\n"
                "Aborting without making any network calls.",
                file=sys.stderr,
            )
            return 1

        model = args.model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        timeout = args.timeout or float(os.environ.get("GEMINI_TIMEOUT", "30.0"))

        print(f"[LIVE BENCHMARK] Initiating external LLM calls to Gemini API using model '{model}'...")
        print(f"[LIVE BENCHMARK] Request timeout: {timeout}s. Credentials: [REDACTED]")

        extractor = GeminiSemanticExtractor(
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
    else:
        print(f"ERROR: Unknown provider '{provider}'", file=sys.stderr)
        return 1

    runner = SemanticBenchmarkRunner(
        extractor=extractor,
        provider_name=provider,
        model_name=model,
    )

    try:
        fixtures = runner.discover_fixtures(fixture_name=args.fixture)
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"[LIVE BENCHMARK] Discovered {len(fixtures)} benchmark PDF fixtures.")

    summary = runner.run_all(fixtures)

    # Print human-readable summary
    print(format_semantic_terminal_summary(summary))

    # Save machine-readable JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = output_dir / f"semantic_{provider}_{timestamp_slug}.json"

    # Export dictionary without storing any secret credentials
    summary_dict = summary.to_dict()
    json_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    print(f"\n[LIVE BENCHMARK] Report saved to: {json_path}")

    # Run comparison against deterministic if requested
    if args.compare:
        print("\n[LIVE BENCHMARK] Executing deterministic baseline for comparison...")
        det_runner = BenchmarkRunner()
        det_summary = det_runner.run_all(fixtures)
        comparison_report = compare_benchmarks(det_summary, summary)
        print(format_comparison_summary(comparison_report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
