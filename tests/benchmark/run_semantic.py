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
from tests.benchmark.compare import (
    compare_benchmarks,
    compare_semantic_runs,
    format_comparison_summary,
    format_semantic_comparison_terminal,
)
from tests.benchmark.quality_gate import evaluate_quality_gate, format_quality_gate_markdown
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
        f"Representation: {summary.representation} ({summary.extraction_mode}, {summary.pass_count} pass)",
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
        tok = f" [tokens={r.usage.get('total_tokens')}]" if r.usage and r.usage.get("total_tokens") else ""
        lines.append(f"  {r.filename:<40} : {r.status}{diag}{tok}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run opt-in live semantic benchmark on real resumes.")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "gemini", "openrouter", "nvidia"],
        default="gemini",
        help="Semantic LLM provider (default: gemini)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. qwen2.5-coder:7b or gemini-3.5-flash-lite)")
    parser.add_argument(
        "--representation",
        type=str,
        default="two_pass_candidate_b",
        choices=[
            "two_pass_candidate_b",
            "single_pass_candidate_b",
            "candidate_b_compact",
            "candidate_b_two_pass",
            "candidate_b_single_pass",
            "full",
        ],
        help="Payload representation and execution mode (default: two_pass_candidate_b)",
    )
    parser.add_argument(
        "--two-pass",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable/disable two-pass extraction (overrides representation mode default)",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for Ollama (default: http://localhost:11434)")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum output tokens budget (default: provider default)")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for Ollama (default: 8)")
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable thinking reasoning mode for Ollama models (default: --no-think)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["regression_12", "generalization"],
        default="regression_12",
        help="Benchmark suite to execute (default: regression_12)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional path to generalization corpus manifest JSON file",
    )
    parser.add_argument(
        "--fixture",
        type=str,
        default=None,
        help="Run only a specific registered fixture PDF (e.g. 'AditCV_SOL.pdf')",
    )
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save JSON results")
    parser.add_argument("--compare", action="store_true", help="Also run deterministic baseline and print comparison")
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Path to baseline semantic JSON report to compare against (e.g. tests/benchmark/reports/candidate_b_single_pass_baseline.json)",
    )
    parser.add_argument("--quality-gate", action="store_true", help="Evaluate results against production Quality Gate")
    args = parser.parse_args(argv)

    provider = args.provider.lower()

    # Determine two_pass and compact flags
    rep = args.representation.lower()
    if args.two_pass is not None:
        two_pass = args.two_pass
    elif "two_pass" in rep:
        two_pass = True
    else:
        two_pass = False

    compact = (rep != "full")
    resolved_rep = "two_pass_candidate_b" if two_pass else ("full" if not compact else "single_pass_candidate_b")

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
        print(f"[LIVE BENCHMARK] Request timeout: {timeout}s. CPU threads: {num_threads or 8}. Thinking: {args.think}.")

        extractor = OllamaSemanticExtractor(
            base_url=base_url,
            model=model,
            timeout=timeout,
            num_threads=num_threads,
            think=args.think,
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
        timeout = args.timeout or float(os.environ.get("GEMINI_TIMEOUT", "45.0"))

        print(f"[LIVE BENCHMARK] Initiating external LLM calls to Gemini API using model '{model}' ({resolved_rep})...")
        print(f"[LIVE BENCHMARK] Request timeout: {timeout}s. Credentials: [REDACTED]")

        extractor = GeminiSemanticExtractor(
            api_key=api_key,
            model=model,
            timeout=timeout,
            compact=compact,
            two_pass=two_pass,
        )
    elif provider == "openrouter":
        from app.extractors.providers.openrouter import OpenRouterSemanticExtractor

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            try:
                settings = Settings()
                api_key = getattr(settings, "openrouter_api_key", None)
            except Exception:
                pass

        if not api_key:
            print(
                "ERROR: OPENROUTER_API_KEY is not configured.\n"
                "Live OpenRouter benchmark requires setting OPENROUTER_API_KEY environment variable.\n"
                "Aborting without making any network calls.",
                file=sys.stderr,
            )
            return 1

        model = args.model or os.environ.get("OPENROUTER_MODEL", "google/gemini-3.5-flash-lite")
        timeout = args.timeout or float(os.environ.get("OPENROUTER_TIMEOUT", "60.0"))

        print(f"[LIVE BENCHMARK] Initiating external LLM calls to OpenRouter API using model '{model}' ({resolved_rep})...")
        print(f"[LIVE BENCHMARK] Request timeout: {timeout}s. Credentials: [REDACTED]")

        max_tokens_env = os.environ.get("OPENROUTER_MAX_TOKENS")
        max_tokens = args.max_tokens or (int(max_tokens_env) if max_tokens_env else None)

        extractor = OpenRouterSemanticExtractor(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            compact=compact,
            two_pass=two_pass,
        )
    elif provider == "nvidia":
        from app.extractors.providers.nvidia import NvidiaSemanticExtractor

        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            try:
                settings = Settings()
                api_key = getattr(settings, "nvidia_api_key", None)
            except Exception:
                pass

        if not api_key:
            print(
                "ERROR: NVIDIA_API_KEY is not configured.\n"
                "Live NVIDIA benchmark requires setting NVIDIA_API_KEY environment variable.\n"
                "Aborting without making any network calls.",
                file=sys.stderr,
            )
            return 1

        model = args.model or os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
        base_url = args.base_url or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        timeout = args.timeout or float(os.environ.get("NVIDIA_TIMEOUT", "60.0"))

        print(f"[LIVE BENCHMARK] Initiating external LLM calls to NVIDIA NIM API using model '{model}' ({resolved_rep})...")
        print(f"[LIVE BENCHMARK] Base URL: {base_url}. Request timeout: {timeout}s. Credentials: [REDACTED]")

        max_tokens_env = os.environ.get("NVIDIA_MAX_TOKENS")
        max_tokens = args.max_tokens or (int(max_tokens_env) if max_tokens_env else None)

        extractor = NvidiaSemanticExtractor(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens if max_tokens is not None else 16384,
            compact=compact,
            two_pass=two_pass,
        )
    else:
        print(f"ERROR: Unknown provider '{provider}'", file=sys.stderr)
        return 1

    corpus = None
    metadata_registry = None
    if args.suite == "generalization":
        from tests.benchmark.generalization import (
            GENERALIZATION_FIXTURES_DIR,
            GeneralizationCorpus,
            build_generalization_aggregate_report,
        )
        manifest_path = Path(args.manifest) if args.manifest else (GENERALIZATION_FIXTURES_DIR / "manifest.json")
        corpus = GeneralizationCorpus(corpus_dir=GENERALIZATION_FIXTURES_DIR, manifest_file=manifest_path if manifest_path.exists() else None)
        metadata_registry = corpus.fixtures

    runner = SemanticBenchmarkRunner(
        extractor=extractor,
        provider_name=provider,
        model_name=model,
        representation=resolved_rep,
        suite_id=args.suite,
        metadata_registry=metadata_registry,
    )

    try:
        fixtures = runner.discover_fixtures(fixture_name=args.fixture)
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"[LIVE BENCHMARK] Discovered {len(fixtures)} benchmark PDF fixtures ({args.suite} suite).")

    summary = runner.run_all(fixtures)

    # Print human-readable summary
    print(format_semantic_terminal_summary(summary))

    # Save machine-readable JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = output_dir / f"semantic_{provider}_{resolved_rep}_{timestamp_slug}.json"

    # Export dictionary without storing any secret credentials
    summary_dict = summary.to_dict()
    json_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    print(f"\n[LIVE BENCHMARK] Report saved to: {json_path}")

    # Export generalization aggregate report if running generalization suite
    if args.suite == "generalization" and corpus:
        gen_report = build_generalization_aggregate_report(summary.results, corpus)
        gen_json_path = output_dir / f"generalization_aggregate_{provider}_{resolved_rep}_{timestamp_slug}.json"
        gen_report.export_json(gen_json_path)
        print(f"[LIVE BENCHMARK] Generalization aggregate report saved to: {gen_json_path}")

    # Compare against semantic baseline if requested
    if args.compare_baseline:
        print(f"\n[LIVE BENCHMARK] Comparing against semantic baseline '{args.compare_baseline}'...")
        try:
            comp_report = compare_semantic_runs(args.compare_baseline, summary)
            print(format_semantic_comparison_terminal(comp_report))
        except Exception as err:
            print(f"ERROR comparing against baseline: {err}", file=sys.stderr)

    # Run comparison against deterministic if requested
    if args.compare:
        print("\n[LIVE BENCHMARK] Executing deterministic baseline for comparison...")
        det_runner = BenchmarkRunner()
        det_summary = det_runner.run_all(fixtures)
        comparison_report = compare_benchmarks(det_summary, summary)
        print(format_comparison_summary(comparison_report))

    # Evaluate quality gate if requested
    if args.quality_gate:
        print("\n[LIVE BENCHMARK] Evaluating Quality Gate...")
        qg_summary = evaluate_quality_gate(summary.results)
        print(format_quality_gate_markdown(qg_summary))

    return 0


if __name__ == "__main__":
    sys.exit(main())
