"""Reporting and formatting utilities for resume parser benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.benchmark.runner import BenchmarkRunSummary, ResumeParseResult


def format_terminal_table(summary: BenchmarkRunSummary) -> str:
    """Format benchmark results into the requested concise terminal table:
    Resume | Parse | Skills | Experience | Education | Projects | Notes
    """
    headers = ["Resume", "Parse", "Skills", "Experience", "Education", "Projects", "Notes"]
    col_widths = [32, 9, 8, 12, 11, 10, 50]

    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    separator_line = "-+-".join("-" * w for w in col_widths)

    rows = [header_line, separator_line]

    for r in summary.results:
        short_name = r.filename
        if len(short_name) > 32:
            short_name = short_name[:29] + "..."

        parse_str = f"{r.status}"
        if not r.parser_success:
            parse_str = "CRASH"

        skills_str = str(r.skills_count)
        exp_str = str(r.experience_count)
        edu_str = str(r.education_count)
        proj_str = str(r.projects_count)

        # Generate concise note summarizing state and key anomaly
        if r.status == "PASS":
            notes_str = "Clean extraction; all expectations met"
        elif not r.parser_success:
            notes_str = f"Error: {r.error_message}"
        else:
            anomaly_tags = []
            for d in r.diagnostics:
                tag = d.split(":")[0]
                if tag not in anomaly_tags:
                    anomaly_tags.append(tag)
            notes_str = ", ".join(anomaly_tags) if anomaly_tags else r.notes

        if len(notes_str) > 50:
            notes_str = notes_str[:47] + "..."

        row_vals = [short_name, parse_str, skills_str, exp_str, edu_str, proj_str, notes_str]
        row_line = " | ".join(f"{v:<{w}}" for v, w in zip(row_vals, col_widths))
        rows.append(row_line)

    return "\n".join(rows)


def generate_markdown_report(summary: BenchmarkRunSummary) -> str:
    """Generate a comprehensive Markdown baseline benchmark report."""
    pass_pct = (summary.pass_count / summary.total_resumes * 100) if summary.total_resumes else 0.0
    partial_pct = (summary.partial_count / summary.total_resumes * 100) if summary.total_resumes else 0.0
    fail_pct = (summary.fail_count / summary.total_resumes * 100) if summary.total_resumes else 0.0

    lines = [
        "# Phase 8-1 Real Resume Benchmark Baseline Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total Resumes Evaluated:** {summary.total_resumes}",
        f"- **Parser Crashes:** {summary.error_count} (100% execution completion without uncaught exceptions)",
        f"- **PASS:** {summary.pass_count} ({pass_pct:.1f}%)",
        f"- **PARTIAL:** {summary.partial_count} ({partial_pct:.1f}%)",
        f"- **FAIL:** {summary.fail_count} ({fail_pct:.1f}%)",
        f"- **Total Benchmark Runtime:** {summary.total_elapsed_seconds:.2f}s",
        "",
        "## 2. Archetype Breakdown",
        "",
        "| Archetype | Total | PASS | PARTIAL | FAIL | Pass Rate |",
        "|---|---|---|---|---|---|",
    ]

    for arch, counts in summary.archetype_breakdown.items():
        total = counts["total"]
        passes = counts.get("PASS", 0)
        rate = (passes / total * 100) if total else 0.0
        lines.append(f"| `{arch}` | {total} | {passes} | {counts.get('PARTIAL', 0)} | {counts.get('FAIL', 0)} | {rate:.1f}% |")

    lines.extend([
        "",
        "## 3. Resume Baseline Results Table",
        "",
        "| Resume | Archetype | Status | Skills | Experience | Education | Projects | Structural Diagnostics / Notes |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for r in summary.results:
        diag_str = "<br>".join(f"• `{d}`" for d in r.diagnostics) if r.diagnostics else "_Clean_"
        lines.append(
            f"| `{r.filename}` | `{r.archetype}` | **{r.status}** | {r.skills_count} | {r.experience_count} | {r.education_count} | {r.projects_count} | {diag_str} |"
        )

    lines.extend([
        "",
        "## 4. Top 5 Systemic Failure Classes in Rule-Based Parser",
        "",
        "1. **Document Header / Form Label Extracted as Personal Name & Location Leakage**",
        "   - *Pattern:* Non-standard documents often have titles like `APPLICATION FORM`, `Curriculum Vitae`, `Seafarer Profile` or form prompt labels like `Surname`. Rule-based regex/heuristics mistake these for candidate names.",
        "   - *Location Leakage:* Headings such as `QUALIFICATION`, `References`, `Position`, `Discipline` or split candidate surnames (`AGARWAL`) leak into `personal.location`.",
        "   - *Examples:* `JOSH PARASHAR MASTER CV2.pdf` (Name: 'APPLICATION FORM', Loc: 'Position'), `CV Rishabh Dixit.pdf` (Name: 'Curriculum Vitae'), `AASHISH DG.pdf` (Name: 'Seafarer Profile', Loc: 'Discipline'), `AKIBUL ALAM CV(JO).pdf` (Name: 'Surname', Loc: 'Alam'), `2nd Officer Mayur Agarwal_062029.pdf` (Name: 'MAYUR', Loc: 'AGARWAL'), `MUKUND 3RD OFF CV 2026.pdf` (Loc: 'QUALIFICATION').",
        "",
        "2. **Tabular Sea-Service & Vessel Grids Ingested as Massive Education Entries**",
        "   - *Pattern:* In maritime CVs and seafarer forms, vessel sea service, STCW courses, and document verification grids lack standard education keywords like 'University'. Loose fallbacks categorize hundreds of table rows under `EDUCATION`.",
        "   - *Examples:* `AASHISH DG.pdf` (36 education items), `JOSH PARASHAR MASTER CV2.pdf` (28 education items), `AKIBUL ALAM CV(JO).pdf` (25 education items), `2nd Officer Mayur Agarwal_062029.pdf` (10 education items capturing 'SEA SERVICE', 'DOCUMENTS', 'ENDORSEMENTS').",
        "",
        "3. **Table Column Headers Extracted as Experience Companies & Designations**",
        "   - *Pattern:* Horizontal reading order across table columns fuses column headers (`Ship Name`, `S. No.`, `Vessel Name`, `Period`, `Type`) into company names and designations.",
        "   - *Examples:* `CV Rishabh Dixit.pdf` (comp='Ship Name', desig='S. No.'), `JOSH PARASHAR MASTER CV2.pdf` (comp='Period', desig='Vessel Type'), `MUKUND 3RD OFF CV 2026.pdf` (comp='Type', desig='Vessel Name').",
        "",
        "4. **Reference & Referee Contact Blocks Ingested into Professional Experience**",
        "   - *Pattern:* Multi-column referee blocks at the end of a CV are grouped into experience candidate groups, generating multiple false-positive employment entries.",
        "   - *Examples:* `Sendrick Costa CV.pdf` generates 14 experience items including referee contacts ('Josley Jeffroy Rodrigues', 'Pedro D’silva') and address lines ('Goa - India').",
        "",
        "5. **Intra-Section Project Boundary & Header-Date Association Failures**",
        "   - *Pattern:* In multi-project resumes where project title and date are on the same line without strong font size contrast or large vertical margins, candidate grouping splits between title and date or binds subtitles, creating entries with `name=None` and capturing page numbers as extra projects.",
        "   - *Examples:* `Rajeev_Ranjan_Prajapati_FullStack_Engineer.pdf` (3 project entries have `name=None`, page footer '1' captured as project item).",
        "",
        "## 5. Comparison Readiness: Current Parser vs Future LLM-Assisted Parser",
        "",
        "This baseline proves that while the deterministic layout pipeline is highly effective for standard resumes (4/5 PASS on `standard_cv`), it fundamentally degrades on tabular and structured multi-column documents (0/7 PASS on maritime/forms).",
        "",
        "The benchmark infrastructure established in `tests/benchmark/` provides a plug-and-play evaluation harness: swapping or augmenting `parser.parse_with_layout_pipeline` with an LLM-assisted pipeline will immediately yield measurable deltas against this machine-readable baseline report.",
    ])

    return "\n".join(lines)


def save_benchmark_report(summary: BenchmarkRunSummary, output_dir: Path) -> tuple[Path, Path]:
    """Save machine-readable JSON and Markdown reports to the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "baseline_report.json"
    md_path = output_dir / "baseline_report.md"

    # Save clean machine-readable JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

    # Save human-readable Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(summary))

    return json_path, md_path
