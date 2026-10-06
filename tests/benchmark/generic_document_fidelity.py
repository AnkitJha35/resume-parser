"""Generic Document Fidelity Benchmark Runner.

Evaluates fidelity across the entire 22-document corpus:
- 12 regression PDFs in tests/fixtures/
- 10 generalization PDFs in tests/fixtures/generalization/

Computes concrete coverage, grounding, heading, list, and table metrics.
Outputs terminal summary and saves JSON/Markdown reports.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.domain.document import document_from_text_blocks
from app.domain.document_structure import DocumentBlock, DocumentSection, DocumentStructure
from app.domain.semantic_contract import build_semantic_input
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor
from app.validation.generic_document_validator import validate_generic_document_structure


@dataclass(frozen=True)
class DocumentFidelityMetrics:
    filename: str
    category: str
    page_count: int
    source_meaningful_blocks: int
    emitted_meaningful_blocks: int
    covered_source_blocks: int
    source_coverage_pct: float
    grounded_emitted_blocks: int
    unsupported_blocks: int
    duplicate_blocks: int
    unassigned_blocks: int
    headings_detected: int
    paragraphs_detected: int
    lists_detected: int
    tables_detected: int
    provenance_violations: int
    status: str
    elapsed_seconds: float


@dataclass(frozen=True)
class CorpusFidelitySummary:
    total_documents: int
    total_pages: int
    total_source_blocks: int
    total_emitted_blocks: int
    total_covered_source_blocks: int
    average_coverage_pct: float
    min_coverage_pct: float
    max_coverage_pct: float
    total_unassigned_blocks: int
    total_unsupported_blocks: int
    total_duplicate_blocks: int
    total_headings_detected: int
    total_paragraphs_detected: int
    total_lists_detected: int
    total_tables_detected: int
    total_violations: int
    pass_count: int
    fail_count: int
    total_elapsed_seconds: float
    documents: list[DocumentFidelityMetrics]


def _collect_emitted_items(sections: list[DocumentSection]) -> tuple[list[DocumentBlock], list[DocumentSection]]:
    blocks: list[DocumentBlock] = []
    all_sections: list[DocumentSection] = []

    def _walk(sec_list: list[DocumentSection]) -> None:
        for s in sec_list:
            all_sections.append(s)
            blocks.extend(s.blocks)
            if s.subsections:
                _walk(s.subsections)

    _walk(sections)
    return blocks, all_sections


def evaluate_document_fidelity(pdf_path: Path, category: str, parser: ResumeParser) -> DocumentFidelityMetrics:
    start_t = time.perf_counter()
    raw_bytes = pdf_path.read_bytes()
    filename = pdf_path.name

    # 1. Parse document structure
    doc_structure = parser.parse_document_structure(raw_bytes, document_id=filename)

    # 2. Extract semantic input
    phys = document_from_text_blocks(PDFExtractor.extract(raw_bytes))
    recon = reconstruct_document(phys)
    layout = interpret_layout(recon)
    sem_inp = build_semantic_input(layout, document_id=filename)

    # Source meaningful blocks
    source_blocks = [b for b in sem_inp.blocks if b.text.strip()]
    source_bids = {b.block_id for b in source_blocks}
    source_count = len(source_blocks)

    # Emitted items
    emitted_blocks, all_sections = _collect_emitted_items(doc_structure.sections)
    emitted_count = len(emitted_blocks)

    # Cited source block IDs
    cited_bids: set[str] = set()
    for b in emitted_blocks:
        cited_bids.update(b.source_block_ids)
    for s in all_sections:
        cited_bids.update(s.source_block_ids)

    covered_bids = source_bids & cited_bids
    covered_count = len(covered_bids)
    unassigned_count = len(source_bids - cited_bids)
    coverage_pct = (covered_count / source_count * 100.0) if source_count > 0 else 100.0

    # Provenance validation
    violations = validate_generic_document_structure(doc_structure, semantic_input=sem_inp)
    violation_count = len(violations)

    # Count block types
    para_count = sum(1 for b in emitted_blocks if b.type == "paragraph")
    list_count = sum(1 for b in emitted_blocks if b.type == "list_item")
    table_count = sum(1 for b in emitted_blocks if b.type == "table")
    headings_count = sum(1 for s in all_sections if s.heading is not None)

    # Check duplicates and unsupported
    unsupported_count = sum(1 for v in violations if "UNGROUNDED_CONTENT" in v or "MISSING_PROVENANCE" in v)
    duplicate_count = sum(1 for v in violations if "DUPLICATE_OWNERSHIP" in v)
    grounded_count = emitted_count - unsupported_count

    elapsed = time.perf_counter() - start_t
    status = "PASS" if (coverage_pct >= 98.0 and violation_count == 0) else "FAIL"

    return DocumentFidelityMetrics(
        filename=filename,
        category=category,
        page_count=doc_structure.page_count,
        source_meaningful_blocks=source_count,
        emitted_meaningful_blocks=emitted_count,
        covered_source_blocks=covered_count,
        source_coverage_pct=round(coverage_pct, 2),
        grounded_emitted_blocks=grounded_count,
        unsupported_blocks=unsupported_count,
        duplicate_blocks=duplicate_count,
        unassigned_blocks=unassigned_count,
        headings_detected=headings_count,
        paragraphs_detected=para_count,
        lists_detected=list_count,
        tables_detected=table_count,
        provenance_violations=violation_count,
        status=status,
        elapsed_seconds=round(elapsed, 3),
    )


def run_fidelity_benchmark() -> CorpusFidelitySummary:
    start_total = time.perf_counter()
    parser = ResumeParser()

    reg_dir = Path("tests/fixtures")
    gen_dir = Path("tests/fixtures/generalization")

    pdf_files: list[tuple[Path, str]] = []
    for p in sorted(reg_dir.glob("*.pdf")):
        pdf_files.append((p, "regression"))
    for p in sorted(gen_dir.glob("*.pdf")):
        pdf_files.append((p, "generalization"))

    metrics_list: list[DocumentFidelityMetrics] = []
    for p, cat in pdf_files:
        m = evaluate_document_fidelity(p, cat, parser)
        metrics_list.append(m)

    total_elapsed = time.perf_counter() - start_total

    total_source = sum(m.source_meaningful_blocks for m in metrics_list)
    total_emitted = sum(m.emitted_meaningful_blocks for m in metrics_list)
    total_covered = sum(m.covered_source_blocks for m in metrics_list)
    total_pages = sum(m.page_count for m in metrics_list)
    coverages = [m.source_coverage_pct for m in metrics_list]
    avg_cov = sum(coverages) / len(coverages) if coverages else 100.0
    min_cov = min(coverages) if coverages else 100.0
    max_cov = max(coverages) if coverages else 100.0

    pass_cnt = sum(1 for m in metrics_list if m.status == "PASS")
    fail_cnt = sum(1 for m in metrics_list if m.status != "PASS")

    return CorpusFidelitySummary(
        total_documents=len(metrics_list),
        total_pages=total_pages,
        total_source_blocks=total_source,
        total_emitted_blocks=total_emitted,
        total_covered_source_blocks=total_covered,
        average_coverage_pct=round(avg_cov, 2),
        min_coverage_pct=round(min_cov, 2),
        max_coverage_pct=round(max_cov, 2),
        total_unassigned_blocks=sum(m.unassigned_blocks for m in metrics_list),
        total_unsupported_blocks=sum(m.unsupported_blocks for m in metrics_list),
        total_duplicate_blocks=sum(m.duplicate_blocks for m in metrics_list),
        total_headings_detected=sum(m.headings_detected for m in metrics_list),
        total_paragraphs_detected=sum(m.paragraphs_detected for m in metrics_list),
        total_lists_detected=sum(m.lists_detected for m in metrics_list),
        total_tables_detected=sum(m.tables_detected for m in metrics_list),
        total_violations=sum(m.provenance_violations for m in metrics_list),
        pass_count=pass_cnt,
        fail_count=fail_cnt,
        total_elapsed_seconds=round(total_elapsed, 2),
        documents=metrics_list,
    )


def format_fidelity_table(summary: CorpusFidelitySummary) -> str:
    lines = [
        "=" * 130,
        f"{'GENERIC DOCUMENT FIDELITY BENCHMARK':^130}",
        "=" * 130,
        f"{'Document':<42} {'Category':<15} {'Pages':<6} {'Src Blks':<9} {'Cov %':<8} {'Headings':<9} {'Paras':<7} {'Lists':<7} {'Tables':<7} {'Viol':<5} {'Status':<6}",
        "-" * 130,
    ]

    for d in summary.documents:
        lines.append(
            f"{d.filename[:40]:<42} {d.category:<15} {d.page_count:<6} {d.source_meaningful_blocks:<9} "
            f"{d.source_coverage_pct:<8.1f} {d.headings_detected:<9} {d.paragraphs_detected:<7} {d.lists_detected:<7} "
            f"{d.tables_detected:<7} {d.provenance_violations:<5} {d.status:<6}"
        )

    lines.append("-" * 130)
    lines.append(
        f"{'TOTAL / AGGREGATE':<42} {'ALL (22)':<15} {summary.total_pages:<6} {summary.total_source_blocks:<9} "
        f"{summary.average_coverage_pct:<8.1f} {summary.total_headings_detected:<9} {summary.total_paragraphs_detected:<7} "
        f"{summary.total_lists_detected:<7} {summary.total_tables_detected:<7} {summary.total_violations:<5} "
        f"{'PASS' if summary.fail_count == 0 else 'FAIL':<6}"
    )
    lines.append("=" * 130)
    return "\n".join(lines)


def save_fidelity_reports(summary: CorpusFidelitySummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "generic_document_fidelity_report.json"
    md_path = output_dir / "generic_document_fidelity_report.md"

    # JSON export
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    # Markdown export
    md_lines = [
        "# Generic Document Fidelity Benchmark Report",
        "",
        f"- **Evaluated Documents**: {summary.total_documents} ({summary.pass_count} PASS, {summary.fail_count} FAIL)",
        f"- **Total Pages**: {summary.total_pages}",
        f"- **Total Source Blocks**: {summary.total_source_blocks}",
        f"- **Total Covered Source Blocks**: {summary.total_covered_source_blocks}",
        f"- **Average Coverage**: {summary.average_coverage_pct}% (Min: {summary.min_coverage_pct}%, Max: {summary.max_coverage_pct}%)",
        f"- **Unassigned Source Blocks**: {summary.total_unassigned_blocks}",
        f"- **Unsupported / Invented Blocks**: {summary.total_unsupported_blocks}",
        f"- **Duplicate Block Claims**: {summary.total_duplicate_blocks}",
        f"- **Provenance Violations**: {summary.total_violations}",
        f"- **Elapsed Execution Time**: {summary.total_elapsed_seconds:.2f}s",
        "",
        "## Summary Table",
        "",
        "| Document | Category | Pages | Source Blocks | Coverage % | Headings | Paragraphs | Lists | Tables | Violations | Status |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]

    for d in summary.documents:
        md_lines.append(
            f"| `{d.filename}` | {d.category} | {d.page_count} | {d.source_meaningful_blocks} | {d.source_coverage_pct:.1f}% | "
            f"{d.headings_detected} | {d.paragraphs_detected} | {d.lists_detected} | {d.tables_detected} | {d.provenance_violations} | **{d.status}** |"
        )

    md_lines.extend([
        f"| **TOTAL / AGGREGATE** | **Corpus (22)** | **{summary.total_pages}** | **{summary.total_source_blocks}** | **{summary.average_coverage_pct:.1f}%** | "
        f"**{summary.total_headings_detected}** | **{summary.total_paragraphs_detected}** | **{summary.total_lists_detected}** | **{summary.total_tables_detected}** | **{summary.total_violations}** | **{'PASS' if summary.fail_count == 0 else 'FAIL'}** |",
        "",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return json_path, md_path


def main() -> None:
    print("Running Generic Document Fidelity Benchmark across 22 fixtures...\n")
    summary = run_fidelity_benchmark()
    print(format_fidelity_table(summary))

    reports_dir = Path(__file__).resolve().parent / "reports"
    json_p, md_p = save_fidelity_reports(summary, reports_dir)
    print(f"\nSaved JSON report to: {json_p}")
    print(f"Saved Markdown report to: {md_p}")


if __name__ == "__main__":
    main()
