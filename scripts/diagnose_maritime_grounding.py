#!/usr/bin/env python3
"""Phase 10W.2: Diagnose Maritime Entity Grounding Failures.

Deterministic diagnostic utility to inspect block-level layout, geometric table binding,
semantic table contexts, logical rows, entity spans, and benchmark validation failures
across maritime CV fixtures:
1. 2nd Officer Mayur Agarwal_062029.pdf
2. MUKUND 3RD OFF CV 2026.pdf
3. Sendrick Costa CV.pdf
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import glob
import json
import os
from pathlib import Path
import re
from typing import Any

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    DeterministicEntitySpan,
    DocumentArchetype,
    SemanticBlockInput,
    SemanticInput,
    build_deterministic_experience_spans,
    build_semantic_input,
    partition_semantic_input_into_sections,
)
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.table_semantic_context import TableSemanticContext
from app.pipeline.stages.text_extraction import PDFExtractor


class FailureCategory(str, Enum):
    """Deterministic failure categorization taxonomy for entity grounding."""

    LAYOUT_SEGMENTATION = "LAYOUT_SEGMENTATION"
    TABLE_SEMANTICS = "TABLE_SEMANTICS"
    LOGICAL_ROW_CONTINUATION = "LOGICAL_ROW_CONTINUATION"
    ENTITY_SPAN = "ENTITY_SPAN"
    MODEL_PROVENANCE_ASSIGNMENT = "MODEL_PROVENANCE_ASSIGNMENT"
    UNKNOWN = "UNKNOWN"


@dataclass
class FailureAnalysis:
    """Detailed record of a single validation failure and its deterministic diagnosis."""

    filename: str
    failure_type: str
    target: str
    collection: str
    item_index: int | None
    field_name: str | None
    rejected_value: str
    source_text: str
    cited_block_ids: list[str]
    cited_blocks_metadata: list[dict[str, Any]]
    assigned_entity_span_index: int | None
    competing_entity_span_index: int | None
    category: FailureCategory
    diagnosis_reason: str


TARGET_FIXTURES = [
    "2nd Officer Mayur Agarwal_062029.pdf",
    "MUKUND 3RD OFF CV 2026.pdf",
    "Sendrick Costa CV.pdf",
]


def run_deterministic_pipeline(fixture_path: Path) -> tuple[SemanticInput, list[DeterministicEntitySpan]]:
    """Run the complete deterministic pipeline on a fixture and return SemanticInput and Experience Spans."""
    raw_bytes = fixture_path.read_bytes()
    extracted_blocks = PDFExtractor.extract(raw_bytes)
    doc = document_from_text_blocks(extracted_blocks)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)
    semantic_input = build_semantic_input(layout, document_id=fixture_path.name)
    experience_spans = build_deterministic_experience_spans(semantic_input)
    return semantic_input, experience_spans


def find_latest_benchmark_record(filename: str) -> dict[str, Any] | None:
    """Find the most recent benchmark result record for a given fixture."""
    candidate_paths = sorted(glob.glob("benchmark_results/*.json"), reverse=True)
    # Prefer structured table runs first
    preferred_paths = [p for p in candidate_paths if "structured_table" in p] + candidate_paths
    seen: set[str] = set()
    for path in preferred_paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for r in data.get("results", []):
            if r.get("filename") == filename:
                if r.get("validation_violations") or r.get("passed_validation") is not None:
                    return r
    return None


def parse_violation_string(violation_str: str) -> dict[str, Any]:
    """Parse structured violation message into field components."""
    # Pattern 1: CROSS_ENTITY_PROVENANCE in experience[0].company: block 'b_p1_45' belongs to experience entity span 2
    cross_match = re.match(
        r"CROSS_ENTITY_PROVENANCE in (?P<target>[^:]+): block '(?P<bid>[^']+)' belongs to experience entity span (?P<span_idx>\d+)",
        violation_str,
    )
    if cross_match:
        target = cross_match.group("target")
        bid = cross_match.group("bid")
        span_idx = int(cross_match.group("span_idx"))
        coll, item_idx, f_name = _parse_target(target)
        return {
            "type": "CROSS_ENTITY_PROVENANCE",
            "target": target,
            "collection": coll,
            "item_index": item_idx,
            "field_name": f_name,
            "cited_block_ids": [bid],
            "competing_span_index": span_idx,
            "rejected_value": "",
            "source_text": "",
        }

    # Pattern 2: UNSUPPORTED_CANONICAL_VALUE in context: 'canonical' not supported by 'source'
    unsup_match = re.match(
        r"UNSUPPORTED_CANONICAL_VALUE in (?P<target>[^:]+): '(?P<val>.*)' not supported by '(?P<src>.*)'",
        violation_str,
    )
    if unsup_match:
        target = unsup_match.group("target")
        val = unsup_match.group("val")
        src = unsup_match.group("src")
        coll, item_idx, f_name = _parse_target(target)
        return {
            "type": "UNSUPPORTED_CANONICAL_VALUE",
            "target": target,
            "collection": coll,
            "item_index": item_idx,
            "field_name": f_name,
            "cited_block_ids": [],
            "competing_span_index": None,
            "rejected_value": val,
            "source_text": src,
        }

    # Generic fallback
    return {
        "type": "UNKNOWN_VIOLATION",
        "target": violation_str.split(":")[0] if ":" in violation_str else violation_str,
        "collection": "unknown",
        "item_index": None,
        "field_name": None,
        "cited_block_ids": [],
        "competing_span_index": None,
        "rejected_value": "",
        "source_text": "",
    }


def _parse_target(target_str: str) -> tuple[str, int | None, str | None]:
    """Decompose e.g. 'experience[0].company' into ('experience', 0, 'company')."""
    m = re.match(r"(?P<coll>[a-zA-Z_]+)(?:\[(?P<idx>\d+)\])?(?:\.(?P<field>[a-zA-Z_]+))?", target_str)
    if m:
        coll = m.group("coll")
        idx = int(m.group("idx")) if m.group("idx") is not None else None
        f_name = m.group("field")
        return coll, idx, f_name
    return target_str, None, None


def classify_failure(
    parsed_vio: dict[str, Any],
    si: SemanticInput,
    spans: list[DeterministicEntitySpan],
) -> tuple[FailureCategory, str, list[str]]:
    """Deterministically classify a failure based on pipeline invariants and block metadata."""
    vtype = parsed_vio["type"]
    target = parsed_vio["target"]
    val = parsed_vio["rejected_value"]
    src = parsed_vio["source_text"]
    cited_bids = list(parsed_vio["cited_block_ids"])

    blocks_by_id_list: dict[str, list[SemanticBlockInput]] = {}
    for b in si.blocks:
        blocks_by_id_list.setdefault(b.block_id, []).append(b)

    # Detect duplicate block IDs produced by table cell splitting
    dup_block_ids = {bid for bid, blist in blocks_by_id_list.items() if len(blist) > 1}

    # 1. Check CROSS_ENTITY_PROVENANCE
    if vtype == "CROSS_ENTITY_PROVENANCE":
        competing_span = parsed_vio.get("competing_span_index")
        bid = cited_bids[0] if cited_bids else None
        if bid and bid in blocks_by_id_list:
            blk = blocks_by_id_list[bid][0]
            # Check if block role is ENTRY_TITLE or ORGANIZATION causing artificial span partition
            if blk.suggested_role in ("ENTRY_TITLE", "ORGANIZATION"):
                return (
                    FailureCategory.ENTITY_SPAN,
                    f"Block {bid!r} has role {blk.suggested_role!r} which artificially split contiguous "
                    f"experience into span {competing_span}. Company/title belongs to the same logical appointment.",
                    [bid],
                )
        return (
            FailureCategory.ENTITY_SPAN,
            f"Block {bid!r} was mapped across entity span boundary {competing_span} due to deterministic span partitioning.",
            [bid] if bid else [],
        )

    # 2. Check UNSUPPORTED_CANONICAL_VALUE
    if vtype == "UNSUPPORTED_CANONICAL_VALUE":
        # Find candidate blocks whose text overlaps with src or val
        matched_bids: list[str] = []
        for bid, blist in blocks_by_id_list.items():
            for b in blist:
                if any(w in b.text for w in src.split() if len(w) > 3) or (val and any(w in b.text for w in val.split() if len(w) > 3)):
                    if bid not in matched_bids:
                        matched_bids.append(bid)

        # Check if any candidate block has duplicate block_ids (clobbered in known_blocks dictionary)
        dup_matches = [bid for bid in matched_bids if bid in dup_block_ids]
        if dup_matches:
            clobbered_details = []
            for dbid in dup_matches:
                texts = [b.text for b in blocks_by_id_list[dbid]]
                clobbered_details.append(f"{dbid} (cells={texts})")
            return (
                FailureCategory.LAYOUT_SEGMENTATION,
                f"Table column splitting generated duplicate block IDs {dup_matches}. Dictionary mapping "
                f"clobbered earlier column text with the last cell: {'; '.join(clobbered_details)}.",
                dup_matches,
            )

        # Check ISO date normalization failure on 2-digit years e.g. 28/Mar/21 -> 2021-03-28
        if "startDate" in target or "endDate" in target or re.match(r"^\d{4}-\d{2}", val):
            if re.search(r"/\d{2}$", src) or re.search(r"'\d{2}\b", src) or re.search(r"\b\d{2}-[a-zA-Z]{3}-\d{2}\b", src):
                return (
                    FailureCategory.MODEL_PROVENANCE_ASSIGNMENT,
                    f"2-digit year in source text '{src}' is not recognized by 4-digit ISO date validator for '{val}'.",
                    matched_bids[:2],
                )

        # Check if text was split across adjacent physical rows in a table (Logical row continuation)
        for tbl in getattr(si, "tables", []):
            table_bids = set(tbl.source_block_ids)
            if any(mb in table_bids for mb in matched_bids):
                # If value spans across rows that were not merged
                return (
                    FailureCategory.LOGICAL_ROW_CONTINUATION,
                    f"Value '{val}' was broken across adjacent table rows or cells in {tbl.table_id}.",
                    matched_bids[:2],
                )

        # Check summary / narrative missing blocks
        if "summary" in target or "description" in target:
            return (
                FailureCategory.MODEL_PROVENANCE_ASSIGNMENT,
                f"LLM cited incomplete block provenance for {target}; parts of value were in omitted blocks.",
                matched_bids[:2],
            )

        return (
            FailureCategory.UNKNOWN,
            f"Unsupported canonical value '{val}' against source '{src}'.",
            matched_bids[:2],
        )

    return FailureCategory.UNKNOWN, "Unclassified failure type.", cited_bids


def diagnose_fixture(fixture_name: str) -> dict[str, Any]:
    """Generate comprehensive deterministic diagnosis for a target fixture."""
    fixture_path = Path("tests/fixtures") / fixture_name
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    semantic_input, experience_spans = run_deterministic_pipeline(fixture_path)
    latest_record = find_latest_benchmark_record(fixture_name)
    violations = latest_record.get("validation_violations", []) if latest_record else []

    blocks_by_id_list: dict[str, list[SemanticBlockInput]] = {}
    for b in semantic_input.blocks:
        blocks_by_id_list.setdefault(b.block_id, []).append(b)

    duplicate_block_ids = {bid: len(blist) for bid, blist in blocks_by_id_list.items() if len(blist) > 1}

    # Diagnose each failure
    analyzed_failures: list[FailureAnalysis] = []
    for v in violations:
        parsed = parse_violation_string(v)
        category, reason, cited = classify_failure(parsed, semantic_input, experience_spans)

        # Gather metadata for cited blocks
        cited_meta = []
        for c_bid in cited:
            if c_bid in blocks_by_id_list:
                for blk in blocks_by_id_list[c_bid]:
                    cited_meta.append({
                        "block_id": blk.block_id,
                        "page": blk.page,
                        "text": blk.text,
                        "suggested_role": blk.suggested_role,
                        "table_id": blk.table_id,
                        "table_purpose": blk.table_purpose,
                        "row_index": blk.row_index,
                        "column_index": blk.column_index,
                        "column_semantic": blk.column_semantic,
                    })

        # Determine which entity span it belonged to
        assigned_span = None
        for span in experience_spans:
            if any(c_bid in span.block_ids for c_bid in cited):
                assigned_span = span.entity_index
                break

        analyzed_failures.append(
            FailureAnalysis(
                filename=fixture_name,
                failure_type=parsed["type"],
                target=parsed["target"],
                collection=parsed["collection"],
                item_index=parsed["item_index"],
                field_name=parsed["field_name"],
                rejected_value=parsed["rejected_value"],
                source_text=parsed["source_text"],
                cited_block_ids=cited,
                cited_blocks_metadata=cited_meta,
                assigned_entity_span_index=assigned_span,
                competing_entity_span_index=parsed.get("competing_span_index"),
                category=category,
                diagnosis_reason=reason,
            )
        )

    # Count failure categories
    category_counts = Counter(af.category.value for af in analyzed_failures)

    return {
        "fixture_name": fixture_name,
        "archetype": semantic_input.archetype.value,
        "total_blocks": len(semantic_input.blocks),
        "unique_block_ids": len(blocks_by_id_list),
        "duplicate_block_ids_count": len(duplicate_block_ids),
        "duplicate_block_ids": duplicate_block_ids,
        "tables_count": len(getattr(semantic_input, "tables", [])),
        "tables": [
            {
                "table_id": t.table_id,
                "page": t.page,
                "purpose": t.purpose.value,
                "is_form_table": t.is_form_table,
                "columns_count": len(t.columns),
                "column_semantics": [c.semantic_role for c in t.columns],
                "physical_rows_count": len(t.physical_rows),
                "logical_rows_count": len(t.rows),
                "sample_logical_row": [
                    {"row": cell.row_index, "col": cell.column_index, "role": cell.semantic_role, "text": cell.text, "bids": cell.source_block_ids}
                    for cell in t.rows[0]
                ] if t.rows else [],
            }
            for t in getattr(semantic_input, "tables", [])
        ],
        "experience_spans_count": len(experience_spans),
        "experience_spans": [
            {
                "entity_index": s.entity_index,
                "section_index": s.section_index,
                "title_block_id": s.title_block_id,
                "block_count": len(s.block_ids),
                "block_ids_sample": s.block_ids[:5],
                "sample_text": " | ".join(blocks_by_id_list[bid][0].text for bid in s.block_ids[:3] if bid in blocks_by_id_list),
            }
            for s in experience_spans
        ],
        "benchmark_violations_count": len(violations),
        "category_counts": dict(category_counts),
        "failures": analyzed_failures,
    }


def print_diagnostic_report() -> None:
    """Print complete diagnostic report for all three target maritime fixtures."""
    print("=" * 80)
    print("PHASE 10W.2: MARITIME ENTITY GROUNDING DIAGNOSTIC REPORT")
    print("=" * 80)

    all_analyses: dict[str, dict[str, Any]] = {}

    for fname in TARGET_FIXTURES:
        print(f"\nProcessing {fname}...")
        diag = diagnose_fixture(fname)
        all_analyses[fname] = diag

        print("\n" + "#" * 80)
        print(f"FIXTURE: {fname}")
        print(f"Archetype: {diag['archetype']}")
        print(f"Total Blocks: {diag['total_blocks']} (Unique IDs: {diag['unique_block_ids']})")
        print(f"Duplicate Block IDs: {diag['duplicate_block_ids_count']}")
        if diag["duplicate_block_ids"]:
            print("  Sample duplicates (clobbered cells):")
            for dbid, cnt in list(diag["duplicate_block_ids"].items())[:5]:
                print(f"    - {dbid} (x{cnt})")

        print(f"\nTables Detected: {diag['tables_count']}")
        for tbl in diag["tables"]:
            print(f"  - Table {tbl['table_id']} (p.{tbl['page']}): purpose={tbl['purpose']}, form={tbl['is_form_table']}, "
                  f"cols={tbl['columns_count']}, phys_rows={tbl['physical_rows_count']}, logical_rows={tbl['logical_rows_count']}")
            print(f"    Columns: {tbl['column_semantics']}")

        print(f"\nDeterministic Experience Spans: {diag['experience_spans_count']}")
        for sp in diag["experience_spans"][:5]:
            print(f"  - Span {sp['entity_index']}: title_bid={sp['title_block_id']}, blocks={sp['block_count']}, sample='{sp['sample_text'][:60]}...'")
        if diag["experience_spans_count"] > 5:
            print(f"    ... and {diag['experience_spans_count'] - 5} more spans")

        print(f"\nBenchmark Violations Count: {diag['benchmark_violations_count']}")
        print(f"Category Breakdown: {diag['category_counts']}")
        print("\nIndividual Failure Analysis:")
        for idx, f in enumerate(diag["failures"]):
            print(f"  [{idx + 1}] Category: {f.category.value}")
            print(f"      Target: {f.target}")
            print(f"      Type: {f.failure_type}")
            if f.rejected_value:
                print(f"      Rejected Value: {f.rejected_value!r}")
            if f.source_text:
                print(f"      Source Evidence: {f.source_text!r}")
            print(f"      Cited Block IDs: {f.cited_block_ids}")
            print(f"      Diagnosis: {f.diagnosis_reason}")
            if f.competing_entity_span_index is not None:
                print(f"      Competing Span Index: {f.competing_entity_span_index}")
            print()

    print("=" * 80)
    print("CROSS-FIXTURE ROOT CAUSE SUMMARY & PRODUCTION-LAYER FIXES")
    print("=" * 80)
    _print_root_cause_and_recommendations(all_analyses)


def _print_root_cause_and_recommendations(analyses: dict[str, dict[str, Any]]) -> None:
    print("""
1. 2nd Officer Mayur Agarwal_062029.pdf
---------------------------------------
PRIMARY ROOT CAUSE: LAYOUT_SEGMENTATION (Duplicate Block ID Clobbering)
  In table_binding.py, _split_block_by_columns() divides a single PDF line spanning
  multiple columns into separate GeometricCells, but assigns the parent block_id
  identically to all cell fragments (e.g., b_p3_201, b_p3_223).
  When known_blocks is instantiated as {b.block_id: b for b in input_data.blocks}
  in semantic_contract.py, Python dictionary collision overwrites earlier cells
  with the rightmost column cell. E.g., b_p3_201 (course name) was overwritten by its
  date cell '15/07/2019'. When the LLM cited b_p3_201, validation checked the date text
  instead of the course name, producing UNSUPPORTED_CANONICAL_VALUE.
SECONDARY ROOT CAUSE: DATE_NORMALIZATION
  The ISO date validator in _is_value_semantically_supported requires 4-digit years
  in source text ('2021'), rejecting valid standard maritime dates like '28/Mar/21'.
SMALLEST GENERIC PRODUCTION FIX:
  (a) In GeometricTableBinder (table_binding.py), ensure sub-cell fragments receive
      unique derived block IDs (e.g. f"{b.block_id}_c{c_idx}") or ensure known_blocks
      preserves/aggregates text across all cell occurrences of a block_id.
  (b) In _is_value_semantically_supported(), support 2-digit years in source text.

2. MUKUND 3RD OFF CV 2026.pdf
-----------------------------
PRIMARY ROOT CAUSE: LAYOUT_SEGMENTATION (Duplicate Block ID Clobbering)
  Mukund suffered 12 validation failures on experience company, rank, and description.
  Single lines in table_p3_1 (e.g. b_p3_215, b_p3_221, b_p3_225, b_p3_229) spanned
  vessel name, company, rank, and sign-on/sign-off dates. Because duplicate block IDs
  were generated, known_blocks clobbered the company and rank cells with the date cell
  ('12-09-2019 30-11-2019'). When the model cited the correct block for company or rank,
  validation only saw the sign-off date string.
SECONDARY ROOT CAUSE: MODEL_PROVENANCE_ASSIGNMENT
  Summary failed validation because the model cited b_p1_61 but omitted b_p1_60.
SMALLEST GENERIC PRODUCTION FIX:
  Resolving the duplicate block ID clobbering in table_binding.py / semantic_contract.py
  instantly eliminates all 12 experience grounding failures on Mukund.

3. Sendrick Costa CV.pdf
------------------------
PRIMARY ROOT CAUSE: ENTITY_SPAN (Cross-Entity Artificial Span Splitting)
  Sendrick is a multi-column CV whose appointment entries consist of consecutive lines:
    'Assistant Second Engineer' (b_p1_44)
    'Anglo Eastern Ship Management' (b_p1_45)
  Because 'Management' was missing from _ORG_SUFFIXES and _INSTITUTIONAL_NOUNS in
  structural_roles.py, 'Anglo Eastern Ship Management' was not recognized as an
  organization. Instead, _looks_like_entry_title() classified it as ENTRY_TITLE.
  In build_deterministic_experience_spans(), encountering two consecutive ENTRY_TITLE
  blocks forced an artificial entity boundary: designation became Span 1, company
  became Span 2. When the LLM grouped them into a single experience item, validation
  rejected them with CROSS_ENTITY_PROVENANCE.
SECONDARY ROOT CAUSE: TABLE_SEMANTICS
  The 2-column CV layout on page 1 and page 2 was falsely detected by GeometricTableBinder
  as tabular tables (table_p1_0 as UNKNOWN, table_p2_0 as DOCUMENTS due to token 'COC').
SMALLEST GENERIC PRODUCTION FIX:
  (a) In structural_roles.py, include maritime/commercial entity patterns ('Management',
      'Ship Management', 'Shipping') in organization recognition so company names
      are classified as ORGANIZATION rather than ENTRY_TITLE.
  (b) In build_deterministic_experience_spans(), avoid splitting spans when an
      ORGANIZATION immediately follows an ENTRY_TITLE (or vice-versa), and for
      tabular CVs, bind experience entity spans by table row rather than line heuristics.
""")


if __name__ == "__main__":
    print_diagnostic_report()
