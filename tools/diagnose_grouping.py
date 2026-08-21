#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from app.pipeline.stages.text_extraction import PDFExtractor
from app.pipeline.stages.reading_order import ReadingOrder
from app.pipeline.stages.normalization import TextNormalizer
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.block_classification import classify_blocks
from app.pipeline.stages.candidate_grouping import group_candidates


FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "swe_experienced_resume.pdf"


def format_block(cb):
    b = cb.original
    text = (b.text or "").replace("\n", " ")
    return f"{cb.label} | {cb.score:.2f} | {b.y0}-{b.y1} | {b.x0}-{b.x1} | {text}"


def main():
    if not FIXTURE.exists():
        print("Fixture not found:", FIXTURE)
        sys.exit(2)

    raw = FIXTURE.read_bytes()

    try:
        blocks = PDFExtractor.extract(raw)
    except Exception as e:
        print("PDF extraction failed:", e)
        sys.exit(3)

    # reading order
    ordered = ReadingOrder.reorder(blocks)

    # normalization
    normalized = TextNormalizer.normalize_blocks(ordered)

    # section detection
    detector = SectionDetector()
    sections = detector.detect(normalized)

    # classification
    classified_sections = {}
    for sec, blks in sections.items():
        classified_sections[sec] = classify_blocks(blks)

    # grouping and printing
    interesting = ["EXPERIENCE", "PROJECTS", "EDUCATION"]

    for sec in interesting:
        cb_list = classified_sections.get(sec, [])
        groups = group_candidates(cb_list, section=sec)
        print(f"\n=== SECTION: {sec} — {len(groups)} groups ===\n")
        for i, g in enumerate(groups, start=1):
            print("--- GROUP {} ---".format(i))
            print(f"section: {g.section}")
            print(f"page: {g.page_number}")
            print(f"column: {g.column_id}")
            print(f"start_index: {g.start_index}")
            print(f"end_index: {g.end_index}")
            print(f"summary: {g.summary_text}\n")
            for cb in g.blocks:
                print(format_block(cb))
            print()


if __name__ == "__main__":
    main()
