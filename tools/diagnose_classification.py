#!/usr/bin/env python3
"""Diagnostic script: classify blocks from a PDF fixture using the block classifier.

This script tries to use the existing PDFExtractor to extract TextBlocks.
If PyMuPDF is not installed, it will instruct the user how to run it locally.
"""

import sys
from pathlib import Path

try:
    from app.pipeline.stages.text_extraction import PDFExtractor
    from app.pipeline.stages.block_classification import classify_blocks
except Exception as e:
    print("Diagnostic cannot run: missing runtime dependency (PyMuPDF?) or import error.")
    print("Error:", e)
    print("To run locally, install dependencies:")
    print("  python -m pip install -e .[dev]")
    sys.exit(1)

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "swe_experienced_resume.pdf"

if not FIXTURE.exists():
    print("Fixture not found:", FIXTURE)
    sys.exit(1)

pdf_bytes = FIXTURE.read_bytes()
blocks = PDFExtractor.extract(pdf_bytes)
classified = classify_blocks(blocks)

for cb in classified:
    text = getattr(cb.original, 'text', '')
    print('---')
    print('text:', text)
    print('label:', cb.label)
    print('score:', cb.score)
    print('reasons:', cb.reasons)
