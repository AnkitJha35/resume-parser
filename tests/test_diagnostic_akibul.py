"""Temporary diagnostic test for AKIBUL benchmark fixture using Gemini 3.5 Flash Lite + Candidate B serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
from dotenv import load_dotenv

from app.core.config import Settings
from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    build_semantic_input,
    validate_semantic_output,
)
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor

load_dotenv()

FIXTURE_PATH = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
OUTPUT_DIR = Path("benchmark_results")


def test_diagnostic_akibul_gemini_semantic_output(capsys):
    """Invoke Gemini with Candidate B compact serialization on AKIBUL fixture and dump SemanticOutput."""
    if not os.environ.get("RUN_LIVE_GEMINI_DIAGNOSTIC"):
        pytest.skip("RUN_LIVE_GEMINI_DIAGNOSTIC is not set; skipping live Gemini API call.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            settings = Settings()
            api_key = settings.gemini_api_key
        except Exception:
            pass

    if not api_key:
        pytest.skip("GEMINI_API_KEY is not available in environment.")

    assert FIXTURE_PATH.exists(), f"Fixture {FIXTURE_PATH} not found"

    raw_bytes = FIXTURE_PATH.read_bytes()
    physical_doc = document_from_text_blocks(PDFExtractor.extract(raw_bytes))
    reconstructed_doc = reconstruct_document(physical_doc)
    layout_doc = interpret_layout(reconstructed_doc)

    semantic_input = build_semantic_input(layout_doc, document_id=FIXTURE_PATH.name)
    assert len(semantic_input.blocks) > 0

    extractor = GeminiSemanticExtractor(
        api_key=api_key,
        model="gemini-3.5-flash-lite",
        compact=True,
    )

    output = extractor.extract(semantic_input)
    violations = validate_semantic_output(output, semantic_input)

    output_data = output.model_dump()
    usage_metadata = getattr(extractor, "last_usage_metadata", None)

    diagnostic_payload = {
        "fixture": FIXTURE_PATH.name,
        "model": "gemini-3.5-flash-lite",
        "representation": "candidate_b_compact",
        "validation_passed": len(violations) == 0,
        "violations": violations,
        "usage": usage_metadata,
        "semantic_output": output_data,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "akibul_semantic_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("AKIBUL DIAGNOSTIC SEMANTIC OUTPUT")
    print("=" * 80)
    print(f"Validation Passed: {len(violations) == 0}")
    if violations:
        print("Violations:", violations)
    if usage_metadata:
        print(f"Tokens: prompt={usage_metadata.get('prompt_tokens')}, output={usage_metadata.get('output_tokens')}, total={usage_metadata.get('total_tokens')}, latency={usage_metadata.get('latency_ms')}ms")

    print("\n--- PERSONAL ---")
    print(json.dumps(output_data.get("personal"), indent=2))

    print("\n--- SUMMARY ---")
    print(json.dumps(output_data.get("summary"), indent=2))

    print("\n--- SKILLS ---")
    print(json.dumps(output_data.get("skills"), indent=2))

    print(f"\n--- EXPERIENCE ({len(output_data.get('experience', []))} items) ---")
    print(json.dumps(output_data.get("experience"), indent=2))

    print(f"\n--- EDUCATION ({len(output_data.get('education', []))} items) ---")
    print(json.dumps(output_data.get("education"), indent=2))

    print(f"\n--- PROJECTS ({len(output_data.get('projects', []))} items) ---")
    print(json.dumps(output_data.get("projects"), indent=2))

    print(f"\n--- CERTIFICATIONS ({len(output_data.get('certifications', []))} items) ---")
    print(json.dumps(output_data.get("certifications"), indent=2))

    print(f"\n--- LANGUAGES ({len(output_data.get('languages', []))} items) ---")
    print(json.dumps(output_data.get("languages"), indent=2))

    print(f"\n--- ACHIEVEMENTS ({len(output_data.get('achievements', []))} items) ---")
    print(json.dumps(output_data.get("achievements"), indent=2))

    print(f"\nWrote full diagnostic JSON to: {out_file}")
    print("=" * 80 + "\n")
