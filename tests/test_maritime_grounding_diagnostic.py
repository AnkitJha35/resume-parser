"""Phase 10W.2 Tests: Deterministic Maritime Entity Grounding Failure Diagnostic."""

from __future__ import annotations

from pathlib import Path
import pytest

from scripts.diagnose_maritime_grounding import (
    TARGET_FIXTURES,
    FailureCategory,
    diagnose_fixture,
    parse_violation_string,
    run_deterministic_pipeline,
)


@pytest.mark.parametrize("fixture_name", TARGET_FIXTURES)
def test_deterministic_pipeline_runs_on_maritime_fixtures(fixture_name: str):
    """Verify that deterministic pipeline executes cleanly across all 3 maritime fixtures."""
    p = Path("tests/fixtures") / fixture_name
    if not p.exists():
        pytest.skip(f"Fixture {p} not found")

    semantic_input, experience_spans = run_deterministic_pipeline(p)
    assert semantic_input.document_id == fixture_name
    assert len(semantic_input.blocks) > 50
    assert isinstance(experience_spans, list)


def test_diagnose_fixture_mayur_duplicate_blocks_detected():
    """Verify that Mayur Agarwal has zero duplicate block IDs after table cell splitting repair."""
    diag = diagnose_fixture("2nd Officer Mayur Agarwal_062029.pdf")
    assert diag["duplicate_block_ids_count"] == 0


def test_diagnose_fixture_mukund_clobbered_blocks_diagnosed():
    """Verify that Mukund Kumar has zero duplicate block IDs after table cell splitting repair."""
    diag = diagnose_fixture("MUKUND 3RD OFF CV 2026.pdf")
    assert diag["duplicate_block_ids_count"] == 0


def test_diagnose_fixture_sendrick_entity_spans_diagnosed():
    """Verify that Sendrick Costa detects artificial span partition causing cross-entity failures."""
    diag = diagnose_fixture("Sendrick Costa CV.pdf")
    assert diag["duplicate_block_ids_count"] == 0
    assert FailureCategory.ENTITY_SPAN.value in diag["category_counts"]
    # All 4 violations should be classified as ENTITY_SPAN
    assert diag["category_counts"][FailureCategory.ENTITY_SPAN.value] == 4


def test_parse_violation_string_patterns():
    """Verify parsing of CROSS_ENTITY_PROVENANCE and UNSUPPORTED_CANONICAL_VALUE strings."""
    v1 = "CROSS_ENTITY_PROVENANCE in experience[0].company: block 'b_p1_45' belongs to experience entity span 2"
    p1 = parse_violation_string(v1)
    assert p1["type"] == "CROSS_ENTITY_PROVENANCE"
    assert p1["collection"] == "experience"
    assert p1["item_index"] == 0
    assert p1["field_name"] == "company"
    assert p1["cited_block_ids"] == ["b_p1_45"]
    assert p1["competing_span_index"] == 2

    v2 = "UNSUPPORTED_CANONICAL_VALUE in certifications[1]: 'Basic Training' not supported by '15/07/2019'"
    p2 = parse_violation_string(v2)
    assert p2["type"] == "UNSUPPORTED_CANONICAL_VALUE"
    assert p2["collection"] == "certifications"
    assert p2["item_index"] == 1
    assert p2["rejected_value"] == "Basic Training"
    assert p2["source_text"] == "15/07/2019"
