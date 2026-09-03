"""Phase 8-4: Deterministic unit tests for Candidate B compact semantic serialization."""

from __future__ import annotations

import copy
import json
import pytest

from app.domain.semantic_contract import SemanticBlockInput, SemanticInput
from app.extractors.semantic_prompt import (
    build_compact_extraction_prompt,
    build_extraction_prompt,
    serialize_compact_semantic_input,
    serialize_semantic_input_full,
)


def _make_block(
    block_id: str = "b_1",
    text: str = "Sample text",
    page: int = 1,
    reading_order: int = 1,
    is_bold: bool | None = None,
    suggested_role: str | None = None,
    table_id: str | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
    cell_role: str | None = None,
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        text=text,
        page=page,
        bbox=[0.0, 0.0, 100.0, 20.0],
        region_id="r1",
        region_kind="body",
        reading_order=reading_order,
        column_id=None,
        is_bold=is_bold,
        font_size=12.0,
        suggested_role=suggested_role,
        table_id=table_id,
        row_index=row_index,
        column_index=column_index,
        cell_role=cell_role,
    )


def test_compact_serialization_single_page_omits_page_field():
    """Single-page resumes (page=1) must not include redundant 'page' keys in blocks."""
    blocks = [
        _make_block("b_1", "Aditi Anand", page=1, reading_order=1, suggested_role="HEADER"),
        _make_block("b_2", "Email: aditi@example.com", page=1, reading_order=2, suggested_role="CONTACT"),
    ]
    sem_in = SemanticInput(document_id="doc_single", page_count=1, pages=[], blocks=blocks)

    raw_json = serialize_compact_semantic_input(sem_in)
    parsed = json.loads(raw_json)

    assert parsed["doc_id"] == "doc_single"
    assert len(parsed["blocks"]) == 2
    for b in parsed["blocks"]:
        assert "page" not in b
        assert "bbox" not in b
        assert "region_id" not in b
        assert "font_size" not in b


def test_compact_serialization_multipage_includes_page_only_when_greater_than_one():
    """Multi-page blocks must include 'page' only when page > 1."""
    blocks = [
        _make_block("b_p1", "Page 1 Header", page=1, reading_order=1),
        _make_block("b_p2", "Page 2 Header", page=2, reading_order=2),
        _make_block("b_p3", "Page 3 Header", page=3, reading_order=3),
    ]
    sem_in = SemanticInput(document_id="doc_multi", page_count=3, pages=[], blocks=blocks)

    raw_json = serialize_compact_semantic_input(sem_in)
    parsed = json.loads(raw_json)

    assert "page" not in parsed["blocks"][0]
    assert parsed["blocks"][1]["page"] == 2
    assert parsed["blocks"][2]["page"] == 3


def test_compact_serialization_preserves_reading_order():
    """Blocks provided in unordered sequence must be deterministically sorted by page, reading_order, block_id."""
    blocks = [
        _make_block("b_p2_2", "P2 B2", page=2, reading_order=2),
        _make_block("b_p1_2", "P1 B2", page=1, reading_order=2),
        _make_block("b_p1_1", "P1 B1", page=1, reading_order=1),
        _make_block("b_p2_1", "P2 B1", page=2, reading_order=1),
    ]
    sem_in = SemanticInput(document_id="doc_sort", page_count=2, pages=[], blocks=blocks)

    raw_json = serialize_compact_semantic_input(sem_in)
    parsed = json.loads(raw_json)

    expected_ids = ["b_p1_1", "b_p1_2", "b_p2_1", "b_p2_2"]
    actual_ids = [b["id"] for b in parsed["blocks"]]
    assert actual_ids == expected_ids


def test_compact_serialization_omits_null_fields():
    """Candidate B compact representation must not contain any null or None fields in JSON."""
    block = _make_block("b_1", "Simple text", is_bold=None, suggested_role=None, table_id=None)
    sem_in = SemanticInput(document_id="doc_test_1", page_count=1, pages=[], blocks=[block])

    raw_json = serialize_compact_semantic_input(sem_in)
    assert ":null" not in raw_json
    assert ": null" not in raw_json
    parsed = json.loads(raw_json)
    assert parsed["blocks"][0] == {"id": "b_1", "text": "Simple text"}


def test_compact_serialization_omits_unknown_role():
    """Suggested role 'UNKNOWN' must be omitted, but non-UNKNOWN roles must be preserved."""
    blocks = [
        _make_block("b_1", "Name", suggested_role="HEADER"),
        _make_block("b_2", "Random sentence", suggested_role="UNKNOWN"),
        _make_block("b_3", "None role", suggested_role=None),
        _make_block("b_4", "Experience header", suggested_role="SECTION_HEADING"),
    ]
    sem_in = SemanticInput(document_id="doc_roles", page_count=1, pages=[], blocks=blocks)

    raw_json = serialize_compact_semantic_input(sem_in)
    parsed = json.loads(raw_json)

    assert parsed["blocks"][0]["role"] == "HEADER"
    assert "role" not in parsed["blocks"][1]
    assert "role" not in parsed["blocks"][2]
    assert parsed["blocks"][3]["role"] == "SECTION_HEADING"


def test_compact_serialization_preserves_bold_true_only():
    """Only is_bold=True is serialized as bold: true; False or None are omitted."""
    blocks = [
        _make_block("b_1", "Bold Title", is_bold=True),
        _make_block("b_2", "Regular text", is_bold=False),
        _make_block("b_3", "Unspecified text", is_bold=None),
    ]
    sem_in = SemanticInput(document_id="doc_bold", page_count=1, pages=[], blocks=blocks)

    raw_json = serialize_compact_semantic_input(sem_in)
    parsed = json.loads(raw_json)

    assert parsed["blocks"][0]["bold"] is True
    assert "bold" not in parsed["blocks"][1]
    assert "bold" not in parsed["blocks"][2]


def test_compact_serialization_preserves_table_metadata():
    """Table metadata (table_id, row_index, column_index, non-DATA cell_role) must be preserved."""
    blocks = [
        _make_block(
            "b_tbl_header",
            "Company Name",
            reading_order=1,
            table_id="tbl_exp",
            row_index=0,
            column_index=0,
            cell_role="HEADER",
        ),
        _make_block(
            "b_tbl_data",
            "Google LLC",
            reading_order=2,
            table_id="tbl_exp",
            row_index=1,
            column_index=0,
            cell_role="DATA",
        ),
        _make_block(
            "b_tbl_no_grid",
            "Table text without cell indices",
            reading_order=3,
            table_id="tbl_exp",
            row_index=None,
            column_index=None,
            cell_role=None,
        ),
    ]
    sem_in = SemanticInput(document_id="doc_table", page_count=1, pages=[], blocks=blocks)

    raw_json = serialize_compact_semantic_input(sem_in)
    parsed = json.loads(raw_json)

    # Header cell
    assert parsed["blocks"][0]["table"] == "tbl_exp"
    assert parsed["blocks"][0]["row"] == 0
    assert parsed["blocks"][0]["col"] == 0
    assert parsed["blocks"][0]["cell_role"] == "HEADER"

    # Data cell (DATA is default, omitted for brevity)
    assert parsed["blocks"][1]["table"] == "tbl_exp"
    assert parsed["blocks"][1]["row"] == 1
    assert parsed["blocks"][1]["col"] == 0
    assert "cell_role" not in parsed["blocks"][1]

    # Cell without row/col indices
    assert parsed["blocks"][2]["table"] == "tbl_exp"
    assert "row" not in parsed["blocks"][2]
    assert "col" not in parsed["blocks"][2]


def test_compact_serialization_is_deterministic():
    """Repeated serialization invocations must produce bit-for-bit identical output."""
    blocks = [
        _make_block("b_2", "Line 2", reading_order=2, suggested_role="CONTACT"),
        _make_block("b_1", "Line 1", reading_order=1, suggested_role="HEADER", is_bold=True),
        _make_block("b_3", "Line 3", reading_order=3, table_id="t1", row_index=0, column_index=0),
    ]
    sem_in = SemanticInput(document_id="doc_det", page_count=1, pages=[], blocks=blocks)

    first_run = serialize_compact_semantic_input(sem_in)
    for _ in range(50):
        assert serialize_compact_semantic_input(sem_in) == first_run


def test_compact_serialization_does_not_mutate_input():
    """Serialization must be a pure read operation with zero side-effects on the SemanticInput."""
    blocks = [
        _make_block("b_2", "Line 2", reading_order=2),
        _make_block("b_1", "Line 1", reading_order=1),
    ]
    sem_in = SemanticInput(document_id="doc_pure", page_count=1, pages=[], blocks=blocks)
    original_dict = sem_in.model_dump()

    _ = serialize_compact_semantic_input(sem_in)
    _ = build_compact_extraction_prompt(sem_in)
    _ = build_extraction_prompt(sem_in)

    assert sem_in.model_dump() == original_dict
    assert [b.block_id for b in sem_in.blocks] == ["b_2", "b_1"]


def test_reference_full_serializer_coexists_with_compact():
    """Reference full serializer must remain available and unperturbed."""
    blocks = [_make_block("b_1", "Full check", reading_order=1, suggested_role="HEADER")]
    sem_in = SemanticInput(document_id="doc_full", page_count=1, pages=[], blocks=blocks)

    full_json = serialize_semantic_input_full(sem_in)
    compact_json = serialize_compact_semantic_input(sem_in)

    parsed_full = json.loads(full_json)
    parsed_compact = json.loads(compact_json)

    assert "document_id" in parsed_full
    assert "page_count" in parsed_full
    assert "bbox" in parsed_full["blocks"][0]

    assert "doc_id" in parsed_compact
    assert "bbox" not in parsed_compact["blocks"][0]
    assert len(compact_json) < len(full_json)
