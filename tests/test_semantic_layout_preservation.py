"""Phase 10B-2A: Tests for preserving deterministic layout structure in complex semantic extraction."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.domain.document import (
    BoundingBox,
    Document,
    Line,
    Page,
    Region,
    Span,
    TextStyle,
    document_from_text_blocks,
)
from app.domain.semantic_contract import (
    DocumentArchetype,
    SemanticBlockInput,
    SemanticInput,
    build_semantic_input,
    classify_document_archetype,
)
from app.extractors.semantic_prompt import (
    build_compact_extraction_prompt,
    serialize_compact_semantic_input,
)
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor


def _make_dummy_standard_doc() -> Document:
    """Create a minimal standard single/two column resume."""
    l1 = Line(
        line_id="l1",
        page_number=1,
        bbox=BoundingBox(50.0, 50.0, 200.0, 70.0),
        spans=[Span("s1", "John Doe", BoundingBox(50.0, 50.0, 200.0, 70.0))],
        text="John Doe",
        style=TextStyle(font_size=16.0, bold=True),
        reading_order=1,
    )
    l2 = Line(
        line_id="l2",
        page_number=1,
        bbox=BoundingBox(50.0, 80.0, 200.0, 95.0),
        spans=[Span("s2", "john.doe@example.com", BoundingBox(50.0, 80.0, 200.0, 95.0))],
        text="john.doe@example.com",
        style=TextStyle(font_size=10.0),
        reading_order=2,
    )
    r1 = Region(
        region_id="page-1-region-0",
        kind="header",
        bbox=BoundingBox(50.0, 50.0, 500.0, 100.0),
        lines=[l1, l2],
    )
    p1 = Page(page_number=1, width=612.0, height=792.0, regions=[r1])
    return Document(pages=[p1])


def _make_dummy_maritime_form_doc() -> Document:
    """Create a multi-column maritime application form document."""
    # Col 0: Surname
    l1 = Line(
        line_id="p1-c0-l1",
        page_number=1,
        bbox=BoundingBox(30.0, 100.0, 150.0, 120.0),
        spans=[Span("s1", "APPLICATION FORM FOR CADET", BoundingBox(30.0, 100.0, 150.0, 120.0))],
        text="APPLICATION FORM FOR CADET",
        style=TextStyle(bold=True),
        reading_order=1,
    )
    l2 = Line(
        line_id="p1-c0-l2",
        page_number=1,
        bbox=BoundingBox(30.0, 130.0, 150.0, 150.0),
        spans=[Span("s2", "Surname", BoundingBox(30.0, 130.0, 150.0, 150.0))],
        text="Surname",
        style=TextStyle(),
        reading_order=2,
    )
    l3 = Line(
        line_id="p1-c0-l3",
        page_number=1,
        bbox=BoundingBox(30.0, 155.0, 150.0, 175.0),
        spans=[Span("s3", "Alam", BoundingBox(30.0, 155.0, 150.0, 175.0))],
        text="Alam",
        style=TextStyle(),
        reading_order=3,
    )
    r0 = Region(
        region_id="page-1-region-0",
        kind="column",
        bbox=BoundingBox(30.0, 100.0, 150.0, 700.0),
        column_id=0,
        lines=[l1, l2, l3],
    )

    # Col 1: First Name & Maritime details
    l4 = Line(
        line_id="p1-c1-l1",
        page_number=1,
        bbox=BoundingBox(200.0, 130.0, 350.0, 150.0),
        spans=[Span("s4", "First Name", BoundingBox(200.0, 130.0, 350.0, 150.0))],
        text="First Name",
        style=TextStyle(),
        reading_order=4,
    )
    l5 = Line(
        line_id="p1-c1-l2",
        page_number=1,
        bbox=BoundingBox(200.0, 155.0, 350.0, 175.0),
        spans=[Span("s5", "Akibul", BoundingBox(200.0, 155.0, 350.0, 175.0))],
        text="Akibul",
        style=TextStyle(),
        reading_order=5,
    )
    l6 = Line(
        line_id="p1-c1-l3",
        page_number=1,
        bbox=BoundingBox(200.0, 180.0, 350.0, 200.0),
        spans=[Span("s6", "CDC No: C/O/12345", BoundingBox(200.0, 180.0, 350.0, 200.0))],
        text="CDC No: C/O/12345",
        style=TextStyle(),
        reading_order=6,
    )
    r1 = Region(
        region_id="page-1-region-1",
        kind="column",
        bbox=BoundingBox(200.0, 100.0, 350.0, 700.0),
        column_id=1,
        lines=[l4, l5, l6],
    )

    # Col 2: Vessel & Rank details
    l7 = Line(
        line_id="p1-c2-l1",
        page_number=1,
        bbox=BoundingBox(400.0, 130.0, 550.0, 150.0),
        spans=[Span("s7", "Vessel Name", BoundingBox(400.0, 130.0, 550.0, 150.0))],
        text="Vessel Name",
        style=TextStyle(),
        reading_order=7,
    )
    l8 = Line(
        line_id="p1-c2-l2",
        page_number=1,
        bbox=BoundingBox(400.0, 155.0, 550.0, 175.0),
        spans=[Span("s8", "MT Ocean Star", BoundingBox(400.0, 155.0, 550.0, 175.0))],
        text="MT Ocean Star",
        style=TextStyle(),
        reading_order=8,
    )
    r2 = Region(
        region_id="page-1-region-2",
        kind="column",
        bbox=BoundingBox(400.0, 100.0, 550.0, 700.0),
        column_id=2,
        lines=[l7, l8],
    )

    p1 = Page(page_number=1, width=612.0, height=792.0, regions=[r0, r1, r2])
    return Document(pages=[p1])


def test_standard_cv_candidate_b_output_unchanged():
    """Requirement 3 & 9: STANDARD_CV Candidate B output remains unchanged."""
    doc = _make_dummy_standard_doc()
    sem_input = build_semantic_input(doc, document_id="std-doc-1")

    assert sem_input.archetype == DocumentArchetype.STANDARD_CV

    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)

    assert payload["doc_id"] == "std-doc-1"
    assert "archetype" not in payload
    assert len(payload["blocks"]) == 2
    b0 = payload["blocks"][0]
    assert b0["id"] == "b_p1_0"
    assert b0["text"] == "John Doe"
    assert b0.get("bold") is True
    assert "region" not in b0
    assert "col" not in b0
    assert "kind" not in b0


def test_complex_archetype_retains_region_column_context():
    """Requirement 4 & 9: Complex archetypes retain deterministic region/column context."""
    doc = _make_dummy_maritime_form_doc()
    sem_input = build_semantic_input(doc, document_id="maritime-form-1")

    assert sem_input.archetype in (
        DocumentArchetype.STRUCTURED_FORM,
        DocumentArchetype.MARITIME_TABULAR,
        DocumentArchetype.MARITIME_CV,
    )

    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)

    assert payload["doc_id"] == "maritime-form-1"
    assert payload["archetype"] == sem_input.archetype.value
    assert len(payload["blocks"]) == 8

    # Verify column 0 blocks
    col0_blocks = [b for b in payload["blocks"] if b.get("col") == 0]
    assert len(col0_blocks) == 3
    assert [b["text"] for b in col0_blocks] == ["APPLICATION FORM FOR CADET", "Surname", "Alam"]
    for b in col0_blocks:
        assert b["region"] == "page-1-region-0"
        assert b["kind"] == "column"

    # Verify column 1 blocks
    col1_blocks = [b for b in payload["blocks"] if b.get("col") == 1]
    assert len(col1_blocks) == 3
    assert [b["text"] for b in col1_blocks] == ["First Name", "Akibul", "CDC No: C/O/12345"]
    for b in col1_blocks:
        assert b["region"] == "page-1-region-1"
        assert b["kind"] == "column"

    # Verify column 2 blocks
    col2_blocks = [b for b in payload["blocks"] if b.get("col") == 2]
    assert len(col2_blocks) == 2
    assert [b["text"] for b in col2_blocks] == ["Vessel Name", "MT Ocean Star"]
    for b in col2_blocks:
        assert b["region"] == "page-1-region-2"
        assert b["kind"] == "column"


def test_existing_block_ids_preserved():
    """Requirement 9: Every block retains its exact deterministic block_id."""
    doc = _make_dummy_maritime_form_doc()
    sem_input = build_semantic_input(doc, document_id="test-doc")
    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)

    input_block_ids = [b.block_id for b in sem_input.blocks]
    serialized_block_ids = [b["id"] for b in payload["blocks"]]

    assert set(serialized_block_ids) == set(input_block_ids)
    assert len(serialized_block_ids) == len(input_block_ids)


def test_no_fabricated_table_metadata():
    """Standard CV documents must have no fabricated table rows, columns, or cell roles."""
    doc = _make_dummy_standard_doc()
    sem_input = build_semantic_input(doc, document_id="test-doc")
    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)

    for b in sem_input.blocks:
        assert b.table_id is None
        assert b.row_index is None
        assert b.column_index is None
        assert b.cell_role is None

    for b in payload["blocks"]:
        assert "table" not in b
        assert "row" not in b
        assert "cell_role" not in b


def test_akibul_fixture_preprocessing_and_representation():
    """Requirement 9: AKIBUL preprocessing produces richer structural representation with preserved regions and tables."""
    fixture_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    raw_bytes = fixture_path.read_bytes()
    extracted = PDFExtractor.extract(raw_bytes)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)

    sem_input = build_semantic_input(layout, document_id=fixture_path.name)

    assert sem_input.archetype == DocumentArchetype.STRUCTURED_FORM
    assert sem_input.page_count == 4
    # 322 blocks in legacy merged IR; 328 blocks with column-aware separation; 329 with border-defined tables
    assert len(sem_input.blocks) in (322, 328, 329)

    # Verify Candidate B serialization
    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)

    assert payload["doc_id"] == fixture_path.name
    assert payload["archetype"] == "structured_form"
    assert len(payload["blocks"]) in (322, 328, 329)

    # Ensure blocks on page 1 contain region and column information
    p1_blocks = [b for b in payload["blocks"] if b.get("page") == 1 or "page" not in b]
    has_region = any("region" in b for b in p1_blocks)
    has_col = any("col" in b for b in p1_blocks)
    assert has_region
    assert has_col

    # Verify deterministic table metadata is populated on tabular pages
    p4_table_blocks = [b for b in sem_input.blocks if b.page == 4 and b.table_id is not None]
    assert len(p4_table_blocks) >= 50


def test_deterministic_serialization_order():
    """Requirement 8: Serialization is strictly deterministic and stable across multiple calls."""
    doc = _make_dummy_maritime_form_doc()
    sem_input = build_semantic_input(doc, document_id="stable-doc")

    str1 = serialize_compact_semantic_input(sem_input)
    str2 = serialize_compact_semantic_input(sem_input)

    assert str1 == str2


def test_table_serialization_is_row_major_not_region_major():
    """Requirement 11 & 12: Table blocks spanning multiple regions serialize row-by-row across columns."""
    blocks = [
        # Region 0: Col 0 (Row 0, Row 1)
        SemanticBlockInput(
            block_id="b_r0_c0_row0",
            text="Header Col 0",
            page=1,
            bbox=[10.0, 10.0, 50.0, 20.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=1,
            column_id=0,
            table_id="tbl_1",
            row_index=0,
            column_index=0,
            cell_role="HEADER",
        ),
        SemanticBlockInput(
            block_id="b_r0_c0_row1",
            text="Data Row 1 Col 0",
            page=1,
            bbox=[10.0, 30.0, 50.0, 40.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=2,
            column_id=0,
            table_id="tbl_1",
            row_index=1,
            column_index=0,
        ),
        # Region 1: Col 1 (Row 0, Row 1)
        SemanticBlockInput(
            block_id="b_r1_c1_row0",
            text="Header Col 1",
            page=1,
            bbox=[60.0, 10.0, 100.0, 20.0],
            region_id="reg-1",
            region_kind="column",
            reading_order=3,
            column_id=1,
            table_id="tbl_1",
            row_index=0,
            column_index=1,
            cell_role="HEADER",
        ),
        SemanticBlockInput(
            block_id="b_r1_c1_row1",
            text="Data Row 1 Col 1",
            page=1,
            bbox=[60.0, 30.0, 100.0, 40.0],
            region_id="reg-1",
            region_kind="column",
            reading_order=4,
            column_id=1,
            table_id="tbl_1",
            row_index=1,
            column_index=1,
        ),
    ]
    sem_input = SemanticInput(
        document_id="multi-reg-table",
        page_count=1,
        archetype=DocumentArchetype.STRUCTURED_FORM,
        pages=[],
        blocks=blocks,
    )

    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)
    serialized_ids = [b["id"] for b in payload["blocks"]]

    # Row-major: [Row 0 Col 0, Row 0 Col 1, Row 1 Col 0, Row 1 Col 1]
    assert serialized_ids == [
        "b_r0_c0_row0",
        "b_r1_c1_row0",
        "b_r0_c0_row1",
        "b_r1_c1_row1",
    ]


def test_table_serialization_multiple_blocks_in_same_cell_contiguous():
    """Requirement 11: Multiple blocks in the same physical cell remain together and preserve reading order."""
    blocks = [
        SemanticBlockInput(
            block_id="b_cell_1a",
            text="Line 1 of Cell",
            page=1,
            bbox=[10.0, 30.0, 50.0, 40.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=2,
            table_id="tbl_1",
            row_index=1,
            column_index=0,
        ),
        SemanticBlockInput(
            block_id="b_cell_1b",
            text="Line 2 of Cell",
            page=1,
            bbox=[10.0, 42.0, 50.0, 52.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=3,
            table_id="tbl_1",
            row_index=1,
            column_index=0,
        ),
        SemanticBlockInput(
            block_id="b_cell_2",
            text="Col 1 Content",
            page=1,
            bbox=[60.0, 30.0, 100.0, 40.0],
            region_id="reg-1",
            region_kind="column",
            reading_order=4,
            table_id="tbl_1",
            row_index=1,
            column_index=1,
        ),
    ]
    sem_input = SemanticInput(
        document_id="cell-contiguous",
        page_count=1,
        archetype=DocumentArchetype.STRUCTURED_FORM,
        pages=[],
        blocks=blocks,
    )

    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)
    serialized_ids = [b["id"] for b in payload["blocks"]]

    assert serialized_ids == ["b_cell_1a", "b_cell_1b", "b_cell_2"]


def test_multiple_tables_on_same_page_remain_separate():
    """Requirement 11: Separate tables on the same page are not interleaved."""
    blocks = [
        # Table 1
        SemanticBlockInput(
            block_id="t1_r0",
            text="T1 Header",
            page=1,
            bbox=[10.0, 10.0, 100.0, 20.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=1,
            table_id="tbl_1",
            row_index=0,
            column_index=0,
        ),
        SemanticBlockInput(
            block_id="t1_r1",
            text="T1 Data",
            page=1,
            bbox=[10.0, 30.0, 100.0, 40.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=2,
            table_id="tbl_1",
            row_index=1,
            column_index=0,
        ),
        # Table 2
        SemanticBlockInput(
            block_id="t2_r0",
            text="T2 Header",
            page=1,
            bbox=[10.0, 100.0, 100.0, 110.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=3,
            table_id="tbl_2",
            row_index=0,
            column_index=0,
        ),
        SemanticBlockInput(
            block_id="t2_r1",
            text="T2 Data",
            page=1,
            bbox=[10.0, 120.0, 100.0, 130.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=4,
            table_id="tbl_2",
            row_index=1,
            column_index=0,
        ),
    ]
    sem_input = SemanticInput(
        document_id="multi-tbl",
        page_count=1,
        archetype=DocumentArchetype.STRUCTURED_FORM,
        pages=[],
        blocks=blocks,
    )

    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)
    serialized_ids = [b["id"] for b in payload["blocks"]]

    assert serialized_ids == ["t1_r0", "t1_r1", "t2_r0", "t2_r1"]


def test_non_table_blocks_retain_reading_order():
    """Requirement 11: Non-table blocks maintain their reading order relative to tables."""
    blocks = [
        SemanticBlockInput(
            block_id="header_text",
            text="Section Title",
            page=1,
            bbox=[10.0, 5.0, 100.0, 15.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=0,
        ),
        SemanticBlockInput(
            block_id="t1_r0",
            text="Table Header",
            page=1,
            bbox=[10.0, 20.0, 100.0, 30.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=1,
            table_id="tbl_1",
            row_index=0,
            column_index=0,
        ),
        SemanticBlockInput(
            block_id="footer_text",
            text="Declaration / Footer",
            page=1,
            bbox=[10.0, 200.0, 100.0, 210.0],
            region_id="reg-0",
            region_kind="column",
            reading_order=2,
        ),
    ]
    sem_input = SemanticInput(
        document_id="mixed-blocks",
        page_count=1,
        archetype=DocumentArchetype.STRUCTURED_FORM,
        pages=[],
        blocks=blocks,
    )

    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)
    serialized_ids = [b["id"] for b in payload["blocks"]]

    assert serialized_ids == ["header_text", "t1_r0", "footer_text"]


def test_akibul_page4_table_row_major_regression():
    """Requirement 11 & 12: AKIBUL Page 4 table serializes row-major across regions."""
    fixture_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    raw_bytes = fixture_path.read_bytes()
    extracted = PDFExtractor.extract(raw_bytes)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)
    sem_input = build_semantic_input(layout, document_id=fixture_path.name)

    compact_str = serialize_compact_semantic_input(sem_input)
    payload = json.loads(compact_str)

    p4_blocks = [b for b in payload["blocks"] if b.get("page") == 4]
    table_p4_blocks = [b for b in p4_blocks if b.get("table") == "table_p4_0"]
    assert len(table_p4_blocks) > 0

    # 1. Monotonic row progression: row 0 -> row 1 -> row 2
    rows = [b["row"] for b in table_p4_blocks]
    assert sorted(rows) == rows, f"Rows are not monotonically ordered: {rows}"

    unique_rows = []
    for r in rows:
        if not unique_rows or unique_rows[-1] != r:
            unique_rows.append(r)
    assert unique_rows == [0, 1, 2]

    # 2. Within Row 1, columns monotonically progress
    r1_blocks = [b for b in table_p4_blocks if b["row"] == 1]
    r1_cols = [b["col"] for b in r1_blocks]
    assert sorted(r1_cols) == r1_cols, f"Row 1 cols are not monotonic: {r1_cols}"

    # 3. Row 1 contains vessel, rank, and service dates in the same contiguous row slice
    r1_text = " ".join(b["text"] for b in r1_blocks)
    assert "MT Furano" in r1_text
    assert "Galaxy" in r1_text
    assert "Cadet" in r1_text
    assert "18-02-" in r1_text
    assert "2023" in r1_text
    assert "14-08-" in r1_text

    # 4. Regression check: MUST NOT serialize region-major
    # Row 0 Col 2 (in region 1) must appear BEFORE Row 1 Col 0 (in region 0)
    r0_reg1_idx = next(i for i, b in enumerate(table_p4_blocks) if b["row"] == 0 and b["col"] == 2)
    r1_reg0_idx = next(i for i, b in enumerate(table_p4_blocks) if b["row"] == 1 and b["col"] == 0)
    assert r0_reg1_idx < r1_reg0_idx, (
        f"Row 0 Col 2 (idx {r0_reg1_idx}) should appear before Row 1 Col 0 (idx {r1_reg0_idx})"
    )

