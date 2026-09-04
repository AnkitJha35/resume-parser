"""Tests for generic deterministic geometric table binding."""

from __future__ import annotations

from pathlib import Path
import pytest

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    DocumentArchetype,
    SemanticBlockInput,
    SemanticInput,
    build_semantic_input,
)
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.table_binding import GeometricTableBinder
from app.pipeline.stages.text_extraction import PDFExtractor


def _make_block(
    block_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page: int = 1,
    role: str = "UNKNOWN",
    spans: list[dict] | None = None,
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        text=text,
        page=page,
        bbox=[x0, y0, x1, y1],
        region_id=f"page-{page}-region-0",
        region_kind="physical_region",
        reading_order=1,
        suggested_role=role,
        spans=spans or [],
    )


# =====================================================================
# 1. Synthetic Bounding-Box Unit Tests
# =====================================================================


def test_simple_2x2_table():
    """Simple 2x2 table with header and single data row."""
    binder = GeometricTableBinder(row_tolerance=5.0)
    blocks = [
        # Row 0: Header
        _make_block("b_h0", "Company", 50.0, 100.0, 150.0, 115.0),
        _make_block("b_h1", "Role", 200.0, 100.0, 300.0, 115.0),
        # Row 1: Data
        _make_block("b_d0", "Acme Corp", 50.0, 130.0, 150.0, 145.0),
        _make_block("b_d1", "Engineer", 200.0, 130.0, 300.0, 145.0),
    ]

    bound = binder.bind_document_tables(blocks)
    assert len(bound) == 4

    by_id = {b.block_id: b for b in bound}
    assert by_id["b_h0"].table_id == "table_p1_0"
    assert by_id["b_h0"].row_index == 0
    assert by_id["b_h0"].column_index == 0
    assert by_id["b_h0"].cell_role == "HEADER"

    assert by_id["b_h1"].table_id == "table_p1_0"
    assert by_id["b_h1"].row_index == 0
    assert by_id["b_h1"].column_index == 1
    assert by_id["b_h1"].cell_role == "HEADER"

    assert by_id["b_d0"].table_id == "table_p1_0"
    assert by_id["b_d0"].row_index == 1
    assert by_id["b_d0"].column_index == 0
    assert by_id["b_d0"].cell_role == "DATA"

    assert by_id["b_d1"].table_id == "table_p1_0"
    assert by_id["b_d1"].row_index == 1
    assert by_id["b_d1"].column_index == 1
    assert by_id["b_d1"].cell_role == "DATA"


def test_uneven_column_widths():
    """Table with varying column widths."""
    binder = GeometricTableBinder(row_tolerance=5.0)
    blocks = [
        # Header (Col 0 narrow, Col 1 very wide, Col 2 medium)
        _make_block("h0", "No.", 10.0, 50.0, 30.0, 65.0),
        _make_block("h1", "Full Legal Organization Description", 50.0, 50.0, 350.0, 65.0),
        _make_block("h2", "Status", 380.0, 50.0, 450.0, 65.0),
        # Data
        _make_block("d0", "1", 10.0, 80.0, 30.0, 95.0),
        _make_block("d1", "Acme International Shipping Holdings Ltd", 50.0, 80.0, 350.0, 95.0),
        _make_block("d2", "Active", 380.0, 80.0, 450.0, 95.0),
    ]

    bound = binder.bind_document_tables(blocks)
    by_id = {b.block_id: b for b in bound}

    assert by_id["d0"].column_index == 0
    assert by_id["d1"].column_index == 1
    assert by_id["d2"].column_index == 2


def test_wrapped_cell_text_and_multiple_blocks():
    """Multi-line cells where a physical cell spans multiple wrapped blocks."""
    binder = GeometricTableBinder(row_tolerance=5.0)
    blocks = [
        # Header
        _make_block("b_h0", "Course Name", 30.0, 100.0, 180.0, 115.0),
        _make_block("b_h1", "Certificate No", 220.0, 100.0, 320.0, 115.0),
        # Data Row 1 (Col 0 has 2 wrapped blocks, Col 1 has 1 block)
        _make_block("b_d0_1", "Advanced Fire", 30.0, 130.0, 180.0, 142.0),
        _make_block("b_d0_2", "Fighting Level 2", 30.0, 143.0, 180.0, 155.0),
        _make_block("b_d1", "045.232822", 220.0, 130.0, 320.0, 145.0),
    ]

    bound = binder.bind_document_tables(blocks)
    by_id = {b.block_id: b for b in bound}

    # Both blocks in Col 0 belong to table_p1_0, row 1, col 0
    assert by_id["b_d0_1"].table_id == "table_p1_0"
    assert by_id["b_d0_1"].row_index == 1
    assert by_id["b_d0_1"].column_index == 0

    assert by_id["b_d0_2"].table_id == "table_p1_0"
    assert by_id["b_d0_2"].row_index == 1
    assert by_id["b_d0_2"].column_index == 0

    assert by_id["b_d1"].table_id == "table_p1_0"
    assert by_id["b_d1"].row_index == 1
    assert by_id["b_d1"].column_index == 1


def test_compound_multi_line_header():
    """Compound headers spanning multiple lines and subheaders within the same column track."""
    binder = GeometricTableBinder(row_tolerance=5.0)
    blocks = [
        # Header (Col 0 has 2 lines, Col 1 has 2 lines)
        _make_block("h0_1", "Name of Owners /", 50.0, 100.0, 180.0, 112.0),
        _make_block("h0_2", "Manager", 70.0, 115.0, 140.0, 128.0),
        _make_block("h1_1", "Name of", 200.0, 100.0, 260.0, 112.0),
        _make_block("h1_2", "Vessel", 200.0, 115.0, 260.0, 128.0),
        # Data
        _make_block("d0", "Unix Line PTE LTD", 50.0, 140.0, 180.0, 155.0),
        _make_block("d1", "MT Furano", 200.0, 140.0, 260.0, 155.0),
    ]

    bound = binder.bind_document_tables(blocks)
    by_id = {b.block_id: b for b in bound}

    assert by_id["h0_1"].column_index == 0
    assert by_id["h0_2"].column_index == 0
    assert by_id["h1_1"].column_index == 1
    assert by_id["h1_2"].column_index == 1
    assert by_id["d0"].column_index == 0
    assert by_id["d1"].column_index == 1


def test_different_x_offsets_within_same_cell():
    """Indented or offset text within the same cell must not create new columns."""
    binder = GeometricTableBinder(row_tolerance=5.0)
    blocks = [
        _make_block("h0", "Description", 50.0, 100.0, 250.0, 115.0),
        _make_block("h1", "Code", 300.0, 100.0, 380.0, 115.0),
        # Row 1 has offset/indented bullet in Col 0
        _make_block("d0_main", "Primary Equipment", 50.0, 130.0, 250.0, 142.0),
        _make_block("d0_sub", "  - Sub-assembly part A", 65.0, 143.0, 250.0, 155.0),
        _make_block("d1", "EQ-001", 300.0, 130.0, 380.0, 145.0),
    ]

    bound = binder.bind_document_tables(blocks)
    by_id = {b.block_id: b for b in bound}

    assert by_id["d0_main"].column_index == 0
    assert by_id["d0_sub"].column_index == 0
    assert by_id["d1"].column_index == 1


def test_nearby_unrelated_content():
    """Unrelated text above or below table remains unbound (table_id is None)."""
    binder = GeometricTableBinder(row_tolerance=5.0)
    blocks = [
        # Outside paragraph above table
        _make_block("p_top", "Summary of qualifications and professional profile.", 50.0, 20.0, 400.0, 35.0),
        # 2x2 Table
        _make_block("th0", "Header A", 50.0, 100.0, 150.0, 115.0),
        _make_block("th1", "Header B", 200.0, 100.0, 300.0, 115.0),
        _make_block("td0", "Data A", 50.0, 130.0, 150.0, 145.0),
        _make_block("td1", "Data B", 200.0, 130.0, 300.0, 145.0),
        # Outside paragraph below table
        _make_block("p_bot", "Declaration: All information provided above is true.", 50.0, 250.0, 400.0, 265.0),
    ]

    bound = binder.bind_document_tables(blocks)
    by_id = {b.block_id: b for b in bound}

    assert by_id["p_top"].table_id is None
    assert by_id["p_top"].row_index is None

    assert by_id["th0"].table_id == "table_p1_0"
    assert by_id["td0"].table_id == "table_p1_0"

    assert by_id["p_bot"].table_id is None
    assert by_id["p_bot"].row_index is None


def test_multiple_independent_tables_on_one_page():
    """Multiple independent tables on the same page are detected with distinct table IDs."""
    binder = GeometricTableBinder(row_tolerance=5.0)
    blocks = [
        # Table 1: Top
        _make_block("t1_h0", "Degree", 50.0, 50.0, 150.0, 65.0),
        _make_block("t1_h1", "Year", 200.0, 50.0, 280.0, 65.0),
        _make_block("t1_d0", "B.Sc", 50.0, 75.0, 150.0, 90.0),
        _make_block("t1_d1", "2020", 200.0, 75.0, 280.0, 90.0),
        # Table 2: Bottom (separated by > 25 pt gap)
        _make_block("t2_h0", "Skill", 50.0, 200.0, 150.0, 215.0),
        _make_block("t2_h1", "Level", 200.0, 200.0, 280.0, 215.0),
        _make_block("t2_d0", "Python", 50.0, 225.0, 150.0, 240.0),
        _make_block("t2_d1", "Expert", 200.0, 225.0, 280.0, 240.0),
    ]

    bound = binder.bind_document_tables(blocks)
    by_id = {b.block_id: b for b in bound}

    assert by_id["t1_h0"].table_id == "table_p1_0"
    assert by_id["t1_d0"].table_id == "table_p1_0"

    assert by_id["t2_h0"].table_id == "table_p1_1"
    assert by_id["t2_d0"].table_id == "table_p1_1"


def test_deterministic_repeated_binding():
    """Table binding is 100% deterministic across multiple repeated runs."""
    binder = GeometricTableBinder(row_tolerance=5.0)
    blocks = [
        _make_block("b1", "Col 1", 50.0, 100.0, 150.0, 115.0),
        _make_block("b2", "Col 2", 200.0, 100.0, 300.0, 115.0),
        _make_block("b3", "Val 1", 50.0, 130.0, 150.0, 145.0),
        _make_block("b4", "Val 2", 200.0, 130.0, 300.0, 145.0),
    ]

    res1 = binder.bind_document_tables(blocks)
    res2 = binder.bind_document_tables(blocks)

    for b1, b2 in zip(res1, res2):
        assert b1.block_id == b2.block_id
        assert b1.table_id == b2.table_id
        assert b1.row_index == b2.row_index
        assert b1.column_index == b2.column_index
        assert b1.text == b2.text


# =====================================================================
# 2. AKIBUL ALAM CV(JO).pdf Integration & Regression Tests
# =====================================================================


def test_akibul_integration_table_geometry():
    """AKIBUL table geometry verification:
    - Page 4 sea-service table has 8 to 10 logical columns (not 14 over-segmented columns) and 2 data rows;
    - Page 3 education table with 4 columns and 1 data row;
    - Page 2 course/certificate table with 5 columns and multiple rows;
    - Original block IDs remain attached to cells.
    """
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

    blocks = sem_input.blocks

    # 1. Page 4 Sea Service Table
    p4_table_blocks = [b for b in blocks if b.page == 4 and b.table_id is not None]
    assert len(p4_table_blocks) >= 50

    p4_cols = set(b.column_index for b in p4_table_blocks)
    p4_data_rows = set(b.row_index for b in p4_table_blocks if b.cell_role == "DATA")
    p4_header_rows = set(b.row_index for b in p4_table_blocks if b.cell_role == "HEADER")

    # Over-segmentation regression: column count must not be 14
    assert len(p4_cols) != 14, "Page 4 Sea Service table suffered 14-column over-segmentation regression"
    assert 8 <= len(p4_cols) <= 10, f"Expected 8 to 10 logical columns on page 4, got {len(p4_cols)}"
    assert len(p4_data_rows) == 2, f"Expected 2 data rows on page 4 sea service, got {len(p4_data_rows)}"
    assert len(p4_header_rows) >= 1

    # Check that original block IDs remain attached
    p4_block_ids = {b.block_id for b in p4_table_blocks}
    assert "b_p4_257" in p4_block_ids  # Header
    assert "b_p4_261" in p4_block_ids  # Row 1 '1'
    assert "b_p4_266" in p4_block_ids  # Row 2 '2'

    # 2. Page 3 Education Table
    p3_table_blocks = [b for b in blocks if b.page == 3 and b.table_id is not None]
    # Identify the education table (table containing 'Pre-sea' or 'International Maritime Academy')
    edu_blocks = [
        b for b in p3_table_blocks
        if any(kw in b.text for kw in ["Institute", "Academy", "Nautical Science", "01-04-2020"])
    ]
    assert len(edu_blocks) >= 4
    edu_table_id = edu_blocks[0].table_id

    edu_table_all_blocks = [b for b in p3_table_blocks if b.table_id == edu_table_id]
    edu_cols = set(b.column_index for b in edu_table_all_blocks)
    assert len(edu_cols) == 4, f"Expected 4 columns for page 3 education table, got {len(edu_cols)}"

    # 3. Page 2 Course / Certificate Table
    p2_table_blocks = [b for b in blocks if b.page == 2 and b.table_id is not None]
    assert len(p2_table_blocks) >= 40
    p2_tables = set(b.table_id for b in p2_table_blocks)
    assert len(p2_tables) >= 1

    # Check primary course table has 5 columns
    p2_cols_per_table = [
        len(set(b.column_index for b in p2_table_blocks if b.table_id == tid))
        for tid in p2_tables
    ]
    assert any(num_cols >= 5 for num_cols in p2_cols_per_table)


# =====================================================================
# 3. Word-Level Geometric Column Boundary Splitting Tests
# =====================================================================


def test_cross_column_block_splitting_and_provenance():
    """Verify that a text block spanning across a column boundary is geometrically split by word positions."""
    binder = GeometricTableBinder(row_tolerance=5.0)

    # 2-column table: Col 0 is [50, 150], Col 1 is [200, 300]. Separator is ~175.0.
    # Data row contains a single block spanning from 50 to 300 with text "LeftValue RightValue"
    blocks = [
        # Row 0: Headers
        _make_block("h0", "LeftColumn", 50.0, 100.0, 150.0, 115.0),
        _make_block("h1", "RightColumn", 200.0, 100.0, 300.0, 115.0),
        # Row 1: Crossing Data Block and Single Data Block on same row slice
        _make_block("d_cross", "LeftValue RightValue", 50.0, 130.0, 300.0, 145.0),
        _make_block("d_single", "StandaloneRight", 200.0, 130.0, 300.0, 145.0),
    ]

    bound = binder.bind_document_tables(blocks)

    # A. Single-column block remains exactly one cell
    single_cells = [b for b in bound if b.block_id == "d_single"]
    assert len(single_cells) == 1
    assert single_cells[0].text == "StandaloneRight"
    assert single_cells[0].column_index == 1
    assert single_cells[0].row_index == 1

    # B. Crossing block is split into 2 cells
    cross_cells = [b for b in bound if b.block_id == "d_cross"]
    assert len(cross_cells) == 2

    # C. Words on left and right of the boundary are assigned correctly
    left_cell = next(c for c in cross_cells if c.column_index == 0)
    right_cell = next(c for c in cross_cells if c.column_index == 1)
    assert left_cell.text == "LeftValue"
    assert right_cell.text == "RightValue"

    # D. Row assignment remains unchanged (Row 1)
    assert left_cell.row_index == 1
    assert right_cell.row_index == 1

    # E. Provenance remains valid (block_id matches original source block)
    assert left_cell.block_id == "d_cross"
    assert right_cell.block_id == "d_cross"
    assert left_cell.cell_role == "DATA"
    assert right_cell.cell_role == "DATA"


def test_cross_column_block_splitting_with_actual_spans():
    """Verify that when actual word/span geometry is provided on a block, it is used directly for splitting."""
    binder = GeometricTableBinder(row_tolerance=5.0)

    # 2-column table: Col 0 is [50, 150], Col 1 is [200, 300]. Separator is ~175.0.
    blocks = [
        # Row 0: Headers
        _make_block("h0", "LeftColumn", 50.0, 100.0, 150.0, 115.0),
        _make_block("h1", "RightColumn", 200.0, 100.0, 300.0, 115.0),
        # Row 1: Crossing block with explicit underlying spans
        _make_block(
            "d_spans",
            "LeftPart RightPart",
            50.0,
            130.0,
            300.0,
            145.0,
            spans=[
                {"text": "LeftPart", "bbox": [50.0, 130.0, 140.0, 145.0]},
                {"text": "RightPart", "bbox": [210.0, 130.0, 300.0, 145.0]},
            ],
        ),
        # Row 2: Standard Data Row (giving 2 multi-column slices)
        _make_block("d2_0", "Row2Left", 50.0, 160.0, 150.0, 175.0),
        _make_block("d2_1", "Row2Right", 200.0, 160.0, 300.0, 175.0),
    ]

    bound = binder.bind_document_tables(blocks)

    cross_cells = [b for b in bound if b.block_id == "d_spans"]
    assert len(cross_cells) == 2

    left_cell = next(c for c in cross_cells if c.column_index == 0)
    right_cell = next(c for c in cross_cells if c.column_index == 1)

    assert left_cell.text == "LeftPart"
    assert left_cell.bbox == [50.0, 130.0, 140.0, 145.0]
    assert left_cell.row_index == 1
    assert left_cell.block_id == "d_spans"

    assert right_cell.text == "RightPart"
    assert right_cell.bbox == [210.0, 130.0, 300.0, 145.0]
    assert right_cell.row_index == 1
    assert right_cell.block_id == "d_spans"
