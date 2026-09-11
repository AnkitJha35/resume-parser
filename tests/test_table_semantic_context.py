"""Phase 10W: Tests for generic table semantic context interpretation.

Covers:
A) Tabular grid purpose classification (SEA_SERVICE, EDUCATION, DOCUMENTS, COURSES_CERTIFICATIONS, etc.).
B) Form-style key-value table classification (PERSONAL_DATA).
C) Multi-block column header recognition.
D) Column semantic roles inference for SEA_SERVICE (rank, vessel, dates, grt, engine, company).
E) Column semantic roles inference for EDUCATION (degree, institution, year, grade).
F) Column semantic roles inference for DOCUMENTS / CERTIFICATIONS.
G) Protection against TECHNOLOGY classification for table cells.
H) Conservative fallback to UNKNOWN when table headers are ambiguous.
I) Continuation row / multi-line cell handling.
J) Integration with AKIBUL ALAM CV(JO).pdf.
K) Integration with CV Rishabh Dixit.pdf.
L) Integration with AASHISH DG.pdf.
M) Backwards compatibility of SemanticBlockInput and SemanticInput.
N) Serialization tests verifying compact representation and size reduction.
O) Surrounding section heading contextual reinforcement.
P) Anti-technology structural role precedence.
Q) Structured table serialization with source block IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    DocumentArchetype,
    SemanticBlockInput,
    SemanticInput,
    build_semantic_input,
)
from app.extractors.semantic_prompt import (
    serialize_compact_semantic_input,
    serialize_structured_table_semantic_input,
)
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.structural_roles import _looks_like_technology
from app.pipeline.stages.table_binding import (
    GeometricCell,
    GeometricTable,
    GeometricTableBinder,
)
from app.pipeline.stages.table_semantic_context import (
    CellSemanticDescriptor,
    ColumnSemanticDescriptor,
    TablePurpose,
    TableSemanticContext,
    apply_table_semantics_to_blocks,
    build_table_semantic_contexts,
    detect_is_form_table,
    infer_column_semantics,
    infer_table_purpose,
)
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
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        text=text,
        page=page,
        bbox=[x0, y0, x1, y1],
        region_id=f"reg_p{page}_0",
        region_kind="column",
        reading_order=0,
        suggested_role=role,
    )


def _make_table_with_cells(
    table_id: str,
    page: int,
    rows: list[list[tuple[str, str]]],  # list of rows, each row has list of (block_id, text)
) -> GeometricTable:
    num_rows = len(rows)
    num_cols = max(len(r) for r in rows) if rows else 0
    cells: list[GeometricCell] = []
    for r_idx, row in enumerate(rows):
        role = "HEADER" if r_idx == 0 else "DATA"
        for c_idx, (b_id, txt) in enumerate(row):
            cells.append(
                GeometricCell(
                    block_id=b_id,
                    text=txt,
                    bbox=[c_idx * 100.0, r_idx * 20.0, (c_idx + 1) * 100.0, (r_idx + 1) * 20.0],
                    table_id=table_id,
                    row_index=r_idx,
                    column_index=c_idx,
                    cell_role=role,
                )
            )
    return GeometricTable(
        table_id=table_id,
        page=page,
        num_columns=num_cols,
        num_rows=num_rows,
        num_data_rows=max(0, num_rows - 1),
        column_bands=[(i * 100.0, (i + 1) * 100.0) for i in range(num_cols)],
        cells=cells,
    )


# -----------------------------------------------------------------------------
# Test A: Tabular grid purpose classification
# -----------------------------------------------------------------------------
def test_tabular_grid_purpose_classification():
    """Verify standard tabular grids identify their domain purpose from column headers."""
    # Sea service
    t_sea = _make_table_with_cells(
        "t_sea",
        1,
        [
            [("b1", "Rank"), ("b2", "Ship Name"), ("b3", "Sign On"), ("b4", "Sign Off")],
            [("b5", "Chief Engineer"), ("b6", "MT Apollo"), ("b7", "2020-01-01"), ("b8", "2020-07-01")],
        ],
    )
    purpose, conf, ev = infer_table_purpose(t_sea, [])
    assert purpose == TablePurpose.SEA_SERVICE
    assert conf >= 0.5
    assert any("rank" in e or "ship name" in e for e in ev)

    # Education
    t_edu = _make_table_with_cells(
        "t_edu",
        1,
        [
            [("b1", "Degree"), ("b2", "Name of Institute"), ("b3", "Passing Year"), ("b4", "Marks")],
            [("b5", "B.Tech"), ("b6", "IIT Delhi"), ("b7", "2019"), ("b8", "85%")],
        ],
    )
    purpose, conf, ev = infer_table_purpose(t_edu, [])
    assert purpose == TablePurpose.EDUCATION
    assert conf >= 0.5

    # Documents
    t_doc = _make_table_with_cells(
        "t_doc",
        1,
        [
            [("b1", "Document Name"), ("b2", "Number"), ("b3", "Date of Issue"), ("b4", "Date of Expiry")],
            [("b5", "Passport"), ("b6", "Z1234567"), ("b7", "2018-05-10"), ("b8", "2028-05-09")],
        ],
    )
    purpose, conf, ev = infer_table_purpose(t_doc, [])
    assert purpose == TablePurpose.DOCUMENTS
    assert conf >= 0.5


# -----------------------------------------------------------------------------
# Test B: Form-style key-value table classification
# -----------------------------------------------------------------------------
def test_form_style_table_classification():
    """Verify Style B key-value form tables identify purpose and set is_form_table."""
    t_form = _make_table_with_cells(
        "t_pers",
        1,
        [
            [("b1", "Height (in Cms) :"), ("b2", "180"), ("b3", "Weight :"), ("b4", "75 kg")],
            [("b5", "Date of Birth :"), ("b6", "1995-04-12"), ("b7", "Nationality :"), ("b8", "Indian")],
            [("b9", "Marital Status :"), ("b10", "Single"), ("b11", "Blood Group :"), ("b12", "O+")],
        ],
    )
    assert detect_is_form_table(t_form) is True
    purpose, conf, ev = infer_table_purpose(t_form, [])
    assert purpose == TablePurpose.PERSONAL_DATA
    assert conf >= 0.5


# -----------------------------------------------------------------------------
# Test C: Multi-block column header recognition
# -----------------------------------------------------------------------------
def test_multiblock_column_header_recognition():
    """Verify column headers composed of multiple blocks combine correctly."""
    cells = [
        GeometricCell("b_h1", "Name of", [0, 0, 100, 15], "t1", 0, 0, "HEADER"),
        GeometricCell("b_h2", "Vessel", [0, 15, 100, 30], "t1", 0, 0, "HEADER"),
        GeometricCell("b_h3", "Gross", [100, 0, 200, 15], "t1", 0, 1, "HEADER"),
        GeometricCell("b_h4", "Tonnage (GRT)", [100, 15, 200, 30], "t1", 0, 1, "HEADER"),
    ]
    t = GeometricTable("t1", 1, 2, 1, 0, [(0, 100), (100, 200)], cells)
    cols = infer_column_semantics(t, TablePurpose.SEA_SERVICE)
    assert len(cols) == 2
    assert cols[0].header_text == "Name of Vessel"
    assert cols[0].semantic_role == "vessel_name"
    assert "b_h1" in cols[0].source_block_ids and "b_h2" in cols[0].source_block_ids

    assert "Gross Tonnage (GRT)" in cols[1].header_text
    assert cols[1].semantic_role == "grt"


# -----------------------------------------------------------------------------
# Test D: Column semantic roles inference for SEA_SERVICE
# -----------------------------------------------------------------------------
def test_sea_service_column_semantics():
    """Verify all major maritime sea service columns map to semantic roles."""
    headers = [
        "S. No.",
        "Rank / Capacity",
        "Name of Vessel",
        "Type of Vessel",
        "FLAG",
        "DWT / GRT",
        "Engine BHP",
        "Sign On (From)",
        "Sign Off (To)",
        "Duration (Months)",
        "Company / Manager",
    ]
    row0 = [(f"b_{i}", h) for i, h in enumerate(headers)]
    t = _make_table_with_cells("t_sea", 1, [row0])
    cols = infer_column_semantics(t, TablePurpose.SEA_SERVICE)

    roles = [c.semantic_role for c in cols]
    assert roles[0] == "serial_no"
    assert roles[1] == "rank"
    assert roles[2] == "vessel_name"
    assert roles[3] == "vessel_type"
    assert roles[4] == "flag"
    assert roles[5] in ("dwt", "grt")
    assert roles[6] == "engine_type"
    assert roles[7] == "sign_on"
    assert roles[8] == "sign_off"
    assert roles[9] == "duration"
    assert roles[10] == "company"


# -----------------------------------------------------------------------------
# Test E: Column semantic roles inference for EDUCATION
# -----------------------------------------------------------------------------
def test_education_column_semantics():
    """Verify education table columns map to degree, institution, dates, grade."""
    headers = ["Degree / Standard", "College / University", "From", "Passing Year", "Grade / Marks"]
    row0 = [(f"b_{i}", h) for i, h in enumerate(headers)]
    t = _make_table_with_cells("t_edu", 1, [row0])
    cols = infer_column_semantics(t, TablePurpose.EDUCATION)

    roles = [c.semantic_role for c in cols]
    assert roles[0] == "degree"
    assert roles[1] == "institution"
    assert roles[2] == "start_date"
    assert roles[3] == "end_date"
    assert roles[4] == "grade"


# -----------------------------------------------------------------------------
# Test F: Column semantic roles inference for DOCUMENTS / CERTIFICATIONS
# -----------------------------------------------------------------------------
def test_documents_and_courses_column_semantics():
    """Verify documents and training courses tables map to expected column roles."""
    headers_doc = ["Documents", "Number", "Date of Issue", "Date of Expiry", "Issued by"]
    t_doc = _make_table_with_cells("t_doc", 1, [[(f"b_{i}", h) for i, h in enumerate(headers_doc)]])
    cols_doc = infer_column_semantics(t_doc, TablePurpose.DOCUMENTS)
    roles_doc = [c.semantic_role for c in cols_doc]
    assert roles_doc[0] == "document_name"
    assert roles_doc[1] == "document_number"
    assert roles_doc[2] == "issue_date"
    assert roles_doc[3] == "expiry_date"
    assert roles_doc[4] == "issuing_authority"

    headers_course = ["Details of Courses & Certificates", "Certificate No", "Date of Issue", "Valid Until"]
    t_crs = _make_table_with_cells("t_crs", 1, [[(f"b_{i}", h) for i, h in enumerate(headers_course)]])
    cols_crs = infer_column_semantics(t_crs, TablePurpose.COURSES_CERTIFICATIONS)
    roles_crs = [c.semantic_role for c in cols_crs]
    assert roles_crs[0] == "course_name"
    assert roles_crs[1] == "certificate_number"
    assert roles_crs[2] == "issue_date"
    assert roles_crs[3] == "expiry_date"


# -----------------------------------------------------------------------------
# Test G: Protection against TECHNOLOGY classification for table cells
# -----------------------------------------------------------------------------
def test_protection_against_technology_classification():
    """Verify table cells are protected from false-positive TECHNOLOGY role."""
    # 1. Test _looks_like_technology guard
    for term in ["Rank", "Vessel", "Nationality", "Passport", "Height", "Weight", "Flag", "Sign On", "CDC"]:
        assert _looks_like_technology(term, following=("Other Cell",), previous_text="Some Text") is False

    # 2. Test apply_table_semantics_to_blocks precedence override
    t = _make_table_with_cells(
        "t_sea",
        1,
        [
            [("b_h1", "Rank"), ("b_h2", "Vessel Name")],
            [("b_d1", "Deck Cadet"), ("b_d2", "MT Swarna")],
        ],
    )
    blocks = [
        _make_block("b_h1", "Rank", 50, 100, 150, 115, role="TECHNOLOGY"),
        _make_block("b_h2", "Vessel Name", 200, 100, 300, 115, role="TECHNOLOGY"),
        _make_block("b_d1", "Deck Cadet", 50, 130, 150, 145, role="TECHNOLOGY"),
        _make_block("b_d2", "MT Swarna", 200, 130, 300, 145, role="TECHNOLOGY"),
    ]
    binder = GeometricTableBinder()
    bound = binder.bind_document_tables(blocks)
    contexts = build_table_semantic_contexts(binder.last_detected_tables, bound)
    enriched = apply_table_semantics_to_blocks(contexts, bound)

    by_id = {b.block_id: b for b in enriched}
    assert by_id["b_h1"].suggested_role == "TABLE_HEADER"
    assert by_id["b_h2"].suggested_role == "TABLE_HEADER"
    assert by_id["b_d1"].column_semantic == "rank"
    assert by_id["b_d2"].column_semantic == "vessel_name"
    assert by_id["b_d1"].suggested_role == "DESCRIPTION"
    assert by_id["b_d2"].suggested_role == "DESCRIPTION"
    assert all(b.suggested_role != "TECHNOLOGY" for b in enriched)


# -----------------------------------------------------------------------------
# Test H: Conservative fallback to UNKNOWN when table headers are ambiguous
# -----------------------------------------------------------------------------
def test_conservative_fallback_to_unknown():
    """Verify tables with ambiguous or non-standard headers fall back to UNKNOWN."""
    t_ambiguous = _make_table_with_cells(
        "t_amb",
        1,
        [
            [("b1", "Alpha"), ("b2", "Beta"), ("b3", "Gamma")],
            [("b4", "Data 1"), ("b5", "Data 2"), ("b6", "Data 3")],
        ],
    )
    purpose, conf, ev = infer_table_purpose(t_ambiguous, [])
    assert purpose == TablePurpose.UNKNOWN
    assert conf == 0.0
    assert ev == []


# -----------------------------------------------------------------------------
# Test I: Continuation row / multi-line cell handling
# -----------------------------------------------------------------------------
def test_continuation_row_multiline_cells():
    """Verify multiple blocks in the same cell coordinate are combined and preserve block IDs."""
    cells = [
        GeometricCell("b1", "Product", [0, 20, 100, 30], "t1", 1, 0, "DATA"),
        GeometricCell("b2", "Tanker", [0, 30, 100, 40], "t1", 1, 0, "DATA"),
    ]
    t = GeometricTable("t1", 1, 1, 2, 1, [(0, 100)], cells)
    blocks = [
        _make_block("b1", "Product", 0, 20, 100, 30),
        _make_block("b2", "Tanker", 0, 30, 100, 40),
    ]
    contexts = build_table_semantic_contexts([t], blocks)
    assert len(contexts) == 1
    ctx = contexts[0]
    cell_desc = ctx.rows[1][0]
    assert cell_desc.text == "Product Tanker"
    assert "b1" in cell_desc.source_block_ids and "b2" in cell_desc.source_block_ids


# -----------------------------------------------------------------------------
# Test J: Integration with AKIBUL ALAM CV(JO).pdf
# -----------------------------------------------------------------------------
def test_akibul_fixture_table_interpretation():
    """Verify table purposes and roles on real AKIBUL ALAM CV fixture."""
    fixture_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    raw = fixture_path.read_bytes()
    extracted = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)

    si = build_semantic_input(layout, document_id=fixture_path.name)
    assert len(si.tables) > 0

    tables_by_id = {t.table_id: t for t in si.tables}

    # Page 1 table is personal form
    assert tables_by_id["table_p1_0"].purpose == TablePurpose.PERSONAL_DATA
    assert tables_by_id["table_p1_0"].is_form_table is True

    # Page 2 table 1 is courses & certificates
    assert tables_by_id["table_p2_1"].purpose == TablePurpose.COURSES_CERTIFICATIONS

    # Page 3 table 2 is education
    assert tables_by_id["table_p3_2"].purpose == TablePurpose.EDUCATION

    # Page 4 table is sea service
    assert tables_by_id["table_p4_0"].purpose == TablePurpose.SEA_SERVICE
    sea_cols = {c.semantic_role for c in tables_by_id["table_p4_0"].columns}
    assert "rank" in sea_cols
    assert "vessel_name" in sea_cols
    assert "sign_on" in sea_cols
    assert "sign_off" in sea_cols


# -----------------------------------------------------------------------------
# Test K: Integration with CV Rishabh Dixit.pdf
# -----------------------------------------------------------------------------
def test_rishabh_fixture_table_interpretation():
    """Verify sea service table on real CV Rishabh Dixit fixture."""
    fixture_path = Path("tests/fixtures/CV Rishabh Dixit.pdf")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    raw = fixture_path.read_bytes()
    extracted = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)

    si = build_semantic_input(layout, document_id=fixture_path.name)
    tables_by_id = {t.table_id: t for t in si.tables}

    assert "table_p1_0" in tables_by_id
    assert tables_by_id["table_p1_0"].purpose == TablePurpose.SEA_SERVICE
    col_roles = [c.semantic_role for c in tables_by_id["table_p1_0"].columns]
    assert "rank" in col_roles
    assert "vessel_name" in col_roles
    assert "sign_on" in col_roles
    assert "sign_off" in col_roles

    # Page 2 has documents table
    assert "table_p2_0" in tables_by_id
    assert tables_by_id["table_p2_0"].purpose == TablePurpose.DOCUMENTS


# -----------------------------------------------------------------------------
# Test L: Integration with AASHISH DG.pdf
# -----------------------------------------------------------------------------
def test_aashish_fixture_table_interpretation():
    """Verify form-style tables on AASHISH DG.pdf seafarer profile."""
    fixture_path = Path("tests/fixtures/AASHISH DG.pdf")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    raw = fixture_path.read_bytes()
    extracted = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)

    si = build_semantic_input(layout, document_id=fixture_path.name)
    assert len(si.tables) >= 5

    purposes = {t.purpose for t in si.tables}
    assert TablePurpose.PERSONAL_DATA in purposes
    assert TablePurpose.COURSES_CERTIFICATIONS in purposes
    assert TablePurpose.SEA_SERVICE in purposes


# -----------------------------------------------------------------------------
# Test M: Backwards compatibility of SemanticBlockInput and SemanticInput
# -----------------------------------------------------------------------------
def test_backwards_compatibility():
    """Verify existing block schema and layout preservation assertions are 100% maintained."""
    fixture_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    raw = fixture_path.read_bytes()
    extracted = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)

    si = build_semantic_input(layout, document_id=fixture_path.name)
    # The 322 blocks assertion from Phase 10B must remain intact
    assert len(si.blocks) == 322

    # Check that new fields exist on blocks without breaking serialization
    b0 = si.blocks[0]
    assert hasattr(b0, "table_purpose")
    assert hasattr(b0, "column_semantic")
    assert hasattr(si, "tables")


# -----------------------------------------------------------------------------
# Test N: Serialization tests verifying compact representation and size reduction
# -----------------------------------------------------------------------------
def test_compact_and_structured_table_serialization():
    """Verify JSON serialization includes tables and structured format reduces size."""
    fixture_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    raw = fixture_path.read_bytes()
    extracted = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)

    si = build_semantic_input(layout, document_id=fixture_path.name)

    # 1. Compact serialization includes tables and blocks
    compact_json = serialize_compact_semantic_input(si)
    payload = json.loads(compact_json)
    assert "blocks" in payload
    assert len(payload["blocks"]) == 322
    assert "tables" in payload
    assert len(payload["tables"]) > 0

    # 2. Structured table serialization reduces size and contains explicit structured records
    structured_json = serialize_structured_table_semantic_input(si)
    struct_payload = json.loads(structured_json)
    assert "tables" in struct_payload
    assert "blocks" in struct_payload
    # Non-table blocks should be significantly fewer than 322
    assert len(struct_payload["blocks"]) < len(payload["blocks"])
    # Verify structured tables contain columns and rows with source block IDs
    first_tbl = struct_payload["tables"][0]
    assert "columns" in first_tbl
    assert "rows" in first_tbl
    if first_tbl["rows"]:
        assert "source_block_ids" in first_tbl["rows"][0][0]


# -----------------------------------------------------------------------------
# Test O: Surrounding section heading contextual reinforcement
# -----------------------------------------------------------------------------
def test_surrounding_section_heading_reinforcement():
    """Verify heading block directly above table reinforces purpose inference."""
    heading = _make_block("b_hd", "SEA SERVICE DETAILS", 0, 50, 200, 70, role="SECTION_HEADING")
    cells = [
        GeometricCell("b1", "Vsl", [0, 80, 100, 100], "t_sea_min", 0, 0, "HEADER"),
        GeometricCell("b2", "Period", [100, 80, 200, 100], "t_sea_min", 0, 1, "HEADER"),
        GeometricCell("b3", "Alpha", [0, 105, 100, 125], "t_sea_min", 1, 0, "DATA"),
        GeometricCell("b4", "2021-2022", [100, 105, 200, 125], "t_sea_min", 1, 1, "DATA"),
    ]
    table = GeometricTable("t_sea_min", 1, 2, 2, 1, [(0, 100), (100, 200)], cells)

    purpose, conf, ev = infer_table_purpose(table, [heading])
    assert purpose == TablePurpose.SEA_SERVICE
    assert any("heading:sea service" in e for e in ev)
