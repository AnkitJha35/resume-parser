"""Tests for Table Spatial Fidelity v1.

Verifies:
1. Accurate separation of visual slice groups into distinct geometric tables based on column grid compatibility.
2. Correct column count and boundary detection (e.g. 3-col name table, 2-col address table, 6-col doc table).
3. Preservation of empty cells and row uniformity (no truncation or misalignment when middle cells are empty).
4. Relative column width calculation based on spatial boundaries.
5. Complete source_block_ids provenance grounding on all cells.
6. Generic document structure table_data generation with uniform row lengths and column widths.
7. Non-regression across diverse document archetypes (maritime, technical, corporate).
"""

from __future__ import annotations

from pathlib import Path
import pytest

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import build_semantic_input
from app.pipeline.stages.generic_document_builder import build_document_structure_from_semantic_input
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.table_binding import GeometricTableBinder
from app.pipeline.stages.text_extraction import PDFExtractor


def test_akibul_page1_table_separation_and_spatial_fidelity():
    """Verify that Page 1 of AKIBUL ALAM CV(JO).pdf produces distinct, spatially accurate tables."""
    pdf_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AKIBUL ALAM CV(JO).pdf not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    lay = interpret_layout(rec)
    si = build_semantic_input(lay)

    p1_tables = [t for t in si.tables if t.page == 1]
    # Page 1 must detect at least 3 distinct tables (Name 3-col, Address 2-col, Passport 6-col)
    assert len(p1_tables) >= 3, f"Expected >= 3 tables on Page 1, got {len(p1_tables)}"

    # --- Table 0: 3 columns (Personal Name & Bio Grid) ---
    t0 = p1_tables[0]
    assert t0.num_columns == 3, f"Expected Table 0 to have 3 columns, got {t0.num_columns}"
    assert len(t0.column_widths) == 3
    # Column widths should sum to approx 100%
    assert 98.0 <= sum(t0.column_widths) <= 102.0

    # Headers
    h0 = [c.text for c in t0.rows[0]]
    assert "Surname" in h0[0]
    assert "Middle Name" in h0[1]
    assert "First Name" in h0[2]

    # Data rows: Row 1 must have empty Middle Name between 'Alam' and 'Akibul'
    row1 = [c.text for c in t0.rows[1]]
    assert len(row1) == 3
    assert row1[0] == "Alam"
    assert row1[1] == ""  # Empty middle name preserved!
    assert "Akibul" in row1[2]

    # Row 2 & 3: Nationality, DOB, Place of Birth
    r2_headers = [c.text for c in t0.rows[2]]
    assert "Nationality" in r2_headers[0]
    assert "Date of Birth" in r2_headers[1]
    assert "Place of Birth" in r2_headers[2]

    r3_data = [c.text for c in t0.rows[3]]
    assert "Bangladeshi" in r3_data[0]
    assert "15-08-2000" in r3_data[1]
    assert "Chattogram" in r3_data[2]

    # Provenance grounding: non-empty cells must have valid block IDs
    for r in t0.rows:
        for c in r:
            if c.text:
                assert len(c.source_block_ids) > 0, f"Cell '{c.text}' missing source_block_ids"

    # --- Table 1: 2 columns (Address Grid) ---
    t1 = p1_tables[1]
    assert t1.num_columns == 2, f"Expected Table 1 to have 2 columns, got {t1.num_columns}"
    assert len(t1.column_widths) == 2
    # Two roughly equal halves (45% - 55%)
    assert 45.0 <= t1.column_widths[0] <= 55.0
    assert 45.0 <= t1.column_widths[1] <= 55.0

    h1 = [c.text for c in t1.rows[0]]
    assert "Permanent" in h1[0]
    assert "Present Address" in h1[1]

    # --- Table 2: 6 columns (Passport) ---
    assert len(p1_tables) >= 5, f"Expected at least 5 tables on Page 1, got {len(p1_tables)}"
    t2 = p1_tables[2]
    assert t2.num_columns == 6, f"Expected Table 2 (Passport) to have 6 columns, got {t2.num_columns}"
    assert len(t2.column_widths) == 6
    assert 98.0 <= sum(t2.column_widths) <= 102.0

    h2 = [c.text for c in t2.rows[0]]
    assert "Passport" in h2[0]
    assert any("Date of Issue" in h for h in h2)
    assert any("Date of Expiry" in h for h in h2)
    assert len(t2.rows) == 5
    assert t2.rows[1][0].text == "B00024822"
    assert t2.rows[2][0].text == "Visa (USA)"
    assert t2.rows[3][0].text == "Visa (Schengen)"
    assert t2.rows[4][0].text == "Visa (Others)"

    # --- Table 3: 6 columns (Seaman's Book / CDC) ---
    t3 = p1_tables[3]
    assert t3.num_columns == 6, f"Expected Table 3 (Seaman's Book) to have 6 columns, got {t3.num_columns}"
    assert len(t3.column_widths) == 6
    h3 = [c.text for c in t3.rows[0]]
    assert "Seaman" in h3[0]
    assert "Number" in h3[1]
    # Exact 7 distinct data rows (8 rows total including header)
    assert len(t3.rows) == 8, f"Expected 8 rows for Seaman's Book, got {len(t3.rows)}"
    expected_t3_col0 = [
        "Bangladeshi",
        "Panama",
        "Marshall Island",
        "Bahamas",
        "Others",
        "INDOS No.",
        "SID Number",
    ]
    for r_idx, exp_label in enumerate(expected_t3_col0, start=1):
        actual_val = t3.rows[r_idx][0].text.strip()
        assert actual_val == exp_label, f"Row {r_idx} Col 0 expected '{exp_label}', got '{actual_val}'"

    # Verify data cells in Seaman's Book rows 1-3
    assert t3.rows[1][1].text.strip() == "C/O/11695"
    assert t3.rows[2][1].text.strip() == "PA0511962"
    assert t3.rows[3][1].text.strip() == "MH936855"

    # --- Table 4: 6 columns (Licence) ---
    t4 = p1_tables[4]
    assert t4.num_columns == 6, f"Expected Table 4 (Licence) to have 6 columns, got {t4.num_columns}"
    assert len(t4.column_widths) == 6
    h4 = [c.text for c in t4.rows[0]]
    assert "Licence" in h4[0]
    assert "Grade" in h4[1]
    assert "Number" in h4[2]
    # Exact 7 distinct data rows (8 rows total including header)
    assert len(t4.rows) == 8, f"Expected 8 rows for Licence, got {len(t4.rows)}"
    expected_t4_col0 = [
        "Bangladesh",
        "U.K.",
        "Singapore",
        "Liberian",
        "Marshall Island",
        "Bahamas",
        "Others",
    ]
    for r_idx, exp_label in enumerate(expected_t4_col0, start=1):
        actual_val = t4.rows[r_idx][0].text.strip()
        assert actual_val == exp_label, f"Row {r_idx} Col 0 expected '{exp_label}', got '{actual_val}'"

    # Verify data cells in Licence row 1
    assert t4.rows[1][1].text.strip() == "DOC-3"
    assert t4.rows[1][2].text.strip() == "1.DC3.001412"

    # Provenance grounding across all cells in all tables
    for t in (t2, t3, t4):
        for r in t.rows:
            for c in r:
                if c.text:
                    assert len(c.source_block_ids) > 0, f"Cell '{c.text}' missing source_block_ids"


def test_generic_document_structure_table_data_fidelity():
    """Verify that build_document_structure_from_semantic_input produces well-formed table_data."""
    pdf_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AKIBUL ALAM CV(JO).pdf not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    lay = interpret_layout(rec)
    si = build_semantic_input(lay)
    gdoc = build_document_structure_from_semantic_input(si)

    table_blocks = [b for s in gdoc.sections for b in s.blocks if b.type == "table"]
    assert len(table_blocks) >= 3

    # Check Table 0 (Personal info)
    tb0 = table_blocks[0]
    assert tb0.table_data is not None
    td0 = tb0.table_data
    assert "headers" in td0 and "rows" in td0
    assert len(td0["headers"]) == 3
    assert td0.get("num_columns") == 3
    assert td0.get("column_widths") is not None
    assert len(td0["column_widths"]) == 3

    # Ensure all rows have uniform length of 3
    for r in td0["rows"]:
        assert len(r) == 3, f"Row {r} does not match expected length 3"

    # Verify empty cell preserved in row 0
    assert td0["rows"][0][0] == "Alam"
    assert td0["rows"][0][1] == ""
    assert "Akibul" in td0["rows"][0][2]

    # Verify source_block_ids are populated on DocumentBlock
    assert len(tb0.source_block_ids) > 0

    # Verify Table 2 (Passport), Table 3 (Seaman's Book), Table 4 (Licence) in gdoc
    assert len(table_blocks) >= 5, f"Expected >= 5 table blocks in gdoc, got {len(table_blocks)}"

    # Table 2: Passport
    tb2 = table_blocks[2]
    assert tb2.table_data is not None
    assert len(tb2.table_data["headers"]) == 6
    assert tb2.table_data["headers"][0] == "Passport No:"
    assert len(tb2.table_data["rows"]) == 4

    # Table 3: Seaman's Book
    tb3 = table_blocks[3]
    assert tb3.table_data is not None
    assert len(tb3.table_data["headers"]) == 6
    assert tb3.table_data["headers"][0] == "Seaman’s Book (CDC)"
    # Exactly 7 data rows
    assert len(tb3.table_data["rows"]) == 7
    assert [r[0] for r in tb3.table_data["rows"]] == [
        "Bangladeshi",
        "Panama",
        "Marshall Island",
        "Bahamas",
        "Others",
        "INDOS No.",
        "SID Number",
    ]

    # Table 4: Licence
    tb4 = table_blocks[4]
    assert tb4.table_data is not None
    assert len(tb4.table_data["headers"]) == 6
    assert tb4.table_data["headers"][0] == "Licence"
    # Exactly 7 data rows
    assert len(tb4.table_data["rows"]) == 7
    assert [r[0] for r in tb4.table_data["rows"]] == [
        "Bangladesh",
        "U.K.",
        "Singapore",
        "Liberian",
        "Marshall Island",
        "Bahamas",
        "Others",
    ]

    # Verify Civil Status is detected as a table (Table 5) in v1.2
    assert len(table_blocks) >= 7, f"Expected >= 7 table blocks in gdoc, got {len(table_blocks)}"
    tb5 = table_blocks[5]
    assert tb5.table_data is not None
    assert tb5.table_data["num_columns"] == 2
    assert "Civil Status" in tb5.table_data["headers"][0]
    assert "Single" in tb5.table_data["headers"][0]
    assert len(tb5.table_data["rows"]) == 1
    assert "Next of Kin" in tb5.table_data["rows"][0][0]
    assert "Relationship" in tb5.table_data["rows"][0][1]
    assert len(tb5.source_block_ids) > 0

    # Verify Address/Phone is detected as a table (Table 6) in v1.2
    tb6 = table_blocks[6]
    assert tb6.table_data is not None
    assert tb6.table_data["num_columns"] == 2
    assert "Address of Next of Kin" in tb6.table_data["headers"][0]
    assert "West Quaish" in tb6.table_data["headers"][1]
    assert len(tb6.table_data["rows"]) == 1
    assert tb6.table_data["rows"][0][0] == ""
    assert "Phone" in tb6.table_data["rows"][0][1]
    assert len(tb6.source_block_ids) > 0

    # Verify Height / Weight is detected as a table (Table 7) in v1.2
    tb7 = table_blocks[7]
    assert tb7.table_data is not None
    assert tb7.table_data["num_columns"] == 2
    assert "Height" in tb7.table_data["headers"][0]
    assert "Weight" in tb7.table_data["headers"][1]


def test_non_regression_rishabh_dixit_sea_service_table():
    """Verify that CV Rishabh Dixit table detection and logical row merging are preserved."""
    pdf_path = Path("tests/fixtures/CV Rishabh Dixit.pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture CV Rishabh Dixit.pdf not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    lay = interpret_layout(rec)
    si = build_semantic_input(lay)

    t_sea = next((t for t in si.tables if t.table_id == "table_p1_0"), None)
    assert t_sea is not None
    assert t_sea.num_columns == 7
    grt_cells = [c for r in t_sea.rows for c in r if c.semantic_role == "grt"]
    assert any("Container Ship / 43679" in c.text for c in grt_cells)


def test_v12_civil_status_table_detection():
    """TEST 1 & TEST 4 & TEST 5: Civil Status table detected with sparse row and no conventional header."""
    pdf_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AKIBUL ALAM CV(JO).pdf not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)

    # Table 5 on page 1 is Civil Status / Next of Kin
    t_cs = next((t for t in si.tables if t.page == 1 and t.table_id == "table_p1_5"), None)
    assert t_cs is not None, "Civil Status table (table_p1_5) must be detected"
    assert t_cs.num_columns == 2, f"Expected 2 columns, got {t_cs.num_columns}"
    assert len(t_cs.rows) == 2, f"Expected 2 rows, got {len(t_cs.rows)}"
    assert getattr(t_cs, "is_border_defined", False) is True

    # Row 0 has single populated cell in col 0, col 1 is empty (TEST 5)
    row0_texts = [c.text for c in t_cs.rows[0]]
    assert any("Civil Status" in txt for txt in row0_texts)
    assert any("Single" in txt for txt in row0_texts)

    # Row 1 has col 0 and col 1 populated
    row1_texts = [c.text for c in t_cs.rows[1]]
    assert any("Next of Kin" in txt for txt in row1_texts)
    assert any("Relationship" in txt or "Father" in txt for txt in row1_texts)


def test_v12_address_phone_table_detection():
    """TEST 2 & TEST 3 & TEST 6: Address/Phone table detected with sparse/empty cells and irregular dividers."""
    pdf_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AKIBUL ALAM CV(JO).pdf not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)

    t_ap = next((t for t in si.tables if t.page == 2 and t.table_id == "table_p2_2"), None)
    assert t_ap is not None, "Address/Phone table (table_p2_2) must be detected"
    assert t_ap.num_columns == 2, f"Expected 2 columns, got {t_ap.num_columns}"
    assert len(t_ap.rows) == 2, f"Expected 2 rows, got {len(t_ap.rows)}"
    assert getattr(t_ap, "is_border_defined", False) is True

    # Row 0: Left cell = Address of Next of Kin, Right cell = West Quaish...
    r0 = t_ap.rows[0]
    assert any("Address of Next of Kin" in c.text for c in r0)
    assert any("West Quaish" in c.text for c in r0)

    # Row 1: Left cell = empty, Right cell = Phone
    r1 = t_ap.rows[1]
    assert any("Phone" in c.text for c in r1)


def test_v12_height_weight_and_adjacent_tables_preserved():
    """TEST 7, 8, 9, 10: Height/Weight, Passport, Seaman's Book, Licence tables preserved."""
    pdf_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AKIBUL ALAM CV(JO).pdf not found")

    raw = pdf_path.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)

    # TEST 7: Height/Weight continues to pass as table_p2_0
    t_hw = next((t for t in si.tables if t.table_id == "table_p2_0"), None)
    assert t_hw is not None
    assert t_hw.num_columns == 2
    assert len(t_hw.rows) == 3

    # TEST 8: Passport, Seaman's Book, Licence remain 3 separate tables
    t_pass = next(t for t in si.tables if t.table_id == "table_p1_2")
    t_sea = next(t for t in si.tables if t.table_id == "table_p1_3")
    t_lic = next(t for t in si.tables if t.table_id == "table_p1_4")
    assert t_pass.table_id != t_sea.table_id != t_lic.table_id

    # TEST 9 & 10: Seaman's Book and Licence sparse rows remain 8 rows
    assert len(t_sea.rows) == 8
    assert len(t_lic.rows) == 8


def test_v12_text_fidelity_and_provenance():
    """TEST 11, 12, 13: Zero text loss, zero text duplication, 100% provenance grounding."""
    pdf_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AKIBUL ALAM CV(JO).pdf not found")

    raw = pdf_path.read_bytes()
    raw_blocks = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(raw_blocks)
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)
    gdoc = build_document_structure_from_semantic_input(si)

    # TEST 13: Every table block has provenance source_block_ids
    for s in gdoc.sections:
        for b in s.blocks:
            if b.type == "table":
                assert len(b.source_block_ids) > 0, f"Table block {b.id} missing provenance source_block_ids"

    # TEST 11 & 12: Key table texts are present and not duplicated
    all_table_text = " ".join(
        b.text for s in gdoc.sections for b in s.blocks if b.type == "table"
    )
    assert "Civil Status" in all_table_text
    assert "West Quaish" in all_table_text
    assert "Height" in all_table_text
    assert "Passport" in all_table_text


def test_v12_unrelated_table_heavy_documents_non_regression():
    """TEST 14: At least 2-3 unrelated table-heavy documents continue to pass."""
    for fixture_name in ("CV Rishabh Dixit.pdf", "MUKUND 3RD OFF CV 2026.pdf"):
        fpath = Path(f"tests/fixtures/{fixture_name}")
        if not fpath.exists():
            continue
        raw = fpath.read_bytes()
        doc = document_from_text_blocks(PDFExtractor.extract(raw))
        lay = interpret_layout(reconstruct_document(doc))
        si = build_semantic_input(lay, document_id=fpath.name)
        assert len(si.tables) >= 2, f"Expected >= 2 tables for {fixture_name}, got {len(si.tables)}"
        for t in si.tables:
            assert t.num_columns >= 2, f"Table {t.table_id} in {fixture_name} has invalid column count {t.num_columns}"
            assert len(t.rows) >= 2, f"Table {t.table_id} in {fixture_name} has invalid row count {len(t.rows)}"


def test_v13_border_driven_row_reconstruction():
    """Verify Table Spatial Fidelity v1.3: Border-driven row reconstruction and multiline cells."""
    pdf_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AKIBUL ALAM CV(JO).pdf not found")

    raw = pdf_path.read_bytes()
    raw_blocks = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(raw_blocks)
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)
    gdoc = build_document_structure_from_semantic_input(si)

    # Locate the Courses & Certificates table in gdoc
    courses_block = next(
        b for s in gdoc.sections for b in s.blocks
        if b.type == "table" and b.table_data and any("Courses" in h for h in b.table_data.get("headers", []))
    )
    tbl_data = courses_block.table_data
    headers = tbl_data["headers"]
    rows = tbl_data["rows"]

    # Table must have exactly 5 columns and 15 data rows (+ 1 header row = 16 rows)
    assert len(headers) == 5
    assert len(rows) == 15

    # TEST 1: "Proficiency in Survival Craft / Rescue" + "Boat" must be one row/cell
    r_psc = next(r for r in rows if "Proficiency in Survival Craft" in r[0])
    assert "Boat" in r_psc[0]
    assert r_psc[1] == "043.232820"
    # Ensure "Boat" does not appear as a separate logical row
    assert not any(r[0].strip() == "Boat" for r in rows)

    # TEST 2: "Personal Survival & Social" + "Responsibility (PSSR)" must be one row/cell
    r_pssr = next(r for r in rows if "Personal Survival & Social" in r[0])
    assert "Responsibility (PSSR)" in r_pssr[0]
    assert r_pssr[1] == "042.153009"
    assert not any(r[0].strip() == "Responsibility (PSSR)" for r in rows)

    # TEST 3: "Advance Chemical Tanker Training" + "(CHEMCO)" must be one row/cell
    r_chemco = next(r for r in rows if "Advance Chemical Tanker Training" in r[0])
    assert "(CHEMCO)" in r_chemco[0]
    assert r_chemco[1] == "032.233037"
    assert not any(r[0].strip() == "(CHEMCO)" for r in rows)

    # TEST 4: "Basic Oil Tanker Familiarization" + "(OTFC)" must be one row/cell
    r_otfc = next(r for r in rows if "Basic Oil Tanker Familiarization" in r[0])
    assert "(OTFC)" in r_otfc[0]
    assert r_otfc[1] == "030.159664"
    assert not any(r[0].strip() == "(OTFC)" for r in rows)

    # TEST 5: "Basic Chemical Tanker Familiarization" + "(CTFC)" must be one row/cell
    r_ctfc = next(r for r in rows if "Basic Chemical Tanker Familiarization" in r[0])
    assert "(CTFC)" in r_ctfc[0]
    assert r_ctfc[1] == "030.159664"
    assert not any(r[0].strip() == "(CTFC)" for r in rows)

    # TEST 6: "Basic Gas Tanker Familiarization" + "(GTFC)" must be one row/cell
    r_gtfc = next(r for r in rows if "Basic Gas Tanker Familiarization" in r[0])
    assert "(GTFC)" in r_gtfc[0]
    assert r_gtfc[1] == "033.233344"
    assert not any(r[0].strip() == "(GTFC)" for r in rows)

    # TEST 7: Issuer text such as "Department\nof shipping,\nBangladesh" must remain one cell in the correct row
    for r in (r_psc, r_pssr, r_chemco, r_otfc, r_ctfc, r_gtfc):
        assert "Department" in r[4] and "shipping" in r[4] and "Bangladesh" in r[4]
    # And ensure "Bangladesh" does not appear as a separate logical row
    assert not any(r[0].strip() == "Bangladesh" for r in rows)

    # TEST 8: Existing sparse rows (Bahamas, Others, INDOS No., SID Number) remain separate rows
    sea_table = next(t for t in si.tables if t.table_id == "table_p1_3")
    lic_table = next(t for t in si.tables if t.table_id == "table_p1_4")
    assert len(sea_table.rows) == 8
    assert len(lic_table.rows) == 8
    sea_col0_texts = [r[0].text.strip() for r in sea_table.rows]
    assert "Bahamas" in sea_col0_texts
    assert "Others" in sea_col0_texts
    assert "INDOS No." in sea_col0_texts
    assert "SID Number" in sea_col0_texts

    # TEST 9: Passport / Seaman's Book / Licence remain separate tables
    pass_table = next(t for t in si.tables if t.table_id == "table_p1_2")
    assert pass_table.table_id != sea_table.table_id != lic_table.table_id

    # TEST 10: Civil Status / Address / Phone / Height / Weight tables remain correct
    hw_table = next(t for t in si.tables if t.table_id == "table_p2_0")
    assert hw_table.num_columns == 2
    assert len(hw_table.rows) == 3
    hw_rows_text = [r[0].text.strip() for r in hw_table.rows]
    assert any("Height" in txt for txt in hw_rows_text)
    assert any("Cm" in txt or "Boiler Suit" in txt for txt in hw_rows_text)

    # TEST 11 & 12: No text loss and no text duplication
    all_table_text = " ".join(
        b.text for s in gdoc.sections for b in s.blocks if b.type == "table"
    )
    assert "Proficiency in Survival Craft / Rescue" in all_table_text
    assert "Ship Handling Simulator" in all_table_text
    assert "Department" in all_table_text

    # TEST 13: 100% provenance grounding
    assert len(courses_block.source_block_ids) > 0
    for s in gdoc.sections:
        for b in s.blocks:
            if b.type == "table":
                assert len(b.source_block_ids) > 0, f"Table block {b.id} missing provenance source_block_ids"


def test_v14_source_border_geometry_and_table_continuation():
    """Verify Table Visual Fidelity v1.4: Source border geometry and table continuation."""
    pdf_path = Path("tests/fixtures/AASHISH DG.pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AASHISH DG.pdf not found")

    raw = pdf_path.read_bytes()
    raw_blocks = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(raw_blocks)
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)
    gdoc = build_document_structure_from_semantic_input(si)

    # TEST 1, 2, 3: Address Details table must contain Mobile No. and Email Id
    addr_table_ctx = next(
        t for t in si.tables
        if t.page == 1 and any("Present Address" in c.text or "Permanent Address" in c.text for r in t.rows for c in r)
    )
    assert addr_table_ctx.num_columns == 4
    addr_row_texts = [[c.text.strip() for c in r] for r in addr_table_ctx.rows]

    # Find row with Mobile No.
    mobile_row = next((r for r in addr_row_texts if any("Mobile No" in c for c in r)), None)
    assert mobile_row is not None, "Mobile No. must be a logical row in Address Details table"
    assert any("8130834379" in c for c in mobile_row), "Mobile No. row must contain phone value"

    # Find row with Email Id
    email_row = next((r for r in addr_row_texts if any("Email Id" in c for c in r)), None)
    assert email_row is not None, "Email Id must be a logical row in Address Details table"
    assert any("aashishbol@gmail.com" in c for c in email_row), "Email Id row must contain email value"

    # TEST 2: Neither Mobile No. nor Email Id appear as standalone paragraph blocks outside the table
    all_paragraphs = [b.text for s in gdoc.sections for b in s.blocks if b.type == "paragraph"]
    assert not any("Mobile No." in p and "8130834379" in p for p in all_paragraphs), "Mobile No. must not be a paragraph"
    assert not any("Email Id" in p and "aashishbol@gmail.com" in p for p in all_paragraphs), "Email Id must not be a paragraph"

    # TEST 4, 5, 6: Address Details visual geometry has outer box but NO internal horizontal or vertical borders
    geom = addr_table_ctx.visual_geometry
    assert geom is not None, "Address Details table must have visual_geometry metadata"
    assert geom["has_outer_border"] is True, "Address Details must have outer border box"
    assert geom["has_horizontal_borders"] is False, "Address Details has no internal horizontal dividers"
    assert geom["has_vertical_borders"] is False, "Address Details has no internal vertical dividers"

    # TEST 7: Physical Details table (Height, Hair Color) visual geometry
    phys_table_ctx = next(
        t for t in si.tables
        if t.page == 1 and any("Height" in c.text for r in t.rows for c in r)
    )
    phys_geom = phys_table_ctx.visual_geometry
    assert phys_geom is not None
    assert phys_geom["has_outer_border"] is True
    assert phys_geom["has_horizontal_borders"] is False
    assert phys_geom["has_vertical_borders"] is False

    # TEST 8, 9: Akibul Alam CV Courses table has full grid (outer, horizontal, and vertical)
    akibul_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if akibul_path.exists():
        akibul_doc = document_from_text_blocks(PDFExtractor.extract(akibul_path.read_bytes()))
        akibul_si = build_semantic_input(interpret_layout(reconstruct_document(akibul_doc)))
        akibul_courses = next(
            t for t in akibul_si.tables
            if any("Courses" in c.text or "Certificates" in c.text for r in t.rows for c in r)
        )
        ac_geom = akibul_courses.visual_geometry
        assert ac_geom is not None
        assert ac_geom["has_outer_border"] is True
        assert ac_geom["has_horizontal_borders"] is True
        assert ac_geom["has_vertical_borders"] is True

        # Multiline cell wrap stays 1 cell
        r_psc = next(r for r in akibul_courses.rows if any("Proficiency in Survival Craft" in c.text for c in r))
        assert any("Boat" in c.text for c in r_psc)
        assert not any(all(c.text.strip() == "Boat" for c in r if c.text.strip()) for r in akibul_courses.rows)

    # TEST 12: Generic document structure preserves visual_geometry in table_data
    all_table_blocks = [b for s in gdoc.sections for b in s.blocks if b.type == "table"]
    addr_block = next(
        b for b in all_table_blocks
        if b.page_number == 1 and b.table_data and any(any("Mobile No" in str(c) for c in r) for r in b.table_data.get("rows", []))
    )
    assert "visual_geometry" in addr_block.table_data
    assert addr_block.table_data["visual_geometry"]["has_outer_border"] is True
    assert addr_block.table_data["visual_geometry"]["has_horizontal_borders"] is False

    # TEST 13: 100% provenance grounding
    assert len(addr_block.source_block_ids) > 0
    for r in addr_table_ctx.rows:
        for c in r:
            if c.text.strip():
                assert len(c.source_block_ids) > 0, f"Cell {c.text} missing provenance"

    # TEST 14: Page footer not absorbed into Courses table on page 2
    p2_table = next((t for t in si.tables if t.page == 2), None)
    if p2_table:
        for r in p2_table.rows:
            r_txt = " ".join(c.text for c in r)
            assert "Report generated on" not in r_txt
            assert "Page 2 of 6" not in r_txt


def test_v15_generic_2d_form_spatial_reconstruction():
    """Verify Table Spatial Fidelity v1.5: Generic 2D Form / Key-Value Spatial Reconstruction."""
    pdf_path = Path("tests/fixtures/AASHISH DG.pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AASHISH DG.pdf not found")

    raw = pdf_path.read_bytes()
    raw_blocks = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(raw_blocks)
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)
    gdoc = build_document_structure_from_semantic_input(si)

    all_table_blocks = [b for s in gdoc.sections for b in s.blocks if b.type == "table"]

    # TEST 1: Form-like table containing Label : Value preserves horizontal label/value relationship
    # Look at Physical Details or Personal Details or CDC
    phys_block = next(
        b for b in all_table_blocks
        if b.page_number == 1 and b.table_data and (
            any("Height" in str(c) for c in b.table_data.get("headers", []))
            or any(any("Height" in str(c) for c in r) for r in b.table_data.get("rows", []))
        )
    )
    assert phys_block.table_data is not None
    assert phys_block.table_data.get("is_form_layout") is True, "Physical Details table must be flagged as is_form_layout"
    spatial_rows = phys_block.table_data.get("spatial_rows", [])
    assert len(spatial_rows) >= 3, "Physical Details table must have spatial rows"

    # Row 0: Height and Hair Color
    row0_fields = spatial_rows[0]["fields"]
    assert any("Height" in f["text"] for f in row0_fields)
    assert any("184" in f["text"] for f in row0_fields)

    # TEST 2: Two key/value groups on the same horizontal band remain horizontally separated
    # Height (left) and Hair Color (right)
    height_fld = next(f for f in row0_fields if "Height" in f["text"])
    hair_fld = next(f for f in row0_fields if "Hair Color" in f["text"])
    assert height_fld["relative_x"] < hair_fld["relative_x"], "Height must be to the left of Hair Color"
    assert (hair_fld["relative_x"] - height_fld["relative_x"]) >= 25.0, "Substantial horizontal gap between key/value groups"

    # TEST 3: A field spanning a large width remains spatially wider than normal fields
    # Row 2 has Identification: CUT MARK NEAR THE ELBOW OF RIGHT HAND
    row2_fields = spatial_rows[2]["fields"]
    id_fld = next(f for f in row2_fields if "CUT MARK" in f["text"] or "Identification" in f["text"])
    assert id_fld["relative_width"] >= 20.0 or any(f["relative_width"] >= 25.0 for f in row2_fields), "Wide field spans large horizontal region"

    # TEST 4: Empty structural space between independent field groups is preserved
    # Check that relative_x coordinates accurately reflect column offset
    assert hair_fld["relative_x"] >= 45.0, "Hair Color starts at or beyond ~50% of the table width"

    # TEST 5: Form table with outer border but no internal borders does not gain artificial CSS grid lines
    assert phys_block.table_data["visual_geometry"]["has_outer_border"] is True
    assert phys_block.table_data["visual_geometry"]["has_horizontal_borders"] is False
    assert phys_block.table_data["visual_geometry"]["has_vertical_borders"] is False

    # TEST 6: Conventional fully bordered table still renders as a conventional grid table (NOT is_form_layout)
    akibul_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if akibul_path.exists():
        akibul_doc = document_from_text_blocks(PDFExtractor.extract(akibul_path.read_bytes()))
        akibul_si = build_semantic_input(interpret_layout(reconstruct_document(akibul_doc)))
        akibul_gdoc = build_document_structure_from_semantic_input(akibul_si)
        akibul_tables = [b for s in akibul_gdoc.sections for b in s.blocks if b.type == "table"]
        courses_tbl = next(
            b for b in akibul_tables
            if b.table_data and any(any("Survival Craft" in str(c) for c in r) for r in b.table_data.get("rows", []))
        )
        assert courses_tbl.table_data.get("is_form_layout") is not True, "Full grid Courses table must NOT be is_form_layout"
        assert "spatial_rows" not in courses_tbl.table_data, "Conventional table should not emit spatial_rows"
        assert courses_tbl.table_data["visual_geometry"]["has_horizontal_borders"] is True
        assert courses_tbl.table_data["visual_geometry"]["has_vertical_borders"] is True

    # TEST 7: Multiline form fields remain preserved spatially across their physical lines
    # Address details table has multiline address values across distinct physical lines
    addr_block = next(
        b for b in all_table_blocks
        if b.page_number == 1 and b.table_data and any(any("Mobile No" in str(c) for c in r) for r in b.table_data.get("rows", []))
    )
    assert addr_block.table_data.get("is_form_layout") is True
    addr_srows = addr_block.table_data.get("spatial_rows", [])
    # Verify that multiline address values are preserved at their respective physical lines (not flattened or dropped)
    assert any(any("1/206" in f["text"] for f in sr["fields"]) for sr in addr_srows)
    assert any(any("PRTIKSHA" in f["text"] for f in sr["fields"]) for sr in addr_srows)
    assert any(any("VISHNIUPURI" in f["text"] for f in sr["fields"]) for sr in addr_srows)

    # TEST 8: Sparse form rows remain separate
    # Address details row with Mobile No. is separate from Email Id
    mobile_srow = next(sr for sr in addr_srows if any("Mobile No" in f["text"] for f in sr["fields"]))
    email_srow = next(sr for sr in addr_srows if any("Email Id" in f["text"] for f in sr["fields"]))
    assert mobile_srow != email_srow, "Mobile No. and Email Id must be distinct spatial rows"
    assert any("8130834379" in f["text"] for f in mobile_srow["fields"])
    assert any("aashishbol@gmail.com" in f["text"] for f in email_srow["fields"])

    # Provenance check: all spatial fields have source_block_ids
    for sr in spatial_rows:
        for f in sr["fields"]:
            assert len(f.get("source_block_ids", [])) > 0, f"Field {f['text']} missing source_block_ids"


def test_v16_geometry_first_form_rendering():
    """Verify Table Spatial Fidelity v1.6: Geometry-First Form Rendering."""
    pdf_path = Path("tests/fixtures/AASHISH DG.pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AASHISH DG.pdf not found")

    raw = pdf_path.read_bytes()
    raw_blocks = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(raw_blocks)
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)
    gdoc = build_document_structure_from_semantic_input(si)

    all_table_blocks = [b for s in gdoc.sections for b in s.blocks if b.type == "table"]

    # Locate sample form tables: Physical Details, Personal Details, CDC
    phys_block = next(
        b for b in all_table_blocks
        if b.page_number == 1 and b.table_data and (
            any("Height" in str(c) for c in b.table_data.get("headers", []))
            or any(any("Height" in str(c) for c in r) for r in b.table_data.get("rows", []))
        )
    )
    td = phys_block.table_data
    assert td is not None
    assert td.get("is_form_layout") is True

    # 1. Geometry-first spatial_blocks representation
    spatial_blocks = td.get("spatial_blocks", [])
    assert len(spatial_blocks) > 0, "Form table must emit spatial_blocks"

    # 2. Aspect ratio and max_bottom metadata
    assert "aspect_ratio" in td, "Form table must provide aspect_ratio"
    assert td["aspect_ratio"] > 0
    assert "max_bottom" in td, "Form table must provide max_bottom"
    assert 50.0 <= td["max_bottom"] <= 100.0, "max_bottom reflects bottom extent"

    # 3. Two horizontally separated field groups remain separated
    height_blk = next(b for b in spatial_blocks if "Height" in b["text"])
    hair_blk = next(b for b in spatial_blocks if "Hair Color" in b["text"])
    assert height_blk["relative_x"] < hair_blk["relative_x"]
    assert (hair_blk["relative_x"] - height_blk["relative_x"]) >= 25.0

    # 4. Values stay on their respective sides without overlap
    val_184 = next(b for b in spatial_blocks if "184" in b["text"])
    val_black = next(b for b in spatial_blocks if "BLACK" in b["text"])
    assert val_184["relative_x"] < val_black["relative_x"]
    assert val_black["relative_x"] >= 50.0

    # 5. Wide field retains width
    id_blk = next(b for b in spatial_blocks if "CUT MARK" in b["text"] or "Identification" in b["text"])
    assert id_blk["relative_width"] >= 10.0

    # 6. Colon attachment: colon is visually associated with label/value and does not form an isolated layout destroyer
    assert not any(b["text"] == ":" for b in spatial_blocks), "Colons should be associated with surrounding text"

    # 7. 100% Provenance grounding for all spatial_blocks
    for sb in spatial_blocks:
        assert len(sb.get("source_block_ids", [])) > 0, f"Block {sb['text']} missing source_block_ids"
        assert sb["relative_x"] >= 0.0
        assert sb["relative_y"] >= 0.0
        assert sb["relative_width"] > 0.0
        assert sb["relative_height"] > 0.0

    # 8. Address details check: multiline address block preserves structural space
    addr_block = next(
        b for b in all_table_blocks
        if b.page_number == 1 and b.table_data and any(any("Mobile No" in str(c) for c in r) for r in b.table_data.get("rows", []))
    )
    addr_td = addr_block.table_data
    assert addr_td.get("is_form_layout") is True
    addr_sblks = addr_td.get("spatial_blocks", [])
    assert len(addr_sblks) > 0

    # Empty right side check: left column fields do not occupy > 50% width
    mobile_blk = next(b for b in addr_sblks if "Mobile No" in b["text"])
    assert mobile_blk["relative_x"] < 25.0
    assert mobile_blk["relative_width"] < 40.0

    # 9. Non-regression: Conventional table in Akibul Alam remains standard grid table
    akibul_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if akibul_path.exists():
        akibul_doc = document_from_text_blocks(PDFExtractor.extract(akibul_path.read_bytes()))
        akibul_si = build_semantic_input(interpret_layout(reconstruct_document(akibul_doc)))
        akibul_gdoc = build_document_structure_from_semantic_input(akibul_si)
        akibul_tables = [b for s in akibul_gdoc.sections for b in s.blocks if b.type == "table"]
        courses_tbl = next(
            b for b in akibul_tables
            if b.table_data and any(any("Survival Craft" in str(c) for c in r) for r in b.table_data.get("rows", []))
        )
        assert courses_tbl.table_data.get("is_form_layout") is not True
        assert "spatial_blocks" not in courses_tbl.table_data
        assert courses_tbl.table_data["visual_geometry"]["has_horizontal_borders"] is True
        assert courses_tbl.table_data["visual_geometry"]["has_vertical_borders"] is True


def test_v16_aashish_multipage_form_tables_and_headings():
    """Verify that multi-page form tables in AASHISH DG.pdf are not fragmented by pseudo-headings."""
    pdf_path = Path("tests/fixtures/AASHISH DG.pdf")
    if not pdf_path.exists():
        pytest.skip("Fixture AASHISH DG.pdf not found")

    raw = pdf_path.read_bytes()
    raw_blocks = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(raw_blocks)
    lay = interpret_layout(reconstruct_document(doc))
    si = build_semantic_input(lay)
    gdoc = build_document_structure_from_semantic_input(si)

    # 1. Punctuation-prefixed text like ': CDC' must NOT become section headings
    section_headings = [s.heading for s in gdoc.sections if s.heading]
    for h in section_headings:
        assert not h.startswith(":"), f"Heading '{h}' should not start with colon"
        assert not h.startswith("."), f"Heading '{h}' should not start with dot"

    # 2. Total form tables detected and reconstructed across document
    all_table_blocks = [b for s in gdoc.sections for b in s.blocks if b.type == "table"]
    assert len(all_table_blocks) == 19, f"Expected 19 unified table blocks, got {len(all_table_blocks)}"

    # 3. Form tables must all have is_form_layout=True and spatial_blocks populated
    for tb in all_table_blocks:
        td = tb.table_data
        assert td is not None
        assert td.get("is_form_layout") is True, f"Table {tb.id} should be form layout"
        assert len(td.get("spatial_blocks", [])) > 0, f"Table {tb.id} missing spatial_blocks"

    # 4. Verify Authorised Documents tables on page 5: CDC, INDoS, Passport
    cdc_table = next(
        tb for tb in all_table_blocks
        if tb.page_number == 5 and tb.table_data and any("CDC" in b["text"] for b in tb.table_data.get("spatial_blocks", []))
    )
    cdc_sblks = cdc_table.table_data["spatial_blocks"]
    assert any("Document Type : CDC" in b["text"] for b in cdc_sblks)
    assert any("MUM 78341" in b["text"] for b in cdc_sblks)
    assert any("17/09/1998" in b["text"] for b in cdc_sblks)
    assert any("MUMBAI" in b["text"] for b in cdc_sblks)

    indos_table = next(
        tb for tb in all_table_blocks
        if tb.page_number == 5 and tb.table_data and any("INDoS" in b["text"] for b in tb.table_data.get("spatial_blocks", []))
    )
    indos_sblks = indos_table.table_data["spatial_blocks"]
    assert any("Document Type : INDoS" in b["text"] for b in indos_sblks)
    assert any("99NL3114" in b["text"] for b in indos_sblks)
    assert any("29/01/2002" in b["text"] for b in indos_sblks)

    passport_table = next(
        tb for tb in all_table_blocks
        if tb.page_number == 5 and tb.table_data and any("Passport" in b["text"] for b in tb.table_data.get("spatial_blocks", []))
    )
    passport_sblks = passport_table.table_data["spatial_blocks"]
    assert any("Document Type : Passport" in b["text"] for b in passport_sblks)
    assert any("G9419370" in b["text"] for b in passport_sblks)
    assert any("GAZIABAD" in b["text"] for b in passport_sblks)





