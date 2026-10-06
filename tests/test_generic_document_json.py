"""Tests for Generic Document JSON Output v1.

Verifies:
1. Generic JSON serialization contract.
2. Every supported block type (paragraph, heading, field, record, table, list, text).
3. Table preservation (headers, rows, columns).
4. Empty table cells preservation.
5. Multiline table cells preservation.
6. Lists preservation.
7. Unheaded introductory content.
8. Nested/record content.
9. Provenance preservation.
10. Exact source text preservation.
11. AASHISH DG.pdf JSON generation.
12. AKIBUL ALAM CV(JO).pdf JSON generation.
13. No fixed resume heading dependency.
14. JSON serialization does not mutate DocumentStructure.
"""

from __future__ import annotations

import copy
from pathlib import Path
import pytest

from app.domain.document_structure import (
    DocumentBlock,
    DocumentSection,
    DocumentStructure,
)
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.generic_document_json import serialize_document_structure_to_json

FIXTURES_DIR = Path("tests/fixtures")


def test_1_generic_json_serialization_contract():
    """1. Top-level contract contains 'document' with 'pages' and 'sections'."""
    doc = DocumentStructure(
        page_count=2,
        sections=[
            DocumentSection(
                heading="ABOUT",
                level=1,
                blocks=[DocumentBlock(text="Some intro text.", type="paragraph", source_block_ids=["b0"])],
                source_block_ids=["h0", "b0"],
            )
        ],
        metadata={"archetype": "CV"},
    )
    result = serialize_document_structure_to_json(doc)

    assert "document" in result
    doc_json = result["document"]
    assert doc_json["pages"] == 2
    assert len(doc_json["sections"]) == 1
    assert doc_json["sections"][0]["heading"] == "ABOUT"
    assert doc_json["sections"][0]["level"] == 1
    assert doc_json["sections"][0]["source_block_ids"] == ["h0", "b0"]
    assert doc_json["metadata"]["archetype"] == "CV"


def test_2_every_supported_block_type():
    """2. Serializes paragraph, heading, field, record, table, list, and text."""
    blocks = [
        DocumentBlock(text="A regular paragraph.", type="paragraph", source_block_ids=["b_p"]),
        DocumentBlock(text="Subsection Heading", type="heading", metadata={"level": 2}, source_block_ids=["b_h"]),
        DocumentBlock(text="Status : Active Candidate", type="paragraph", source_block_ids=["b_f"]),
        DocumentBlock(
            type="table",
            table_data={
                "headers": ["Key", "Val"],
                "rows": [["K1", "V1"]],
                "num_columns": 2,
                "spatial_blocks": [
                    {"text": "Rank : Captain", "source_block_ids": ["b_r1"]},
                    {"text": "Nationality : Indian", "source_block_ids": ["b_r2"]},
                ],
            },
            source_block_ids=["tbl_1"],
        ),
        DocumentBlock(text="First list item", type="list_item", source_block_ids=["li_1"]),
        DocumentBlock(text="Second list item", type="list_item", source_block_ids=["li_2"]),
        DocumentBlock(text="Raw snippet text", type="text", source_block_ids=["t_1"]),
    ]
    doc = DocumentStructure(
        page_count=1,
        sections=[DocumentSection(heading="OVERVIEW", blocks=blocks, source_block_ids=["ov"])],
    )
    result = serialize_document_structure_to_json(doc)
    serialized_blocks = result["document"]["sections"][0]["blocks"]

    types = [b["type"] for b in serialized_blocks]
    assert "paragraph" in types
    assert "heading" in types
    assert "field" in types
    assert "table" in types
    assert "list" in types
    assert "text" in types

    # Check field block
    field_b = next(b for b in serialized_blocks if b["type"] == "field")
    assert field_b["label"] == "Status"
    assert field_b["value"] == "Active Candidate"

    # Check table block with record/fields
    tbl_b = next(b for b in serialized_blocks if b["type"] == "table")
    assert "records" in tbl_b
    assert tbl_b["records"][0]["fields"][0]["label"] == "Rank"
    assert tbl_b["records"][0]["fields"][0]["value"] == "Captain"


def test_3_table_preservation():
    """3. Table preserves exact headers, rows, columns, and geometry."""
    tbl_block = DocumentBlock(
        type="table",
        page_number=2,
        table_data={
            "headers": ["Course Name", "Certificate No", "Issue Date"],
            "rows": [
                ["Basic Safety", "BS-1029", "12/05/2018"],
                ["Advanced Fire Fighting", "AFF-902", "14/06/2019"],
            ],
            "num_columns": 3,
            "visual_geometry": {
                "outer_bounds": [30.5, 100.2, 550.0, 250.8],
            },
        },
        source_block_ids=["tbl_cert"],
    )
    doc = DocumentStructure(
        page_count=2,
        sections=[DocumentSection(heading="COURSES", blocks=[tbl_block])],
    )
    result = serialize_document_structure_to_json(doc)
    tbl_json = result["document"]["sections"][0]["blocks"][0]

    assert tbl_json["type"] == "table"
    assert tbl_json["headers"] == ["Course Name", "Certificate No", "Issue Date"]
    assert len(tbl_json["rows"]) == 2
    assert tbl_json["rows"][0] == ["Basic Safety", "BS-1029", "12/05/2018"]
    assert tbl_json["columns"] == 3
    assert tbl_json["geometry"]["page"] == 2
    assert tbl_json["geometry"]["x"] == 30.5
    assert tbl_json["geometry"]["y"] == 100.2
    assert tbl_json["geometry"]["width"] == round(550.0 - 30.5, 2)
    assert tbl_json["geometry"]["height"] == round(250.8 - 100.2, 2)


def test_4_empty_table_cells():
    """4. Empty table cells are preserved and never silently dropped."""
    tbl_block = DocumentBlock(
        type="table",
        table_data={
            "headers": ["Vessel", "Flag", "Sign On", "Sign Off"],
            "rows": [
                ["MT Ocean Star", "", "01/01/2020", ""],
                ["", "Liberia", "", "01/08/2021"],
            ],
            "num_columns": 4,
        },
        source_block_ids=["t0"],
    )
    doc = DocumentStructure(
        page_count=1,
        sections=[DocumentSection(heading="EXPERIENCE", blocks=[tbl_block])],
    )
    result = serialize_document_structure_to_json(doc)
    tbl_json = result["document"]["sections"][0]["blocks"][0]

    assert tbl_json["rows"][0] == ["MT Ocean Star", "", "01/01/2020", ""]
    assert tbl_json["rows"][1] == ["", "Liberia", "", "01/08/2021"]
    assert len(tbl_json["rows"][0]) == 4
    assert len(tbl_json["rows"][1]) == 4


def test_5_multiline_table_cells():
    """5. Multiline table cells with newlines are preserved verbatim."""
    multiline_cell = "Engine Room Department\nChief Engineer Watchkeeper\nSenior Officer"
    tbl_block = DocumentBlock(
        type="table",
        table_data={
            "headers": ["Role Description", "Code"],
            "rows": [
                [multiline_cell, "ENG-01"],
            ],
            "num_columns": 2,
        },
        source_block_ids=["t1"],
    )
    doc = DocumentStructure(
        page_count=1,
        sections=[DocumentSection(heading="SERVICE", blocks=[tbl_block])],
    )
    result = serialize_document_structure_to_json(doc)
    tbl_json = result["document"]["sections"][0]["blocks"][0]

    assert tbl_json["rows"][0][0] == multiline_cell
    assert "\n" in tbl_json["rows"][0][0]


def test_6_lists():
    """6. List items are grouped into a list block with items and provenance."""
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="CORE COMPETENCIES",
                blocks=[
                    DocumentBlock(text="Radar Navigation", type="list_item", source_block_ids=["li_1"]),
                    DocumentBlock(text="Bridge Team Management", type="list_item", source_block_ids=["li_2"]),
                    DocumentBlock(text="ECDIS Type Specific", type="list_item", source_block_ids=["li_3"]),
                ],
            )
        ],
    )
    result = serialize_document_structure_to_json(doc)
    blocks = result["document"]["sections"][0]["blocks"]

    assert len(blocks) == 1
    assert blocks[0]["type"] == "list"
    assert blocks[0]["items"] == ["Radar Navigation", "Bridge Team Management", "ECDIS Type Specific"]
    assert "li_1" in blocks[0]["source_block_ids"]
    assert "li_2" in blocks[0]["source_block_ids"]
    assert "li_3" in blocks[0]["source_block_ids"]


def test_7_unheaded_introductory_content():
    """7. Unheaded introductory content preserves heading: None and its blocks."""
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading=None,
                level=1,
                blocks=[
                    DocumentBlock(text="Application Form - Seafarer Division", type="paragraph", source_block_ids=["h0"]),
                    DocumentBlock(text="Generated : 2024-10-15", type="paragraph", source_block_ids=["h1"]),
                ],
                source_block_ids=["h0", "h1"],
            ),
            DocumentSection(
                heading="PERSONAL DETAILS",
                level=1,
                blocks=[DocumentBlock(text="John Doe", type="paragraph", source_block_ids=["b0"])],
                source_block_ids=["p0", "b0"],
            ),
        ],
    )
    result = serialize_document_structure_to_json(doc)
    sections = result["document"]["sections"]

    assert len(sections) == 2
    assert sections[0]["heading"] is None
    assert sections[0]["level"] == 1
    assert len(sections[0]["blocks"]) == 2
    assert sections[0]["blocks"][0]["text"] == "Application Form - Seafarer Division"
    assert sections[1]["heading"] == "PERSONAL DETAILS"


def test_8_nested_record_content():
    """8. Subsections and nested records are properly represented."""
    sub_sec = DocumentSection(
        heading="Tanker Endorsements",
        level=2,
        blocks=[
            DocumentBlock(text="OIL TANKER SPECIALIZATION", type="paragraph", source_block_ids=["sub_b1"])
        ],
        source_block_ids=["sub_h", "sub_b1"],
    )
    main_sec = DocumentSection(
        heading="CERTIFICATES",
        level=1,
        blocks=[DocumentBlock(text="Master Mariner Certificate", type="paragraph", source_block_ids=["m_b1"])],
        subsections=[sub_sec],
        source_block_ids=["m_h", "m_b1"],
    )
    doc = DocumentStructure(page_count=1, sections=[main_sec])
    result = serialize_document_structure_to_json(doc)
    sec_json = result["document"]["sections"][0]

    assert sec_json["heading"] == "CERTIFICATES"
    assert len(sec_json["subsections"]) == 1
    assert sec_json["subsections"][0]["heading"] == "Tanker Endorsements"
    assert sec_json["subsections"][0]["level"] == 2
    assert sec_json["subsections"][0]["blocks"][0]["text"] == "OIL TANKER SPECIALIZATION"


def test_9_provenance_preservation():
    """9. Source block IDs are faithfully preserved on every block type."""
    blocks = [
        DocumentBlock(text="Paragraph", type="paragraph", source_block_ids=["bid_p1", "bid_p2"]),
        DocumentBlock(text="City : Mumbai", type="paragraph", source_block_ids=["bid_f1"]),
        DocumentBlock(text="Item 1", type="list_item", source_block_ids=["bid_li1"]),
        DocumentBlock(type="table", table_data={"rows": [["Cell"]]}, source_block_ids=["bid_t1"]),
    ]
    doc = DocumentStructure(page_count=1, sections=[DocumentSection(heading="SEC", blocks=blocks, source_block_ids=["sec_h"])])
    result = serialize_document_structure_to_json(doc)
    s_blocks = result["document"]["sections"][0]["blocks"]

    assert s_blocks[0]["source_block_ids"] == ["bid_p1", "bid_p2"]
    assert s_blocks[1]["source_block_ids"] == ["bid_f1"]
    assert s_blocks[2]["source_block_ids"] == ["bid_li1"]
    assert s_blocks[3]["source_block_ids"] == ["bid_t1"]


def test_10_exact_source_text_preservation():
    """10. Text is never normalized, lowercase-folded, or spell-checked."""
    weird_text = "   PrOtOcOl   v1.0a   [SPECIALL!!]  --  w/   newlines\n\n\tand   tabs   "
    doc = DocumentStructure(
        page_count=1,
        sections=[DocumentSection(heading="RAW", blocks=[DocumentBlock(text=weird_text, type="paragraph", source_block_ids=["w0"])])],
    )
    result = serialize_document_structure_to_json(doc)

    assert result["document"]["sections"][0]["blocks"][0]["text"] == weird_text


def test_11_aashish_json_generation():
    """11. AASHISH DG.pdf generates populated generic JSON with all authentic sections and tables."""
    fixture_path = FIXTURES_DIR / "AASHISH DG.pdf"
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    ds = ResumeParser().parse_document_structure(fixture_path.read_bytes(), document_id=fixture_path.name)
    result = serialize_document_structure_to_json(ds)

    doc_json = result["document"]
    assert doc_json["pages"] == 6
    headings = [s["heading"] for s in doc_json["sections"]]

    # Authentic headings must be present
    assert "Seafarer Profile" in headings
    assert "Personal Details" in headings
    assert "Address Details" in headings
    assert "Physical Details" in headings
    assert "Modular Course Details" in headings
    assert "Sea Going Service" in headings
    assert "Certificate of Endorsement" in headings
    assert "Certificate of Competency" in headings
    assert "Authorised Documents" in headings
    assert "DECLARATION TO BE MADE BY CANDIDATE :" in headings

    # Table values must NOT become section headings
    assert "JAWAHAR LAL NEHRU" not in headings
    assert "NANGA PARBAT" not in headings
    assert "ANNAPURNA" not in headings
    assert "DC ENDORSEMENT NAUTICAL" not in headings

    # Verify Personal Details section contains table with form_fields
    personal_sec = next(s for s in doc_json["sections"] if s["heading"] == "Personal Details")
    assert any(b["type"] == "table" for b in personal_sec["blocks"])
    tbl = next(b for b in personal_sec["blocks"] if b["type"] == "table")
    assert "form_fields" in tbl
    labels = [f["label"] for f in tbl["form_fields"]]
    assert "Name" in labels or "INDoS No." in labels


def test_12_akibul_json_generation():
    """12. AKIBUL ALAM CV(JO).pdf produces clean generic JSON without regression."""
    fixture_path = FIXTURES_DIR / "AKIBUL ALAM CV(JO).pdf"
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    ds = ResumeParser().parse_document_structure(fixture_path.read_bytes(), document_id=fixture_path.name)
    result = serialize_document_structure_to_json(ds)

    doc_json = result["document"]
    assert doc_json["pages"] == 4
    headings = [s["heading"] for s in doc_json["sections"]]
    assert "PERSONAL DATA." in headings

    # Tables preserved
    p_sec = next(s for s in doc_json["sections"] if s["heading"] == "PERSONAL DATA.")
    tables = [b for b in p_sec["blocks"] if b["type"] == "table"]
    assert len(tables) >= 10


def test_13_no_fixed_resume_heading_dependency():
    """13. Arbitrary/unknown headings serialize cleanly without schema rejection."""
    arbitrary_headings = [
        "METALLURGICAL ASSAYS",
        "HYDROLOGICAL SAMPLES",
        "GEOPHYSICAL LOGS",
        "BOREHOLE CORE RECORDS",
    ]
    sections = [
        DocumentSection(heading=h, level=1, blocks=[DocumentBlock(text=f"Data for {h}", type="paragraph")])
        for h in arbitrary_headings
    ]
    doc = DocumentStructure(page_count=3, sections=sections)
    result = serialize_document_structure_to_json(doc)

    serialized_headings = [s["heading"] for s in result["document"]["sections"]]
    assert serialized_headings == arbitrary_headings


def test_14_json_serialization_does_not_mutate_document_structure():
    """14. Invariant: serialize_document_structure_to_json must not mutate DocumentStructure."""
    doc = DocumentStructure(
        page_count=2,
        sections=[
            DocumentSection(
                heading="ORIGINAL HEADING",
                level=1,
                blocks=[
                    DocumentBlock(text="Field : Value", type="paragraph", source_block_ids=["b1"]),
                    DocumentBlock(type="table", table_data={"headers": ["A"], "rows": [["1"]]}, source_block_ids=["t1"]),
                ],
                source_block_ids=["h1", "b1", "t1"],
            )
        ],
        metadata={"custom": "value"},
    )
    doc_copy = copy.deepcopy(doc)
    doc_dump_before = doc.model_dump()

    result = serialize_document_structure_to_json(doc)

    assert doc.model_dump() == doc_dump_before
    assert doc == doc_copy


def test_15_multiline_form_field_preservation_aashish():
    """15. AASHISH DG.pdf Address Details preserves multiline fields (PRTIKSHA, VISHNIUPURI) and canonical rows."""
    fixture_path = FIXTURES_DIR / "AASHISH DG.pdf"
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found")

    ds = ResumeParser().parse_document_structure(fixture_path.read_bytes(), document_id=fixture_path.name)
    result = serialize_document_structure_to_json(ds)

    addr_sec = next(s for s in result["document"]["sections"] if s["heading"] == "Address Details")
    tbl = next(b for b in addr_sec["blocks"] if b["type"] == "table")

    assert "form_fields" in tbl
    fields = tbl["form_fields"]
    assert len(fields) == 14

    addr_present = next(f for f in fields if f["label"] == "Address" and f["value"])
    # 1. Multiline preserved with newline
    assert "\n" in addr_present["value"]
    # 2. Meaningful tokens retained
    assert "1/206 OLD 1/366 NEW" in addr_present["value"]
    assert "PRTIKSHA" in addr_present["value"]
    assert "VISHNIUPURI" in addr_present["value"]
    assert addr_present["value"] == "1/206 OLD 1/366 NEW\nPRTIKSHA\nVISHNIUPURI"

    # 3. No standalone colon values in form_fields
    for f in fields:
        assert f["value"] != ":"
        assert not f["value"].startswith(": ")

    # 4. Canonical agreement between rows, form_fields, and records
    assert tbl["records"][0]["fields"] == fields
    assert tbl["rows"][0][0] == "Address"
    assert tbl["rows"][0][1] == "1/206 OLD 1/366 NEW\nPRTIKSHA\nVISHNIUPURI"
    assert tbl["rows"][0][2] == "Address"
    assert tbl["rows"][0][3] == ""


def test_16_synthetic_form_table_multiline_continuation():
    """16. Synthetic form table with continuation lines merges correctly and drops separator colons."""
    spatial_rows = [
        {
            "fields": [
                {"text": "Ship Name : MV PACIFIC", "relative_x": 10.0, "source_block_ids": ["b1"]},
                {"text": "Port : SINGAPORE", "relative_x": 50.0, "source_block_ids": ["b2"]},
            ]
        },
        {
            "fields": [
                {"text": ": VOYAGE 42", "relative_x": 10.0, "source_block_ids": ["b3"]},
                {"text": ": TERMINAL 3", "relative_x": 50.0, "source_block_ids": ["b4"]},
            ]
        },
    ]
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="VOYAGES",
                level=1,
                blocks=[
                    DocumentBlock(
                        type="table",
                        source_block_ids=["b1", "b2", "b3", "b4"],
                        table_data={
                            "is_form_layout": True,
                            "headers": ["Ship Name", "MV PACIFIC\n:\nVOYAGE 42", "Port", "SINGAPORE\n:\nTERMINAL 3"],
                            "rows": [["Ship Name", "MV PACIFIC\n:\nVOYAGE 42", "Port", "SINGAPORE\n:\nTERMINAL 3"]],
                            "spatial_rows": spatial_rows,
                        },
                    )
                ],
                source_block_ids=["h1", "b1", "b2", "b3", "b4"],
            )
        ],
    )
    result = serialize_document_structure_to_json(doc)
    tbl = result["document"]["sections"][0]["blocks"][0]

    assert len(tbl["form_fields"]) == 2
    assert tbl["form_fields"][0]["label"] == "Ship Name"
    assert tbl["form_fields"][0]["value"] == "MV PACIFIC\nVOYAGE 42"
    assert tbl["form_fields"][0]["source_block_ids"] == ["b1", "b3"]

    assert tbl["form_fields"][1]["label"] == "Port"
    assert tbl["form_fields"][1]["value"] == "SINGAPORE\nTERMINAL 3"
    assert tbl["form_fields"][1]["source_block_ids"] == ["b2", "b4"]

    # Canonical rows have separator colons cleaned
    assert tbl["rows"][0][1] == "MV PACIFIC\nVOYAGE 42"
    assert tbl["rows"][0][3] == "SINGAPORE\nTERMINAL 3"


def test_17_colons_in_legitimate_content_preserved():
    """17. Colons inside timestamps, ratios, or text are never globally deleted."""
    spatial_rows = [
        {
            "fields": [
                {"text": "Departure Time : 14:30:00", "relative_x": 10.0, "source_block_ids": ["b1"]},
                {"text": "Fuel Ratio : 1:4", "relative_x": 50.0, "source_block_ids": ["b2"]},
            ]
        }
    ]
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="LOGS",
                level=1,
                blocks=[
                    DocumentBlock(
                        type="table",
                        source_block_ids=["b1", "b2"],
                        table_data={
                            "is_form_layout": True,
                            "headers": [],
                            "rows": [["Departure Time", "14:30:00", "Fuel Ratio", "1:4"]],
                            "spatial_rows": spatial_rows,
                        },
                    )
                ],
                source_block_ids=["h1", "b1", "b2"],
            )
        ],
    )
    result = serialize_document_structure_to_json(doc)
    tbl = result["document"]["sections"][0]["blocks"][0]

    assert tbl["form_fields"][0]["value"] == "14:30:00"
    assert tbl["form_fields"][1]["value"] == "1:4"
    assert tbl["rows"][0][1] == "14:30:00"
    assert tbl["rows"][0][3] == "1:4"


def test_18_clean_data_json_serialization_no_forbidden_metadata():
    """Verify clean Data JSON contains NO source_block_ids, geometry, or block-level page."""
    from app.pipeline.stages.generic_document_json import serialize_document_structure_to_data_json
    from app.validation.generic_document_json_validator import validate_document_json

    doc = DocumentStructure(
        page_count=2,
        metadata={"archetype": "CV"},
        sections=[
            DocumentSection(
                heading="PERSONAL",
                level=1,
                source_block_ids=["h1", "b1", "b2"],
                blocks=[
                    DocumentBlock(
                        type="paragraph",
                        text="Name : John Doe",
                        source_block_ids=["b1"],
                        page_number=1,
                    ),
                    DocumentBlock(
                        type="table",
                        source_block_ids=["b2"],
                        table_data={
                            "is_form_layout": True,
                            "headers": ["Key", "Val"],
                            "rows": [["Address", "Line 1\nLine 2"]],
                            "num_columns": 2,
                            "spatial_rows": [
                                {
                                    "fields": [
                                        {"text": "Address : Line 1\nLine 2", "relative_x": 0.0, "source_block_ids": ["b2"]}
                                    ]
                                }
                            ],
                        },
                    ),
                ],
            )
        ],
    )
    clean_json = serialize_document_structure_to_data_json(doc)
    assert "document" in clean_json
    doc_dict = clean_json["document"]
    assert doc_dict["pages"] == 2
    sec = doc_dict["sections"][0]
    assert sec["heading"] == "PERSONAL"
    # Sections have NO source_block_ids or geometry or page
    assert "source_block_ids" not in sec
    assert "geometry" not in sec
    assert "page" not in sec

    # Blocks have NO source_block_ids, geometry, page
    field_blk = sec["blocks"][0]
    assert field_blk["type"] == "field"
    assert field_blk["label"] == "Name"
    assert field_blk["value"] == "John Doe"
    assert "source_block_ids" not in field_blk
    assert "geometry" not in field_blk
    assert "page" not in field_blk

    tbl_blk = sec["blocks"][1]
    assert tbl_blk["type"] == "table"
    assert "source_block_ids" not in tbl_blk
    assert "records" not in tbl_blk
    assert tbl_blk["form_fields"][0]["label"] == "Address"
    assert tbl_blk["form_fields"][0]["value"] == "Line 1\nLine 2"
    assert "source_block_ids" not in tbl_blk["form_fields"][0]

    # Validate passes fidelity check
    rep = validate_document_json(doc, clean_json)
    assert rep.valid is True
    assert rep.missing_blocks == 0
    assert rep.duplicate_blocks == 0
    assert rep.provenance_errors == 0

