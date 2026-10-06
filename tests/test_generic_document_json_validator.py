"""Focused unit tests for Generic JSON Fidelity Diagnostics validator.

Covers:
1. Perfect JSON -> PASS
2. Missing source block -> FAIL
3. Duplicate source ownership -> FAIL
4. Unknown provenance ID -> FAIL
5. Unsupported block -> FAIL
6. Exact text mismatch -> FAIL
7. Empty table cells preserved -> PASS
8. Multiline table cell preserved -> PASS
9. List provenance -> PASS
10. Section hierarchy -> PASS
11. Legitimate many-source-blocks-to-one-table-cell -> PASS
12. Validator does not mutate inputs -> PASS
13. API response model contains json_fidelity -> PASS
"""

from __future__ import annotations

import copy
from typing import Any
import pytest

from app.domain.document_structure import (
    DocumentBlock,
    DocumentSection,
    DocumentStructure,
)
from app.pipeline.stages.generic_document_json import serialize_document_structure_to_json
from app.validation.generic_document_json_validator import (
    JsonFidelityReport,
    validate_document_json,
)
from app.api.v1.models import ParseResponse, ParseMetadata, ParseStatus


def _create_sample_doc() -> DocumentStructure:
    """Create a multi-section DocumentStructure with tables, lists, and fields."""
    return DocumentStructure(
        page_count=2,
        metadata={"archetype": "structured_form"},
        sections=[
            DocumentSection(
                heading=None,
                level=1,
                source_block_ids=["b_h0_1"],
                blocks=[
                    DocumentBlock(
                        type="paragraph",
                        text="Header metadata text",
                        source_block_ids=["b_h0_1"],
                        page_number=1,
                    )
                ],
            ),
            DocumentSection(
                heading="Personal Details",
                level=1,
                source_block_ids=["b_sec_1", "b_p1_1", "b_p1_2"],
                blocks=[
                    DocumentBlock(
                        type="paragraph",
                        text="Name: John Doe",
                        source_block_ids=["b_p1_1"],
                        page_number=1,
                    ),
                    DocumentBlock(
                        type="paragraph",
                        text="Standard paragraph description.",
                        source_block_ids=["b_p1_2"],
                        page_number=1,
                    ),
                ],
            ),
            DocumentSection(
                heading="Qualifications",
                level=1,
                source_block_ids=["b_sec_2", "b_l_1", "b_l_2"],
                blocks=[
                    DocumentBlock(
                        type="list_item",
                        text="• Bachelor of Science",
                        source_block_ids=["b_l_1"],
                        page_number=1,
                    ),
                    DocumentBlock(
                        type="list_item",
                        text="• Master of Science",
                        source_block_ids=["b_l_2"],
                        page_number=1,
                    ),
                ],
            ),
            DocumentSection(
                heading="Records & Schedule",
                level=1,
                source_block_ids=["b_sec_3", "b_t_1", "b_t_2", "b_t_3"],
                blocks=[
                    DocumentBlock(
                        type="table",
                        text="",
                        source_block_ids=["b_t_1", "b_t_2", "b_t_3"],
                        page_number=2,
                        table_data={
                            "headers": ["Course", "Status", "Notes"],
                            "rows": [
                                ["Safety Training", "Completed", "Valid"],
                                ["First Aid", "", "Renew\nRequired"],
                            ],
                            "num_columns": 3,
                            "visual_geometry": {
                                "outer_bounds": [30.0, 50.0, 500.0, 150.0],
                            },
                        },
                    )
                ],
            ),
        ],
    )


def test_perfect_json_passes():
    """1. Perfect JSON produced by serializer passes with 0 violations."""
    doc = _create_sample_doc()
    doc_json = serialize_document_structure_to_json(doc)

    report = validate_document_json(doc, doc_json)

    assert report.valid is True
    assert report.missing_blocks == 0
    assert report.duplicate_blocks == 0
    assert report.unsupported_blocks == 0
    assert report.provenance_errors == 0
    assert len(report.issues) == 0
    assert report.source_blocks == report.json_owned_blocks
    assert report["valid"] is True
    assert report.get("source_blocks") > 0


def test_missing_source_block_fails():
    """2. Omitting a source block from JSON results in FAIL."""
    doc = _create_sample_doc()
    doc_json = serialize_document_structure_to_json(doc)

    # Corrupt by removing one source block from a paragraph
    doc_json["document"]["sections"][1]["blocks"][0]["source_block_ids"] = []

    report = validate_document_json(doc, doc_json)

    assert report.valid is False
    assert report.missing_blocks >= 1
    assert any("MISSING_SOURCE_BLOCKS" in issue for issue in report.issues)


def test_duplicate_source_ownership_fails():
    """3. Claiming the same source block in two separate JSON blocks results in FAIL."""
    doc = _create_sample_doc()
    doc_json = serialize_document_structure_to_json(doc)

    # Claim block 'b_p1_1' in section 2 as well
    doc_json["document"]["sections"][2]["blocks"][0]["source_block_ids"].append("b_p1_1")

    report = validate_document_json(doc, doc_json)

    assert report.valid is False
    assert report.duplicate_blocks >= 1
    assert any("DUPLICATE_OWNERSHIP" in issue for issue in report.issues)


def test_unknown_provenance_id_fails():
    """4. Fabricated provenance ID not present in DocumentStructure results in FAIL."""
    doc = _create_sample_doc()
    doc_json = serialize_document_structure_to_json(doc)

    # Add an invented provenance ID
    doc_json["document"]["sections"][0]["blocks"][0]["source_block_ids"].append("b_invented_999")

    report = validate_document_json(doc, doc_json)

    assert report.valid is False
    assert report.provenance_errors >= 1
    assert any("UNKNOWN_PROVENANCE_ID" in issue for issue in report.issues)


def test_unsupported_block_fails():
    """5. An unsupported DocumentBlock type results in FAIL."""
    doc = _create_sample_doc()
    # Add an unsupported block type
    doc.sections[0].blocks.append(
        DocumentBlock(
            type="unsupported_3d_mesh",
            text="mesh data",
            source_block_ids=["b_mesh_1"],
            page_number=1,
        )
    )
    doc_json = serialize_document_structure_to_json(doc)

    report = validate_document_json(doc, doc_json)

    assert report.valid is False
    assert report.unsupported_blocks >= 1
    assert any("UNSUPPORTED_BLOCK_TYPE" in issue for issue in report.issues)


def test_exact_text_mismatch_fails():
    """6. Text paraphrasing or mismatch between DocumentStructure and JSON results in FAIL."""
    doc = _create_sample_doc()
    doc_json = serialize_document_structure_to_json(doc)

    # Paraphrase paragraph text
    doc_json["document"]["sections"][1]["blocks"][1]["text"] = "Altered paragraph summary."

    report = validate_document_json(doc, doc_json)

    assert report.valid is False
    assert any("TEXT_MISMATCH" in issue for issue in report.issues)


def test_empty_table_cells_preserved_passes():
    """7. Empty table cells (empty strings) are faithfully preserved and pass."""
    doc = _create_sample_doc()
    # Ensure there is an empty cell in the test doc
    assert doc.sections[3].blocks[0].table_data["rows"][1][1] == ""

    doc_json = serialize_document_structure_to_json(doc)
    # Verify empty string preserved in serialized JSON
    assert doc_json["document"]["sections"][3]["blocks"][0]["rows"][1][1] == ""

    report = validate_document_json(doc, doc_json)
    assert report.valid is True

    # If empty cell is altered (e.g. replaced with None or 'N/A')
    corrupted_json = copy.deepcopy(doc_json)
    corrupted_json["document"]["sections"][3]["blocks"][0]["rows"][1][1] = "N/A"
    corrupted_report = validate_document_json(doc, corrupted_json)
    assert corrupted_report.valid is False
    assert any("TABLE_ROWS_MISMATCH" in issue for issue in corrupted_report.issues)


def test_multiline_table_cell_preserved_passes():
    """8. Multiline table cells containing newlines are faithfully preserved and pass."""
    doc = _create_sample_doc()
    assert "\n" in doc.sections[3].blocks[0].table_data["rows"][1][2]

    doc_json = serialize_document_structure_to_json(doc)
    assert "\n" in doc_json["document"]["sections"][3]["blocks"][0]["rows"][1][2]

    report = validate_document_json(doc, doc_json)
    assert report.valid is True


def test_list_provenance_passes():
    """9. Grouped list items preserve granular provenance and pass."""
    doc = _create_sample_doc()
    doc_json = serialize_document_structure_to_json(doc)

    list_block = doc_json["document"]["sections"][2]["blocks"][0]
    assert list_block["type"] == "list"
    assert len(list_block["item_details"]) == 2
    assert list_block["item_details"][0]["source_block_ids"] == ["b_l_1"]
    assert list_block["item_details"][1]["source_block_ids"] == ["b_l_2"]

    report = validate_document_json(doc, doc_json)
    assert report.valid is True

    # Corrupt list item provenance
    corrupted_json = copy.deepcopy(doc_json)
    corrupted_json["document"]["sections"][2]["blocks"][0]["item_details"][0]["source_block_ids"] = ["b_l_corrupt"]
    corrupted_report = validate_document_json(doc, corrupted_json)
    assert corrupted_report.valid is False
    assert any("LIST_ITEM_PROVENANCE_MISMATCH" in issue for issue in corrupted_report.issues)


def test_section_hierarchy_passes():
    """10. Section hierarchies and levels are validated."""
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="Parent Section",
                level=1,
                source_block_ids=["b_p_1"],
                blocks=[
                    DocumentBlock(
                        type="paragraph",
                        text="Parent description",
                        source_block_ids=["b_p_1"],
                        page_number=1,
                    )
                ],
                subsections=[
                    DocumentSection(
                        heading="Child Subsection",
                        level=2,
                        source_block_ids=["b_c_1"],
                        blocks=[
                            DocumentBlock(
                                type="paragraph",
                                text="Child content",
                                source_block_ids=["b_c_1"],
                                page_number=1,
                            )
                        ],
                    )
                ],
            )
        ],
    )

    doc_json = serialize_document_structure_to_json(doc)
    report = validate_document_json(doc, doc_json)
    assert report.valid is True

    # Corrupt subsection heading level
    corrupted_json = copy.deepcopy(doc_json)
    corrupted_json["document"]["sections"][0]["subsections"][0]["level"] = 1
    corrupted_report = validate_document_json(doc, corrupted_json)
    assert corrupted_report.valid is False
    assert any("HEADING_LEVEL_MISMATCH" in issue for issue in corrupted_report.issues)


def test_many_source_blocks_to_one_table_cell():
    """11. Legitimate many-source-blocks represented by a table block pass ownership."""
    many_bids = [f"b_cell_src_{i}" for i in range(25)]
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="Data Grid",
                level=1,
                source_block_ids=["b_grid_h"] + many_bids,
                blocks=[
                    DocumentBlock(
                        type="table",
                        text="",
                        source_block_ids=many_bids,
                        page_number=1,
                        table_data={
                            "headers": ["Col A", "Col B"],
                            "rows": [["Merged content", "Value"]],
                            "num_columns": 2,
                        },
                    )
                ],
            )
        ],
    )

    doc_json = serialize_document_structure_to_json(doc)
    report = validate_document_json(doc, doc_json)

    assert report.valid is True
    assert report.source_blocks == 26  # 1 heading + 25 table blocks
    assert report.json_owned_blocks == 26
    assert report.missing_blocks == 0
    assert report.duplicate_blocks == 0


def test_validator_does_not_mutate_inputs():
    """12. Validator operates strictly read-only and does not mutate either input."""
    doc = _create_sample_doc()
    doc_json = serialize_document_structure_to_json(doc)

    doc_before = repr(doc)
    json_before = repr(doc_json)

    report = validate_document_json(doc, doc_json)
    assert report.valid is True

    assert repr(doc) == doc_before
    assert repr(doc_json) == json_before


def test_api_response_contains_json_fidelity():
    """13. ParseResponse model accepts and serializes json_fidelity."""
    resp = ParseResponse(
        success=True,
        status=ParseStatus.SUCCESS,
        document_structure={"sections": []},
        document_json={"document": {"pages": 1, "sections": []}},
        json_fidelity={
            "valid": True,
            "source_blocks": 611,
            "json_owned_blocks": 611,
            "missing_blocks": 0,
            "duplicate_blocks": 0,
            "unsupported_blocks": 0,
            "provenance_errors": 0,
            "issues": [],
        },
        metadata=ParseMetadata(
            filename="sample.pdf",
            pageCount=1,
            ocrUsed=False,
            provider="mock",
            model="mock",
            representation="compact",
            latencyMs=10.0,
            requestCount=1,
        ),
    )

    dumped = resp.model_dump()
    assert "json_fidelity" in dumped
    assert dumped["json_fidelity"]["valid"] is True
    assert dumped["json_fidelity"]["source_blocks"] == 611
    assert dumped["json_fidelity"]["json_owned_blocks"] == 611
    assert dumped["json_fidelity"]["missing_blocks"] == 0


def test_missing_source_text_fails():
    """14. When meaningful source text from table spatial blocks is dropped, validator flags MISSING_SOURCE_TEXT."""
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="ADDRESS",
                level=1,
                blocks=[
                    DocumentBlock(
                        type="table",
                        source_block_ids=["b1", "b2"],
                        table_data={
                            "is_form_layout": True,
                            "headers": [],
                            "rows": [["Address", "1/206 OLD NEW"]],
                            "spatial_blocks": [
                                {"text": "Address : 1/206 OLD NEW", "source_block_ids": ["b1"]},
                                {"text": ": PRTIKSHA", "source_block_ids": ["b2"]},
                            ],
                            "spatial_rows": [],
                        },
                    )
                ],
                source_block_ids=["h1", "b1", "b2"],
            )
        ],
    )
    # Serialized JSON intentionally omits PRTIKSHA
    corrupted_json = {
        "document": {
            "pages": 1,
            "sections": [
                {
                    "heading": "ADDRESS",
                    "level": 1,
                    "source_block_ids": ["h1", "b1", "b2"],
                    "blocks": [
                        {
                            "type": "table",
                            "headers": [],
                            "rows": [["Address", "1/206 OLD NEW"]],
                            "form_fields": [{"label": "Address", "value": "1/206 OLD NEW", "source_block_ids": ["b1", "b2"]}],
                            "source_block_ids": ["b1", "b2"],
                        }
                    ],
                }
            ],
        }
    }
    report = validate_document_json(doc, corrupted_json)
    assert report.valid is False
    assert any("MISSING_SOURCE_TEXT" in issue and "PRTIKSHA" in issue for issue in report.issues)


def test_standalone_colon_cell_fails():
    """15. Standalone colon ':' in a table cell fails validation."""
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="DATA",
                level=1,
                blocks=[
                    DocumentBlock(
                        type="table",
                        source_block_ids=["b1"],
                        table_data={"headers": ["A"], "rows": [[":"]]},
                    )
                ],
                source_block_ids=["h1", "b1"],
            )
        ],
    )
    doc_json = {
        "document": {
            "pages": 1,
            "sections": [
                {
                    "heading": "DATA",
                    "level": 1,
                    "source_block_ids": ["h1", "b1"],
                    "blocks": [
                        {"type": "table", "headers": ["A"], "rows": [[":"]], "source_block_ids": ["b1"]}
                    ],
                }
            ],
        }
    }
    report = validate_document_json(doc, doc_json)
    assert report.valid is False
    assert any("STANDALONE_COLON_CELL" in issue for issue in report.issues)


def test_standalone_colon_line_fails():
    """16. Standalone colon line inside multiline cell fails validation."""
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="DATA",
                level=1,
                blocks=[
                    DocumentBlock(
                        type="table",
                        source_block_ids=["b1"],
                        table_data={"headers": ["A"], "rows": [["Line 1\n:\nLine 2"]]},
                    )
                ],
                source_block_ids=["h1", "b1"],
            )
        ],
    )
    doc_json = {
        "document": {
            "pages": 1,
            "sections": [
                {
                    "heading": "DATA",
                    "level": 1,
                    "source_block_ids": ["h1", "b1"],
                    "blocks": [
                        {"type": "table", "headers": ["A"], "rows": [["Line 1\n:\nLine 2"]], "source_block_ids": ["b1"]}
                    ],
                }
            ],
        }
    }
    report = validate_document_json(doc, doc_json)
    assert report.valid is False
    assert any("STANDALONE_COLON_LINE" in issue for issue in report.issues)


def test_multiline_field_value_lost_fails():
    """17. Truncation of multiline table value in form_fields fails validation."""
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="DATA",
                level=1,
                blocks=[
                    DocumentBlock(
                        type="table",
                        source_block_ids=["b1"],
                        table_data={"is_form_layout": True, "headers": [], "rows": [["Street", "Line 1\nLine 2"]]},
                    )
                ],
                source_block_ids=["h1", "b1"],
            )
        ],
    )
    # Truncate value to single line
    corrupted_json = {
        "document": {
            "pages": 1,
            "sections": [
                {
                    "heading": "DATA",
                    "level": 1,
                    "source_block_ids": ["h1", "b1"],
                    "blocks": [
                        {
                            "type": "table",
                            "headers": [],
                            "rows": [["Street", "Line 1\nLine 2"]],
                            "form_fields": [{"label": "Street", "value": "Line 1", "source_block_ids": ["b1"]}],
                            "source_block_ids": ["b1"],
                        }
                    ],
                }
            ],
        }
    }
    report = validate_document_json(doc, corrupted_json)
    assert report.valid is False
    assert any("MULTILINE_FIELD_VALUE_LOST" in issue for issue in report.issues)


def test_records_form_fields_mismatch_fails():
    """18. Disagreement between records[0]['fields'] and form_fields fails validation."""
    doc = DocumentStructure(
        page_count=1,
        sections=[
            DocumentSection(
                heading="DATA",
                level=1,
                blocks=[
                    DocumentBlock(
                        type="table",
                        source_block_ids=["b1"],
                        table_data={"is_form_layout": True, "headers": [], "rows": [["Name", "John"]]},
                    )
                ],
                source_block_ids=["h1", "b1"],
            )
        ],
    )
    corrupted_json = {
        "document": {
            "pages": 1,
            "sections": [
                {
                    "heading": "DATA",
                    "level": 1,
                    "source_block_ids": ["h1", "b1"],
                    "blocks": [
                        {
                            "type": "table",
                            "headers": [],
                            "rows": [["Name", "John"]],
                            "form_fields": [{"label": "Name", "value": "John", "source_block_ids": ["b1"]}],
                            "records": [{"fields": [{"label": "Name", "value": "DIFFERENT", "source_block_ids": ["b1"]}]}],
                            "source_block_ids": ["b1"],
                        }
                    ],
                }
            ],
        }
    }
    report = validate_document_json(doc, corrupted_json)
    assert report.valid is False
    assert any("RECORDS_FORM_FIELDS_MISMATCH" in issue for issue in report.issues)


def test_aashish_and_akibul_pass_validation():
    """19. Live fixtures AASHISH DG.pdf and AKIBUL ALAM CV(JO).pdf pass complete validation."""
    from pathlib import Path
    from app.pipeline.parser import ResumeParser

    fixtures_dir = Path("tests/fixtures")
    for fname in ["AASHISH DG.pdf", "AKIBUL ALAM CV(JO).pdf"]:
        p = fixtures_dir / fname
        if not p.exists():
            continue
        ds = ResumeParser().parse_document_structure(p.read_bytes(), document_id=fname)
        doc_json = serialize_document_structure_to_json(ds)
        report = validate_document_json(ds, doc_json)

        assert report.valid is True, f"{fname} failed validation: {report.issues}"
        assert report.missing_blocks == 0
        assert report.duplicate_blocks == 0
        assert report.unsupported_blocks == 0
        assert report.provenance_errors == 0
        assert len(report.issues) == 0


def test_clean_data_json_validation_on_live_fixtures():
    """20. Live fixtures produce clean Data JSON passing complete fidelity validation with zero forbidden metadata."""
    from pathlib import Path
    from app.pipeline.parser import ResumeParser
    from app.pipeline.stages.generic_document_json import serialize_document_structure_to_data_json

    fixtures_dir = Path("tests/fixtures")
    for fname in ["AASHISH DG.pdf", "AKIBUL ALAM CV(JO).pdf"]:
        p = fixtures_dir / fname
        if not p.exists():
            continue
        ds = ResumeParser().parse_document_structure(p.read_bytes(), document_id=fname)
        clean_json = serialize_document_structure_to_data_json(ds)
        report = validate_document_json(ds, clean_json)

        assert report.valid is True, f"{fname} clean JSON failed validation: {report.issues}"
        assert report.missing_blocks == 0
        assert report.duplicate_blocks == 0
        assert report.unsupported_blocks == 0
        assert report.provenance_errors == 0
        assert len(report.issues) == 0


