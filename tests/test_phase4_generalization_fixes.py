"""Phase 4 generalization fix tests:
1. optional experience cross-entity location clearing
2. valid same-entity location remains
3. cross-entity provenance protection (company/designation/dates not silently accepted)
4. multi-row variable-height table continuation
5. wrapped cells in later table rows
6. stable column ownership across wrapped lines
7. adjacent-column text is not absorbed into another column
8. consulting-like table structure handled generically
9. provenance remains strict
10. arbitrary unsupported semantic additions still fail
"""

from __future__ import annotations

from pathlib import Path
import pytest

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedString,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    build_semantic_input,
    repair_semantic_output_provenance,
    validate_semantic_output,
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
    reading_order: int = 1,
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        text=text,
        page=page,
        bbox=[x0, y0, x1, y1],
        region_id=f"page-{page}-region-0",
        region_kind="physical_region",
        reading_order=reading_order,
        suggested_role=role,
    )


# =====================================================================
# 1-3. Optional Experience Subfield Cross-Entity Sanitization Tests
# =====================================================================


def test_experience_cross_entity_location_cleared_when_no_in_span_evidence():
    """When LLM cites location from a different experience span and current span has no location,
    the optional location field and its provenance are cleared deterministically.
    """
    blocks = [
        # Span 0: Experience 0 (Page 1)
        _make_block("b_p1_0", "Senior Research Fellow", 50.0, 100.0, 250.0, 115.0, role="ENTRY_TITLE", reading_order=1),
        _make_block("b_p1_1", "Stanford University", 50.0, 120.0, 200.0, 135.0, role="ORGANIZATION", reading_order=2),
        _make_block("b_p1_2", "Stanford, CA", 50.0, 140.0, 150.0, 155.0, role="LOCATION", reading_order=3),
        # Span 1: Experience 1 (Page 2) - contains title, company, but NO location
        _make_block("b_p2_10", "Teaching Assistant", 50.0, 100.0, 200.0, 115.0, page=2, role="ENTRY_TITLE", reading_order=10),
        _make_block("b_p2_11", "MIT", 50.0, 120.0, 100.0, 135.0, page=2, role="ORGANIZATION", reading_order=11),
    ]
    sem_input = SemanticInput(
        document_id="test_doc",
        page_count=2,
        archetype=DocumentArchetype.ACADEMIC_CV,
        blocks=blocks,
    )

    output = SemanticOutput(
        document_archetype=DocumentArchetype.ACADEMIC_CV,
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Senior Research Fellow", source_block_ids=["b_p1_0"]),
                company=GroundedString(value="Stanford University", source_block_ids=["b_p1_1"]),
                location=GroundedString(value="Stanford, CA", source_block_ids=["b_p1_2"]),
                source_block_ids=["b_p1_0", "b_p1_1", "b_p1_2"],
            ),
            GroundedExperienceItem(
                designation=GroundedString(value="Teaching Assistant", source_block_ids=["b_p2_10"]),
                company=GroundedString(value="MIT", source_block_ids=["b_p2_11"]),
                # LLM cross-cited b_p1_2 from span 0!
                location=GroundedString(value="Stanford, CA", source_block_ids=["b_p1_2"]),
                source_block_ids=["b_p2_10", "b_p2_11", "b_p1_2"],
            ),
        ],
    )

    repaired, repairs = repair_semantic_output_provenance(output, sem_input)

    # Location on experience[1] must be cleared
    assert repaired.experience[1].location is None
    # Cross-entity block b_p1_2 must be pruned from experience[1].source_block_ids
    assert "b_p1_2" not in repaired.experience[1].source_block_ids
    assert repaired.experience[1].source_block_ids == ["b_p2_10", "b_p2_11"]

    # Location on experience[0] must remain intact
    assert repaired.experience[0].location is not None
    assert repaired.experience[0].location.value == "Stanford, CA"
    assert repaired.experience[0].location.source_block_ids == ["b_p1_2"]

    # Semantic validation must pass without CROSS_ENTITY_PROVENANCE violations
    violations = validate_semantic_output(repaired, sem_input)
    assert not any("CROSS_ENTITY_PROVENANCE" in v for v in violations)


def test_experience_same_entity_location_remains_intact():
    """Valid same-entity location remains untouched."""
    blocks = [
        _make_block("b_p1_0", "Lead Architect", 50.0, 100.0, 200.0, 115.0, role="ENTRY_TITLE", reading_order=1),
        _make_block("b_p1_1", "Google LLC", 50.0, 120.0, 150.0, 135.0, role="ORGANIZATION", reading_order=2),
        _make_block("b_p1_2", "Mountain View, CA", 50.0, 140.0, 200.0, 155.0, role="LOCATION", reading_order=3),
    ]
    sem_input = SemanticInput(
        document_id="test_doc",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        blocks=blocks,
    )

    output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Lead Architect", source_block_ids=["b_p1_0"]),
                company=GroundedString(value="Google LLC", source_block_ids=["b_p1_1"]),
                location=GroundedString(value="Mountain View, CA", source_block_ids=["b_p1_2"]),
                source_block_ids=["b_p1_0", "b_p1_1", "b_p1_2"],
            ),
        ],
    )

    repaired, repairs = repair_semantic_output_provenance(output, sem_input)
    assert repaired.experience[0].location is not None
    assert repaired.experience[0].location.value == "Mountain View, CA"
    assert repaired.experience[0].location.source_block_ids == ["b_p1_2"]

    violations = validate_semantic_output(repaired, sem_input)
    assert not violations


def test_experience_cross_entity_location_rebound_when_unique_in_span_candidate_exists():
    """When LLM cites cross-entity location, but a unique valid location block exists in the same span,
    it rebinds to the in-span block.
    """
    blocks = [
        # Span 0
        _make_block("b_p1_0", "Engineer 1", 50.0, 100.0, 150.0, 115.0, role="ENTRY_TITLE", reading_order=1),
        _make_block("b_p1_1", "Austin, TX", 50.0, 120.0, 150.0, 135.0, role="LOCATION", reading_order=2),
        # Span 1
        _make_block("b_p1_10", "Engineer 2", 50.0, 200.0, 150.0, 215.0, role="ENTRY_TITLE", reading_order=10),
        _make_block("b_p1_11", "Seattle, WA", 50.0, 220.0, 150.0, 235.0, role="LOCATION", reading_order=11),
    ]
    sem_input = SemanticInput(
        document_id="test_doc",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        blocks=blocks,
    )

    output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Engineer 1", source_block_ids=["b_p1_0"]),
                location=GroundedString(value="Austin, TX", source_block_ids=["b_p1_1"]),
                source_block_ids=["b_p1_0", "b_p1_1"],
            ),
            GroundedExperienceItem(
                designation=GroundedString(value="Engineer 2", source_block_ids=["b_p1_10"]),
                # LLM cited b_p1_1 from span 0, but emitted Seattle, WA which matches b_p1_11 in span 1!
                location=GroundedString(value="Seattle, WA", source_block_ids=["b_p1_1"]),
                source_block_ids=["b_p1_10", "b_p1_1"],
            ),
        ],
    )

    repaired, repairs = repair_semantic_output_provenance(output, sem_input)
    assert repaired.experience[1].location is not None
    assert repaired.experience[1].location.value == "Seattle, WA"
    # Rebound to b_p1_11
    assert repaired.experience[1].location.source_block_ids == ["b_p1_11"]
    assert "b_p1_1" not in repaired.experience[1].source_block_ids
    assert "b_p1_11" in repaired.experience[1].source_block_ids


def test_experience_cross_entity_core_fields_not_cleared():
    """Core experience fields (company, designation, startDate, endDate) must NEVER be silently cleared
    when cited across entities; they must strictly fail validation.
    """
    blocks = [
        _make_block("b_sh", "WORK EXPERIENCE", 50.0, 50.0, 200.0, 65.0, role="SECTION_HEADING", reading_order=0),
        # Span 0
        _make_block("b_p1_0", "Software Engineer", 50.0, 100.0, 200.0, 115.0, role="ENTRY_TITLE", reading_order=1),
        _make_block("b_p1_1", "Acme Inc", 50.0, 120.0, 150.0, 135.0, role="ORGANIZATION", reading_order=2),
        # Span 1
        _make_block("b_p1_10", "Engineering Manager", 50.0, 200.0, 200.0, 215.0, role="ENTRY_TITLE", reading_order=10),
        _make_block("b_p1_11", "Beta LLC", 50.0, 220.0, 150.0, 235.0, role="ORGANIZATION", reading_order=11),
    ]
    sem_input = SemanticInput(
        document_id="test_doc",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        blocks=blocks,
    )

    output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Software Engineer", source_block_ids=["b_p1_0"]),
                company=GroundedString(value="Acme Inc", source_block_ids=["b_p1_1"]),
                source_block_ids=["b_p1_0", "b_p1_1"],
            ),
            GroundedExperienceItem(
                # Cross-entity designation cited from span 0 on an appointment that has its own title b_p1_10!
                designation=GroundedString(value="Software Engineer", source_block_ids=["b_p1_0"]),
                company=GroundedString(value="Beta LLC", source_block_ids=["b_p1_11"]),
                source_block_ids=["b_p1_0", "b_p1_10", "b_p1_11"],
            ),
        ],
    )

    repaired, repairs = repair_semantic_output_provenance(output, sem_input)
    # Core field must NOT be cleared
    assert repaired.experience[1].designation is not None
    assert repaired.experience[1].designation.value == "Software Engineer"

    # Validation must detect CROSS_ENTITY_PROVENANCE
    violations = validate_semantic_output(repaired, sem_input)
    assert any("CROSS_ENTITY_PROVENANCE in experience[1].designation" in v for v in violations)


# =====================================================================
# 4-8. Multi-Row Variable-Height Table Continuation Tests
# =====================================================================


def test_multi_row_variable_height_table_continuation():
    """Multi-row table with variable-height rows and inter-row spacing > 25pt
    continues across all rows based on established consensus column geometry.
    """
    binder = GeometricTableBinder()
    blocks = [
        # Row 0: Header (4 columns: Col 0: 50..120, Col 1: 150..220, Col 2: 250..320, Col 3: 350..420)
        _make_block("h0", "Col A", 50.0, 100.0, 120.0, 115.0),
        _make_block("h1", "Col B", 150.0, 100.0, 220.0, 115.0),
        _make_block("h2", "Col C", 250.0, 100.0, 320.0, 115.0),
        _make_block("h3", "Col D", 350.0, 100.0, 420.0, 115.0),
        # Row 1: Data row 1 (tight gap 10pt from header)
        _make_block("r1_0", "Val A1", 50.0, 125.0, 120.0, 140.0),
        _make_block("r1_1", "Val B1", 150.0, 125.0, 220.0, 140.0),
        _make_block("r1_2", "Val C1", 250.0, 125.0, 320.0, 140.0),
        _make_block("r1_3", "Val D1", 350.0, 125.0, 420.0, 140.0),
        # Row 2: Data row 2 (large row spacing 35pt > 25pt)
        _make_block("r2_0", "Val A2", 50.0, 175.0, 120.0, 190.0),
        _make_block("r2_1", "Val B2", 150.0, 175.0, 220.0, 190.0),
        _make_block("r2_2", "Val C2", 250.0, 175.0, 320.0, 190.0),
        _make_block("r2_3", "Val D2", 350.0, 175.0, 420.0, 190.0),
        # Row 3: Data row 3 (large row spacing 40pt > 25pt)
        _make_block("r3_0", "Val A3", 50.0, 230.0, 120.0, 245.0),
        _make_block("r3_1", "Val B3", 150.0, 230.0, 220.0, 245.0),
        _make_block("r3_2", "Val C3", 250.0, 230.0, 320.0, 245.0),
        _make_block("r3_3", "Val D3", 350.0, 230.0, 420.0, 245.0),
    ]

    tables = binder.detect_page_tables(blocks, 1)
    assert len(tables) == 1
    t = tables[0]
    assert t.num_columns == 4
    # All 4 rows (1 header + 3 data rows) must be captured in the same table
    assert t.num_rows == 4
    assert t.num_data_rows == 3

    rows_found = set(c.row_index for c in t.cells)
    assert rows_found == {0, 1, 2, 3}


def test_table_continuation_preserves_wrapped_cells_in_later_rows():
    """Multi-line cell wraps in later table rows preserve column ownership and row grouping."""
    binder = GeometricTableBinder()
    blocks = [
        # Row 0: Header (3 columns: 50..120, 150..220, 250..450)
        _make_block("h0", "Project", 50.0, 100.0, 120.0, 115.0),
        _make_block("h1", "Role", 150.0, 100.0, 220.0, 115.0),
        _make_block("h2", "Scope & Impact", 250.0, 100.0, 450.0, 115.0),
        # Row 1
        _make_block("r1_0", "Alpha", 50.0, 125.0, 120.0, 140.0),
        _make_block("r1_1", "Lead", 150.0, 125.0, 220.0, 140.0),
        _make_block("r1_2", "Built pipeline", 250.0, 125.0, 450.0, 140.0),
        # Row 2 (30pt gap > 25pt, with 3 wrapped lines in Col 2)
        _make_block("r2_0", "Beta", 50.0, 170.0, 120.0, 185.0),
        _make_block("r2_1", "Senior", 150.0, 170.0, 220.0, 185.0),
        _make_block("r2_2a", "Migration of core services to cloud infrastructure", 250.0, 170.0, 450.0, 185.0),
        _make_block("r2_2b", "and optimization of database clusters", 250.0, 187.0, 450.0, 202.0),
        _make_block("r2_2c", "resulting in zero downtime.", 250.0, 204.0, 380.0, 219.0),
    ]

    tables = binder.detect_page_tables(blocks, 1)
    assert len(tables) == 1
    t = tables[0]
    assert t.num_rows == 3

    # Check row 2 cells
    r2_cells = [c for c in t.cells if c.row_index == 2]
    # Wrapped lines in Col 2 must all have column_index == 2
    r2_col2_texts = [c.text for c in r2_cells if c.column_index == 2]
    assert "Migration of core services to cloud infrastructure" in r2_col2_texts
    assert "and optimization of database clusters" in r2_col2_texts
    assert "resulting in zero downtime." in r2_col2_texts


def test_table_continuation_halts_at_section_heading():
    """Table continuation halts immediately when encountering a section heading."""
    binder = GeometricTableBinder()
    blocks = [
        # Header (3 cols: 50..120, 150..250, 280..380)
        _make_block("h0", "Degree", 50.0, 100.0, 120.0, 115.0),
        _make_block("h1", "Institution", 150.0, 100.0, 250.0, 115.0),
        _make_block("h2", "Year", 280.0, 100.0, 380.0, 115.0),
        # Row 1
        _make_block("r1_0", "B.S. CS", 50.0, 125.0, 120.0, 140.0),
        _make_block("r1_1", "Stanford", 150.0, 125.0, 250.0, 140.0),
        _make_block("r1_2", "2018", 280.0, 125.0, 380.0, 140.0),
        # Section Heading below table
        _make_block("sh", "WORK EXPERIENCE", 50.0, 170.0, 250.0, 185.0, role="SECTION_HEADING"),
        # Blocks after section heading
        _make_block("exp0", "Google", 50.0, 200.0, 150.0, 215.0),
        _make_block("exp1", "Software Engineer", 160.0, 200.0, 300.0, 215.0),
        _make_block("exp2", "2018 - Present", 310.0, 200.0, 400.0, 215.0),
    ]

    tables = binder.detect_page_tables(blocks, 1)
    assert len(tables) == 1
    t = tables[0]
    # Must only contain the education table (2 rows: header + 1 data row)
    assert t.num_rows == 2
    assert not any("WORK EXPERIENCE" in c.text for c in t.cells)
    assert not any("Google" in c.text for c in t.cells)


def test_adjacent_column_text_not_absorbed_into_another_column():
    """Adjacent column text stays strictly within its own column band."""
    binder = GeometricTableBinder()
    blocks = [
        # 3 cols: Col 0: 50..150, Col 1: 200..300, Col 2: 350..450
        _make_block("h0", "Col 0", 50.0, 100.0, 150.0, 115.0),
        _make_block("h1", "Col 1", 200.0, 100.0, 300.0, 115.0),
        _make_block("h2", "Col 2", 350.0, 100.0, 450.0, 115.0),
        # Row 1
        _make_block("r1_0", "Text A", 50.0, 125.0, 150.0, 140.0),
        _make_block("r1_1", "Text B", 200.0, 125.0, 300.0, 140.0),
        _make_block("r1_2", "Text C", 350.0, 125.0, 450.0, 140.0),
        # Row 2
        _make_block("r2_0", "Text D", 50.0, 160.0, 150.0, 175.0),
        _make_block("r2_1", "Text E", 200.0, 160.0, 300.0, 175.0),
        _make_block("r2_2", "Text F", 350.0, 160.0, 450.0, 175.0),
    ]

    tables = binder.detect_page_tables(blocks, 1)
    assert len(tables) == 1
    t = tables[0]
    cells_by_col = {}
    for c in t.cells:
        cells_by_col.setdefault(c.column_index, []).append(c.text)

    assert "Text B" in cells_by_col[1]
    assert "Text B" not in cells_by_col[0]
    assert "Text B" not in cells_by_col[2]
    assert "Text E" in cells_by_col[1]
    assert "Text E" not in cells_by_col[0]


def test_consulting_projects_table_e2e_structure():
    """The consulting projects table in gen_007_table_heavy_consulting_projects.pdf
    is detected with 4 columns and all 5 rows (1 header + 4 projects), with all wrapped cell
    blocks properly preserved in their respective cells.
    """
    pdf_path = Path("tests/fixtures/generalization/table_heavy_consulting_projects.pdf")
    if not pdf_path.exists():
        pytest.skip("Consulting fixture not present")

    raw_bytes = pdf_path.read_bytes()
    doc = interpret_layout(reconstruct_document(document_from_text_blocks(PDFExtractor.extract(raw_bytes))))
    sem_input = build_semantic_input(doc)

    assert len(sem_input.tables) == 1
    t = sem_input.tables[0]
    assert len(t.columns) == 4
    assert len(t.rows) == 5

    # Check header
    header_texts = [c.text for c in t.rows[0]]
    assert "Client Industry" in header_texts
    assert "Role & Period" in header_texts
    assert "Engagement Scope" in header_texts
    assert "Business Impact & ROI" in header_texts

    # Check Row 2 (Tier-1 Investment Bank)
    row2_texts = [c.text for c in t.rows[2]]
    assert "Tier-1 Investment Bank" in row2_texts
    # Column 2 scope includes wrapped block 'compliance.'
    assert any("Core transaction processing" in txt and "compliance." in txt for txt in row2_texts)
    # Column 3 impact includes 'zero operational downtime.'
    assert any("zero operational downtime." in txt for txt in row2_texts)

    # Check Row 3 (Multinational Healthcare)
    row3_texts = [c.text for c in t.rows[3]]
    assert "Multinational Healthcare" in row3_texts
    # Column 3 impact includes wrapped block 'months of acquisition.'
    assert any("months of acquisition." in txt for txt in row3_texts)


# =====================================================================
# 9-10. Provenance Strictness Tests
# =====================================================================


def test_provenance_strictness_rejects_arbitrary_unsupported_text():
    """Arbitrary unsupported semantic additions not present in source blocks strictly fail validation."""
    blocks = [
        _make_block("b_p1_0", "John Doe", 50.0, 50.0, 150.0, 65.0, role="HEADER"),
        _make_block("b_p1_1", "Senior Developer", 50.0, 100.0, 200.0, 115.0, role="ENTRY_TITLE"),
        _make_block("b_p1_2", "Google", 50.0, 120.0, 150.0, 135.0, role="ORGANIZATION"),
    ]
    sem_input = SemanticInput(
        document_id="test_strict",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        blocks=blocks,
    )

    output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
            # Hallucinated location not in source!
            location=GroundedString(value="Tokyo, Japan", source_block_ids=["b_p1_0"]),
        ),
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Senior Developer", source_block_ids=["b_p1_1"]),
                company=GroundedString(value="Google", source_block_ids=["b_p1_2"]),
                # Hallucinated description completely unsupported by source!
                description=GroundedString(
                    value="Architected global quantum computing data centers and managed $500M budget.",
                    source_block_ids=["b_p1_1", "b_p1_2"],
                ),
                source_block_ids=["b_p1_1", "b_p1_2"],
            ),
        ],
    )

    violations = validate_semantic_output(output, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].description" in v for v in violations)
