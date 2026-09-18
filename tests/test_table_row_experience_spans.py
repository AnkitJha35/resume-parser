from __future__ import annotations

from app.domain.semantic_contract import (
    DocumentArchetype,
    SemanticBlockInput,
    SemanticInput,
    SemanticPageMeta,
    build_deterministic_experience_spans,
)


def _make_block(
    block_id: str,
    text: str,
    role: str,
    reading_order: int,
    table_id: str | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
    cell_role: str | None = None,
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        text=text,
        page=1,
        bbox=[50.0, 100.0 + reading_order * 15, 200.0, 115.0 + reading_order * 15],
        region_id="page-1-region-0",
        region_kind="physical_region",
        reading_order=reading_order,
        suggested_role=role,
        table_id=table_id,
        row_index=row_index,
        column_index=column_index,
        cell_role=cell_role,
    )


def test_table_row_grouped_into_single_span_and_headers_excluded():
    """Table row with ENTRY_TITLE, ORGANIZATION, and DATE stays in 1 span, and row 0 header is skipped."""
    blocks = [
        _make_block("b_heading", "SEA SERVICE EXPERIENCE", "SECTION_HEADING", 0),
        # Header row (row 0)
        _make_block("b_h0", "Rank", "HEADER", 1, table_id="tbl_1", row_index=0, column_index=0, cell_role="HEADER"),
        _make_block("b_h1", "Vessel", "HEADER", 2, table_id="tbl_1", row_index=0, column_index=1, cell_role="HEADER"),
        _make_block("b_h2", "Dates", "HEADER", 3, table_id="tbl_1", row_index=0, column_index=2, cell_role="HEADER"),
        # Row 1
        _make_block("b_r1_c0", "Third Officer", "ENTRY_TITLE", 4, table_id="tbl_1", row_index=1, column_index=0, cell_role="DATA"),
        _make_block("b_r1_c1", "MT Maersk Tong", "ORGANIZATION", 5, table_id="tbl_1", row_index=1, column_index=1, cell_role="DATA"),
        _make_block("b_r1_c2", "2021 - 2022", "DATE", 6, table_id="tbl_1", row_index=1, column_index=2, cell_role="DATA"),
        # Row 2
        _make_block("b_r2_c0", "Second Officer", "ENTRY_TITLE", 7, table_id="tbl_1", row_index=2, column_index=0, cell_role="DATA"),
        _make_block("b_r2_c1", "MT Nordic Star", "ORGANIZATION", 8, table_id="tbl_1", row_index=2, column_index=1, cell_role="DATA"),
        _make_block("b_r2_c2", "2022 - 2023", "DATE", 9, table_id="tbl_1", row_index=2, column_index=2, cell_role="DATA"),
    ]

    sem = SemanticInput(
        document_id="doc_table_exp",
        page_count=1,
        archetype=DocumentArchetype.MARITIME_CV,
        pages=[SemanticPageMeta(page_number=1, width=600.0, height=800.0)],
        blocks=blocks,
    )

    spans = build_deterministic_experience_spans(sem)

    # Exactly 2 spans (row 1 and row 2); header row 0 must NOT become a span!
    assert len(spans) == 2

    # Span 0 = Row 1
    assert spans[0].entity_index == 0
    assert spans[0].title_block_id == "b_r1_c0"
    assert spans[0].block_ids == ["b_r1_c0", "b_r1_c1", "b_r1_c2"]

    # Span 1 = Row 2
    assert spans[1].entity_index == 1
    assert spans[1].title_block_id == "b_r2_c0"
    assert spans[1].block_ids == ["b_r2_c0", "b_r2_c1", "b_r2_c2"]


def test_mixed_table_and_nontable_experience():
    """Non-table appointment followed by a table row produces cleanly separated spans."""
    blocks = [
        _make_block("b_heading", "Work Experience", "SECTION_HEADING", 0),
        # Non-table appointment
        _make_block("b_nt_title", "Software Engineer", "ENTRY_TITLE", 1),
        _make_block("b_nt_org", "Acme Corp", "ORGANIZATION", 2),
        _make_block("b_nt_desc", "Developed backend services using Python.", "DESCRIPTION", 3),
        # Table row
        _make_block("b_t_h0", "Role", "HEADER", 4, table_id="tbl_2", row_index=0, column_index=0, cell_role="HEADER"),
        _make_block("b_t_h1", "Company", "HEADER", 5, table_id="tbl_2", row_index=0, column_index=1, cell_role="HEADER"),
        _make_block("b_t_c0", "Data Engineer", "ENTRY_TITLE", 6, table_id="tbl_2", row_index=1, column_index=0, cell_role="DATA"),
        _make_block("b_t_c1", "BigData Inc", "ORGANIZATION", 7, table_id="tbl_2", row_index=1, column_index=1, cell_role="DATA"),
    ]

    sem = SemanticInput(
        document_id="doc_mixed_exp",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=600.0, height=800.0)],
        blocks=blocks,
    )

    spans = build_deterministic_experience_spans(sem)

    assert len(spans) == 2
    # Span 0: non-table
    assert spans[0].title_block_id == "b_nt_title"
    assert spans[0].block_ids == ["b_nt_title", "b_nt_org", "b_nt_desc"]
    # Span 1: table row 1
    assert spans[1].title_block_id == "b_t_c0"
    assert spans[1].block_ids == ["b_t_c0", "b_t_c1"]
