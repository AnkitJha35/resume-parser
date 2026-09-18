from __future__ import annotations

from app.pipeline.stages.table_semantic_context import (
    CellSemanticDescriptor,
    ColumnSemanticDescriptor,
    TablePurpose,
    TableSemanticContext,
)
from app.domain.semantic_contract import _is_false_unknown_table


def test_rejects_contact_dominated_header_when_purpose_unknown():
    """A table with UNKNOWN purpose whose headers are email/phone/github/linkedin is rejected."""
    ctx = TableSemanticContext(
        table_id="tbl_contact",
        page=1,
        purpose=TablePurpose.UNKNOWN,
        columns=[
            ColumnSemanticDescriptor(column_index=0, header_text="shubham@example.com", semantic_role="unknown"),
            ColumnSemanticDescriptor(column_index=1, header_text="+91 9876543210", semantic_role="unknown"),
            ColumnSemanticDescriptor(column_index=2, header_text="https://github.com/user", semantic_role="unknown"),
        ],
        rows=[
            [
                CellSemanticDescriptor(row_index=1, column_index=0, text="Row 1 Col 0"),
                CellSemanticDescriptor(row_index=1, column_index=1, text="Row 1 Col 1"),
                CellSemanticDescriptor(row_index=1, column_index=2, text="Row 1 Col 2"),
            ],
        ],
    )
    assert _is_false_unknown_table(ctx) is True


def test_rejects_prose_paragraphs_and_section_headings_in_rows():
    """A candidate table containing section headings or long prose paragraphs is rejected."""
    ctx = TableSemanticContext(
        table_id="tbl_prose",
        page=1,
        purpose=TablePurpose.UNKNOWN,
        columns=[
            ColumnSemanticDescriptor(column_index=0, header_text="Col A", semantic_role="unknown"),
            ColumnSemanticDescriptor(column_index=1, header_text="Col B", semantic_role="unknown"),
            ColumnSemanticDescriptor(column_index=2, header_text="Col C", semantic_role="unknown"),
        ],
        rows=[
            # Row with section headings
            [
                CellSemanticDescriptor(row_index=1, column_index=0, text="EXPERIENCE"),
                CellSemanticDescriptor(row_index=1, column_index=1, text="SKILLS"),
                CellSemanticDescriptor(row_index=1, column_index=2, text="EDUCATION"),
            ],
            # Row with long prose paragraph
            [
                CellSemanticDescriptor(
                    row_index=2,
                    column_index=0,
                    text="• Led design and development of end-to-end scalable backend architecture across distributed cloud systems using Django and Python.",
                ),
                CellSemanticDescriptor(
                    row_index=2,
                    column_index=1,
                    text="• Managed team of 15 engineers to implement production CI/CD pipelines and microservices infrastructure.",
                ),
                CellSemanticDescriptor(row_index=2, column_index=2, text="Details"),
            ],
        ],
    )
    assert _is_false_unknown_table(ctx) is True


def test_keeps_genuine_table_with_unknown_purpose():
    """A genuine structured table with UNKNOWN purpose (e.g. consulting engagements) is retained."""
    ctx = TableSemanticContext(
        table_id="tbl_consulting",
        page=1,
        purpose=TablePurpose.UNKNOWN,
        columns=[
            ColumnSemanticDescriptor(column_index=0, header_text="Client Industry", semantic_role="unknown"),
            ColumnSemanticDescriptor(column_index=1, header_text="Role & Period", semantic_role="unknown"),
            ColumnSemanticDescriptor(column_index=2, header_text="Engagement Scope", semantic_role="unknown"),
            ColumnSemanticDescriptor(column_index=3, header_text="Business Impact & ROI", semantic_role="unknown"),
        ],
        rows=[
            [
                CellSemanticDescriptor(row_index=1, column_index=0, text="Global Retail Conglomerate"),
                CellSemanticDescriptor(row_index=1, column_index=1, text="Engagement Lead 2022 - 2023"),
                CellSemanticDescriptor(row_index=1, column_index=2, text="Omni-channel logistics supply chain."),
                CellSemanticDescriptor(row_index=1, column_index=3, text="Reduced logistics costs by $38M."),
            ],
        ],
    )
    assert _is_false_unknown_table(ctx) is False


def test_keeps_known_purpose_tables_even_with_contact_or_prose():
    """Tables with a verified purpose (e.g. PERSONAL_DATA, SEA_SERVICE) are never false unknown tables."""
    ctx = TableSemanticContext(
        table_id="tbl_personal",
        page=1,
        purpose=TablePurpose.PERSONAL_DATA,
        columns=[
            ColumnSemanticDescriptor(column_index=0, header_text="Email", semantic_role="contact"),
            ColumnSemanticDescriptor(column_index=1, header_text="Phone", semantic_role="contact"),
        ],
        rows=[
            [
                CellSemanticDescriptor(row_index=1, column_index=0, text="user@example.com"),
                CellSemanticDescriptor(row_index=1, column_index=1, text="+1 555-0199"),
            ],
        ],
    )
    assert _is_false_unknown_table(ctx) is False
