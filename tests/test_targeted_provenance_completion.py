"""Targeted unit tests for generic provenance completion and structural label normalization.

Covers:
1. Wrapped table-cell value completion ('3rd' + 'Officer' in same cell).
2. Wrapped non-table continuation in same region/entity.
3. Multi-column grounded value (subset of cited blocks strictly ground value).
4. Rejection of adjacent unrelated column (cannot invent ungrounded text).
5. Structural prefix removal when core value is grounded ('Vessel Name: X' -> 'X').
6. Refusal to normalize unsupported semantic text (hallucinated content remains invalid).
7. Current-marker date normalization ('Till Now' -> 'Present').
8. Bullet/source-block mapping ('•' -> adjacent text block).
9. Cross-entity provenance protection (cannot steal blocks from other entity spans).
10. Rajeev-style unsupported value remains invalid (no fake provenance).
"""

import pytest

from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedBool,
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedProjectItem,
    GroundedString,
    SemanticBlockCategory,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    SemanticPageMeta,
    repair_semantic_output_provenance,
    validate_semantic_output,
)


def _make_block(
    block_id: str,
    text: str,
    page: int = 1,
    reading_order: int = 0,
    table_id: str | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
    suggested_role: str = "UNKNOWN",
    bbox: list[float] | None = None,
    region_kind: str = "body",
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        page=page,
        reading_order=reading_order,
        text=text,
        suggested_role=suggested_role,
        is_bold=False,
        font_size=10.0,
        bbox=bbox or [10.0, float(reading_order * 15), 200.0, float(reading_order * 15 + 12)],
        table_id=table_id,
        row_index=row_index,
        column_index=column_index,
        cell_role=None,
        region_id=f"page-{page}-region-0",
        region_kind=region_kind,
    )


def test_1_wrapped_table_cell_value_completion():
    """1. Wrapped table-cell value completion ('3rd' + 'Officer' in same cell)."""
    blocks = [
        _make_block("b_hdr", "Sea Service", 1, 0, suggested_role="SECTION_HEADING"),
        _make_block("b_c0", "3rd", 1, 1, table_id="t1", row_index=1, column_index=0, suggested_role="ENTRY_TITLE"),
        _make_block("b_c1", "Officer", 1, 2, table_id="t1", row_index=1, column_index=0, suggested_role="ENTRY_TITLE"),
        _make_block("b_co", "Maersk Line", 1, 3, table_id="t1", row_index=1, column_index=1, suggested_role="ORGANIZATION"),
    ]
    sem_input = SemanticInput(
        document_id="doc_wrap_cell",
        page_count=1,
        archetype=DocumentArchetype.MARITIME_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # LLM cited only b_c0 ('3rd') for emitted designation '3rd Officer'
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.MARITIME_CV,
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Maersk Line", source_block_ids=["b_co"]),
                designation=GroundedString(value="3rd Officer", source_block_ids=["b_c0"]),
                source_block_ids=["b_c0", "b_co"],
            )
        ],
    )
    # Prior to repair, designation fails validation because 'Officer' is not in b_c0
    viols_before = validate_semantic_output(raw_output, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].designation" in v for v in viols_before)

    # After repair, b_c1 is appended because it is in the exact same cell and completes the string verbatim
    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    assert "b_c1" in repaired.experience[0].designation.source_block_ids
    assert "b_c0" in repaired.experience[0].designation.source_block_ids

    viols_after = validate_semantic_output(repaired, sem_input)
    assert not any("experience[0].designation" in v for v in viols_after)


def test_1b_table_cell_slice_completion_dropping_unrelated_tail():
    """1b. Wrapped cell slice completion where cited blocks include cell tail ('3rd', 'Seapeak') but cell has ('3rd', 'Officer', 'Seapeak')."""
    blocks = [
        _make_block("b_hdr", "Sea Service", 1, 0, suggested_role="SECTION_HEADING"),
        _make_block("b_c0", "3rd", 1, 1, table_id="t1", row_index=1, column_index=0, suggested_role="ENTRY_TITLE"),
        _make_block("b_c1", "Officer", 1, 2, table_id="t1", row_index=1, column_index=0, suggested_role="ENTRY_TITLE"),
        _make_block("b_c2", "Seapeak", 1, 3, table_id="t1", row_index=1, column_index=0, suggested_role="ENTRY_TITLE"),
    ]
    sem_input = SemanticInput(
        document_id="doc_wrap_cell_slice",
        page_count=1,
        archetype=DocumentArchetype.MARITIME_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # LLM cited b_c0 ('3rd') and b_c2 ('Seapeak') for designation '3rd Officer', omitting b_c1 ('Officer')
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.MARITIME_CV,
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="3rd Officer", source_block_ids=["b_c0", "b_c2"]),
                source_block_ids=["b_c0", "b_c2"],
            )
        ],
    )
    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    assert repaired.experience[0].designation.source_block_ids == ["b_c0", "b_c1"]
    viols_after = validate_semantic_output(repaired, sem_input)
    assert not any("experience[0].designation" in v for v in viols_after)


def test_2_wrapped_non_table_continuation_in_same_region():
    """2. Wrapped non-table continuation in same region/entity."""
    blocks = [
        _make_block("b_exp_h", "EXPERIENCE", 1, 0, suggested_role="SECTION_HEADING"),
        _make_block("b_t0", "Chief Officer", 1, 1, suggested_role="ENTRY_TITLE"),
        _make_block("b_co0", "Columbia Ship Management , Darya Shipping & The Shipping Corporation of", 1, 2, suggested_role="ORGANIZATION"),
        _make_block("b_co1", "India Ltd | Oct 2022 – Till Now", 1, 3, suggested_role="DESCRIPTION"),
    ]
    sem_input = SemanticInput(
        document_id="doc_wrap_nontable",
        page_count=1,
        archetype=DocumentArchetype.MARITIME_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # LLM cited only b_co0 for emitted full company name
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.MARITIME_CV,
        experience=[
            GroundedExperienceItem(
                company=GroundedString(
                    value="Columbia Ship Management , Darya Shipping & The Shipping Corporation of India Ltd",
                    source_block_ids=["b_co0"],
                ),
                designation=GroundedString(value="Chief Officer", source_block_ids=["b_t0"]),
                source_block_ids=["b_t0", "b_co0", "b_co1"],
            )
        ],
    )
    viols_before = validate_semantic_output(raw_output, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].company" in v for v in viols_before)

    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    assert "b_co1" in repaired.experience[0].company.source_block_ids

    viols_after = validate_semantic_output(repaired, sem_input)
    assert not any("experience[0].company" in v for v in viols_after)


def test_3_multi_column_grounded_value_and_pruning():
    """3. Multi-column grounded value with non-contributing block pruning."""
    blocks = [
        _make_block("b_p36", "Strategic procurement optimization", 1, 1),
        _make_block("b_p37", "Achieved 14% direct material cost reduction", 1, 2),
        _make_block("b_p39", "& automated supplier bidding", 1, 3),
        _make_block("b_p40", "across 6 global business units.", 1, 4),
        _make_block("b_p41", "portal.", 1, 5),
    ]
    sem_input = SemanticInput(
        document_id="doc_multicol",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # LLM cited all 5 blocks, but description only used b_p36, b_p39, b_p40
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="Procurement", source_block_ids=["b_p36"]),
                description=GroundedString(
                    value="Strategic procurement optimization & automated supplier bidding across 6 global business units.",
                    source_block_ids=["b_p36", "b_p37", "b_p39", "b_p40", "b_p41"],
                ),
                source_block_ids=["b_p36", "b_p37", "b_p39", "b_p40", "b_p41"],
            )
        ],
    )
    viols_before = validate_semantic_output(raw_output, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in projects[0].description" in v for v in viols_before)

    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    # Non-contributing blocks b_p37 and b_p41 must be pruned
    assert "b_p37" not in repaired.projects[0].description.source_block_ids
    assert "b_p41" not in repaired.projects[0].description.source_block_ids
    assert set(repaired.projects[0].description.source_block_ids) == {"b_p36", "b_p39", "b_p40"}

    viols_after = validate_semantic_output(repaired, sem_input)
    assert not any("projects[0].description" in v for v in viols_after)


def test_4_rejection_of_adjacent_unrelated_column():
    """4. Rejection of adjacent unrelated column: cannot add ungrounded text."""
    blocks = [
        _make_block("b_c0", "Core migration", 1, 1),
        _make_block("b_c1", "Unrelated financial budget $10M", 1, 2),
    ]
    sem_input = SemanticInput(
        document_id="doc_unrelated",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # LLM hallucinated 'with high security' not in any block
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="Migration", source_block_ids=["b_c0"]),
                description=GroundedString(
                    value="Core migration with high security",
                    source_block_ids=["b_c0"],
                ),
                source_block_ids=["b_c0"],
            )
        ],
    )
    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    viols = validate_semantic_output(repaired, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in projects[0].description" in v for v in viols)


def test_5_structural_prefix_removal_when_core_value_grounded():
    """5. Structural prefix removal when core value is grounded ('Vessel Name: X' -> 'X')."""
    blocks = [
        _make_block("b_v0", "MT Furano Galaxy", 1, 1),
        _make_block("b_t0", "2nd Officer", 1, 2),
    ]
    sem_input = SemanticInput(
        document_id="doc_prefix",
        page_count=1,
        archetype=DocumentArchetype.MARITIME_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.MARITIME_CV,
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="2nd Officer", source_block_ids=["b_t0"]),
                description=GroundedString(
                    value="Vessel Name: MT Furano Galaxy",
                    source_block_ids=["b_v0"],
                ),
                source_block_ids=["b_t0", "b_v0"],
            )
        ],
    )
    viols_before = validate_semantic_output(raw_output, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].description" in v for v in viols_before)

    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    assert repaired.experience[0].description.value == "MT Furano Galaxy"

    viols_after = validate_semantic_output(repaired, sem_input)
    assert not any("experience[0].description" in v for v in viols_after)


def test_6_refusal_to_normalize_unsupported_semantic_text():
    """6. Refusal to normalize unsupported semantic text (hallucinated content remains invalid)."""
    blocks = [
        _make_block("b_v0", "MT Furano Galaxy", 1, 1),
    ]
    sem_input = SemanticInput(
        document_id="doc_refuse",
        page_count=1,
        archetype=DocumentArchetype.MARITIME_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # LLM hallucinated a completely different vessel name
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.MARITIME_CV,
        experience=[
            GroundedExperienceItem(
                description=GroundedString(
                    value="Vessel Name: USS Enterprise Galaxy",
                    source_block_ids=["b_v0"],
                ),
                source_block_ids=["b_v0"],
            )
        ],
    )
    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    viols = validate_semantic_output(repaired, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].description" in v for v in viols)


def test_7_current_marker_date_normalization():
    """7. Current-marker date normalization ('Till Now' -> 'Present')."""
    blocks = [
        _make_block("b_d0", "Oct 2022 – Till Now", 1, 1, suggested_role="DATE"),
        _make_block("b_c0", "Acme Corp", 1, 2, suggested_role="ORGANIZATION"),
    ]
    sem_input = SemanticInput(
        document_id="doc_date_curr",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Acme Corp", source_block_ids=["b_c0"]),
                startDate=GroundedString(value="2022-10", source_block_ids=["b_d0"]),
                endDate=GroundedString(value="Present", source_block_ids=["b_d0"]),
                current=GroundedBool(value=True, source_block_ids=["b_d0"]),
                source_block_ids=["b_c0", "b_d0"],
            )
        ],
    )
    # Supported immediately under strict current-marker normalization
    viols = validate_semantic_output(raw_output, sem_input)
    assert not any("experience[0].endDate" in v for v in viols)


def test_8_bullet_source_block_mapping():
    """8. Bullet/source-block mapping ('•' -> adjacent text block)."""
    blocks = [
        _make_block("b_sec", "LANGUAGES", 1, 0, suggested_role="SECTION_HEADING"),
        _make_block("b_txt0", "English", 1, 1, bbox=[40.0, 100.0, 80.0, 112.0], suggested_role="DESCRIPTION"),
        _make_block("b_bul0", "•", 1, 2, bbox=[20.0, 100.0, 30.0, 112.0], suggested_role="BULLET"),
        _make_block("b_txt1", "Hindi", 1, 3, bbox=[40.0, 120.0, 80.0, 132.0], suggested_role="DESCRIPTION"),
        _make_block("b_bul1", "•", 1, 4, bbox=[20.0, 120.0, 30.0, 132.0], suggested_role="BULLET"),
    ]
    sem_input = SemanticInput(
        document_id="doc_bullet",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # LLM mistakenly cited bullet block '•' for English and Hindi
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        languages=[
            GroundedString(value="English", source_block_ids=["b_bul0"]),
            GroundedString(value="Hindi", source_block_ids=["b_bul1"]),
        ],
    )
    viols_before = validate_semantic_output(raw_output, sem_input)
    assert any("languages[0]" in v for v in viols_before)
    assert any("languages[1]" in v for v in viols_before)

    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    assert repaired.languages[0].source_block_ids == ["b_txt0"]
    assert repaired.languages[1].source_block_ids == ["b_txt1"]

    viols_after = validate_semantic_output(repaired, sem_input)
    assert not any("languages" in v for v in viols_after)


def test_9_cross_entity_provenance_protection():
    """9. Cross-entity provenance protection (cannot steal blocks from other entity spans)."""
    blocks = [
        _make_block("b_exp_h", "EXPERIENCE", 1, 0, suggested_role="SECTION_HEADING"),
        # Entity 0
        _make_block("b_t0", "Software Engineer", 1, 1, suggested_role="ENTRY_TITLE"),
        _make_block("b_c0", "Google", 1, 2, suggested_role="ORGANIZATION"),
        _make_block("b_d0", "2020 - 2022", 1, 3, suggested_role="DATE"),
        # Entity 1
        _make_block("b_t1", "Staff Engineer", 1, 4, suggested_role="ENTRY_TITLE"),
        _make_block("b_c1", "Meta", 1, 5, suggested_role="ORGANIZATION"),
        _make_block("b_d1", "2022 - Present", 1, 6, suggested_role="DATE"),
    ]
    sem_input = SemanticInput(
        document_id="doc_cross_ent",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # Entity 0 illegally claims Meta from Entity 1
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Software Engineer", source_block_ids=["b_t0"]),
                company=GroundedString(value="Meta", source_block_ids=["b_c1"]),
                source_block_ids=["b_t0", "b_c0", "b_d0", "b_c1"],
            ),
            GroundedExperienceItem(
                designation=GroundedString(value="Staff Engineer", source_block_ids=["b_t1"]),
                company=GroundedString(value="Meta", source_block_ids=["b_c1"]),
                source_block_ids=["b_t1", "b_c1", "b_d1"],
            ),
        ],
    )
    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    viols = validate_semantic_output(repaired, sem_input)
    assert any("CROSS_ENTITY_PROVENANCE" in v for v in viols)


def test_10_rajeev_unsupported_location_remains_invalid():
    """10. Rajeev-style unsupported value remains invalid (no fabricated provenance)."""
    blocks = [
        _make_block("b_n", "Rajeev Ranjan", 1, 0, suggested_role="NAME"),
        _make_block("b_e", "rajeev@example.com", 1, 1, suggested_role="CONTACT"),
    ]
    sem_input = SemanticInput(
        document_id="doc_rajeev",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )
    # LLM inferred location 'India' from phone code or background knowledge, without text evidence
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="Rajeev Ranjan", source_block_ids=["b_n"]),
            email=GroundedString(value="rajeev@example.com", source_block_ids=["b_e"]),
            location=GroundedString(value="India", source_block_ids=[]),
        ),
    )
    repaired, repairs = repair_semantic_output_provenance(raw_output, sem_input)
    viols = validate_semantic_output(repaired, sem_input)
    # Must remain a missing provenance violation!
    assert any("MISSING_PROVENANCE in personal.location" in v for v in viols)
