"""Tests for Phase 8-2A SemanticInput and SemanticOutput contracts and invariants."""

from __future__ import annotations

import json
import pytest

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.domain.semantic_contract import (
    BlockClassification,
    DocumentArchetype,
    GroundedBool,
    GroundedEducationItem,
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedProjectItem,
    GroundedString,
    SemanticBlockCategory,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    SemanticPageMeta,
    _is_multiblock_text_semantically_supported,
    _is_value_semantically_supported,
    build_semantic_input,
    semantic_output_to_resume,
    validate_semantic_output,
)


def _make_test_document() -> Document:
    # Page 1 Header Region
    line1 = Line(
        line_id="p1-l1",
        page_number=1,
        bbox=BoundingBox(50.0, 60.0, 250.0, 80.0),
        spans=[Span("s1", "John Doe", BoundingBox(50.0, 60.0, 250.0, 80.0))],
        text="John Doe",
        style=TextStyle(font_size=16.0, bold=True),
        reading_order=1,
    )
    line2 = Line(
        line_id="p1-l2",
        page_number=1,
        bbox=BoundingBox(50.0, 85.0, 250.0, 100.0),
        spans=[Span("s2", "john.doe@example.com", BoundingBox(50.0, 85.0, 250.0, 100.0))],
        text="john.doe@example.com",
        style=TextStyle(font_size=10.0),
        reading_order=2,
    )
    line3 = Line(
        line_id="p1-l3",
        page_number=1,
        bbox=BoundingBox(50.0, 105.0, 200.0, 120.0),
        spans=[Span("s3", "San Francisco, CA", BoundingBox(50.0, 105.0, 200.0, 120.0))],
        text="San Francisco, CA",
        style=TextStyle(font_size=10.0),
        reading_order=3,
    )
    line_phone = Line(
        line_id="p1-l-phone",
        page_number=1,
        bbox=BoundingBox(50.0, 125.0, 200.0, 140.0),
        spans=[Span("s-phone", "+91 98295 19017", BoundingBox(50.0, 125.0, 200.0, 140.0))],
        text="+91 98295 19017",
        style=TextStyle(font_size=10.0),
        reading_order=4,
    )
    r_header = Region("r0", "header", BoundingBox(50.0, 50.0, 500.0, 150.0), [line1, line2, line3, line_phone])

    # Page 1 Body Region
    line4 = Line(
        line_id="p1-l4",
        page_number=1,
        bbox=BoundingBox(50.0, 200.0, 200.0, 220.0),
        spans=[Span("s4", "Acme Corporation", BoundingBox(50.0, 200.0, 200.0, 220.0))],
        text="Acme Corporation",
        style=TextStyle(font_size=12.0, bold=True),
        reading_order=5,
    )
    line5 = Line(
        line_id="p1-l5",
        page_number=1,
        bbox=BoundingBox(50.0, 225.0, 200.0, 240.0),
        spans=[Span("s5", "Senior Software Engineer", BoundingBox(50.0, 225.0, 200.0, 240.0))],
        text="Senior Software Engineer",
        style=TextStyle(font_size=11.0),
        reading_order=6,
    )
    line_date = Line(
        line_id="p1-l-date",
        page_number=1,
        bbox=BoundingBox(50.0, 245.0, 200.0, 260.0),
        spans=[Span("s-date", "July 2018 – Present", BoundingBox(50.0, 245.0, 200.0, 260.0))],
        text="July 2018 – Present",
        style=TextStyle(font_size=10.0),
        reading_order=7,
    )
    line_ship = Line(
        line_id="p1-l-ship",
        page_number=1,
        bbox=BoundingBox(50.0, 300.0, 200.0, 315.0),
        spans=[Span("s-ship", "Darya Shaan", BoundingBox(50.0, 300.0, 200.0, 315.0))],
        text="Darya Shaan",
        style=TextStyle(font_size=10.0),
        reading_order=8,
    )
    line_past_date = Line(
        line_id="p1-l-past",
        page_number=1,
        bbox=BoundingBox(50.0, 320.0, 200.0, 335.0),
        spans=[Span("s-past", "Jan 2015 – Dec 2017", BoundingBox(50.0, 320.0, 200.0, 335.0))],
        text="Jan 2015 – Dec 2017",
        style=TextStyle(font_size=10.0),
        reading_order=9,
    )
    line_ref = Line(
        line_id="p1-l-ref",
        page_number=1,
        bbox=BoundingBox(50.0, 500.0, 250.0, 515.0),
        spans=[Span("s-ref", "Captain Smith (Referee)", BoundingBox(50.0, 500.0, 250.0, 515.0))],
        text="Captain Smith (Referee)",
        style=TextStyle(font_size=10.0),
        reading_order=10,
    )
    r_body = Region(
        "r1",
        "physical_region",
        BoundingBox(50.0, 200.0, 500.0, 600.0),
        [line4, line5, line_date, line_ship, line_past_date, line_ref],
    )

    page = Page(page_number=1, width=612.0, height=792.0, regions=[r_header, r_body])
    return Document(pages=[page])


def test_build_semantic_input_no_block_duplication():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    assert sem_input.document_id == "test-doc-1"
    assert sem_input.page_count == 1
    # Check that pages only contain metadata (no blocks field inside page)
    assert hasattr(sem_input.pages[0], "page_number")
    assert not hasattr(sem_input.pages[0], "blocks")

    # Canonical blocks live only at top level
    assert len(sem_input.blocks) == 10
    assert sem_input.blocks[0].text == "John Doe"
    assert sem_input.blocks[0].region_kind == "header"
    assert sem_input.blocks[0].suggested_role is not None

    # Serialization check
    json_str = sem_input.model_dump_json()
    parsed = json.loads(json_str)
    assert "blocks" in parsed
    assert "blocks" not in parsed["pages"][0]


def test_date_phone_and_current_status_normalization():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    # 1. "July 2018" supports "2018-07"
    # 2. "+91 98295 19017" supports "+919829519017"
    # 3. "Present" supports current=True
    # 4. raw_value is preserved
    output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", raw_value="John Doe", source_block_ids=["b_p1_0"]),
            email=GroundedString(value="john.doe@example.com", source_block_ids=["b_p1_1"]),
            phone=GroundedString(value="+919829519017", raw_value="+91 98295 19017", source_block_ids=["b_p1_3"]),
            location=GroundedString(value="San Francisco, CA", source_block_ids=["b_p1_2"]),
        ),
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Acme Corporation", source_block_ids=["b_p1_4"]),
                designation=GroundedString(value="Senior Software Engineer", source_block_ids=["b_p1_5"]),
                startDate=GroundedString(value="2018-07", raw_value="July 2018", source_block_ids=["b_p1_6"]),
                current=GroundedBool(value=True, source_block_ids=["b_p1_6"]),
            )
        ],
    )

    violations = validate_semantic_output(output, sem_input)
    assert violations == []

    # Verify Resume conversion preserves fields
    resume = semantic_output_to_resume(output)
    assert resume.personal.phone == "+919829519017"
    assert resume.experience[0].startDate == "2018-07"
    assert resume.experience[0].current is True


def test_rejection_of_unsupported_semantic_renaming():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    # "Darya Shaan" (b_p1_7) cannot support "Darya Shipping"
    # "Senior Software Engineer" (b_p1_5) cannot support "Principal Software Engineer"
    output = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Darya Shipping", source_block_ids=["b_p1_7"]),
                designation=GroundedString(value="Principal Software Engineer", source_block_ids=["b_p1_5"]),
            )
        ]
    )

    violations = validate_semantic_output(output, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].company" in v for v in violations)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].designation" in v for v in violations)


def test_rejection_of_unsupported_current_status():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    # b_p1_8 has text "Jan 2015 – Dec 2017" (no present/current marker)
    output = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Acme Corporation", source_block_ids=["b_p1_4"]),
                current=GroundedBool(value=True, source_block_ids=["b_p1_8"]),
            )
        ]
    )

    violations = validate_semantic_output(output, sem_input)
    assert any("UNSUPPORTED_CURRENT_STATUS in experience[0].current" in v for v in violations)


def test_rejection_of_token_combination_fabrication():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    # Attempting to combine "John" from b_p1_0 and "Acme" from b_p1_4 to create "John Acme Corporation"
    # Even if both blocks are referenced, "John Acme Corporation" as a single value cannot be fabricated
    output = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="John Acme Corporation", source_block_ids=["b_p1_0", "b_p1_4"]),
            )
        ]
    )

    violations = validate_semantic_output(output, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].company" in v for v in violations)


def test_structural_header_based_location_validation():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    # b_p1_4 is in region_kind="physical_region" (body), suggested_role="ORGANIZATION", NOT a personal header/contact block
    output = SemanticOutput(
        personal=GroundedPersonal(
            location=GroundedString(value="Acme Corporation", source_block_ids=["b_p1_4"]),
        )
    )

    violations = validate_semantic_output(output, sem_input)
    assert any("LOCATION_OUTSIDE_HEADER_REGION" in v for v in violations)


def test_personal_location_in_physical_region_valid():
    sem_input = SemanticInput(
        document_id="test-doc-physical-region",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=[
            SemanticBlockInput(
                block_id="b_p1_0",
                text="Devon Vance",
                page=1,
                bbox=[50.0, 50.0, 200.0, 70.0],
                region_id="r1",
                region_kind="physical_region",
                reading_order=1,
                suggested_role="HEADER",
            ),
            SemanticBlockInput(
                block_id="b_p1_1",
                text="Seattle, WA | devon.vance@email.com | github.com/devon-vance",
                page=1,
                bbox=[50.0, 75.0, 450.0, 90.0],
                region_id="r1",
                region_kind="physical_region",
                reading_order=2,
                suggested_role="CONTACT",
            ),
        ],
    )
    output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[
            BlockClassification(block_id="b_p1_0", category=SemanticBlockCategory.PERSONAL),
            BlockClassification(block_id="b_p1_1", category=SemanticBlockCategory.PERSONAL),
        ],
        personal=GroundedPersonal(
            name=GroundedString(value="Devon Vance", source_block_ids=["b_p1_0"]),
            location=GroundedString(value="Seattle, WA", source_block_ids=["b_p1_1"]),
            email=GroundedString(value="devon.vance@email.com", source_block_ids=["b_p1_1"]),
        ),
    )
    violations = validate_semantic_output(output, sem_input)
    assert not any("LOCATION_OUTSIDE_HEADER_REGION" in v for v in violations)
    assert not violations


def test_personal_location_in_column_form_context_valid():
    sem_input = SemanticInput(
        document_id="test-doc-form-column",
        page_count=1,
        archetype=DocumentArchetype.STRUCTURED_FORM,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=[
            SemanticBlockInput(
                block_id="b_p1_0",
                text="AKIBUL ALAM",
                page=1,
                bbox=[50.0, 50.0, 200.0, 70.0],
                region_id="r1",
                region_kind="column",
                reading_order=1,
                table_id="table_p1_0",
                row_index=0,
                column_index=0,
                suggested_role="UNKNOWN",
            ),
            SemanticBlockInput(
                block_id="b_p1_1",
                text="PERMANENT ADDRESS : VILL: PASHCHIM KATHALIA, PO: KATHALIA, DIST: JHALAKATHI",
                page=1,
                bbox=[50.0, 75.0, 450.0, 90.0],
                region_id="r1",
                region_kind="column",
                reading_order=2,
                table_id="table_p1_0",
                row_index=1,
                column_index=0,
                suggested_role="UNKNOWN",
            ),
        ],
    )
    output = SemanticOutput(
        document_archetype=DocumentArchetype.STRUCTURED_FORM,
        block_classifications=[
            BlockClassification(block_id="b_p1_0", category=SemanticBlockCategory.PERSONAL),
            BlockClassification(block_id="b_p1_1", category=SemanticBlockCategory.PERSONAL),
        ],
        personal=GroundedPersonal(
            name=GroundedString(value="AKIBUL ALAM", source_block_ids=["b_p1_0"]),
            location=GroundedString(
                value="VILL: PASHCHIM KATHALIA, PO: KATHALIA, DIST: JHALAKATHI",
                source_block_ids=["b_p1_1"],
            ),
        ),
    )
    violations = validate_semantic_output(output, sem_input)
    assert not any("LOCATION_OUTSIDE_HEADER_REGION" in v for v in violations)
    assert not violations


def test_personal_location_in_experience_section_rejected():
    sem_input = SemanticInput(
        document_id="test-doc-exp-location",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=[
            SemanticBlockInput(
                block_id="b_p1_0",
                text="John Doe",
                page=1,
                bbox=[50.0, 50.0, 200.0, 70.0],
                region_id="r0",
                region_kind="header",
                reading_order=1,
                suggested_role="HEADER",
            ),
            SemanticBlockInput(
                block_id="b_p1_1",
                text="Senior Software Engineer - New York, NY",
                page=1,
                bbox=[50.0, 200.0, 300.0, 220.0],
                region_id="r1",
                region_kind="physical_region",
                reading_order=2,
                suggested_role="ENTRY_TITLE",
            ),
        ],
    )
    # Case 1: Rejected because suggested_role is ENTRY_TITLE
    output1 = SemanticOutput(
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
            location=GroundedString(value="New York, NY", source_block_ids=["b_p1_1"]),
        ),
    )
    violations1 = validate_semantic_output(output1, sem_input)
    assert any("LOCATION_OUTSIDE_HEADER_REGION" in v for v in violations1)

    # Case 2: Rejected because block is classified as EXPERIENCE
    output2 = SemanticOutput(
        block_classifications=[
            BlockClassification(block_id="b_p1_1", category=SemanticBlockCategory.EXPERIENCE),
        ],
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
            location=GroundedString(value="New York, NY", source_block_ids=["b_p1_1"]),
        ),
    )
    violations2 = validate_semantic_output(output2, sem_input)
    assert any("LOCATION_OUTSIDE_HEADER_REGION" in v for v in violations2)

    # Case 3: Rejected because block is mapped to experience body collection
    output3 = SemanticOutput(
        personal=GroundedPersonal(
            name=GroundedString(value="John Doe", source_block_ids=["b_p1_0"]),
            location=GroundedString(value="New York, NY", source_block_ids=["b_p1_1"]),
        ),
        experience=[
            GroundedExperienceItem(
                designation=GroundedString(value="Senior Software Engineer", source_block_ids=["b_p1_1"]),
            )
        ],
    )
    violations3 = validate_semantic_output(output3, sem_input)
    assert any("LOCATION_OUTSIDE_HEADER_REGION" in v for v in violations3)


def test_reference_classification_exclusion():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    # b_p1_9 is Captain Smith (Referee), explicitly classified as REFERENCE
    output = SemanticOutput(
        block_classifications=[
            BlockClassification(
                block_id="b_p1_9",
                category=SemanticBlockCategory.REFERENCE,
                exclusion_reason="referee_contact_block",
            )
        ],
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Captain Smith (Referee)", source_block_ids=["b_p1_9"]),
            )
        ],
    )

    violations = validate_semantic_output(output, sem_input)
    assert any("REFERENCE_IN_EXPERIENCE" in v for v in violations)


def test_complete_provenance_on_all_entities():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    # Missing provenance on skills and languages
    output = SemanticOutput(
        skills=[GroundedString(value="Python", source_block_ids=[])],
        languages=[GroundedString(value="English", source_block_ids=[])],
    )

    violations = validate_semantic_output(output, sem_input)
    assert any("MISSING_PROVENANCE in skills[0]" in v for v in violations)
    assert any("MISSING_PROVENANCE in languages[0]" in v for v in violations)


def test_generic_table_metadata_in_input():
    block = SemanticBlockInput(
        block_id="b_table_1",
        text="Vessel Name",
        page=1,
        bbox=[50.0, 100.0, 150.0, 115.0],
        region_id="r_table",
        region_kind="column",
        reading_order=1,
        table_id="tbl_sea_service",
        row_index=0,
        column_index=1,
        cell_role="HEADER",
    )
    assert block.table_id == "tbl_sea_service"
    assert block.row_index == 0
    assert block.column_index == 1
    assert block.cell_role == "HEADER"


def test_constrained_document_archetype_enum():
    assert DocumentArchetype.STANDARD_CV.value == "standard_cv"
    assert DocumentArchetype.MARITIME_CV.value == "maritime_cv"
    assert DocumentArchetype.STRUCTURED_FORM.value == "structured_form"
    assert DocumentArchetype.UNKNOWN.value == "unknown"

    output = SemanticOutput(document_archetype=DocumentArchetype.MARITIME_CV)
    assert output.document_archetype == DocumentArchetype.MARITIME_CV


def test_phone_normalization_rules():
    # 1. (650) 498-1240 -> +16504981240 accepted (1-digit country code prefix)
    assert _is_value_semantically_supported("+16504981240", "(650) 498-1240")
    assert _is_value_semantically_supported("16504981240", "(650) 498-1240")

    # 2. +91 98295 19017 -> +919829519017 accepted (country code already in source)
    assert _is_value_semantically_supported("+919829519017", "+91 98295 19017")

    # 3. Exact phone digit normalization still accepted
    assert _is_value_semantically_supported("6504981240", "(650) 498-1240")
    assert _is_value_semantically_supported("9829519017", "98295-19017")

    # 4. Unrelated canonical number rejected
    assert not _is_value_semantically_supported("+19998887766", "(650) 498-1240")
    assert not _is_value_semantically_supported("+441234567890", "+91 98295 19017")

    # 5. Canonical number with modified subscriber digits rejected
    assert not _is_value_semantically_supported("+16504989999", "(650) 498-1240")
    assert not _is_value_semantically_supported("+919829519099", "+91 98295 19017")

    # 6. Prefix longer than 3 digits rejected
    assert not _is_value_semantically_supported("+12346504981240", "(650) 498-1240")
    assert not _is_value_semantically_supported("999996504981240", "(650) 498-1240")


def test_personal_phone_country_code_e2e_contract_validation():
    doc = _make_test_document()
    sem_input = build_semantic_input(doc, document_id="test-doc-1")

    # b_p1_3 has text "+91 98295 19017"
    output_valid = SemanticOutput(
        personal=GroundedPersonal(
            phone=GroundedString(value="+919829519017", source_block_ids=["b_p1_3"]),
        )
    )
    violations = validate_semantic_output(output_valid, sem_input)
    assert not violations

    # Modified digits should violate UNSUPPORTED_CANONICAL_VALUE
    output_invalid = SemanticOutput(
        personal=GroundedPersonal(
            phone=GroundedString(value="+919829519999", source_block_ids=["b_p1_3"]),
        )
    )
    violations = validate_semantic_output(output_invalid, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in personal.phone" in v for v in violations)


def test_multiblock_text_grounding_consulting_descriptions():
    # Consulting Fixture Row 1
    s1 = "Omni-channel logistics and supply Reduced annual logistics costs by $38M;improved order fulfillment SLA by 40%. chain restructuring across 400 stores."
    v1 = "Omni-channel logistics and supply chain restructuring across 400 stores. Reduced annual logistics costs by $38M; improved order fulfillment SLA by 40%."
    assert not _is_value_semantically_supported(v1, s1)
    assert _is_multiblock_text_semantically_supported(v1, s1)

    # Consulting Fixture Row 2
    s2 = "Core transaction processing cloud Successfully migrated 12M accounts with zero migration & regulatory risk compliance. operational downtime."
    v2 = "Core transaction processing cloud migration & regulatory risk compliance. Successfully migrated 12M accounts with zero operational downtime."
    assert not _is_value_semantically_supported(v2, s2)
    assert _is_multiblock_text_semantically_supported(v2, s2)

    # Consulting Fixture Row 3
    s3 = "Post-merger integration of 14 regional hospital clinical networks.months of acquisition. Realized $24M in operating synergies within 18"
    v3 = "Post-merger integration of 14 regional hospital clinical networks. Realized $24M in operating synergies within 18 months of acquisition."
    assert not _is_value_semantically_supported(v3, s3)
    assert _is_multiblock_text_semantically_supported(v3, s3)

    # Consulting Fixture Row 4
    s4 = "Strategic procurement optimization Achieved 14% direct material cost reduction & automated supplier bidding portal. across 6 global business units."
    v4 = "Strategic procurement optimization & automated supplier bidding portal. Achieved 14% direct material cost reduction across 6 global business units."
    assert not _is_value_semantically_supported(v4, s4)
    assert _is_multiblock_text_semantically_supported(v4, s4)


def test_multiblock_text_grounding_reordered_and_rejections():
    s_text = "part two here. part one first."
    v_reordered = "Part one first. Part two here."
    assert not _is_value_semantically_supported(v_reordered, s_text)
    assert _is_multiblock_text_semantically_supported(v_reordered, s_text)

    # Missing token rejected (e.g. omitted word "two")
    v_missing = "Part one first. Part here."
    assert not _is_multiblock_text_semantically_supported(v_missing, s_text)

    # Extra token rejected (e.g. hallucinated word "extra")
    v_extra = "Part one first. Part two here extra."
    assert not _is_multiblock_text_semantically_supported(v_extra, s_text)

    # Modified number rejected (e.g. "400" -> "500")
    s_num = "Restructured supply chain across 400 stores."
    v_mod_num = "Restructured supply chain across 500 stores."
    assert not _is_multiblock_text_semantically_supported(v_mod_num, s_num)

    # Unsupported date rejected
    s_date = "· 'Architecting Design Tokens for Enterprise Scale' - Published on Medium"
    v_date = "2022"
    assert not _is_value_semantically_supported(v_date, s_date)
    assert not _is_multiblock_text_semantically_supported(v_date, s_date)


def test_multiblock_description_e2e_contract_validation():
    sem_input = SemanticInput(
        document_id="test-doc-multiblock-desc",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=[
            SemanticBlockInput(
                block_id="b_p1_0",
                text="Arthur Pendelton",
                page=1,
                bbox=[50.0, 50.0, 200.0, 70.0],
                region_id="r1",
                region_kind="header",
                reading_order=1,
                suggested_role="HEADER",
            ),
            SemanticBlockInput(
                block_id="b_p1_1",
                text="Omni-channel logistics and supply Reduced annual logistics costs by $38M;",
                page=1,
                bbox=[50.0, 100.0, 450.0, 115.0],
                region_id="r1",
                region_kind="physical_region",
                reading_order=2,
                suggested_role="DESCRIPTION",
            ),
            SemanticBlockInput(
                block_id="b_p1_2",
                text="improved order fulfillment SLA by 40%. chain restructuring across 400 stores.",
                page=1,
                bbox=[50.0, 120.0, 450.0, 135.0],
                region_id="r1",
                region_kind="physical_region",
                reading_order=3,
                suggested_role="DESCRIPTION",
            ),
        ],
    )

    # Valid reordered multi-block description
    output_valid = SemanticOutput(
        personal=GroundedPersonal(
            name=GroundedString(value="Arthur Pendelton", source_block_ids=["b_p1_0"]),
        ),
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="Omni-channel logistics", source_block_ids=["b_p1_1"]),
                description=GroundedString(
                    value="Omni-channel logistics and supply chain restructuring across 400 stores. Reduced annual logistics costs by $38M; improved order fulfillment SLA by 40%.",
                    source_block_ids=["b_p1_1", "b_p1_2"],
                ),
            )
        ],
    )
    violations = validate_semantic_output(output_valid, sem_input)
    assert not violations

    # Invalid: extra hallucinated token in description
    output_hallucinated = SemanticOutput(
        personal=GroundedPersonal(
            name=GroundedString(value="Arthur Pendelton", source_block_ids=["b_p1_0"]),
        ),
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="Omni-channel logistics", source_block_ids=["b_p1_1"]),
                description=GroundedString(
                    value="Omni-channel logistics and global supply chain restructuring across 400 stores. Reduced annual logistics costs by $38M; improved order fulfillment SLA by 40%.",
                    source_block_ids=["b_p1_1", "b_p1_2"],
                ),
            )
        ],
    )
    violations_h = validate_semantic_output(output_hallucinated, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in projects[0].description" in v for v in violations_h)
