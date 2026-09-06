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
    PersonalSemanticOutput,
    BodySemanticOutput,
    SemanticBlockCategory,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    SemanticPageMeta,
    _is_multiblock_text_semantically_supported,
    _is_value_semantically_supported,
    build_semantic_input,
    has_explicit_skills_evidence,
    normalize_semantic_output_skills,
    sanitize_grounded_skills,
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


def test_explicit_current_status_grounding_suite():
    """Phase 10I: Comprehensive test suite verifying explicit evidence requirements for boolean current status."""
    sem_input = SemanticInput(
        document_id="current-status-test",
        page_count=1,
        blocks=[
            SemanticBlockInput(
                block_id="b_pres",
                text="Acme Corp - Senior Software Engineer (2020 - Present)",
                page=1,
                bbox=[0, 0, 100, 20],
                region_id="r1",
                region_kind="column",
                reading_order=1,
            ),
            SemanticBlockInput(
                block_id="b_curr",
                text="Beta Ltd - Currently working as Lead Architect (2022 - Now)",
                page=1,
                bbox=[0, 20, 100, 40],
                region_id="r1",
                region_kind="column",
                reading_order=2,
            ),
            SemanticBlockInput(
                block_id="b_ongoing",
                text="NIH Grant R01: Neural Decoding (2021 - 2026, Ongoing research)",
                page=1,
                bbox=[0, 40, 100, 60],
                region_id="r1",
                region_kind="column",
                reading_order=3,
            ),
            SemanticBlockInput(
                block_id="b_future_dates_only",
                text="NIH R01-EB028491: Real-Time Neural Signal Decoding ($2.4M, PI, 2021 - 2026)",
                page=1,
                bbox=[0, 60, 100, 80],
                region_id="r1",
                region_kind="column",
                reading_order=4,
            ),
            SemanticBlockInput(
                block_id="b_past_dates_only",
                text="NSF Award #1548201: Cortical Interfaces ($1.1M, PI, 2016 - 2021)",
                page=1,
                bbox=[0, 80, 100, 100],
                region_id="r1",
                region_kind="column",
                reading_order=5,
            ),
        ],
    )

    # 1. Explicit "Present" -> current=True allowed
    out_pres = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Acme Corp", source_block_ids=["b_pres"]),
                current=GroundedBool(value=True, source_block_ids=["b_pres"]),
            )
        ]
    )
    assert validate_semantic_output(out_pres, sem_input) == []

    # 2. Explicit "Current" / "Now" -> current=True allowed
    out_curr = SemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Beta Ltd", source_block_ids=["b_curr"]),
                current=GroundedBool(value=True, source_block_ids=["b_curr"]),
            )
        ]
    )
    assert validate_semantic_output(out_curr, sem_input) == []

    # 3. Explicit "Ongoing" -> current=True allowed
    out_ongoing = SemanticOutput(
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="Neural Decoding", source_block_ids=["b_ongoing"]),
                current=GroundedBool(value=True, source_block_ids=["b_ongoing"]),
            )
        ]
    )
    assert validate_semantic_output(out_ongoing, sem_input) == []

    # 4. "2021 - 2026" without explicit current marker -> current=True REJECTED
    out_future_true = SemanticOutput(
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="Real-Time Neural Signal Decoding", source_block_ids=["b_future_dates_only"]),
                current=GroundedBool(value=True, source_block_ids=["b_future_dates_only"]),
            )
        ]
    )
    viol_future = validate_semantic_output(out_future_true, sem_input)
    assert any("UNSUPPORTED_CURRENT_STATUS in projects[0].current" in v for v in viol_future)

    # 5. "2021 - 2026" with current=None (or omitted) -> ACCEPTED
    out_future_null = SemanticOutput(
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="Real-Time Neural Signal Decoding", source_block_ids=["b_future_dates_only"]),
                startDate=GroundedString(value="2021", source_block_ids=["b_future_dates_only"]),
                endDate=GroundedString(value="2026", source_block_ids=["b_future_dates_only"]),
                current=None,
            )
        ]
    )
    assert validate_semantic_output(out_future_null, sem_input) == []

    # 6. Historical date range without current marker -> current=True REJECTED
    out_past_true = SemanticOutput(
        projects=[
            GroundedProjectItem(
                name=GroundedString(value="Cortical Interfaces", source_block_ids=["b_past_dates_only"]),
                current=GroundedBool(value=True, source_block_ids=["b_past_dates_only"]),
            )
        ]
    )
    viol_past = validate_semantic_output(out_past_true, sem_input)
    assert any("UNSUPPORTED_CURRENT_STATUS in projects[0].current" in v for v in viol_past)


# =====================================================================
# 11. Provenance-Schema Hardening Tests
# =====================================================================


def test_grounded_string_schema_requires_value_and_provenance():
    """GroundedString generated JSON schema explicitly marks value and source_block_ids as required."""
    schema = GroundedString.model_json_schema()
    assert "required" in schema
    assert "value" in schema["required"]
    assert "source_block_ids" in schema["required"]
    assert schema["properties"]["source_block_ids"]["type"] == "array"


def test_grounded_bool_schema_requires_value_and_provenance():
    """GroundedBool generated JSON schema explicitly marks value and source_block_ids as required."""
    schema = GroundedBool.model_json_schema()
    assert "required" in schema
    assert "value" in schema["required"]
    assert "source_block_ids" in schema["required"]
    assert schema["properties"]["source_block_ids"]["type"] == "array"


def test_nested_grounded_fields_inherit_provenance_requirement():
    """Nested grounded fields across SemanticOutput, PersonalSemanticOutput, and BodySemanticOutput inherit required provenance."""
    from app.extractors.semantic_prompt import resolve_schema_defs

    # 1. SemanticOutput model_json_schema contains $defs with required provenance
    full_raw_schema = SemanticOutput.model_json_schema()
    defs = full_raw_schema.get("$defs", {})
    assert "GroundedString" in defs
    assert "value" in defs["GroundedString"]["required"]
    assert "source_block_ids" in defs["GroundedString"]["required"]
    assert "GroundedBool" in defs
    assert "value" in defs["GroundedBool"]["required"]
    assert "source_block_ids" in defs["GroundedBool"]["required"]

    # 2. Inlined PersonalSemanticOutput schema requires provenance on all personal fields
    personal_schema = resolve_schema_defs(PersonalSemanticOutput)
    name_schema = personal_schema["properties"]["personal"]["properties"]["name"]
    name_obj = name_schema["anyOf"][0] if "anyOf" in name_schema else name_schema
    assert "value" in name_obj["required"]
    assert "source_block_ids" in name_obj["required"]

    email_schema = personal_schema["properties"]["personal"]["properties"]["email"]
    email_obj = email_schema["anyOf"][0] if "anyOf" in email_schema else email_schema
    assert "value" in email_obj["required"]
    assert "source_block_ids" in email_obj["required"]

    phone_schema = personal_schema["properties"]["personal"]["properties"]["phone"]
    phone_obj = phone_schema["anyOf"][0] if "anyOf" in phone_schema else phone_schema
    assert "value" in phone_obj["required"]
    assert "source_block_ids" in phone_obj["required"]

    # 3. Inlined BodySemanticOutput schema requires provenance on skills, experience, and boolean current
    body_schema = resolve_schema_defs(BodySemanticOutput)
    skill_item = body_schema["properties"]["skills"]["items"]
    assert "value" in skill_item["required"]
    assert "source_block_ids" in skill_item["required"]

    comp_schema = body_schema["properties"]["experience"]["items"]["properties"]["company"]
    comp_obj = comp_schema["anyOf"][0] if "anyOf" in comp_schema else comp_schema
    assert "value" in comp_obj["required"]
    assert "source_block_ids" in comp_obj["required"]

    curr_schema = body_schema["properties"]["experience"]["items"]["properties"]["current"]
    curr_obj = curr_schema["anyOf"][0] if "anyOf" in curr_schema else curr_schema
    assert "value" in curr_obj["required"]
    assert "source_block_ids" in curr_obj["required"]


def test_grounded_models_runtime_behavior_preserved():
    """Pydantic runtime behavior remains unchanged (defaults source_block_ids to empty list when instantiated)."""
    gs = GroundedString(value="test")
    assert gs.value == "test"
    assert gs.raw_value is None
    assert gs.source_block_ids == []

    gb = GroundedBool(value=True)
    assert gb.value is True
    assert gb.source_block_ids == []

    gs_with_ids = GroundedString(value="test", source_block_ids=["b0"])
    assert gs_with_ids.source_block_ids == ["b0"]


# =====================================================================
# 12. Deterministic Skills-Evidence Guard Tests
# =====================================================================


def _make_block(
    block_id: str,
    text: str,
    page: int = 1,
    reading_order: int = 0,
    region_kind: str = "physical_region",
    suggested_role: str = "UNKNOWN",
    is_bold: bool = False,
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        text=text,
        page=page,
        bbox=[50.0, 50.0, 300.0, 70.0],
        region_id=f"page-{page}-region-0",
        region_kind=region_kind,
        column_id=None,
        reading_order=reading_order,
        is_bold=is_bold,
        font_size=11.0,
        suggested_role=suggested_role,
        spans=[{"text": text, "bbox": [50.0, 50.0, 300.0, 70.0]}],
    )


def test_consulting_style_input_no_skills_section_forces_empty_skills():
    """8a: Consulting-style input with client engagements and table layout but no skills section forces skills = []."""
    blocks = [
        _make_block("b_hdr_0", "Arthur Pendelton", 1, 0, region_kind="header", suggested_role="HEADER"),
        _make_block("b_hdr_1", "Principal Strategy & Operations Consultant", 1, 1, region_kind="header", suggested_role="HEADER"),
        _make_block("b_sec_sum", "EXECUTIVE SUMMARY", 1, 2, region_kind="column", suggested_role="ENTRY_TITLE", is_bold=True),
        _make_block("b_sum_text", "Senior management consultant with 12+ years experience.", 1, 3, region_kind="column", suggested_role="DESCRIPTION"),
        _make_block("b_sec_eng", "MAJOR CLIENT ENGAGEMENTS & PROGRAM PORTFOLIO", 1, 4, region_kind="column", suggested_role="ORGANIZATION", is_bold=True),
        _make_block("b_eng_col1", "Engagement Scope", 1, 5, region_kind="column", suggested_role="TECHNOLOGY"),
        _make_block("b_eng_client1", "Multinational Healthcare", 1, 6, region_kind="column", suggested_role="TECHNOLOGY"),
        _make_block("b_eng_client2", "Industrial Manufacturer", 1, 7, region_kind="column", suggested_role="TECHNOLOGY"),
        _make_block("b_eng_desc1", "chain restructuring across 400", 1, 8, region_kind="column", suggested_role="UNKNOWN"),
        _make_block("b_eng_desc2", "stores.", 1, 9, region_kind="column", suggested_role="DESCRIPTION"),
        _make_block("b_sec_emp", "EMPLOYMENT HISTORY", 2, 10, region_kind="physical_region", suggested_role="SECTION_HEADING", is_bold=True),
        _make_block("b_emp_title", "Principal Consultant", 2, 11, region_kind="physical_region", suggested_role="ENTRY_TITLE"),
        _make_block("b_sec_edu", "EDUCATION", 2, 12, region_kind="physical_region", suggested_role="SECTION_HEADING", is_bold=True),
        _make_block("b_edu_deg", "Master of Business Administration (MBA)", 2, 13, region_kind="physical_region", suggested_role="DESCRIPTION"),
        _make_block("b_sec_cert", "PROFESSIONAL CERTIFICATIONS", 2, 14, region_kind="physical_region", suggested_role="SECTION_HEADING", is_bold=True),
        _make_block("b_cert_val", "Project Management Professional (PMP)", 2, 15, region_kind="physical_region", suggested_role="BULLET"),
    ]
    sem_input = SemanticInput(
        document_id="doc-consulting",
        page_count=2,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0), SemanticPageMeta(page_number=2, width=612.0, height=792.0)],
        blocks=blocks,
    )

    # 1. Structural evidence check
    assert has_explicit_skills_evidence(sem_input) is False

    # 2. LLM incorrectly synthesized skills from job titles and narrative metrics
    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[],
        skills=[
            GroundedString(value="Strategy & Operations", source_block_ids=["b_hdr_1"]),
            GroundedString(value="400 stores", source_block_ids=["b_eng_desc1", "b_eng_desc2"]),
        ],
    )

    # 3. Deterministic normalization forces skills = []
    normalized_output = sanitize_grounded_skills(raw_output, sem_input)
    assert normalized_output.skills == []

    # 4. Normalized output passes validation cleanly
    violations = validate_semantic_output(normalized_output, sem_input)
    assert violations == []


def test_explicit_skills_section_preserves_skills():
    """8b: Explicit skills section preserves LLM-extracted grounded skills."""
    blocks = [
        _make_block("b_hdr", "Jane Smith", 1, 0, region_kind="header", suggested_role="HEADER"),
        _make_block("b_sec_sk", "TECHNICAL SKILLS", 1, 1, region_kind="physical_region", suggested_role="SECTION_HEADING", is_bold=True),
        _make_block("b_sk_1", "Python, Kubernetes, Docker, Go", 1, 2, region_kind="physical_region", suggested_role="SKILL"),
    ]
    sem_input = SemanticInput(
        document_id="doc-tech-skills",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )

    assert has_explicit_skills_evidence(sem_input) is True

    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[],
        skills=[
            GroundedString(value="Python", source_block_ids=["b_sk_1"]),
            GroundedString(value="Docker", source_block_ids=["b_sk_1"]),
        ],
    )

    normalized_output = sanitize_grounded_skills(raw_output, sem_input)
    assert len(normalized_output.skills) == 2
    assert normalized_output.skills[0].value == "Python"
    assert normalized_output.skills[1].value == "Docker"

    violations = validate_semantic_output(normalized_output, sem_input)
    assert violations == []


def test_technology_blocks_in_client_section_not_skills_evidence():
    """8c: TECHNOLOGY blocks inside a non-skills/client section do not activate skills evidence."""
    blocks = [
        _make_block("b_sec_clients", "CLIENTS", 1, 0, region_kind="physical_region", suggested_role="SECTION_HEADING", is_bold=True),
        _make_block("b_client_1", "Multinational Healthcare", 1, 1, region_kind="column", suggested_role="TECHNOLOGY"),
        _make_block("b_client_2", "Industrial Manufacturer", 1, 2, region_kind="column", suggested_role="TECHNOLOGY"),
    ]
    sem_input = SemanticInput(
        document_id="doc-clients-only",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )

    assert has_explicit_skills_evidence(sem_input) is False

    raw_output = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[],
        skills=[
            GroundedString(value="Multinational Healthcare", source_block_ids=["b_client_1"]),
        ],
    )

    normalized_output = sanitize_grounded_skills(raw_output, sem_input)
    assert normalized_output.skills == []


def test_existing_grounded_skills_behavior_remains_unchanged_and_strict():
    """8d: When explicit skills evidence exists, provenance validation remains strictly enforced."""
    blocks = [
        _make_block("b_sec_sk", "SKILLS", 1, 0, region_kind="physical_region", suggested_role="SECTION_HEADING", is_bold=True),
        _make_block("b_sk_1", "Python, SQL", 1, 1, region_kind="physical_region", suggested_role="SKILL"),
    ]
    sem_input = SemanticInput(
        document_id="doc-strict-validation",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks,
    )

    # 1. Hallucinated skill value not supported by source block is flagged
    output_hallucinated = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[],
        skills=[
            GroundedString(value="Rust", source_block_ids=["b_sk_1"]),
        ],
    )
    normalized_hallucinated = sanitize_grounded_skills(output_hallucinated, sem_input)
    assert len(normalized_hallucinated.skills) == 1
    violations = validate_semantic_output(normalized_hallucinated, sem_input)
    assert any("UNSUPPORTED_CANONICAL_VALUE in skills[0]" in v for v in violations)

    # 2. Missing provenance is flagged
    output_missing_prov = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[],
        skills=[
            GroundedString(value="Python", source_block_ids=[]),
        ],
    )
    normalized_missing = sanitize_grounded_skills(output_missing_prov, sem_input)
    violations_missing = validate_semantic_output(normalized_missing, sem_input)
    assert any("MISSING_PROVENANCE in skills[0]" in v for v in violations_missing)


def test_sanitize_grounded_skills_idempotence():
    """8e: sanitize_grounded_skills is strictly idempotent across both presence and absence of skills evidence."""
    import copy

    # Case 1: Absence of skills evidence
    blocks_no_skills = [
        _make_block("b_sec_exp", "EXPERIENCE", 1, 0, region_kind="physical_region", suggested_role="SECTION_HEADING", is_bold=True),
        _make_block("b_exp_title", "Consultant", 1, 1, region_kind="physical_region", suggested_role="ENTRY_TITLE"),
    ]
    sem_input_no_skills = SemanticInput(
        document_id="doc-idempotent-no-skills",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks_no_skills,
    )
    output_no_skills = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[],
        skills=[GroundedString(value="Consulting", source_block_ids=["b_exp_title"])],
    )

    out1 = sanitize_grounded_skills(copy.deepcopy(output_no_skills), sem_input_no_skills)
    out2 = sanitize_grounded_skills(copy.deepcopy(out1), sem_input_no_skills)
    out3 = normalize_semantic_output_skills(copy.deepcopy(out2), sem_input_no_skills)
    assert out1.skills == []
    assert out2.skills == []
    assert out3.skills == []

    # Case 2: Presence of skills evidence
    blocks_with_skills = [
        _make_block("b_sec_sk", "SKILLS", 1, 0, region_kind="physical_region", suggested_role="SECTION_HEADING", is_bold=True),
        _make_block("b_sk_1", "Python, SQL", 1, 1, region_kind="physical_region", suggested_role="SKILL"),
    ]
    sem_input_with_skills = SemanticInput(
        document_id="doc-idempotent-with-skills",
        page_count=1,
        archetype=DocumentArchetype.STANDARD_CV,
        pages=[SemanticPageMeta(page_number=1, width=612.0, height=792.0)],
        blocks=blocks_with_skills,
    )
    output_with_skills = SemanticOutput(
        document_archetype=DocumentArchetype.STANDARD_CV,
        block_classifications=[],
        skills=[
            GroundedString(value="Python", source_block_ids=["b_sk_1"]),
            GroundedString(value="SQL", source_block_ids=["b_sk_1"]),
        ],
    )

    out_s1 = sanitize_grounded_skills(copy.deepcopy(output_with_skills), sem_input_with_skills)
    out_s2 = sanitize_grounded_skills(copy.deepcopy(out_s1), sem_input_with_skills)
    out_s3 = normalize_semantic_output_skills(copy.deepcopy(out_s2), sem_input_with_skills)
    assert len(out_s1.skills) == 2
    assert len(out_s2.skills) == 2
    assert len(out_s3.skills) == 2
    assert [s.value for s in out_s1.skills] == ["Python", "SQL"]
    assert [s.value for s in out_s2.skills] == ["Python", "SQL"]
    assert [s.value for s in out_s3.skills] == ["Python", "SQL"]


