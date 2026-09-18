"""Phase 10U-Fix-2 Tests: Target-Isolated Body Merge & Unified Academic Appointment Provenance.

Validates:
1. Education response containing spurious experience entities being reduced to education-only before merge.
2. Experience response containing unrelated collections being reduced to experience-only.
3. Postdoc final merged experience containing exactly four appointment entities (spurious education experience purged).
4. Authoritative appointment boundaries matching extraction and validation.
5. b_p2_39 belonging to the same appointment boundary in both extraction and validation.
6. Zero CROSS_ENTITY_PROVENANCE for the four postdoc appointments.
7. Tenured-professor academic extraction remaining valid.
8. STANDARD_CV monolithic behavior remaining unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    BodySemanticOutput,
    DeterministicAppointmentGroup,
    DeterministicEntitySpan,
    DocumentArchetype,
    GroundedEducationItem,
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedString,
    PersonalSemanticOutput,
    SemanticInput,
    SemanticOutput,
    build_deterministic_appointment_groups,
    build_deterministic_experience_spans,
    build_semantic_input,
    isolate_unit_target_collections,
    merge_body_outputs,
    partition_semantic_input_into_sections,
    plan_section_aware_body_passes,
    should_use_section_aware_body_extraction,
    validate_semantic_output,
)
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.providers.nvidia import NvidiaSemanticExtractor
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor


# =====================================================================
# 1. Target Isolation: Education Response Purges Experience
# =====================================================================

def test_education_response_spurious_experience_reduced_to_education_only():
    """An education response containing spurious experience entities is reduced to education-only before merge."""
    raw_body = BodySemanticOutput(
        education=[
            GroundedEducationItem(
                institution=GroundedString(value="MIT", source_block_ids=["b_p1_3"]),
                degree=GroundedString(value="Ph.D.", source_block_ids=["b_p1_4"]),
                source_block_ids=["b_p1_3", "b_p1_4"],
            )
        ],
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="MIT", source_block_ids=["b_p1_3"]),
                designation=GroundedString(value="Ph.D. Researcher", source_block_ids=["b_p1_4"]),
                source_block_ids=["b_p1_3", "b_p1_4"],
            )
        ],
        skills=[GroundedString(value="Python", source_block_ids=["b_p1_5"])],
    )

    isolated = isolate_unit_target_collections(raw_body, "education")
    assert len(isolated.education) == 1
    assert isolated.education[0].institution.value == "MIT"
    assert isolated.experience == []
    assert isolated.skills == []
    assert isolated.projects == []
    assert isolated.certifications == []
    assert isolated.languages == []
    assert isolated.achievements == []


# =====================================================================
# 2. Target Isolation: Experience Response Purges Unrelated Collections
# =====================================================================

def test_experience_response_unrelated_collections_reduced_to_experience_only():
    """An experience response containing unrelated collections is reduced to experience-only."""
    raw_body = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Stanford University", source_block_ids=["b_p1_15"]),
                designation=GroundedString(value="Postdoctoral Research Fellow", source_block_ids=["b_p1_13"]),
                source_block_ids=["b_p1_13", "b_p1_15"],
            )
        ],
        education=[
            GroundedEducationItem(
                institution=GroundedString(value="Stanford University", source_block_ids=["b_p1_15"]),
                source_block_ids=["b_p1_15"],
            )
        ],
        skills=[GroundedString(value="Distributed Systems", source_block_ids=["b_p1_16"])],
    )

    isolated = isolate_unit_target_collections(raw_body, "experience")
    assert len(isolated.experience) == 1
    assert isolated.experience[0].designation.value == "Postdoctoral Research Fellow"
    assert isolated.education == []
    assert isolated.skills == []


# =====================================================================
# 3. Postdoc Final Merged Experience Contains Exactly Four Appointments
# =====================================================================

def test_postdoc_final_merged_experience_contains_exactly_four_entities():
    """Mocking all 7 postdoc units with spurious education experience yields exactly 4 experience items."""
    fpath = Path("tests/fixtures/generalization/academic_research_postdoc_cv.pdf")
    raw = fpath.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(layout_doc, document_id=fpath.name, archetype=DocumentArchetype.ACADEMIC_CV)

    units = plan_section_aware_body_passes(sinput)
    assert len(units) == 7

    # Unit 0 is education; simulate model emitting 2 spurious experience items
    edu_out = BodySemanticOutput(
        education=[
            GroundedEducationItem(institution=GroundedString(value="MIT", source_block_ids=["b_p1_3"]))
        ],
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="MIT", source_block_ids=["b_p1_3"]),
                designation=GroundedString(value="Graduate Student", source_block_ids=["b_p1_4"]),
                source_block_ids=["b_p1_3", "b_p1_4"],
            ),
            GroundedExperienceItem(
                company=GroundedString(value="Tsinghua", source_block_ids=["b_p1_5"]),
                designation=GroundedString(value="Undergraduate", source_block_ids=["b_p1_5"]),
                source_block_ids=["b_p1_5"],
            ),
        ],
    )

    # Units 1-4 are legitimate appointments
    appt1_out = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Stanford University", source_block_ids=["b_p1_15"]),
                designation=GroundedString(value="Postdoctoral Research Fellow", source_block_ids=["b_p1_13"]),
                startDate=GroundedString(value="2023 - Present", source_block_ids=["b_p1_14"]),
                endDate=GroundedString(value="2023 - Present", source_block_ids=["b_p1_14"]),
                source_block_ids=["b_p1_13", "b_p1_14", "b_p1_15", "b_p1_16", "b_p1_17"],
            )
        ]
    )
    appt2_out = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="MIT CSAIL", source_block_ids=["b_p1_20"]),
                designation=GroundedString(value="Graduate Research Assistant", source_block_ids=["b_p1_18"]),
                startDate=GroundedString(value="2018 - 2023", source_block_ids=["b_p1_19"]),
                endDate=GroundedString(value="2018 - 2023", source_block_ids=["b_p1_19"]),
                source_block_ids=["b_p1_18", "b_p1_19", "b_p1_20", "b_p1_21", "b_p1_22"],
            )
        ]
    )
    appt3_out = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Stanford University", source_block_ids=["b_p2_39"]),
                designation=GroundedString(value="Teaching Assistant · CS 244B: Distributed Systems", source_block_ids=["b_p2_37"]),
                startDate=GroundedString(value="Spring 2024", source_block_ids=["b_p2_38"]),
                endDate=GroundedString(value="Spring 2024", source_block_ids=["b_p2_38"]),
                source_block_ids=["b_p2_37", "b_p2_38", "b_p2_39"],
            )
        ]
    )
    appt4_out = BodySemanticOutput(
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="MIT EECS", source_block_ids=["b_p2_42"]),
                designation=GroundedString(value="Teaching Assistant · 6.824: Distributed Systems", source_block_ids=["b_p2_40"]),
                startDate=GroundedString(value="Fall 2021, Fall 2022", source_block_ids=["b_p2_41"]),
                endDate=GroundedString(value="Fall 2021, Fall 2022", source_block_ids=["b_p2_41"]),
                source_block_ids=["b_p2_40", "b_p2_41", "b_p2_42"],
            )
        ]
    )

    outputs = [edu_out, appt1_out, appt2_out, appt3_out, appt4_out, BodySemanticOutput(), BodySemanticOutput()]

    isolated_outputs: list[BodySemanticOutput] = []
    for u, o in zip(units, outputs):
        iso = isolate_unit_target_collections(o, u.canonical_target)
        isolated_outputs.append(iso)

    merged = merge_body_outputs(isolated_outputs)
    assert len(merged.experience) == 4
    assert merged.experience[0].designation.value == "Postdoctoral Research Fellow"
    assert merged.experience[1].designation.value == "Graduate Research Assistant"
    assert merged.experience[2].designation.value == "Teaching Assistant · CS 244B: Distributed Systems"
    assert merged.experience[3].designation.value == "Teaching Assistant · 6.824: Distributed Systems"


# =====================================================================
# 4. Authoritative Appointment Boundaries Match Extraction & Validation
# =====================================================================

def test_authoritative_appointment_boundaries_match_extraction_and_validation():
    """Extraction unit block IDs and validation span block IDs are identical."""
    for fixture in ["academic_research_postdoc_cv.pdf", "long_academic_tenured_professor_cv.pdf"]:
        fpath = Path("tests/fixtures/generalization") / fixture
        raw = fpath.read_bytes()
        doc = document_from_text_blocks(PDFExtractor.extract(raw))
        rec = reconstruct_document(doc)
        layout_doc = interpret_layout(rec)
        sinput = build_semantic_input(layout_doc, document_id=fixture, archetype=DocumentArchetype.ACADEMIC_CV)

        units = [u for u in plan_section_aware_body_passes(sinput) if u.is_appointment]
        spans = build_deterministic_experience_spans(sinput)

        assert len(units) == len(spans), f"Count mismatch for {fixture}: {len(units)} units vs {len(spans)} spans"
        for idx, (u, s) in enumerate(zip(units, spans)):
            unit_bids = [b.block_id for b in u.section_input.blocks]
            span_bids = s.block_ids
            assert set(unit_bids) == set(span_bids), f"Mismatch at appointment {idx} for {fixture}: unit={unit_bids} vs span={span_bids}"
            assert u.title_block_id == s.title_block_id, f"Title mismatch at {idx} for {fixture}"


# =====================================================================
# 5. b_p2_39 Belongs to Same Appointment Boundary in Extraction & Validation
# =====================================================================

def test_b_p2_39_belongs_to_same_appointment_boundary_in_extraction_and_validation():
    """b_p2_39 belongs to Appointment 3 in extraction and Span 2 in validation."""
    fpath = Path("tests/fixtures/generalization/academic_research_postdoc_cv.pdf")
    raw = fpath.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(layout_doc, document_id=fpath.name, archetype=DocumentArchetype.ACADEMIC_CV)

    units = [u for u in plan_section_aware_body_passes(sinput) if u.is_appointment]
    spans = build_deterministic_experience_spans(sinput)

    # Appointment 3 is index 2
    appt3_bids = [b.block_id for b in units[2].section_input.blocks]
    span2_bids = spans[2].block_ids

    assert "b_p2_39" in appt3_bids
    assert "b_p2_39" in span2_bids
    assert appt3_bids == ["b_p2_37", "b_p2_38", "b_p2_39"]
    assert span2_bids == ["b_p2_37", "b_p2_38", "b_p2_39"]

    # Appointment 4 / Span 3 must NOT contain b_p2_39
    appt4_bids = [b.block_id for b in units[3].section_input.blocks]
    span3_bids = spans[3].block_ids

    assert "b_p2_39" not in appt4_bids
    assert "b_p2_39" not in span3_bids
    assert appt4_bids == ["b_p2_40", "b_p2_41", "b_p2_42"]
    assert span3_bids == ["b_p2_40", "b_p2_41", "b_p2_42"]


# =====================================================================
# 6. Zero CROSS_ENTITY_PROVENANCE for the Four Postdoc Appointments
# =====================================================================

def test_zero_cross_entity_provenance_for_four_postdoc_appointments():
    """Full extraction of academic_research_postdoc_cv.pdf produces 0 CROSS_ENTITY_PROVENANCE violations."""
    fpath = Path("tests/fixtures/generalization/academic_research_postdoc_cv.pdf")
    raw = fpath.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(layout_doc, document_id=fpath.name, archetype=DocumentArchetype.ACADEMIC_CV)

    # Build semantic output representing clean 4-appointment extraction
    output = SemanticOutput(
        document_archetype=DocumentArchetype.ACADEMIC_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="Alex Chen", source_block_ids=["b_p1_0"]),
            email=GroundedString(value="alex.chen@cs.stanford.edu", source_block_ids=["b_p1_1"]),
        ),
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Stanford University", source_block_ids=["b_p1_15"]),
                designation=GroundedString(value="Postdoctoral Research Fellow", source_block_ids=["b_p1_13"]),
                startDate=GroundedString(value="2023 - Present", source_block_ids=["b_p1_14"]),
                endDate=GroundedString(value="2023 - Present", source_block_ids=["b_p1_14"]),
                description=GroundedString(
                    value="Investigating asynchronous replication protocols for geo-distributed pipelines. Published 2 primary papers in top-tier conferences.",
                    source_block_ids=["b_p1_16", "b_p1_17"],
                ),
                source_block_ids=["b_p1_13", "b_p1_14", "b_p1_15", "b_p1_16", "b_p1_17"],
            ),
            GroundedExperienceItem(
                company=GroundedString(value="Computer Science and Artificial Intelligence Laboratory (CSAIL), MIT", source_block_ids=["b_p1_20"]),
                designation=GroundedString(value="Graduate Research Assistant", source_block_ids=["b_p1_18"]),
                startDate=GroundedString(value="2018 - 2023", source_block_ids=["b_p1_19"]),
                endDate=GroundedString(value="2018 - 2023", source_block_ids=["b_p1_19"]),
                description=GroundedString(
                    value="Designed and benchmarked a novel Byzantine fault-tolerant consensus algorithm. Co-authored 4 peer-reviewed conference publications.",
                    source_block_ids=["b_p1_21", "b_p1_22"],
                ),
                source_block_ids=["b_p1_18", "b_p1_19", "b_p1_20", "b_p1_21", "b_p1_22"],
            ),
            GroundedExperienceItem(
                company=GroundedString(value="Stanford University", source_block_ids=["b_p2_39"]),
                designation=GroundedString(value="Teaching Assistant · CS 244B: Distributed Systems", source_block_ids=["b_p2_37"]),
                startDate=GroundedString(value="Spring 2024", source_block_ids=["b_p2_38"]),
                endDate=GroundedString(value="Spring 2024", source_block_ids=["b_p2_38"]),
                source_block_ids=["b_p2_37", "b_p2_38", "b_p2_39"],
            ),
            GroundedExperienceItem(
                company=GroundedString(value="MIT Department of Electrical Engineering and Computer Science", source_block_ids=["b_p2_42"]),
                designation=GroundedString(value="Teaching Assistant · 6.824: Distributed Systems", source_block_ids=["b_p2_40"]),
                startDate=GroundedString(value="Fall 2021, Fall 2022", source_block_ids=["b_p2_41"]),
                endDate=GroundedString(value="Fall 2021, Fall 2022", source_block_ids=["b_p2_41"]),
                source_block_ids=["b_p2_40", "b_p2_41", "b_p2_42"],
            ),
        ],
    )

    violations = validate_semantic_output(output, sinput)
    cross_entity_violations = [v for v in violations if "CROSS_ENTITY_PROVENANCE" in v]
    assert cross_entity_violations == []


# =====================================================================
# 7. Tenured Professor Academic Extraction Remains Valid
# =====================================================================

def test_tenured_professor_academic_extraction_remaining_valid():
    """long_academic_tenured_professor_cv.pdf extraction remains fully valid with 0 violations."""
    fpath = Path("tests/fixtures/generalization/long_academic_tenured_professor_cv.pdf")
    raw = fpath.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(layout_doc, document_id=fpath.name, archetype=DocumentArchetype.ACADEMIC_CV)

    output = SemanticOutput(
        document_archetype=DocumentArchetype.ACADEMIC_CV,
        personal=GroundedPersonal(
            name=GroundedString(value="Benjamin Thorne", source_block_ids=["b_p1_0"]),
            email=GroundedString(value="bthorne@jhu.edu", source_block_ids=["b_p1_2"]),
        ),
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="Department of Biomedical Engineering, Johns Hopkins University", source_block_ids=["b_p1_6"]),
                designation=GroundedString(value="Professor with Tenure", source_block_ids=["b_p1_4"]),
                startDate=GroundedString(value="2019 - Present", source_block_ids=["b_p1_5"]),
                endDate=GroundedString(value="2019 - Present", source_block_ids=["b_p1_5"]),
                source_block_ids=["b_p1_4", "b_p1_5", "b_p1_6"],
            ),
            GroundedExperienceItem(
                company=GroundedString(value="Department of Biomedical Engineering, Johns Hopkins University", source_block_ids=["b_p1_9"]),
                designation=GroundedString(value="Associate Professor", source_block_ids=["b_p1_7"]),
                startDate=GroundedString(value="2014 - 2019", source_block_ids=["b_p1_8"]),
                endDate=GroundedString(value="2014 - 2019", source_block_ids=["b_p1_8"]),
                source_block_ids=["b_p1_7", "b_p1_8", "b_p1_9"],
            ),
            GroundedExperienceItem(
                company=GroundedString(value="Department of Bioengineering, University of Pennsylvania", source_block_ids=["b_p1_12"]),
                designation=GroundedString(value="Assistant Professor", source_block_ids=["b_p1_10"]),
                startDate=GroundedString(value="2008 - 2014", source_block_ids=["b_p1_11"]),
                endDate=GroundedString(value="2008 - 2014", source_block_ids=["b_p1_11"]),
                source_block_ids=["b_p1_10", "b_p1_11", "b_p1_12"],
            ),
            GroundedExperienceItem(
                company=GroundedString(value="Division of Biology and Bioengineering, California Institute of Technology", source_block_ids=["b_p1_15"]),
                designation=GroundedString(value="Postdoctoral Research Fellow", source_block_ids=["b_p1_13"]),
                startDate=GroundedString(value="2005 - 2008", source_block_ids=["b_p1_14"]),
                endDate=GroundedString(value="2005 - 2008", source_block_ids=["b_p1_14"]),
                source_block_ids=["b_p1_13", "b_p1_14", "b_p1_15"],
            ),
        ],
    )

    violations = validate_semantic_output(output, sinput)
    cross_entity_violations = [v for v in violations if "CROSS_ENTITY_PROVENANCE" in v]
    assert cross_entity_violations == []


# =====================================================================
# 8. STANDARD_CV Monolithic Behavior Remaining Unchanged
# =====================================================================

def test_standard_cv_monolithic_behavior_remaining_unchanged():
    """STANDARD_CV resumes continue using monolithic body extraction and generic spans."""
    fpath = Path("tests/fixtures/generalization/dense_technical_infrastructure_engineer.pdf")
    raw = fpath.read_bytes()
    doc = document_from_text_blocks(PDFExtractor.extract(raw))
    rec = reconstruct_document(doc)
    layout_doc = interpret_layout(rec)
    sinput = build_semantic_input(layout_doc, document_id=fpath.name, archetype=DocumentArchetype.STANDARD_CV)

    assert sinput.archetype == DocumentArchetype.STANDARD_CV
    # Section-aware mode is disabled for STANDARD_CV normal path
    assert should_use_section_aware_body_extraction(sinput) is False

    # build_deterministic_experience_spans uses the generic algorithm
    spans = build_deterministic_experience_spans(sinput)
    assert len(spans) > 0
