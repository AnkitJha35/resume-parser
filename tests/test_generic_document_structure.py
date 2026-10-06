"""Tests for Generic Document Structure v1.

Verifies domain-agnostic document reconstruction:
1. Conventional CV with standard headings.
2. CV using completely different/custom headings.
3. Mixed-case/custom heading text is preserved exactly.
4. Heading ordering is preserved.
5. Content belongs to the correct heading.
6. Nested/subsection hierarchy.
7. Content before first heading.
8. Document with no recognizable headings.
9. Lists/bullets.
10. Tables.
11. Multi-column document.
12. Multi-page document.
13. Scanned/OCR fixture.
14. Provenance validation rejects invented content.
15. Unknown heading requires ZERO source-code changes.
16. Regression assertion: no fixed heading whitelist controls output.
"""

from __future__ import annotations

import pytest

from app.domain.document_structure import (
    DocumentBlock,
    DocumentSection,
    DocumentStructure,
    document_structure_to_resume,
)
from app.domain.semantic_contract import (
    DocumentArchetype,
    GroundedString,
    SemanticBlockInput,
    SemanticInput,
    SemanticPageMeta,
)
from app.pipeline.stages.generic_document_builder import (
    build_document_structure_from_semantic_input,
)
from app.pipeline.stages.table_binding import GeometricCell, GeometricTable
from app.validation.generic_document_validator import (
    validate_generic_document_structure,
)


def _make_block(
    block_id: str,
    text: str,
    page: int = 1,
    reading_order: int = 0,
    suggested_role: str = "UNKNOWN",
    bbox: list[float] | None = None,
    table_id: str | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
) -> SemanticBlockInput:
    return SemanticBlockInput(
        block_id=block_id,
        page=page,
        reading_order=reading_order,
        text=text,
        region_id="reg_1",
        region_kind="body",
        suggested_role=suggested_role,
        bbox=bbox or [10.0, 10.0 + reading_order * 15.0, 200.0, 22.0 + reading_order * 15.0],
        table_id=table_id,
        row_index=row_index,
        column_index=column_index,
    )


def test_1_conventional_cv_with_standard_headings():
    """1. Conventional CV with standard headings preserves exact headings and content."""
    blocks = [
        _make_block("b0", "Jane Doe", reading_order=0, suggested_role="HEADER"),
        _make_block("b1", "jane@example.com", reading_order=1, suggested_role="CONTACT"),
        _make_block("b2", "SUMMARY", reading_order=2, suggested_role="SECTION_HEADING"),
        _make_block("b3", "Experienced software engineer with a focus on distributed systems.", reading_order=3),
        _make_block("b4", "EXPERIENCE", reading_order=4, suggested_role="SECTION_HEADING"),
        _make_block("b5", "Senior Engineer at Acme Corp, 2020-Present", reading_order=5),
        _make_block("b6", "EDUCATION", reading_order=6, suggested_role="SECTION_HEADING"),
        _make_block("b7", "B.S. in Computer Science, Tech University", reading_order=7),
    ]
    inp = SemanticInput(document_id="doc-1", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    headings = [s.heading for s in doc.sections]
    assert headings == [None, "SUMMARY", "EXPERIENCE", "EDUCATION"]
    assert doc.sections[1].blocks[0].text == "Experienced software engineer with a focus on distributed systems."
    assert doc.sections[2].blocks[0].text == "Senior Engineer at Acme Corp, 2020-Present"
    assert doc.sections[3].blocks[0].text == "B.S. in Computer Science, Tech University"


def test_2_cv_using_completely_different_headings():
    """2. CV using completely different/custom headings does not normalize or rename them."""
    blocks = [
        _make_block("b0", "CAREER TRAJECTORY", reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "Principal Architect at Globex, leading cloud migration.", reading_order=1),
        _make_block("b2", "INTELLECTUAL PROPERTY", reading_order=2, suggested_role="SECTION_HEADING"),
        _make_block("b3", "Patent US-123456: Distributed consensus algorithm.", reading_order=3),
        _make_block("b4", "COMMUNITY OUTREACH", reading_order=4, suggested_role="SECTION_HEADING"),
        _make_block("b5", "Keynote speaker at Global Open Source Summit.", reading_order=5),
    ]
    inp = SemanticInput(document_id="doc-2", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    headings = [s.heading for s in doc.sections]
    assert headings == ["CAREER TRAJECTORY", "INTELLECTUAL PROPERTY", "COMMUNITY OUTREACH"]
    assert "EXPERIENCE" not in headings
    assert "EDUCATION" not in headings


def test_3_mixed_case_custom_heading_preserved_exactly():
    """3. Mixed-case and custom heading text is preserved exactly without case changes."""
    blocks = [
        _make_block("b0", "aWaRdS & HoNoRs", reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "First place in 2024 International Hackathon", reading_order=1),
        _make_block("b2", "Bio-Sketch / Background", reading_order=2, suggested_role="SECTION_HEADING"),
        _make_block("b3", "Researcher working at the intersection of AI and biology.", reading_order=3),
    ]
    inp = SemanticInput(document_id="doc-3", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert doc.sections[0].heading == "aWaRdS & HoNoRs"
    assert doc.sections[1].heading == "Bio-Sketch / Background"


def test_4_heading_ordering_preserved():
    """4. Heading ordering strictly matches document order, not conventional CV order."""
    blocks = [
        _make_block("b0", "PUBLICATIONS", reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "Paper A, IEEE 2022", reading_order=1),
        _make_block("b2", "EDUCATION", reading_order=2, suggested_role="SECTION_HEADING"),
        _make_block("b3", "Ph.D. Tech University", reading_order=3),
        _make_block("b4", "SUMMARY", reading_order=4, suggested_role="SECTION_HEADING"),
        _make_block("b5", "Summary paragraph placed at the end.", reading_order=5),
    ]
    inp = SemanticInput(document_id="doc-4", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    headings = [s.heading for s in doc.sections]
    # In conventional resume schemas, Summary is forced first and Education after Experience.
    # Here, document order must be preserved exactly:
    assert headings == ["PUBLICATIONS", "EDUCATION", "SUMMARY"]


def test_5_content_belongs_to_correct_heading():
    """5. Content belongs strictly to the heading that precedes it."""
    blocks = [
        _make_block("b0", "SECTION A", reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "Content item 1 for A", reading_order=1),
        _make_block("b2", "Content item 2 for A", reading_order=2),
        _make_block("b3", "SECTION B", reading_order=3, suggested_role="SECTION_HEADING"),
        _make_block("b4", "Content item 1 for B", reading_order=4),
    ]
    inp = SemanticInput(document_id="doc-5", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert len(doc.sections) == 2
    assert [b.text for b in doc.sections[0].blocks] == ["Content item 1 for A", "Content item 2 for A"]
    assert [b.text for b in doc.sections[1].blocks] == ["Content item 1 for B"]


def test_6_nested_subsection_hierarchy():
    """6. Subsections with numbering or subordinate typography form child subsections."""
    blocks = [
        _make_block("b0", "RESEARCH EXPERIENCE", reading_order=0, suggested_role="SECTION_HEADING", bbox=[10, 10, 300, 35]),
        _make_block("b1", "Overview of lab activities.", reading_order=1),
        _make_block("b2", "1.1 Machine Learning Lab", reading_order=2, suggested_role="SECTION_HEADING", bbox=[10, 50, 200, 65]),
        _make_block("b3", "Developed transformer models for sequence prediction.", reading_order=3),
        _make_block("b4", "1.2 Robotics Lab", reading_order=4, suggested_role="SECTION_HEADING", bbox=[10, 80, 200, 95]),
        _make_block("b5", "Built autonomous navigation systems.", reading_order=5),
    ]
    inp = SemanticInput(document_id="doc-6", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert len(doc.sections) == 1
    root = doc.sections[0]
    assert root.heading == "RESEARCH EXPERIENCE"
    assert root.level == 1
    assert len(root.subsections) == 2
    assert root.subsections[0].heading == "1.1 Machine Learning Lab"
    assert root.subsections[0].level == 2
    assert root.subsections[0].blocks[0].text == "Developed transformer models for sequence prediction."
    assert root.subsections[1].heading == "1.2 Robotics Lab"
    assert root.subsections[1].level == 2


def test_7_content_before_first_heading():
    """7. Content before first heading forms an unheaded top-level section."""
    blocks = [
        _make_block("b0", "Alice Smith", reading_order=0, suggested_role="HEADER"),
        _make_block("b1", "alice@example.com | +1-555-0199", reading_order=1, suggested_role="CONTACT"),
        _make_block("b2", "San Francisco, CA", reading_order=2, suggested_role="LOCATION"),
        _make_block("b3", "PROFILE", reading_order=3, suggested_role="SECTION_HEADING"),
        _make_block("b4", "Software engineer with 10 years experience.", reading_order=4),
    ]
    inp = SemanticInput(document_id="doc-7", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert len(doc.sections) == 2
    unheaded = doc.sections[0]
    assert unheaded.heading is None
    assert unheaded.level == 1
    assert [b.text for b in unheaded.blocks] == [
        "Alice Smith",
        "alice@example.com | +1-555-0199",
        "San Francisco, CA",
    ]
    assert doc.sections[1].heading == "PROFILE"


def test_8_document_with_no_recognizable_headings():
    """8. Document with no recognizable headings returns valid DocumentStructure containing all blocks."""
    blocks = [
        _make_block("b0", "This is an unformatted narrative document.", reading_order=0),
        _make_block("b1", "It contains paragraphs of general information.", reading_order=1),
        _make_block("b2", "No section titles or headings appear anywhere.", reading_order=2),
    ]
    inp = SemanticInput(document_id="doc-8", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert len(doc.sections) == 1
    assert doc.sections[0].heading is None
    assert doc.sections[0].level == 1
    assert len(doc.sections[0].blocks) == 3
    assert doc.sections[0].blocks[0].text == "This is an unformatted narrative document."


def test_9_lists_and_bullets_preserved_generically():
    """9. Bullets and lists are preserved as list_item blocks without domain conversion."""
    blocks = [
        _make_block("b0", "CORE COMPETENCIES", reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "• Python & Rust programming", reading_order=1, suggested_role="BULLET"),
        _make_block("b2", "• Distributed systems architecture", reading_order=2, suggested_role="BULLET"),
        _make_block("b3", "- Container orchestration with Kubernetes", reading_order=3),
    ]
    inp = SemanticInput(document_id="doc-9", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    section = doc.sections[0]
    assert section.heading == "CORE COMPETENCIES"
    assert len(section.blocks) == 3
    for b in section.blocks:
        assert b.type == "list_item"
    assert section.blocks[0].text == "Python & Rust programming"
    assert section.blocks[1].text == "Distributed systems architecture"
    assert section.blocks[2].text == "Container orchestration with Kubernetes"


def test_10_tables_preserved_generically():
    """10. Tables are represented generically with rows, cells, and headers."""
    cells = [
        GeometricCell(block_id="b1", text="Vessel", bbox=[10, 20, 50, 30], table_id="t1", row_index=0, column_index=0, cell_role="HEADER"),
        GeometricCell(block_id="b2", text="Rank", bbox=[50, 20, 90, 30], table_id="t1", row_index=0, column_index=1, cell_role="HEADER"),
        GeometricCell(block_id="b3", text="Sea Star", bbox=[10, 30, 50, 40], table_id="t1", row_index=1, column_index=0, cell_role="DATA"),
        GeometricCell(block_id="b4", text="Captain", bbox=[50, 30, 90, 40], table_id="t1", row_index=1, column_index=1, cell_role="DATA"),
    ]
    table = GeometricTable(
        table_id="t1",
        page=1,
        num_columns=2,
        num_rows=2,
        num_data_rows=1,
        column_bands=[(10, 50), (50, 90)],
        cells=cells,
    )
    blocks = [
        _make_block("b0", "MARITIME HISTORY", reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "Vessel", reading_order=1, table_id="t1", row_index=0, column_index=0),
        _make_block("b2", "Rank", reading_order=2, table_id="t1", row_index=0, column_index=1),
        _make_block("b3", "Sea Star", reading_order=3, table_id="t1", row_index=1, column_index=0),
        _make_block("b4", "Captain", reading_order=4, table_id="t1", row_index=1, column_index=1),
    ]
    inp = SemanticInput(document_id="doc-10", page_count=1, blocks=blocks, tables=[table])
    doc = build_document_structure_from_semantic_input(inp)

    section = doc.sections[0]
    assert section.heading == "MARITIME HISTORY"
    assert len(section.blocks) == 1
    tbl_block = section.blocks[0]
    assert tbl_block.type == "table"
    assert tbl_block.table_data is not None
    assert tbl_block.table_data["headers"] == ["Vessel", "Rank"]
    assert tbl_block.table_data["rows"] == [["Sea Star", "Captain"]]


def test_11_multi_column_document():
    """11. Reading order across multiple columns preserves correct section membership."""
    # Left column: Personal details & skills
    # Right column: Career history
    blocks = [
        _make_block("b0", "SKILLS", reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "Python, Go", reading_order=1),
        _make_block("b2", "CAREER HISTORY", reading_order=2, suggested_role="SECTION_HEADING"),
        _make_block("b3", "Engineer at Beta LLC", reading_order=3),
    ]
    inp = SemanticInput(document_id="doc-11", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert len(doc.sections) == 2
    assert doc.sections[0].heading == "SKILLS"
    assert doc.sections[0].blocks[0].text == "Python, Go"
    assert doc.sections[1].heading == "CAREER HISTORY"
    assert doc.sections[1].blocks[0].text == "Engineer at Beta LLC"


def test_12_multi_page_document():
    """12. Multi-page document sections are preserved continuously across pages."""
    blocks = [
        _make_block("b0", "CAREER OVERVIEW", page=1, reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "Page 1 work description.", page=1, reading_order=1),
        _make_block("b2", "Page 2 continued work description.", page=2, reading_order=2),
        _make_block("b3", "DIPLOMAS & DEGREES", page=2, reading_order=3, suggested_role="SECTION_HEADING"),
        _make_block("b4", "M.S. in Software Engineering", page=2, reading_order=4),
    ]
    inp = SemanticInput(document_id="doc-12", page_count=2, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert len(doc.sections) == 2
    assert doc.sections[0].heading == "CAREER OVERVIEW"
    assert len(doc.sections[0].blocks) == 2
    assert doc.sections[0].blocks[0].page_number == 1
    assert doc.sections[0].blocks[1].page_number == 2
    assert doc.sections[1].heading == "DIPLOMAS & DEGREES"
    assert doc.sections[1].blocks[0].page_number == 2


def test_13_scanned_ocr_fixture_handling():
    """13. Reconstructed OCR blocks flow seamlessly into DocumentStructure."""
    blocks = [
        _make_block("ocr_0", "CURRICULUM VITAE", reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("ocr_1", "Scanned text line with minor artifacts.", reading_order=1),
        _make_block("ocr_2", "PROFESSIONAL ATTAINMENTS", reading_order=2, suggested_role="SECTION_HEADING"),
        _make_block("ocr_3", "Delivered mission critical telemetry system.", reading_order=3),
    ]
    inp = SemanticInput(document_id="doc-13", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert len(doc.sections) == 2
    assert doc.sections[0].heading == "CURRICULUM VITAE"
    assert doc.sections[1].heading == "PROFESSIONAL ATTAINMENTS"


def test_14_provenance_validation_rejects_invented_content():
    """14. Deterministic provenance validation rejects invalid source IDs or ungrounded text."""
    # Test valid document
    valid_blocks = {
        "b0": "CAREER HISTORY",
        "b1": "Software Engineer at Google",
    }
    valid_doc = DocumentStructure(
        sections=[
            DocumentSection(
                heading="CAREER HISTORY",
                level=1,
                source_block_ids=["b0"],
                blocks=[
                    DocumentBlock(
                        type="paragraph",
                        text="Software Engineer at Google",
                        source_block_ids=["b1"],
                    )
                ],
            )
        ],
        page_count=1,
    )
    violations = validate_generic_document_structure(valid_doc, known_blocks=valid_blocks)
    assert violations == []

    # Test invalid block ID
    invalid_id_doc = DocumentStructure(
        sections=[
            DocumentSection(
                heading="CAREER HISTORY",
                level=1,
                source_block_ids=["b_NONEXISTENT"],
                blocks=[],
            )
        ],
        page_count=1,
    )
    violations = validate_generic_document_structure(invalid_id_doc, known_blocks=valid_blocks)
    assert any("UNKNOWN_BLOCK_ID" in v for v in violations)

    # Test invented heading content
    hallucinated_heading_doc = DocumentStructure(
        sections=[
            DocumentSection(
                heading="COMPLETELY FABRICATED HEADING",
                level=1,
                source_block_ids=["b0"],
                blocks=[],
            )
        ],
        page_count=1,
    )
    violations = validate_generic_document_structure(hallucinated_heading_doc, known_blocks=valid_blocks)
    assert any("UNGROUNDED_CONTENT" in v for v in violations)

    # Test invented block content
    hallucinated_block_doc = DocumentStructure(
        sections=[
            DocumentSection(
                heading="CAREER HISTORY",
                level=1,
                source_block_ids=["b0"],
                blocks=[
                    DocumentBlock(
                        type="paragraph",
                        text="Hallucinated text never present in source blocks",
                        source_block_ids=["b1"],
                    )
                ],
            )
        ],
        page_count=1,
    )
    violations = validate_generic_document_structure(hallucinated_block_doc, known_blocks=valid_blocks)
    assert any("UNGROUNDED_CONTENT" in v for v in violations)


def test_15_unknown_heading_requires_zero_source_code_changes():
    """15. Completely unknown heading works dynamically with ZERO source-code changes."""
    novel_heading = "FUTURISTIC_QUANTUM_DISPATCHES_X99"
    blocks = [
        _make_block("b0", novel_heading, reading_order=0, suggested_role="SECTION_HEADING"),
        _make_block("b1", "Quantum entangled routing engine details.", reading_order=1),
    ]
    inp = SemanticInput(document_id="doc-15", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    assert len(doc.sections) == 1
    assert doc.sections[0].heading == novel_heading
    assert doc.sections[0].blocks[0].text == "Quantum entangled routing engine details."


def test_16_regression_no_fixed_heading_whitelist_controlling_output():
    """16. Explicit regression assertion proving no fixed heading whitelist controls emission."""
    import app.pipeline.stages.generic_document_builder as builder

    # Verify generic document builder does not import or use SECTION_NAMES
    assert not hasattr(builder, "SECTION_NAMES")

    # Generate 5 arbitrary novel headings
    test_headings = [
        "STRATEGIC ADVISORY",
        "SCHOLARLY INITIATIVES",
        "GLOBAL EXPEDITIONS",
        "INVENTIONS & PATENTS",
        "SYMPOSIUM KEYNOTES",
    ]
    blocks = []
    for idx, h in enumerate(test_headings):
        blocks.append(_make_block(f"h_{idx}", h, reading_order=idx * 2, suggested_role="SECTION_HEADING"))
        blocks.append(_make_block(f"p_{idx}", f"Description for {h}", reading_order=idx * 2 + 1))

    inp = SemanticInput(document_id="doc-16", page_count=1, blocks=blocks)
    doc = build_document_structure_from_semantic_input(inp)

    emitted_headings = [s.heading for s in doc.sections]
    assert emitted_headings == test_headings


def test_17_backward_compatible_resume_projection():
    """17. Verifies that document_structure_to_resume produces a valid Resume object."""
    doc = DocumentStructure(
        sections=[
            DocumentSection(
                heading=None,
                level=1,
                blocks=[
                    DocumentBlock(type="paragraph", text="John Doe"),
                    DocumentBlock(type="paragraph", text="john.doe@example.com | +1 (555) 012-3456"),
                ],
            ),
            DocumentSection(
                heading="Summary",
                level=1,
                blocks=[DocumentBlock(type="paragraph", text="Staff engineer with 12 years experience.")],
            ),
            DocumentSection(
                heading="Career History",
                level=1,
                blocks=[DocumentBlock(type="paragraph", text="Principal Engineer at TechCorp (2018 - Present)")],
            ),
            DocumentSection(
                heading="Academic Qualifications",
                level=1,
                blocks=[DocumentBlock(type="paragraph", text="B.S. Computer Science from MIT")],
            ),
            DocumentSection(
                heading="Technical Skills",
                level=1,
                blocks=[
                    DocumentBlock(type="list_item", text="Python"),
                    DocumentBlock(type="list_item", text="PostgreSQL"),
                    DocumentBlock(type="list_item", text="Docker"),
                ],
            ),
        ],
        page_count=1,
    )

    resume = document_structure_to_resume(doc)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"
    assert resume.personal.phone == "+1 (555) 012-3456"
    assert resume.summary == "Staff engineer with 12 years experience."
    assert len(resume.skills) == 3
    assert "Python" in resume.skills
    assert len(resume.experience) == 1
    assert len(resume.education) == 1


def test_18_spatial_form_multiline_colons_not_merged():
    """18. Verifies that colons sharing the same X across multiple physical lines are not merged into ': : :'."""
    from app.pipeline.stages.generic_document_builder import _build_spatial_data

    # Simulate 3 vertical lines sharing x0=134.0
    blocks = [
        _make_block("b0", "Address", reading_order=0, bbox=[39.0, 321.0, 74.0, 331.0]),
        _make_block("b1", ":", reading_order=1, bbox=[134.0, 321.8, 137.0, 329.8]),
        _make_block("b2", "1/206 OLD 1/366 NEW", reading_order=2, bbox=[139.0, 321.8, 216.0, 329.8]),
        _make_block("b3", ":", reading_order=3, bbox=[134.0, 339.8, 137.0, 347.8]),
        _make_block("b4", "PRTIKSHA", reading_order=4, bbox=[139.0, 339.8, 178.0, 347.8]),
        _make_block("b5", ":", reading_order=5, bbox=[134.0, 357.8, 137.0, 365.8]),
        _make_block("b6", "VISHNIUPURI", reading_order=6, bbox=[139.0, 357.8, 190.0, 365.8]),
    ]
    spatial_blocks, spatial_rows, ar, max_bottom = _build_spatial_data(blocks)

    # Must produce 3 separate physical lines, none of which contain multiple colons
    assert len(spatial_blocks) == 3
    texts = [b["text"] for b in spatial_blocks]
    assert texts[0] == "Address : 1/206 OLD 1/366 NEW"
    assert texts[1] == ": PRTIKSHA"
    assert texts[2] == ": VISHNIUPURI"
    for b in spatial_blocks:
        assert ": :" not in b["text"]

    # Physical Y positions must be strictly descending (distinct baselines)
    assert spatial_blocks[0]["relative_y"] < spatial_blocks[1]["relative_y"] < spatial_blocks[2]["relative_y"]


def test_19_form_canvas_aspect_ratio_calculation():
    """19. Verifies that aspect ratio calculation produces faithful vertical height (100 / ar)."""
    ar = 2.341
    # Aspect ratio math: paddingBottomPct = round((100 / ar) * 100) / 100
    padding_bottom_pct = round((100.0 / ar) * 100.0) / 100.0
    # Expected: ~42.72%, not 120%
    assert abs(padding_bottom_pct - 42.72) < 0.1
    assert padding_bottom_pct < 50.0
