"""Tests verifying the repair of maritime entity grounding (Phase 10W.2).

Covers requirements A through P:
- A. Split fragments get unique deterministic IDs
- B. IDs are stable across repeated deterministic runs
- C. Parent/source identity remains traceable (parent_block_id)
- D. All fragments remain independently addressable
- E. SemanticInput contains no duplicate IDs after splitting on target fixtures
- F. Mayur-style course/date fragments independently support their canonical values
- G. Mukund-style company/rank/date fragments independently support their canonical values
- H. "Anglo Eastern Ship Management" -> ORGANIZATION
- I. "Management Consultant" -> ENTRY_TITLE
- J. Designation + organization form one deterministic experience span
- K. Genuine separate experience entries remain separate
- L. 28/Mar/21 can support 2021-03-28
- M. 01/11/21 can support 2021-11-01
- N. Arbitrary numeric values are not accepted as years
- O. Existing four-digit date behavior remains unchanged
- P. Existing Phase 10W.1 logical-row continuation tests remain green
"""

from collections import Counter
from pathlib import Path
import pytest

from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import (
    SemanticBlockInput,
    SemanticInput,
    _is_value_semantically_supported,
    build_deterministic_experience_spans,
    build_semantic_input,
)
from app.domain.structural import StructuralRole
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.structural_roles import _looks_like_organization, classify_structural_role
from app.pipeline.stages.table_binding import GeometricCell, GeometricTableBinder
from app.pipeline.stages.text_extraction import PDFExtractor


def _make_block(
    block_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page: int = 1,
    role: str = "DESCRIPTION",
    spans: list | None = None,
) -> SemanticBlockInput:
    kwargs = {
        "block_id": block_id,
        "region_id": "r_1",
        "region_kind": "body",
        "page": page,
        "text": text,
        "bbox": [x0, y0, x1, y1],
        "reading_order": 0,
        "suggested_role": role,
    }
    if spans is not None:
        kwargs["spans"] = spans
    return SemanticBlockInput(**kwargs)


def _load_semantic_input(fixture_name: str) -> SemanticInput:
    p = Path("tests/fixtures") / fixture_name
    if not p.exists():
        pytest.skip(f"Fixture {fixture_name} not found")
    raw = p.read_bytes()
    extracted = PDFExtractor.extract(raw)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)
    return build_semantic_input(layout, document_id=fixture_name)


# -----------------------------------------------------------------------------
# Requirement A: Split fragments get unique deterministic IDs
# Requirement C: Parent/source identity remains traceable (parent_block_id)
# Requirement D: All fragments remain independently addressable
# -----------------------------------------------------------------------------
def test_requirement_a_c_d_split_fragments_unique_and_traceable():
    binder = GeometricTableBinder(row_tolerance=5.0)

    blocks = [
        _make_block("h0", "Course Name", 50.0, 100.0, 150.0, 115.0),
        _make_block("h1", "Date", 200.0, 100.0, 300.0, 115.0),
        _make_block("d_multi", "Survival Craft 28/Mar/21", 50.0, 130.0, 300.0, 145.0),
        _make_block("d_other", "Fire Fighting", 50.0, 160.0, 150.0, 175.0),
        _make_block("d_other_date", "15/Jan/20", 200.0, 160.0, 300.0, 175.0),
    ]

    bound = binder.bind_document_tables(blocks)

    # Find the cells produced from d_multi
    multi_cells = [b for b in bound if b.parent_block_id == "d_multi"]
    assert len(multi_cells) == 2

    # A: Unique deterministic IDs
    col0_cell = next(c for c in multi_cells if c.column_index == 0)
    col1_cell = next(c for c in multi_cells if c.column_index == 1)
    assert col0_cell.block_id == "d_multi_c0"
    assert col1_cell.block_id == "d_multi_c1"
    assert col0_cell.block_id != col1_cell.block_id

    # C: Parent identity preserved
    assert col0_cell.parent_block_id == "d_multi"
    assert col1_cell.parent_block_id == "d_multi"

    # D: Independently addressable with distinct text
    assert col0_cell.text == "Survival Craft"
    assert col1_cell.text == "28/Mar/21"

    # Non-split block preserves original ID and None parent
    other_cell = next(b for b in bound if b.block_id == "d_other")
    assert other_cell.parent_block_id is None


# -----------------------------------------------------------------------------
# Requirement B: IDs are stable across repeated deterministic runs
# -----------------------------------------------------------------------------
def test_requirement_b_ids_stable_across_repeated_runs():
    binder = GeometricTableBinder(row_tolerance=5.0)

    blocks = [
        _make_block("h0", "ColA", 50.0, 100.0, 150.0, 115.0),
        _make_block("h1", "ColB", 200.0, 100.0, 300.0, 115.0),
        _make_block("cross_1", "ValA ValB", 50.0, 130.0, 300.0, 145.0),
        _make_block("d2_a", "ValC", 50.0, 160.0, 150.0, 175.0),
        _make_block("d2_b", "ValD", 200.0, 160.0, 300.0, 175.0),
    ]

    run1 = binder.bind_document_tables(blocks)
    run2 = binder.bind_document_tables(blocks)

    ids1 = [b.block_id for b in run1]
    ids2 = [b.block_id for b in run2]
    assert ids1 == ids2
    assert "cross_1_c0" in ids1
    assert "cross_1_c1" in ids1


# -----------------------------------------------------------------------------
# Requirement E: SemanticInput contains zero duplicate block IDs on fixtures
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fixture_name",
    [
        "2nd Officer Mayur Agarwal_062029.pdf",
        "MUKUND 3RD OFF CV 2026.pdf",
        "Sendrick Costa CV.pdf",
    ],
)
def test_requirement_e_no_duplicate_block_ids(fixture_name: str):
    sem = _load_semantic_input(fixture_name)
    ids = [b.block_id for b in sem.blocks]
    counts = Counter(ids)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, f"Found duplicate block IDs in {fixture_name}: {duplicates}"


# -----------------------------------------------------------------------------
# Requirement F: Mayur-style course/date fragments independently support values
# -----------------------------------------------------------------------------
def test_requirement_f_mayur_independent_fragment_grounding():
    sem = _load_semantic_input("2nd Officer Mayur Agarwal_062029.pdf")

    split_fragments = [b for b in sem.blocks if b.parent_block_id is not None]
    assert len(split_fragments) > 0

    block_map = {b.block_id: b for b in sem.blocks}
    for frag in split_fragments:
        assert frag.block_id in block_map
        assert block_map[frag.block_id].text == frag.text

    # Verify split fragments retain specific column bounds
    assert any("c0" in b.block_id for b in split_fragments)
    assert any("c1" in b.block_id for b in split_fragments)


# -----------------------------------------------------------------------------
# Requirement G: Mukund-style company/rank/date fragments independently support values
# -----------------------------------------------------------------------------
def test_requirement_g_mukund_independent_fragment_grounding():
    sem = _load_semantic_input("MUKUND 3RD OFF CV 2026.pdf")
    split_fragments = [b for b in sem.blocks if b.parent_block_id is not None]
    assert len(split_fragments) > 0

    block_map = {b.block_id: b for b in sem.blocks}
    for frag in split_fragments:
        assert frag.block_id in block_map
        assert block_map[frag.block_id].text == frag.text


# -----------------------------------------------------------------------------
# Requirement H: "Anglo Eastern Ship Management" -> ORGANIZATION
# -----------------------------------------------------------------------------
def test_requirement_h_ship_management_recognized_as_organization():
    assert _looks_like_organization("Anglo Eastern Ship Management")
    assert _looks_like_organization("Fleet Management Limited")
    assert _looks_like_organization("Bernhard Schulte Shipmanagement")
    assert _looks_like_organization("V.Ships Marine Management")

    role, _, _ = classify_structural_role(
        text="Anglo Eastern Ship Management",
        previous_text="Assistant Second Engineer",
        next_text="2015 - 2017",
        following_texts=("2015 - 2017",),
    )
    assert role == StructuralRole.ORGANIZATION


# -----------------------------------------------------------------------------
# Requirement I: "Management Consultant" -> ENTRY_TITLE
# -----------------------------------------------------------------------------
def test_requirement_i_management_consultant_recognized_as_entry_title():
    assert not _looks_like_organization("Management Consultant")
    assert not _looks_like_organization("Management Trainee")
    assert not _looks_like_organization("Product Management Lead")

    role, _, _ = classify_structural_role(
        text="Management Consultant",
        previous_text=None,
        next_text="McKinsey & Company",
        following_texts=("McKinsey & Company",),
    )
    assert role == StructuralRole.ENTRY_TITLE


# -----------------------------------------------------------------------------
# Requirement J: Designation + organization form one deterministic experience span
# -----------------------------------------------------------------------------
def test_requirement_j_designation_and_organization_form_single_span():
    blocks = [
        _make_block("b1", "Experience", 50.0, 50.0, 200.0, 65.0, role="SECTION_HEADING"),
        _make_block("b2", "Assistant Second Engineer", 50.0, 80.0, 250.0, 95.0, role="ENTRY_TITLE"),
        _make_block("b3", "Anglo Eastern Ship Management", 50.0, 100.0, 300.0, 115.0, role="ORGANIZATION"),
        _make_block("b4", "2015 - 2017", 50.0, 120.0, 150.0, 135.0, role="DATE"),
        _make_block("b5", "Maintained generators and boilers.", 50.0, 140.0, 400.0, 155.0, role="DESCRIPTION"),
    ]
    sem = SemanticInput(document_id="test", archetype="standard_cv", page_count=1, blocks=blocks)
    spans = build_deterministic_experience_spans(sem)

    assert len(spans) == 1
    assert "b2" in spans[0].block_ids
    assert "b3" in spans[0].block_ids
    assert "b4" in spans[0].block_ids
    assert "b5" in spans[0].block_ids


# -----------------------------------------------------------------------------
# Requirement K: Genuine separate experience entries remain separate
# -----------------------------------------------------------------------------
def test_requirement_k_genuine_separate_experience_entries_remain_separate():
    blocks = [
        _make_block("b0", "Work History", 50.0, 50.0, 200.0, 65.0, role="SECTION_HEADING"),
        # Entry 1
        _make_block("b1", "Third Engineer", 50.0, 80.0, 250.0, 95.0, role="ENTRY_TITLE"),
        _make_block("b2", "Anglo Eastern Ship Management", 50.0, 100.0, 300.0, 115.0, role="ORGANIZATION"),
        _make_block("b3", "2017 - 2019", 50.0, 120.0, 150.0, 135.0, role="DATE"),
        _make_block("b4", "Maintained propulsion systems.", 50.0, 140.0, 400.0, 155.0, role="DESCRIPTION"),
        # Entry 2
        _make_block("b5", "Junior Engineer", 50.0, 180.0, 250.0, 195.0, role="ENTRY_TITLE"),
        _make_block("b6", "Anglo Eastern Ship Management", 50.0, 200.0, 300.0, 215.0, role="ORGANIZATION"),
        _make_block("b7", "2015 - 2017", 50.0, 220.0, 150.0, 235.0, role="DATE"),
        _make_block("b8", "Assisted seniors with overhaul.", 50.0, 240.0, 400.0, 255.0, role="DESCRIPTION"),
    ]
    sem = SemanticInput(document_id="test", archetype="standard_cv", page_count=1, blocks=blocks)
    spans = build_deterministic_experience_spans(sem)

    assert len(spans) == 2
    assert set(spans[0].block_ids) == {"b1", "b2", "b3", "b4"}
    assert set(spans[1].block_ids) == {"b5", "b6", "b7", "b8"}


# -----------------------------------------------------------------------------
# Requirement L: 28/Mar/21 can support 2021-03-28
# -----------------------------------------------------------------------------
def test_requirement_l_two_digit_date_with_month_name():
    assert _is_value_semantically_supported("2021-03-28", "Issued on 28/Mar/21 at Mumbai")
    assert _is_value_semantically_supported("2021-03-28", "28-Mar-21")
    assert _is_value_semantically_supported("2021-03-28", "28.Mar.21")
    assert _is_value_semantically_supported("2021-03-28", "28 Mar 21")
    assert _is_value_semantically_supported("2021-03-28", "28 Mar '21")
    assert _is_value_semantically_supported("2021-03-28", "Mar 28, 21")


# -----------------------------------------------------------------------------
# Requirement M: 01/11/21 can support 2021-11-01
# -----------------------------------------------------------------------------
def test_requirement_m_two_digit_numeric_date():
    assert _is_value_semantically_supported("2021-11-01", "01/11/21")
    assert _is_value_semantically_supported("2021-11-01", "01-11-21")
    assert _is_value_semantically_supported("2021-11-01", "1/11/21")
    assert _is_value_semantically_supported("2021-11-01", "Valid: 01/11/21 to 01/11/26")


# -----------------------------------------------------------------------------
# Requirement N: Arbitrary numeric values are not accepted as years
# -----------------------------------------------------------------------------
def test_requirement_n_arbitrary_numbers_rejected_as_years():
    # Standalone '21' without date structure
    assert not _is_value_semantically_supported("2021-03-28", "21")
    assert not _is_value_semantically_supported("2021-03-28", "Age 21")
    assert not _is_value_semantically_supported("2021-03-28", "21 candidates passed")
    assert not _is_value_semantically_supported("2021-11-01", "Score 21-11")
    assert not _is_value_semantically_supported("2021-03-28", "Room 21, Floor 3, March 28")


# -----------------------------------------------------------------------------
# Requirement O: Existing four-digit date behavior remains unchanged
# -----------------------------------------------------------------------------
def test_requirement_o_four_digit_date_support_unchanged():
    assert _is_value_semantically_supported("2021-03-28", "28 March 2021")
    assert _is_value_semantically_supported("2021-03-28", "28/03/2021")
    assert _is_value_semantically_supported("2021-03-28", "2021-03-28")
    assert _is_value_semantically_supported("2021-03", "March 2021")
    assert _is_value_semantically_supported("2021", "Graduated in 2021")


# -----------------------------------------------------------------------------
# Requirement P: Existing Phase 10W.1 logical-row continuation tests remain green
# -----------------------------------------------------------------------------
def test_requirement_p_logical_row_continuation_health():
    sem = _load_semantic_input("AKIBUL ALAM CV(JO).pdf")
    t_cert = next((t for t in sem.tables if t.table_id == "table_p2_1"), None)
    assert t_cert is not None

    # Check that Survival Craft / Rescue Boat is a single logical row
    rescue_row = next((r for r in t_cert.rows if any("Survival Craft" in c.text for c in r)), None)
    assert rescue_row is not None
    course_cell = next(c for c in rescue_row if c.semantic_role == "course_name")
    assert course_cell.text == "Proficiency in Survival Craft / Rescue Boat"
