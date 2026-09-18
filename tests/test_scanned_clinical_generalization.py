"""Tests for scanned clinical generalization fixes:
1. State abbreviations and location detection in composite education entry identification.
2. Conservative single-block ordered subphrase grounding in semantic contract.
"""

from app.domain.candidate_entry import EntryType
from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.domain.semantic_contract import _is_value_semantically_supported
from app.domain.structural import StructuralRole
from app.pipeline.stages.candidate_entries import build_candidate_entries
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.structural_roles import (
    _looks_like_composite_education_entry,
    _looks_like_location,
    build_structural_blocks,
)


def _line(line_id: str, text: str, y0: float, *, x0: float = 10.0, font_size: float = 11.0, bold: bool = False) -> Line:
    bbox = BoundingBox(x0, y0, x0 + max(40.0, len(text) * 6.0), y0 + 12.0)
    span = Span(f"{line_id}-span", text, bbox, font_size=font_size, bold=bold)
    return Line(
        line_id,
        1,
        bbox,
        [span],
        text,
        TextStyle(font_size=font_size, bold=bold),
        reading_order=int(y0),
        source_span_ids=[span.span_id],
        reconstruction_method="physical",
    )


def _document(lines: list[Line]) -> Document:
    bbox = BoundingBox(
        min(line.bbox.x0 for line in lines),
        min(line.bbox.y0 for line in lines),
        max(line.bbox.x1 for line in lines),
        max(line.bbox.y1 for line in lines),
    )
    return Document(
        pages=[Page(1, regions=[Region("p1-r1", "physical_region", bbox, lines=lines, reading_order=0, column_id=0)])]
    )


def test_locations_with_us_state_codes_recognized():
    assert _looks_like_location("Boston, MA") is True
    assert _looks_like_location("Palo Alto, CA") is True
    assert _looks_like_location("Philadelphia, PA") is True
    assert _looks_like_location("Austin, TX") is True
    assert _looks_like_location("New York, NY") is True
    assert _looks_like_location("Seattle, WA") is True


def test_location_with_state_code_not_composite_education_entry():
    # Even if nearby date evidence is present, Boston, MA is a location, not a degree
    result = _looks_like_composite_education_entry(
        "Dana-Farber Cancer Institute | Boston, MA",
        following=("2018 - Present",),
    )
    assert result is False


def test_legitimate_composite_education_recognized():
    # Legitimate degree | institution
    result = _looks_like_composite_education_entry(
        "Bachelor of Science | Harvard University",
        following=("2014 - 2018",),
    )
    assert result is True


def test_experience_entry_does_not_split_on_organization_location():
    lines = [
        _line("h1", "PROFESSIONAL EXPERIENCE", 10.0, font_size=14.0, bold=True),
        _line("t1", "Senior Clinical Research Specialist", 30.0, font_size=12.0, bold=True),
        _line("o1", "Dana-Farber Cancer Institute | Boston, MA", 50.0, font_size=10.0),
        _line("d1", "2018 - Present", 70.0, font_size=10.0),
        _line("b1", "Led clinical trials on novel oncology therapeutics.", 90.0, font_size=10.0),
        _line("b2", "Managed regulatory compliance and patient data protocols.", 110.0, font_size=10.0),
    ]
    doc = _document(lines)
    blocks = build_structural_blocks(doc)

    # Dana-Farber block should NOT be ENTRY_TITLE
    df_block = next(b for b in blocks if "Dana-Farber" in b.text)
    assert df_block.role != StructuralRole.ENTRY_TITLE

    # Candidate entries should form exactly ONE experience entry, not two
    sections = build_candidate_sections(blocks)
    entries = build_candidate_entries(sections)
    exp_entries = [e for e in entries if e.entry_type == EntryType.EXPERIENCE]
    assert len(exp_entries) == 1
    texts = [b.text for b in exp_entries[0].blocks]
    assert texts[0] == "Senior Clinical Research Specialist"
    assert "Dana-Farber Cancer Institute | Boston, MA" in texts


def test_single_block_ordered_subphrase_grounding_positive():
    source = "Doctor of Medicine (M.D.) & Master of Public Health (M.P.H.)"
    # Canonical major extracted from composite degree line
    assert _is_value_semantically_supported("Medicine & Public Health", source) is True
    assert _is_value_semantically_supported("Medicine", source) is True
    assert _is_value_semantically_supported("Public Health", source) is True


def test_single_block_ordered_subphrase_extra_unsupported_token_rejected():
    source = "Doctor of Medicine (M.D.) & Master of Public Health (M.P.H.)"
    # "Cardiology" does not exist in source
    assert _is_value_semantically_supported("Medicine & Public Health & Cardiology", source) is False


def test_single_block_ordered_subphrase_reordered_tokens_rejected():
    source = "Doctor of Medicine (M.D.) & Master of Public Health (M.P.H.)"
    # Tokens out of relative order in source
    assert _is_value_semantically_supported("Public Medicine Health", source) is False


def test_single_token_unsupported_substring_rejected():
    source = "Doctor of Medicine (M.D.) & Master of Public Health (M.P.H.)"
    assert _is_value_semantically_supported("Cardiology", source) is False
    assert _is_value_semantically_supported("Biochemistry", source) is False
