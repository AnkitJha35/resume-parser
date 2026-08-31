"""Phase 7B-4: embedded-column parent selection in ``interpret_layout``.

Phase 7B-3 localized the first wrong stage to ``layout``:
``_widest_containing_sibling()`` ranked adoptive parents by *width*, returned
only that single candidate, and ``_merge_embedded_column_groups()`` abandoned
the cluster when ``_is_embedded_subcolumn()`` rejected it instead of trying the
next candidate. On the Shubham CV a 36.7pt skills-chip cluster that is 100%
contained in the right column (width 250.4) was awarded to the left column
(width 255.2, overlap 0.00) because the left column is wider by 4.8pt.

These tests pin the two behavioural properties of the fix -- containment-first
ranking, and retry across candidates -- plus the production consequence.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from app.domain.document import BoundingBox, Line, Span, TextStyle, document_from_text_blocks
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.layout import (
    _bounds,
    _containing_sibling_candidates,
    _is_embedded_subcolumn,
    _merge_embedded_column_groups,
    interpret_layout,
)
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.semantic_compat import semantic_sections_to_text_blocks
from app.pipeline.stages.semantic_paths import detect_region_aware_sections
from app.pipeline.stages.text_extraction import PDFExtractor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _shubham_fixture() -> Path:
    """Resolve the fixture by glob: its filename is NFD-encoded on disk."""
    matches = sorted(glob.glob(str(FIXTURES_DIR / "*Shubham*.pdf")))
    if not matches:
        pytest.skip("missing fixture: *Shubham*.pdf")
    return Path(matches[0])


def _line(line_id: str, text: str, x0: float, y0: float, x1: float, y1: float) -> Line:
    bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    span = Span(span_id=f"{line_id}-span-0", text=text, bbox=bbox, font_size=10.0, bold=False)
    return Line(
        line_id=line_id,
        page_number=1,
        bbox=bbox,
        spans=[span],
        text=text,
        style=TextStyle(font_size=10.0, bold=False),
        source_span_ids=[span.span_id],
    )


def _stack(prefix: str, x0: float, x1: float, tops: list[float]) -> list[Line]:
    return [
        _line(f"{prefix}-{index}", f"{prefix} {index}", x0, top, x1, top + 10.0)
        for index, top in enumerate(tops)
    ]


def _ids(group: list[Line]) -> set[str]:
    return {line.line_id for line in group}


# --- Property 1: containment beats width -----------------------------------


def _shubham_page_geometry() -> tuple[list[Line], list[Line], list[Line]]:
    """The measured Shubham page-1 clusters, reduced to their bounding geometry.

    left  x 28.3-283.5  (w 255.2)  y 119.1-759.7
    right x 317.5-567.9 (w 250.4)  y 119.1-810.8
    chips x 517.8-554.5 (w  36.7)  y 150.0-195.6   <- React JS / AWS / MVC
    """
    left = _stack("left", 28.3, 283.5, [119.1, 300.0, 500.0, 749.7])
    right = _stack("right", 317.5, 567.9, [119.1, 300.0, 500.0, 800.8])
    chips = [
        _line("chip-0", "React JS", 517.8, 150.0, 554.5, 160.0),
        _line("chip-1", "AWS", 517.8, 170.0, 554.5, 180.0),
        _line("chip-2", "MVC", 517.8, 185.6, 554.5, 195.6),
    ]
    return left, right, chips


def test_fully_contained_parent_outranks_slightly_wider_non_overlapping_parent():
    left, right, chips = _shubham_page_geometry()
    bounds = [_bounds(left), _bounds(right), _bounds(chips)]

    ranked = _containing_sibling_candidates(2, bounds)

    # The left column is still admitted -- it enters via the sparse_right
    # escape, and that gate is deliberately unchanged. It must simply lose.
    assert ranked[0] == 1, "the 100%-containing right column must rank first"
    assert 0 in ranked, "the wider left column must remain an admitted fallback"
    assert ranked.index(1) < ranked.index(0)


def test_shubham_chip_cluster_is_absorbed_by_the_containing_column():
    left, right, chips = _shubham_page_geometry()

    merged = _merge_embedded_column_groups([left, right, chips])

    assert len(merged) == 2, "the chip cluster must not survive as a third column"
    owner = next(group for group in merged if "chip-0" in _ids(group))
    assert _ids(owner) >= _ids(chips)
    assert _ids(owner) >= _ids(right), "chips belong to the right column"
    assert not (_ids(owner) & _ids(left))


# --- Property 2: retry past a rejected top-ranked candidate ----------------


def _retry_geometry() -> tuple[list[Line], list[Line], list[Line]]:
    """Two containing parents; the top-ranked one is refused, the next accepts.

    Pure predicate geometry, not a plausible page: both parents fully contain
    the cluster horizontally, so containment ties and width breaks the tie.

    wide  x 90-190 (w 100)  y  95-195 (h 100)  -> ranked first, rejected
    tall  x 95-185 (w  90)  y  50-250 (h 200)  -> ranked second, accepted
    small x 100-180 (w 80)  y 100-190 (h  90)  5 lines
    """
    wide = _stack("wide", 90.0, 190.0, [95.0, 125.0, 155.0, 185.0])
    tall = _stack("tall", 95.0, 185.0, [50.0, 110.0, 180.0, 240.0])
    small = _stack("small", 100.0, 180.0, [100.0, 120.0, 140.0, 160.0, 180.0])
    return wide, tall, small


def test_top_ranked_candidate_is_rejected_but_the_next_one_is_still_tried():
    wide, tall, small = _retry_geometry()
    bounds = [_bounds(wide), _bounds(tall), _bounds(small)]

    ranked = _containing_sibling_candidates(2, bounds)
    assert ranked[:2] == [0, 1], "equal containment must fall back to width order"

    # Pre-condition of the test: the first candidate genuinely fails the
    # unchanged predicate, and the second genuinely passes it.
    assert _is_embedded_subcolumn(small, wide, bounds[2], bounds[0]) is False
    assert _is_embedded_subcolumn(small, tall, bounds[2], bounds[1]) is True

    merged = _merge_embedded_column_groups([wide, tall, small])

    assert len(merged) == 2
    owner = next(group for group in merged if "small-0" in _ids(group))
    assert _ids(owner) >= _ids(tall)
    assert not (_ids(owner) & _ids(wide))


def test_cluster_is_abandoned_only_when_every_candidate_fails():
    wide, _tall, small = _retry_geometry()

    merged = _merge_embedded_column_groups([wide, small])

    assert len(merged) == 2, "no candidate accepts, so the cluster stays separate"


# --- Production integration: Shubham ---------------------------------------


def _shubham_sections() -> dict[str, list]:
    raw = _shubham_fixture().read_bytes()
    layout_document = interpret_layout(
        reconstruct_document(document_from_text_blocks(PDFExtractor.extract(raw)))
    )
    semantic_document = detect_region_aware_sections(layout_document, SectionDetector())
    return semantic_sections_to_text_blocks(semantic_document)


@pytest.fixture(scope="module")
def shubham_resume():
    return ResumeParser().parse_with_layout_pipeline(_shubham_fixture().read_bytes())


def test_shubham_page_one_has_no_spurious_third_column():
    layout_document = interpret_layout(
        reconstruct_document(
            document_from_text_blocks(PDFExtractor.extract(_shubham_fixture().read_bytes()))
        )
    )
    columns = [
        region for region in layout_document.pages[0].regions if region.kind == "column"
    ]
    assert len(columns) == 2, [region.region_id for region in columns]


def test_shubham_chips_are_owned_by_skills_not_experience():
    """The layout fix's direct contract: section *ownership* of the chips."""
    sections = _shubham_sections()
    skills_text = "\n".join(block.text or "" for block in sections.get("SKILLS", []))
    experience_text = "\n".join(block.text or "" for block in sections.get("EXPERIENCE", []))

    for chip in ("React JS", "AWS", "MVC"):
        assert chip in skills_text, f"{chip!r} must be owned by SKILLS"
        assert chip not in experience_text, f"{chip!r} must not be owned by EXPERIENCE"


def test_shubham_has_exactly_three_experience_entries(shubham_resume):
    designations = [entry.designation for entry in shubham_resume.experience]
    assert len(shubham_resume.experience) == 3, designations
    for chip in ("React JS", "AWS", "MVC"):
        assert chip not in designations


def test_shubham_chips_surface_in_skills(shubham_resume):
    """Canonical values, because SkillsExtractor emits canonicals not raw text.

    ``skills.json`` maps ``react -> React``, ``aws -> AWS``, and ``mvc -> MVC``;
    ``\\breact\\b`` matches inside the raw chip "React JS".
    """
    assert "React" in shubham_resume.skills
    assert "AWS" in shubham_resume.skills
    assert "MVC" in shubham_resume.skills


def test_mvc_surfaces_in_skills_vocabulary():
    from app.extractors.skills import SkillsExtractor

    assert "mvc" in SkillsExtractor().skills
