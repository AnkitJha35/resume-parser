"""Phase 7B-2: ExperienceExtractor hardening, verified against the Shubham CV.

Scope is deliberately narrow: only ExperienceExtractor behaviour is asserted
here. The bogus fourth EXPERIENCE group (React JS / AWS / MVC) is an upstream
semantic-ownership defect and is characterized, not fixed, in this phase.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from app.extractors.experience import ExperienceExtractor
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.block_classification import ClassifiedBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup
from app.pipeline.stages.text_extraction import TextBlock

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _shubham_fixture() -> Path:
    """Resolve the fixture by glob: its filename is NFD-encoded on disk."""
    matches = sorted(glob.glob(str(FIXTURES_DIR / "*Shubham*.pdf")))
    if not matches:
        pytest.skip("missing fixture: *Shubham*.pdf")
    return Path(matches[0])


def _classified(text: str, label: str, y0: float = 0.0) -> ClassifiedBlock:
    block = TextBlock(text=text, page_number=1, x0=0.0, y0=y0, x1=100.0, y1=y0 + 10.0)
    return ClassifiedBlock(original=block, label=label, score=1.0, reasons=["test"])


def _group(*blocks: ClassifiedBlock) -> CandidateGroup:
    return CandidateGroup(
        section="EXPERIENCE",
        blocks=list(blocks),
        page_number=1,
        column_id=0,
        start_index=0,
        end_index=len(blocks),
        summary_text="",
    )


@pytest.fixture(scope="module")
def shubham_experience():
    resume = ResumeParser().parse_with_layout_pipeline(_shubham_fixture().read_bytes())
    return resume.experience


# --- FIX 4: title canonicalization -----------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Software Engineer -II", "Software Engineer -II"),
        ("SDE 2", "SDE 2"),
        ("HR Intern", "HR Intern"),
        ("SDE Intern", "SDE Intern"),
        ("Software Engineer", "Software Engineer"),
        ("Software Engineer (SDE)", "Software Engineer (SDE)"),
        ("SOFTWARE ENGINEER", "Software Engineer"),
        ("head of engineering", "Head of Engineering"),
    ],
)
def test_canonicalize_title_preserves_punctuation_prefixed_acronyms(raw, expected):
    assert ExperienceExtractor()._canonicalize_title(raw) == expected


# --- FIX 1: DATE blocks reach the entry ------------------------------------


def test_shubham_experience_dates_are_recovered(shubham_experience):
    assert len(shubham_experience) >= 3
    first, second, third = shubham_experience[0], shubham_experience[1], shubham_experience[2]

    assert first.startDate == "2024-09"
    assert first.endDate is None
    assert first.current is True

    assert second.startDate == "2022-07"
    assert second.endDate == "2024-09"
    assert second.current is False

    assert third.startDate == "2022-02"
    assert third.endDate == "2022-06"
    assert third.current is False


def test_icon_glyph_prefixed_date_still_parses_through_group_path():
    """The calendar glyph decodes to U+0011; the group path must see past it."""
    entries = ExperienceExtractor()._extract_from_groups(
        [
            _group(
                _classified("Software Engineer -II", "JOB_TITLE", y0=0.0),
                _classified("\x11 Sep 2024 – Present", "DATE", y0=12.0),
            )
        ]
    )
    assert entries[0]["startDate"] == "2024-09"
    assert entries[0]["current"] is True


# --- FIX 2: structural company fallback ------------------------------------


def test_shubham_companies_are_recovered(shubham_experience):
    assert shubham_experience[0].company == "Probus Smart Things"
    assert shubham_experience[1].company == "Accenture"
    assert shubham_experience[2].company == "Beyond Byte Solution"


def test_company_fallback_does_not_harvest_body_prose():
    """A comma clause after body content must never become the company."""
    entries = ExperienceExtractor()._extract_from_groups(
        [
            _group(
                _classified("Software Engineer", "JOB_TITLE", y0=0.0),
                _classified("\x11 Sep 2024 – Present", "DATE", y0=12.0),
                _classified("• Built the service", "BULLET", y0=24.0),
                _classified("Probus Smart Things", "UNKNOWN", y0=36.0),
            )
        ]
    )
    assert entries[0]["company"] is None


# --- FIX 3: bare-city location ---------------------------------------------


def test_shubham_locations_are_recovered(shubham_experience):
    assert shubham_experience[0].location == "Noida"
    assert shubham_experience[1].location == "Gurgaon"
    assert shubham_experience[2].location == "Surat"


def test_bare_city_needs_glyph_evidence_to_become_location():
    """Accenture and Gurgaon are both bare capitalised words.

    Only the glyph-prefixed one is a location; the other stays eligible as the
    company, which is what keeps the two rules from stealing each other's line.
    """
    entries = ExperienceExtractor()._extract_from_groups(
        [
            _group(
                _classified("Software Engineer", "JOB_TITLE", y0=0.0),
                _classified("Accenture", "UNKNOWN", y0=12.0),
                _classified("\x11 July 2022 – Sep 2024", "DATE", y0=24.0),
                _classified("1⁄2 Gurgaon", "UNKNOWN", y0=36.0),
            )
        ]
    )
    assert entries[0]["company"] == "Accenture"
    assert entries[0]["location"] == "Gurgaon"


# --- Location false-positive regression ------------------------------------


@pytest.mark.parametrize(
    "prose",
    [
        "from scratch using Django, integrating OpenAI and",
        "for backend and React.js for frontend, building RESTful",
    ],
)
def test_wrapped_bullet_prose_never_becomes_location(prose):
    """These clauses satisfy _is_location_line lexically. The positional
    invariant -- metadata only, before body content -- must reject them."""
    entries = ExperienceExtractor()._extract_from_groups(
        [
            _group(
                _classified("Software Engineer -II", "JOB_TITLE", y0=0.0),
                _classified("\x11 Sep 2024 – Present", "DATE", y0=12.0),
                _classified("• Led development of the platform", "BULLET", y0=24.0),
                _classified(prose, "UNKNOWN", y0=36.0),
            )
        ]
    )
    assert entries[0]["location"] is None
    assert prose in (entries[0]["description"] or "")


def test_genuine_metadata_location_is_still_accepted():
    """The comma form must keep working while still in the metadata region."""
    entries = ExperienceExtractor()._extract_from_groups(
        [
            _group(
                _classified("HR Intern", "JOB_TITLE", y0=0.0),
                _classified("Sonipat, Haryana", "UNKNOWN", y0=12.0),
                _classified("• Ran onboarding", "BULLET", y0=24.0),
            )
        ]
    )
    assert entries[0]["location"] == "Sonipat, Haryana"


def test_shubham_locations_are_not_prose(shubham_experience):
    for entry in shubham_experience:
        location = entry.location or ""
        assert "Django" not in location
        assert "React" not in location
        assert len(location.split()) <= 4


# --- Fixed in Phase 7B-4 ----------------------------------------------------


def test_no_bogus_fourth_experience_group(shubham_experience):
    """React JS / AWS / MVC is a skills grid that EXPERIENCE used to own.

    Phase 7B-2 characterized this as a known upstream defect (a bogus 4th
    entry). Phase 7B-3 localized it to ``layout``: embedded-column parents were
    ranked by width, so the chip cluster was awarded to the wider left column
    instead of the right column that fully contains it. Phase 7B-4 fixed the
    ranking, so the 4th group is gone.
    """
    designations = [entry.designation for entry in shubham_experience]
    assert designations == [
        "Software Engineer -II",
        "Software Engineer",
        "Software Engineer Intern",
    ]
    for chip in ("React JS", "AWS", "MVC"):
        assert chip not in designations
