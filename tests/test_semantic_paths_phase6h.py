"""Phase 6H: skills-like unknown heading + chip/grid content → SKILLS ownership."""

from __future__ import annotations

from app.domain.document import BoundingBox, Document, Line, Page, Region, Span, TextStyle
from app.pipeline.stages.semantic_paths import detect_region_aware_sections


def _line(line_id: str, text: str, y0: float, *, font_size: float = 11.0, bold: bool = False, x0: float = 10.0):
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


def _heading(line_id: str, text: str, y0: float, *, x0: float = 10.0):
    return _line(line_id, text, y0, font_size=13.0, bold=True, x0=x0)


def _doc(lines: list[Line]) -> Document:
    bbox = BoundingBox(
        min(line.bbox.x0 for line in lines),
        min(line.bbox.y0 for line in lines),
        max(line.bbox.x1 for line in lines),
        max(line.bbox.y1 for line in lines),
    )
    return Document(pages=[Page(1, regions=[Region("page-1-region-0", "physical_region", bbox, lines, 0, 0)])])


def _texts(result, section: str) -> list[str]:
    return [line.text for sec in result.sections.get(section, []) for line in sec.lines]


def _section_of(result, text: str) -> str | None:
    for name, sections in result.sections.items():
        for sec in sections:
            for line in sec.lines:
                if (line.text or "").strip() == text:
                    return name
    return None


def test_a_core_hr_skill_grid_under_education_opens_skills():
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "MBA", 30),
            _line("inst", "State University", 45),
            _line("date", "2018-2020", 45, x0=400.0),
            _heading("hr", "CORE HR SKILLS", 80),
            _line("c1", "Recruitment", 100, x0=10.0),
            _line("c2", "Employee Engagement", 100, x0=140.0),
            _line("c3", "Onboarding", 100, x0=320.0),
            _line("c4", "Training", 120, x0=10.0),
        ]
    )
    result = detect_region_aware_sections(document)
    assert "MBA" in _texts(result, "EDUCATION")
    assert _section_of(result, "Recruitment") == "SKILLS"
    assert _section_of(result, "Employee Engagement") == "SKILLS"
    assert _section_of(result, "Onboarding") == "SKILLS"
    assert _section_of(result, "Training") == "SKILLS"
    assert "Recruitment" not in _texts(result, "EDUCATION")


def test_b_technical_skills_under_education_opens_skills():
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "Bachelor of Science", 30),
            _line("date", "2015-2019", 30, x0=400.0),
            _heading("sk", "TECHNICAL SKILLS", 70),
            _line("s1", "Excel", 90),
            _line("s2", "Python", 105),
            _line("s3", "SQL", 120),
        ]
    )
    result = detect_region_aware_sections(document)
    assert "Bachelor of Science" in _texts(result, "EDUCATION")
    assert _section_of(result, "Excel") == "SKILLS"
    assert _section_of(result, "Python") == "SKILLS"
    assert _section_of(result, "SQL") == "SKILLS"


def test_c_soft_skills_grid_opens_skills():
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "MBA", 30),
            _line("date", "2018-2020", 30, x0=400.0),
            _heading("sk", "SOFT SKILLS", 70),
            _line("s1", "Communication", 90),
            _line("s2", "Teamwork", 105),
            _line("s3", "Adaptability", 120),
        ]
    )
    result = detect_region_aware_sections(document)
    assert _section_of(result, "Communication") == "SKILLS"
    assert _section_of(result, "Teamwork") == "SKILLS"
    assert _section_of(result, "Adaptability") == "SKILLS"


def test_d_core_skills_compact_grid_opens_skills():
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "MBA", 30),
            _line("date", "2018-2020", 30, x0=400.0),
            _heading("sk", "CORE SKILLS", 70),
            _line("r1a", "Excel", 90, x0=10.0),
            _line("r1b", "Communication", 90, x0=120.0),
            _line("r1c", "Teamwork", 90, x0=260.0),
            _line("r2a", "HRMS", 110, x0=10.0),
            _line("r2b", "Recruitment", 110, x0=120.0),
            _line("r2c", "Training", 110, x0=260.0),
        ]
    )
    result = detect_region_aware_sections(document)
    for skill in ("Excel", "Communication", "Teamwork", "HRMS", "Recruitment", "Training"):
        assert _section_of(result, skill) == "SKILLS"


def test_e_uppercase_award_under_education_stays_education():
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "MBA", 30),
            _line("date", "2015", 30, x0=400.0),
            _heading("aw", "AWARD RECEIVED", 70),
            _line("org", "State Chamber of Commerce", 90),
            _line("yr", "2015", 105),
        ]
    )
    result = detect_region_aware_sections(document)
    # Uppercase item title must not open SKILLS; content stays under education path.
    assert _texts(result, "SKILLS") == []
    assert _section_of(result, "State Chamber of Commerce") == "EDUCATION"
    assert "AWARD RECEIVED" not in _texts(result, "SKILLS")


def test_f_academic_achievement_stays_non_skills():
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "MBA", 30),
            _line("date", "2015", 30, x0=400.0),
            _heading("ac", "ACADEMIC ACHIEVEMENT", 70),
            _line("org", "University Honors Society", 90),
            _line("yr", "2014", 105),
        ]
    )
    result = detect_region_aware_sections(document)
    assert _texts(result, "SKILLS") == []
    assert "University Honors Society" not in _texts(result, "SKILLS")
    assert "ACADEMIC ACHIEVEMENT" not in _texts(result, "SKILLS")


def test_g_explicit_skills_alias_with_prose_preserves_alias_behavior():
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "MBA", 30),
            _line("date", "2015", 30, x0=400.0),
            _heading("sk", "SKILLS", 70),
            _line(
                "prose",
                "I have extensive experience working with teams and managing projects.",
                90,
            ),
        ]
    )
    result = detect_region_aware_sections(document)
    # Deterministic alias opens SKILLS; prose ownership follows existing alias path.
    assert any(sec.lines for sec in result.sections.get("SKILLS", []))


def test_h_experience_then_core_skills_opens_skills():
    document = _doc(
        [
            _heading("exp", "EXPERIENCE", 10),
            _line("job", "HR Associate", 30),
            _line("co", "Acme Corp LLC", 45),
            _line("dt", "2019-2021", 45, x0=400.0),
            _line("body", "Supported recruiting operations across business units.", 60),
            _heading("sk", "CORE SKILLS", 90),
            _line("s1", "Excel", 110),
            _line("s2", "Communication", 125),
        ]
    )
    result = detect_region_aware_sections(document)
    assert "HR Associate" in _texts(result, "EXPERIENCE")
    assert _section_of(result, "Excel") == "SKILLS"
    assert _section_of(result, "Communication") == "SKILLS"


def test_i_summary_then_core_skills_opens_skills():
    document = _doc(
        [
            _heading("sum", "SUMMARY", 10),
            _line("p", "Results-oriented professional seeking HR opportunities.", 30),
            _heading("sk", "CORE SKILLS", 60),
            _line("s1", "Excel", 80),
            _line("s2", "Communication", 95),
        ]
    )
    result = detect_region_aware_sections(document)
    assert any("Results-oriented" in t for t in _texts(result, "SUMMARY"))
    assert _section_of(result, "Excel") == "SKILLS"
    assert _section_of(result, "Communication") == "SKILLS"


def test_j_projects_then_core_skills_opens_skills():
    document = _doc(
        [
            _heading("pr", "PROJECTS", 10),
            _line("title", "Hiring Workflow Automation", 30),
            _line("dt", "2021", 30, x0=400.0),
            _line("body", "Built dashboards to track candidate pipeline health.", 45),
            _heading("sk", "CORE SKILLS", 80),
            _line("s1", "Python", 100),
            _line("s2", "SQL", 115),
        ]
    )
    result = detect_region_aware_sections(document)
    assert "Hiring Workflow Automation" in _texts(result, "PROJECTS")
    assert _section_of(result, "Python") == "SKILLS"
    assert _section_of(result, "SQL") == "SKILLS"


def test_k_career_details_without_skill_signal_not_skills():
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "MBA", 30),
            _line("date", "2015", 30, x0=400.0),
            _heading("cd", "CAREER DETAILS", 70),
            _line("org", "Regional Operations Group", 90),
            _line("yr", "2016", 105),
        ]
    )
    result = detect_region_aware_sections(document)
    assert _texts(result, "SKILLS") == []
    assert "CAREER DETAILS" not in _texts(result, "SKILLS")
    assert _section_of(result, "Regional Operations Group") == "EDUCATION"


def test_l_skills_like_heading_with_years_only_not_skills():
    """Unknown skills-heading must not open SKILLS on date-heavy content alone."""
    document = _doc(
        [
            _heading("edu", "EDUCATION", 10),
            _line("deg", "MBA", 30),
            _line("date", "2015", 30, x0=400.0),
            _heading("sk", "PROFESSIONAL SKILLS", 70),
            _line("y1", "2019", 90),
            _line("y2", "2020", 105),
            _line("y3", "2021", 120),
        ]
    )
    result = detect_region_aware_sections(document)
    assert _texts(result, "SKILLS") == []
    assert _section_of(result, "2019") == "EDUCATION"
    assert "PROFESSIONAL SKILLS" not in _texts(result, "SKILLS")


def test_achievements_then_professional_skills_opens_skills():
    document = _doc(
        [
            _heading("ach", "ACHIEVEMENTS", 10),
            _line("aw", "Employee of the Year", 30),
            _line("org", "Acme Corp", 45),
            _line("yr", "2018", 60),
            _heading("sk", "PROFESSIONAL SKILLS", 90),
            _line("s1", "Excel", 110),
            _line("s2", "Communication", 125),
        ]
    )
    result = detect_region_aware_sections(document)
    assert "Employee of the Year" in _texts(result, "ACHIEVEMENTS")
    assert _section_of(result, "Excel") == "SKILLS"
    assert _section_of(result, "Communication") == "SKILLS"
