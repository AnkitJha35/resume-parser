from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def _values(text: str) -> list[str]:
    return [item["value"] for item in SkillsExtractor().extract([_make_block(text)], section_name="SKILLS")]


def test_A_existing_engineering_aliases_still_work():
    assert set(_values("Java, Python, SQL, PostgreSQL")) == {"Java", "Python", "SQL", "PostgreSQL"}


def test_B_office_tools_aliases():
    assert set(_values("MS Excel, Google Sheets, Google Docs, HRMS")) == {
        "Excel",
        "Google Sheets",
        "Google Docs",
        "HRMS",
    }


def test_C_hr_aliases():
    assert set(_values("Recruitment, Employee Engagement, Onboarding, HR Analytics")) == {
        "Recruitment",
        "Employee Engagement",
        "Onboarding",
        "HR Analytics",
    }


def test_D_soft_skill_aliases():
    assert set(_values("Communication, Teamwork, Adaptability, Problem Solving")) == {
        "Communication",
        "Teamwork",
        "Adaptability",
        "Problem Solving",
    }


def test_E_compound_google_sheets_and_docs():
    assert set(_values("Google Sheets & Docs")) == {"Google Sheets", "Google Docs"}


def test_F_bullet_excel():
    assert _values("• Excel") == ["Excel"]


def test_G_ratings_are_not_skills():
    assert _values("4") == []
    assert _values("5") == []
    assert _values("3") == []


def test_H_heading_is_not_a_skill():
    assert _values("TECHNICAL SKILLS") == []
    assert _values("Soft Skills") == []


def test_I_arbitrary_prose_is_not_emitted_as_a_skill_phrase():
    prose = "Understanding employee motivation and workplace behavior"
    values = _values(prose)
    assert prose not in values
    assert "Understanding employee motivation and workplace behavior" not in values


def test_J_fontawesome_prefix_recovers_employee_engagement():
    assert "Employee Engagement" in _values("\\faUsers : Employee Engagement")


def test_K_excel_aliases_dedupe():
    assert _values("Excel, MS Excel, Microsoft Excel") == ["Excel"]


def test_L_dense_engineering_comma_list_preserved():
    text = "Languages & Databases: Java, SQL, MySQL, PostgreSQL, MongoDB"
    assert {"Java", "SQL", "MySQL", "PostgreSQL", "MongoDB"} <= set(_values(text))
