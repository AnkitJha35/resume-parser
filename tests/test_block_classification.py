from types import SimpleNamespace

from app.pipeline.stages.block_classification import classify_block


def _make_block(text: str, bold=False, font_size=None):
    return SimpleNamespace(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0, bold=bold, font_size=font_size)


def test_classify_job_titles_and_companies():
    b1 = _make_block("SDE 2", bold=True, font_size=12)
    c1 = classify_block(b1)
    assert c1.label == "JOB_TITLE"

    b2 = _make_block("Probus Smart Things!", bold=False, font_size=12)
    c2 = classify_block(b2)
    assert c2.label == "COMPANY"

    b3 = _make_block("Date : 07/2022 - Present")
    c3 = classify_block(b3)
    assert c3.label == "DATE"

    b4 = _make_block("SDE Intern", bold=True)
    c4 = classify_block(b4)
    assert c4.label == "JOB_TITLE"

    b5 = _make_block("Navyug Infosolutions Pvt. Ltd")
    c5 = classify_block(b5)
    assert c5.label == "COMPANY"


def test_ambiguous_lines_are_unknown_or_low_confidence():
    b = _make_block("Experienced software professional")
    c = classify_block(b)
    # conservative: should not confidently claim JOB_TITLE or COMPANY
    assert c.label in ("UNKNOWN", "DESCRIPTION")


def test_date_classification_ignores_zero_width_trailing_chars():
    c1 = classify_block(_make_block("Date : 02/2022 - 06/2022 \u200b"))
    assert c1.label == "DATE"

    c2 = classify_block(_make_block("Date : 07/2019 - 06/2022 \u200b"))
    assert c2.label == "DATE"


def test_location_and_section_and_bullets_and_dates():
    # location explicit
    b_loc = _make_block("Location: Noida, UP, India")
    c_loc = classify_block(b_loc)
    assert c_loc.label == "LOCATION"

    # ambiguous comma but not a location
    b_comma = _make_block("Languages & Databases: Java, SQL, MySQL, PostgreSQL,")
    c_comma = classify_block(b_comma)
    assert c_comma.label != "LOCATION"

    # section header
    b_section = _make_block("SUMMARY")
    c_section = classify_block(b_section)
    assert c_section.label == "SECTION_HEADER"

    # bullet requires marker
    b_bullet = _make_block("4 years of experience in backend development, specializing")
    c_bullet = classify_block(b_bullet)
    assert c_bullet.label != "BULLET"

    # date variants
    for s in ["Date : 07/2022 - Present", "Date : 02/2022 - 06/2022", "Date : 09/2022 - 10/2023"]:
        cb = classify_block(_make_block(s))
        assert cb.label == "DATE"

    # project titles stay UNKNOWN
    b_proj = _make_block("Network Monitoring System")
    c_proj = classify_block(b_proj)
    assert c_proj.label in ("UNKNOWN", "DESCRIPTION")

    # education examples
    b_edu1 = _make_block("M.C.A NIT Calicut")
    c_edu1 = classify_block(b_edu1)
    assert c_edu1.label in ("UNKNOWN", "DEGREE", "INSTITUTION")

    b_edu2 = _make_block("B.SC-IT Magadh University")
    c_edu2 = classify_block(b_edu2)
    assert c_edu2.label in ("DEGREE", "UNKNOWN")


def test_degree_alias_matching_rejects_ms_excel_but_keeps_ms_degrees():
    assert classify_block(_make_block("MS Excel (VLOOKUP, Pivot Tables, Filters)")).label != "DEGREE"
    assert classify_block(_make_block("MS")).label == "DEGREE"
    assert classify_block(_make_block("M.S.")).label == "DEGREE"
    assert classify_block(_make_block("MS in Computer Science")).label == "DEGREE"
    assert classify_block(_make_block("Master of Science")).label == "DEGREE"


def test_placeholder_degree_lines_are_classified_as_degrees():
    assert classify_block(_make_block("DEGREE NAME / MAJOR")).label == "DEGREE"


def test_descriptive_company_text_is_not_classified_as_location():
    block = _make_block("Developed new filing and organizational practices, saving the company")

    assert classify_block(block).label != "LOCATION"


def test_parenthesized_date_range_is_classified_as_date():
    result = classify_block(_make_block("(June 2017 – August 2019)"))

    assert result.label == "DATE"
    assert result.reasons == ["parenthesized_date_pattern"]
