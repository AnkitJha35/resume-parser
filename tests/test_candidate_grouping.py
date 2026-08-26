from types import SimpleNamespace

from app.pipeline.stages.block_classification import classify_block, ClassifiedBlock
from app.pipeline.stages.candidate_grouping import group_candidates


def _make_block(text: str, page=1, x0=0, y0=0, x1=100, y1=10, bold=False, font_size=None):
    return SimpleNamespace(text=text, page_number=page, x0=x0, y0=y0, x1=x1, y1=y1, bold=bold, font_size=font_size)


def _classify_sequence(texts):
    blocks = []
    for i, t in enumerate(texts):
        y = i * 12
        if isinstance(t, dict):
            kwargs = dict(text=t.get("text"), page=1, x0=t.get("x0", 0), y0=y, x1=100, y1=10 + y, bold=t.get("bold", False), font_size=t.get("font_size", None))
            blocks.append(_make_block(**kwargs))
        else:
            blocks.append(_make_block(t, x0=0, y0=y, y1=10 + y))
    return [classify_block(b) for b in blocks]


def test_experience_simple_grouping():
    seq = [
        {"text": "SDE 2", "bold": True, "font_size": 12},
        "Probus Smart Things",
        "Date : 07/2022 - Present",
        "Location: Noida, India",
        "Worked on backend systems and APIs." * 3,
        {"text": "Senior Engineer", "bold": True, "font_size": 12},
        "Another Company Pvt. Ltd",
        "Date : 01/2019 - 06/2022",
    ]
    classified = _classify_sequence(seq)
    groups = group_candidates(classified, section="EXPERIENCE")
    assert len(groups) == 2


def test_experience_date_first():
    seq = [
        "Date : 07/2022 - Present",
        {"text": "SDE 2", "bold": True, "font_size": 12},
        "Probus Smart Things",
        "Worked on backend systems and APIs." * 2,
        "Date : 01/2019 - 06/2022",
        "Senior Engineer",
        "Another Company Pvt. Ltd",
    ]
    classified = _classify_sequence(seq)
    groups = group_candidates(classified, section="EXPERIENCE")
    assert len(groups) == 2


def test_projects_grouping_without_project_title_label():
    seq = [
        "Network Monitoring System",
        "Date : 03/2021 - 08/2021",
        "Built distributed monitoring agents and dashboards.",
        "Implemented alerting and metrics pipelines.",
        "Solar String Monitoring System",
        "Date : 01/2020 - 12/2020",
        "Led firmware integration and data ingestion.",
    ]
    # create spatial gaps: second project starts after larger y gap
    blocks = []
    for i, t in enumerate(seq):
        if i < 4:
            y = i * 12
        else:
            y = 120 + (i - 4) * 12
        blocks.append(_make_block(t, y0=y, y1=10 + y))

    # introduce varying x0 to simulate spillover spans across the same visual region
    for i, b in enumerate(blocks):
        if i in (0, 2, 3):
            b.x0 += 320

    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="PROJECTS")
    assert len(groups) == 2


def test_ignore_zero_width_and_varying_x0():
    seq = [
        "M.C.A NIT Calicut",
        "\u200b",  # zero-width invisible
        "Date : 07/2019 - 06/2022",
        {"text": "Location: Kerala, India", "x0": 334},
        {"text": "B.SC-IT Magadh University", "x0": 316},
        "Date : 07/2015 - 08/2018",
        {"text": "Location: Patna, India", "x0": 400},
    ]

    blocks = []
    for i, t in enumerate(seq):
        y = i * 12
        if isinstance(t, dict):
            blocks.append(_make_block(t["text"], y0=y, x0=t.get("x0", 0)))
        else:
            blocks.append(_make_block(t, y0=y))

    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="EDUCATION")

    all_labels = [cb.label for g in groups for cb in g.blocks]
    assert "UNKNOWN" not in all_labels or all_labels.count("UNKNOWN") == 1
    assert len(groups) in (1, 2)
    assert all(len(g.blocks) > 0 for g in groups)


def test_projects_title_date_description_grouping():
    # Simulate three projects with title(x0=300), date(x0=300), description(x0=335)
    seq = []
    # Project 1
    seq.append({"text": "Head End System (HES)", "x0": 300})
    seq.append({"text": "Date : 02/2025 - present", "x0": 300})
    seq.append({"text": "Built components ...", "x0": 335})
    seq.append({"text": "Implemented features ...", "x0": 335})

    # Project 2 (starts after a large vertical gap)
    seq.append({"text": "Network Monitoring System", "x0": 300, "y0_offset": 120})
    seq.append({"text": "Date : 11/2023 - 01/2025", "x0": 300, "y0_offset": 120})
    seq.append({"text": "Led firmware integration", "x0": 335, "y0_offset": 120})

    # Project 3 (starts after a large vertical gap)
    seq.append({"text": "Solar String Monitoring System", "x0": 300, "y0_offset": 240})
    seq.append({"text": "Date : 09/2022 - 10/2023", "x0": 300, "y0_offset": 240})
    seq.append({"text": "Led data ingestion", "x0": 335, "y0_offset": 240})

    blocks = []
    for i, t in enumerate(seq):
        y = i * 12
        if isinstance(t, dict):
            y = t.get("y0_offset", 0) + (i % 4) * 12
            blocks.append(_make_block(t.get("text"), y0=y, x0=t.get("x0", 0)))
        else:
            blocks.append(_make_block(t, y0=y))

    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="PROJECTS")

    assert len(groups) == 3
    for g in groups:
        # each group should contain exactly one DATE
        assert sum(1 for cb in g.blocks if cb.label == "DATE") == 1
        # first block should be the project title (UNKNOWN)
        assert g.blocks[0].label == "UNKNOWN"


def test_education_grouping():
    seq = [
        "M.C.A NIT Calicut",
        "Date : 2014 - 2017",
        "Location: Calicut",
        {"text": "B.SC-IT Magadh University", "bold": True, "font_size": 11},
        "Date : 2010 - 2013",
        "Location: Patna",
    ]
    classified = _classify_sequence(seq)
    groups = group_candidates(classified, section="EDUCATION")
    assert len(groups) == 2


def test_resume_1_education_does_not_split_on_repeated_institution():
    seq = [
        "EDUCATIO",
        "DEGREE NAME /",
        "MAJOR",
        "University,",
        "DEGREE NAME / MAJOR",
        "University,",
        "Location 2007 -",
    ]
    classified = _classify_sequence(seq)
    groups = group_candidates(classified, section="EDUCATION")

    assert len(groups) == 1

    texts = [cb.original.text for cb in groups[0].blocks]
    assert texts == seq


def test_section_headers_not_included():
    seq = [
        "EXPERIENCE",
        "SDE 2",
        "Probus Smart Things",
        "Date : 07/2022 - Present",
    ]
    blocks = []
    for i, t in enumerate(seq):
        blocks.append(_make_block(t, y0=i * 12, y1=10 + i * 12))
    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="EXPERIENCE")
    # header should not be part of any group's blocks
    assert all(all(cb.label != "SECTION_HEADER" for cb in g.blocks) for g in groups)


def test_fresher_hr_education_groups_match_right_column_dates():
    from pathlib import Path

    from app.pipeline.stages.normalization import TextNormalizer
    from app.pipeline.stages.reading_order import ReadingOrder
    from app.pipeline.stages.text_extraction import PDFExtractor
    from app.pipeline.stages.sections import SectionDetector

    blocks = TextNormalizer.normalize_blocks(
        ReadingOrder.reorder(PDFExtractor.extract(Path("tests/fixtures/fresher_hr_resume.pdf").read_bytes()))
    )
    sections = SectionDetector().detect(blocks)
    classified = [classify_block(block) for block in sections["EDUCATION"]]
    groups = group_candidates(classified, section="EDUCATION")

    group_texts = [[cb.original.text for cb in group.blocks] for group in groups]
    mba_group = next(group for group in group_texts if "MBA (Human Resource Management) | School Of Open Learning , Delhi University (DU-SOL)" in group)
    bachelor_group = next(group for group in group_texts if "Bachelor of Arts (General) | BIR Tikendrajit University" in group)

    assert "2024 – 2026" in mba_group
    assert "2020 – 2023" in bachelor_group
    assert "Bachelor of Arts (General) | BIR Tikendrajit University" not in mba_group
    assert "MBA (Human Resource Management) | School Of Open Learning , Delhi University (DU-SOL)" not in bachelor_group
    assert all(group not in ({"2024 – 2026"}, {"2020 – 2023"}) for group in map(set, group_texts))


def test_resume_2_experience_date_starts_group_with_following_title():
    from pathlib import Path

    from app.domain.document import document_from_text_blocks
    from app.pipeline.stages.layout import interpret_layout
    from app.pipeline.stages.reconstruction import reconstruct_document
    from app.pipeline.stages.semantic_compat import semantic_sections_to_text_blocks
    from app.pipeline.stages.semantic_paths import detect_region_aware_sections
    from app.pipeline.stages.text_extraction import PDFExtractor

    raw = PDFExtractor.extract(Path("tests/fixtures/resume_2.pdf").read_bytes())
    document = interpret_layout(reconstruct_document(document_from_text_blocks(raw)))
    semantic = detect_region_aware_sections(document)
    blocks = semantic_sections_to_text_blocks(semantic)["EXPERIENCE"]
    groups = group_candidates([classify_block(block) for block in blocks], "EXPERIENCE")

    assert any(
        [cb.original.text for cb in group.blocks][:3]
        == ["(June 2017 – August 2019)", "SECRETARY", "BRIGHT SPOT LTD – Boston, MA"]
        for group in groups
    )


def test_projects_horizontal_title_date_layout():
    # Title at x0=100, date at same y but x0=420 (to the right). Description below.
    blocks = []
    blocks.append(_make_block("Project A", x0=100, y0=0))
    blocks.append(_make_block("Date : 01/2021 - 12/2021", x0=420, y0=0))
    blocks.append(_make_block("Implemented feature X", x0=120, y0=12))

    # Second project after a gap
    blocks.append(_make_block("Project B", x0=100, y0=120))
    blocks.append(_make_block("Date : 02/2020 - 12/2020", x0=420, y0=120))
    blocks.append(_make_block("Implemented feature Y", x0=120, y0=132))

    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="PROJECTS")
    assert len(groups) == 2
    for g in groups:
        assert sum(1 for cb in g.blocks if cb.label == "DATE") == 1
        assert g.blocks[0].label == "UNKNOWN"


def test_projects_indented_body_layout():
    # Title/date aligned left, body indented
    blocks = []
    blocks.append(_make_block("Analytics Dashboard", x0=80, y0=0))
    blocks.append(_make_block("Date : 03/2022 - 08/2022", x0=80, y0=12))
    blocks.append(_make_block("- Built APIs", x0=140, y0=24))

    blocks.append(_make_block("Data Pipeline", x0=80, y0=120))
    blocks.append(_make_block("Date : 01/2021 - 02/2022", x0=80, y0=132))
    blocks.append(_make_block("- Stream processing", x0=140, y0=144))

    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="PROJECTS")
    assert len(groups) == 2


def test_projects_two_consecutive_projects_small_gap():
    # Two projects with small vertical gap but clear DATE markers should split
    blocks = []
    blocks.append(_make_block("Proj One", x0=100, y0=0))
    blocks.append(_make_block("Date : 01/2021 - 06/2021", x0=100, y0=12))
    blocks.append(_make_block("Work A", x0=120, y0=24))

    # Small gap, next title immediately follows
    blocks.append(_make_block("Proj Two", x0=100, y0=36))
    blocks.append(_make_block("Date : 07/2021 - 12/2021", x0=100, y0=48))
    blocks.append(_make_block("Work B", x0=120, y0=60))

    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="PROJECTS")
    assert len(groups) == 2


def test_projects_unknown_title_followed_by_date_lookahead():
    # Unknown title (no bold) followed shortly by DATE should form a project
    blocks = []
    blocks.append(_make_block("Mysterious Project", x0=200, y0=0))
    blocks.append(_make_block("Date : 05/2020 - 10/2020", x0=200, y0=12))
    blocks.append(_make_block("Did important things", x0=220, y0=24))

    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="PROJECTS")
    assert len(groups) == 1
    g = groups[0]
    assert sum(1 for cb in g.blocks if cb.label == "DATE") == 1


def test_projects_bullet_continuation_not_new_group():
    # Bullet lines should not start a new project group; they continue the prior body
    blocks = []
    blocks.append(_make_block("Head End System (HES)", x0=300, y0=0))
    blocks.append(_make_block("Date : 02/2025 - present", x0=300, y0=12))
    blocks.append(_make_block("Architected a scalable microservices-based Head End", x0=335, y0=24))
    blocks.append(_make_block("●", x0=316, y0=36))
    blocks.append(_make_block("containerization, Kubernetes, and Jenkins CI/CD.", x0=335, y0=48))

    # Next project starts after a larger gap
    blocks.append(_make_block("Network Monitoring System", x0=300, y0=160))
    blocks.append(_make_block("Date : 11/2023 - 01/2025", x0=300, y0=172))
    blocks.append(_make_block("Developed a scalable backend", x0=335, y0=184))

    classified = [classify_block(b) for b in blocks]
    groups = group_candidates(classified, section="PROJECTS")
    # should be two groups: first contains the bullet and following line
    assert len(groups) == 2
    assert groups[0].start_index == 0 and groups[0].end_index >= 4
    assert any((getattr(cb, 'label', None) == 'DATE') for cb in groups[0].blocks)
