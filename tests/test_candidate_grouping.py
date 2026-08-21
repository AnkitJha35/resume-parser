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
