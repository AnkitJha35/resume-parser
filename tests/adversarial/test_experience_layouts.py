from types import SimpleNamespace

import pytest

from app.extractors.experience import ExperienceExtractor
from app.pipeline.stages.block_classification import ClassifiedBlock
from app.pipeline.stages.candidate_grouping import CandidateGroup
from app.pipeline.stages.text_extraction import TextBlock


def group(lines):
    blocks = []
    for index, text in enumerate(lines):
        block = TextBlock(text=text, page_number=1, x0=10, y0=index * 20, x1=300, y1=index * 20 + 10)
        label = "DATE" if any(token in text.lower() for token in ("present", "current", "2020", "2021", "2022", "2023", "2024")) else "UNKNOWN"
        blocks.append(ClassifiedBlock(block, label, 1.0, []))
    return CandidateGroup("EXPERIENCE", blocks, 1, 0, 0, len(blocks) - 1, "")


@pytest.mark.parametrize("lines", [
    ["Software Engineer", "Google", "New York", "2021 - 2024", "Designed services"],
    ["Google", "Software Engineer", "New York", "2021 - 2024", "Designed services"],
    ["Software Engineer - 2021 to 2024", "Google, New York", "Designed services"],
    ["Software Engineer", "Google | New York | 2021 - 2024", "Designed services"],
    ["Software Engineer", "Google", "Worked on", "distributed systems"],
    ["Software Engineer", "Google", "Designed and implemented distributed services"],
])
def test_common_experience_layouts_keep_body_text_as_description(lines):
    entries = ExperienceExtractor().extract(groups=[group(lines)])
    assert len(entries) == 1
    assert entries[0]["description"] is None or "distributed" in entries[0]["description"] or "Designed" in entries[0]["description"]
