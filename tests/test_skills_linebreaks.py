from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_multiword_skill_across_blocks_matches_and_original_unchanged():
    extractor = SkillsExtractor()
    b1 = _make_block("Weblogic")
    b2 = _make_block("JMS")
    blocks = [b1, b2]

    skills = extractor.extract(blocks, section_name="SKILLS")
    values = {s['value'] for s in skills}

    assert 'Weblogic JMS' in values
    # ensure original TextBlock.text unchanged
    assert b1.text == "Weblogic"
    assert b2.text == "JMS"


def test_multiword_skill_within_block_newline_matches():
    extractor = SkillsExtractor()
    b = _make_block("Saga\nPattern")
    skills = extractor.extract([b], section_name="SKILLS")
    values = {s['value'] for s in skills}
    assert 'Saga Pattern' in values


def test_existing_single_line_multiword_skill_still_matches():
    extractor = SkillsExtractor()
    b = _make_block("Spring Boot")
    skills = extractor.extract([b], section_name="SKILLS")
    values = {s['value'] for s in skills}
    assert 'Spring Boot' in values
