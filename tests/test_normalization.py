from app.pipeline.stages.normalization import TextNormalizer
from app.pipeline.stages.text_extraction import TextBlock


def test_normalizer_collapses_unnecessary_whitespace():
    block = TextBlock(
        text="   Senior   Engineer\t\t at  Acme Corp   ",
        page_number=1,
        x0=0,
        y0=0,
        x1=0,
        y1=0,
    )

    normalized = TextNormalizer.normalize_blocks([block])[0]

    assert normalized.text == "Senior Engineer at Acme Corp"


def test_normalizer_removes_control_characters_and_ligatures():
    block = TextBlock(
        text="Lead\u000bDeveloper\u000c at Acme\u000dCorp fiﬂ",
        page_number=1,
        x0=0,
        y0=0,
        x1=0,
        y1=0,
    )

    normalized = TextNormalizer.normalize_blocks([block])[0]

    assert "\u000b" not in normalized.text
    assert "\u000c" not in normalized.text
    assert "\u000d" not in normalized.text
    assert "fiﬂ" not in normalized.text
    assert normalized.text == "Lead Developer at Acme\nCorp fifl"


def test_normalizer_preserves_emails_urls_dates():
    block = TextBlock(
        text="Contact: jane.doe@example.com | https://example.com | Jan 2022 - Present",
        page_number=1,
        x0=0,
        y0=0,
        x1=0,
        y1=0,
    )

    normalized = TextNormalizer.normalize_blocks([block])[0]

    assert normalized.text == "Contact: jane.doe@example.com | https://example.com | Jan 2022 - Present"
