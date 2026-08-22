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


def test_normalizer_coalesces_same_visual_line_fragments():
    # fragments on same y0 should merge into one logical line
    b1 = TextBlock(text="JMS", page_number=1, x0=334.8, y0=253.0, x1=350.0, y1=260.0, font_size=10)
    b2 = TextBlock(text="to", page_number=1, x0=363.3, y0=253.0, x1=370.0, y1=260.0, font_size=10)
    b3 = TextBlock(text="manage", page_number=1, x0=383.1, y0=253.0, x1=420.0, y1=260.0, font_size=10)
    b4 = TextBlock(text="high-volume", page_number=1, x0=425.3, y0=253.0, x1=495.0, y1=260.0, font_size=10)

    normalized = TextNormalizer.normalize_blocks([b1, b2, b3, b4])
    assert len(normalized) == 1
    assert normalized[0].text == "JMS to manage high-volume"


def test_normalizer_keeps_different_y_lines_separate():
    b1 = TextBlock(text="Line one", page_number=1, x0=10, y0=10, x1=100, y1=16)
    b2 = TextBlock(text="Line two", page_number=1, x0=10, y0=30, x1=100, y1=36)
    normalized = TextNormalizer.normalize_blocks([b1, b2])
    assert len(normalized) == 2


def test_normalizer_does_not_merge_bullets():
    b1 = TextBlock(text="Item", page_number=1, x0=10, y0=100, x1=50, y1=106, font_size=10)
    b2 = TextBlock(text="●", page_number=1, x0=55, y0=100, x1=60, y1=106, font_size=10)
    b3 = TextBlock(text="detail", page_number=1, x0=70, y0=100, x1=120, y1=106, font_size=10)
    normalized = TextNormalizer.normalize_blocks([b1, b2, b3])
    # bullet should prevent merging into a single block
    assert len(normalized) == 3


def test_normalizer_does_not_merge_across_pages():
    b1 = TextBlock(text="Foo", page_number=1, x0=10, y0=10, x1=30, y1=16)
    b2 = TextBlock(text="Bar", page_number=2, x0=12, y0=10, x1=40, y1=16)
    normalized = TextNormalizer.normalize_blocks([b1, b2])
    assert len(normalized) == 2
