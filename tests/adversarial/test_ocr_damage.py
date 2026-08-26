from app.pipeline.stages.normalization import TextNormalizer
from app.pipeline.stages.text_extraction import TextBlock


def test_normalization_preserves_contact_and_damage_gracefully():
    blocks = [TextBlock("john.doe+tag@example.com", 1, 0, 0, 100, 10), TextBlock("895 555\\xa0555", 1, 0, 20, 100, 30)]
    result = TextNormalizer.normalize_blocks(blocks)
    assert result[0].text == "john.doe+tag@example.com"
    assert result[1].text
