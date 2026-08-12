from app.extractors.contact import ContactExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str, y0: float = 0.0) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=y0, x1=0, y1=0)


def test_contact_extractor_with_all_fields():
    blocks = [
        _make_block("John Doe"),
        _make_block("john.doe@example.com"),
        _make_block("+1 555-123-4567"),
        _make_block("San Francisco, CA"),
        _make_block("https://linkedin.com/in/johndoe"),
        _make_block("https://github.com/johndoe"),
        _make_block("https://johndoe.dev"),
    ]

    result = ContactExtractor.extract(blocks)

    assert result["name"]["value"] == "John Doe"
    assert result["email"]["value"] == "john.doe@example.com"
    assert result["phone"]["value"] == "+1 555-123-4567"
    assert result["location"]["value"] == "San Francisco, CA"
    assert result["linkedin"]["value"] == "https://linkedin.com/in/johndoe"
    assert result["github"]["value"] == "https://github.com/johndoe"
    assert result["portfolio"]["value"] == "https://johndoe.dev"


def test_contact_extractor_without_email():
    blocks = [
        _make_block("Jane Doe"),
        _make_block("+1 555-987-6543"),
    ]

    result = ContactExtractor.extract(blocks)

    assert result["email"]["value"] is None
    assert result["phone"]["value"] == "+1 555-987-6543"
    assert result["name"]["value"] == "Jane Doe"


def test_contact_extractor_without_phone():
    blocks = [
        _make_block("Alex Smith"),
        _make_block("alex.smith@example.com"),
    ]

    result = ContactExtractor.extract(blocks)

    assert result["phone"]["value"] is None
    assert result["email"]["value"] == "alex.smith@example.com"
    assert result["name"]["value"] == "Alex Smith"


def test_contact_extractor_with_linkedin_github_portfolio():
    blocks = [
        _make_block("Morgan Lee"),
        _make_block("morgan.lee@example.com"),
        _make_block("https://linkedin.com/in/morganlee"),
        _make_block("https://github.com/morganlee"),
        _make_block("https://morganlee.dev"),
    ]

    result = ContactExtractor.extract(blocks)

    assert result["linkedin"]["value"] == "https://linkedin.com/in/morganlee"
    assert result["github"]["value"] == "https://github.com/morganlee"
    assert result["portfolio"]["value"] == "https://morganlee.dev"
