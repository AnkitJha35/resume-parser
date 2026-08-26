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


def test_contact_extractor_prefers_letter_spaced_name_over_role_title():
    blocks = [
        _make_block("A N G E L A"),
        _make_block("W I L K I N S O"),
        _make_block("ADMINISTRATIVE ASSISTANT"),
        _make_block("youremail@gmail.com"),
        _make_block("895 555 555"),
        _make_block("Drive Harrisburg, PA"),
    ]

    result = ContactExtractor.extract(blocks)

    assert result["name"]["value"] != "ADMINISTRATIVE ASSISTANT"
    assert result["name"]["value"] == "ANGELA WILKINSO"
    assert result["email"]["value"] == "youremail@gmail.com"
    assert result["phone"]["value"] == "895 555 555"


def test_contact_extractor_supports_scheme_less_linkedin():
    result = ContactExtractor.extract([
        _make_block("John Doe"),
        _make_block("linkedin.com/in/johndoe"),
    ])

    assert result["linkedin"]["value"] == "linkedin.com/in/johndoe"


def test_contact_extractor_normalizes_ligatures_in_linkedin():
    result = ContactExtractor.extract([_make_block("linkedin.com/in/yourproﬁle")])

    assert result["linkedin"]["value"] == "linkedin.com/in/yourprofile"


def test_contact_extractor_prefers_prominent_name_over_spaced_role_title():
    blocks = [
        _make_block("DAVID PÉREZ"),
        _make_block("A d m i n i s t r a t i v e A s s i s t a n t"),
    ]
    blocks[0].font_size = 28
    blocks[1].font_size = 11

    result = ContactExtractor.extract(blocks)

    assert result["name"]["value"] == "DAVID PÉREZ"


def test_contact_extractor_combines_spaced_name_fragments_across_header_text():
    blocks = [
        _make_block("J O H N"),
        _make_block("PROFILE"),
        _make_block("D O E"),
    ]

    result = ContactExtractor.extract(blocks)

    assert result["name"]["value"] == "JOHN DOE"


def test_contact_extractor_preserves_spaced_name_word_boundary():
    result = ContactExtractor.extract([_make_block("M A R G A R E T  T H O M A S O")])

    assert result["name"]["value"] == "MARGARET THOMASO"


def test_contact_extractor_accepts_single_spaced_name_fragment():
    result = ContactExtractor.extract([_make_block("J O H N")])

    assert result["name"]["value"] == "JOHN"


def test_contact_extractor_normalizes_unicode_whitespace_in_phone():
    result = ContactExtractor.extract([_make_block("895 555\xa0555")])

    assert result["phone"]["value"] == "895 555 555"


def test_contact_extractor_rejects_structural_and_non_location_text():
    assert ContactExtractor._find_location(["Summary", "Software Engineer", "A concise profile sentence."], None, None, None, None, None) is None


def test_contact_extractor_rejects_standalone_skill_as_location():
    assert ContactExtractor._find_location(["Innovative"], None, None, None, None, None) is None


def test_contact_extractor_accepts_international_location_structure():
    assert ContactExtractor._find_location(["Singapore"], None, None, None, None, None) == "Singapore"
    assert ContactExtractor._find_location(["Singapore, Singapore"], None, None, None, None, None) == "Singapore, Singapore"


def test_contact_extractor_combines_two_line_address():
    assert ContactExtractor._find_location(
        ["4397 Aaron Smith Drive", "Harrisburg, PA 17101"],
        None, None, None, None, None,
    ) == "4397 Aaron Smith Drive, Harrisburg, PA 17101"


def test_contact_extractor_keeps_standalone_location_support():
    assert ContactExtractor._find_location(["Singapore"], None, None, None, None, None) == "Singapore"
