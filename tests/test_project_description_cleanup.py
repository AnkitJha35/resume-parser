from app.extractors.projects import ProjectExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _b(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_remove_bullet_only_lines_and_join_sentences():
    blocks = [
        _b("Head End System (HES)"),
        _b("Architected a scalable microservices-based Head End"),
        _b("●"),
        _b("System for smart meter data ingestion and analytics."),
        _b("Leveraged REST APIs, Apache Kafka, MQTT, Weblogic"),
        _b("●"),
        _b("JMS to manage high-volume synchronous and"),
        _b("asynchronous data streams."),
    ]

    extractor = ProjectExtractor()
    entries = extractor.extract(blocks)
    assert len(entries) == 1
    desc = entries[0]["description"]
    assert "●" not in desc
    assert "Head End System for smart meter data ingestion" in desc
    assert "Weblogic JMS to manage" in desc


def test_preserve_meaningful_bullet_content_and_paragraphs():
    blocks = [
        _b("Project X"),
        _b("Implemented features:"),
        _b("- Feature A: did something."),
        _b("- Feature B: did something else."),
        _b("") ,
        _b("Notes:"),
        _b("This is a separate paragraph."),
    ]
    extractor = ProjectExtractor()
    entries = extractor.extract(blocks)
    desc = entries[0]["description"]
    assert "Feature A" in desc
    assert "Feature B" in desc
    assert "This is a separate paragraph." in desc


def test_original_textblocks_unchanged():
    b1 = _b("Saga")
    b2 = _b("Pattern")
    blocks = [b1, b2]
    extractor = ProjectExtractor()
    entries = extractor.extract(blocks)
    # ensure original blocks remain unchanged
    assert b1.text == "Saga"
    assert b2.text == "Pattern"