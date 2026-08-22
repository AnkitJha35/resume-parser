from app.extractors.experience import ExperienceExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _b(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_experience_bullet_cleanup_and_continuation_joining():
    blocks = [
        _b("SDE 2"),
        _b("Company X"),
        _b("• Engineered scalable Spring Boot microservices using SOLID"),
        _b("principles, Saga pattern, and Spring Cloud for distributed"),
        _b("consistency."),
        _b("• Developed high-throughput event-driven data pipelines utilizing"),
        _b("Kafka and MQTT for asynchronous communication."),
        _b("• Optimized performance via Ignite caching and Hibernate query"),
        _b("tuning; implemented CI/CD pipelines with Jenkins and GitHub"),
        _b("Actions."),
    ]

    extractor = ExperienceExtractor()
    entries = extractor.extract(blocks)
    assert len(entries) == 1
    exp = entries[0]
    desc = exp["description"]
    assert "•" in desc
    assert "Engineered scalable Spring Boot microservices using SOLID principles, Saga pattern, and Spring Cloud for distributed consistency." in desc
    assert "Developed high-throughput event-driven data pipelines utilizing Kafka and MQTT for asynchronous communication." in desc
    assert "Optimized performance via Ignite caching and Hibernate query tuning; implemented CI/CD pipelines with Jenkins and GitHub Actions." in desc


def test_experience_description_preserves_separate_bullets_and_removes_pdf_artifact():
    text = "• Built APIs\n●\n• Improved reliability\nwith retries and caching."
    cleaned = ExperienceExtractor()._clean_experience_description(text)
    assert cleaned == "• Built APIs\n• Improved reliability with retries and caching."


def test_company_word_boundary_regression_principles_not_company():
    assert ExperienceExtractor()._is_company_line("principles, Saga pattern, and Spring Cloud for distributed") is False
    assert ExperienceExtractor()._is_company_line("Company X") is True


def test_original_textblocks_unchanged():
    b1 = _b("• Engineered scalable Spring Boot microservices using SOLID")
    b2 = _b("principles, Saga pattern, and Spring Cloud for distributed")
    blocks = [b1, b2]
    extractor = ExperienceExtractor()
    _ = extractor.extract(blocks)
    assert b1.text == "• Engineered scalable Spring Boot microservices using SOLID"
    assert b2.text == "principles, Saga pattern, and Spring Cloud for distributed"