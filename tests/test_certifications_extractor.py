from app.extractors.certifications import CertificationExtractor
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_certification_extractor_with_all_fields():
    blocks = [
        _make_block("AWS Certified Solutions Architect"),
        _make_block("Amazon"),
        _make_block("Issued: Jun 2022"),
        _make_block("Expires: Jun 2025"),
        _make_block("Credential ID: ABCD-1234"),
        _make_block("https://www.credly.com/badges/12345"),
    ]

    extractor = CertificationExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "AWS Certified Solutions Architect"
    assert entry["issuingOrganization"] == "Amazon"
    assert entry["issueDate"] == "2022-06"
    assert entry["expiryDate"] == "2025-06"
    assert entry["credentialId"] == "ABCD-1234"
    assert entry["credentialUrl"] == "https://www.credly.com/badges/12345"
    assert entry["confidence"] == 0.85


def test_certification_extractor_with_missing_expiry():
    blocks = [
        _make_block("Google Cloud Professional Data Engineer"),
        _make_block("Google"),
        _make_block("Issued on 2023"),
    ]

    extractor = CertificationExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["issueDate"] == "2023"
    assert entry["expiryDate"] is None


def test_certification_extractor_splits_description_properly():
    blocks = [
        _make_block("Certified Kubernetes Administrator"),
        _make_block("Linux Foundation"),
        _make_block("Credential ID: CKA-5678"),
        _make_block("Maintained Kubernetes clusters in production."),
    ]

    extractor = CertificationExtractor()
    entries = extractor.extract(blocks)

    assert len(entries) == 1
    assert "Maintained Kubernetes clusters" in entries[0]["description"]
