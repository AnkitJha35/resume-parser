"""Phase 6L: labeled LinkedIn handle extraction."""

from __future__ import annotations

from pathlib import Path

from app.extractors.contact import ContactExtractor
from app.pipeline.parser import ResumeParser
from app.pipeline.stages.text_extraction import TextBlock


def _block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=100, y1=10)


def _linkedin(text: str) -> str | None:
    return ContactExtractor.extract([_block(text)])["linkedin"]["value"]


def test_labeled_handle_linkedin_colon_username():
    assert _linkedin("linkedin: username") == "https://www.linkedin.com/in/username"


def test_labeled_handle_linkedin_title_case():
    assert _linkedin("LinkedIn: username") == "https://www.linkedin.com/in/username"


def test_labeled_handle_linked_in_spaced():
    assert _linkedin("linked in: username") == "https://www.linkedin.com/in/username"


def test_labeled_handle_linkedin_profile():
    assert _linkedin("LinkedIn profile: username") == "https://www.linkedin.com/in/username"


def test_scheme_less_linkedin_url_unchanged():
    assert _linkedin("linkedin.com/in/username") == "linkedin.com/in/username"


def test_full_https_linkedin_url_unchanged():
    assert _linkedin("https://www.linkedin.com/in/username") == "https://www.linkedin.com/in/username"


def test_bare_username_rejected():
    assert _linkedin("username") is None


def test_github_labeled_username_rejected():
    assert _linkedin("github: username") is None


def test_email_with_linkedin_domain_rejected():
    assert _linkedin("email: someone@linkedin.com") is None


def test_linkedin_na_rejected():
    assert _linkedin("linkedin: N/A") is None
    assert _linkedin("linkedin: -") is None
    assert _linkedin("linkedin: none") is None


def test_linkedin_foreign_url_rejected():
    assert _linkedin("linkedin: http://some-random-site.com") is None


def test_resume_a_linkedin_populated_from_handle():
    path = Path("tests/fixtures/AditCV_SOL.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.personal.linkedin == "https://www.linkedin.com/in/aditianand99"


def test_resume_b_linkedin_url_unchanged():
    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.personal.linkedin == "https://www.linkedin.com/in/aditianand99/"


def test_resume_c_linkedin_url_unchanged():
    path = Path("tests/fixtures/swe_experienced_resume.pdf")
    if not path.exists():
        return
    resume = ResumeParser().parse_with_layout_pipeline(path.read_bytes())
    assert resume.personal.linkedin == "https://linkedin.com/in/ankitjhajavafullstackdeveloper"
