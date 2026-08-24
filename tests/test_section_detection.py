from pathlib import Path

from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.text_extraction import TextBlock


def _make_block(text: str) -> TextBlock:
    return TextBlock(text=text, page_number=1, x0=0, y0=0, x1=0, y1=0)


def test_section_detection_single_column_resume():
    blocks = [
        _make_block("Profile"),
        _make_block("Experienced engineer"),
        _make_block("Experience"),
        _make_block("Company A"),
        _make_block("Education"),
        _make_block("University X"),
    ]

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert sections["SUMMARY"] == [blocks[1]]
    assert sections["EXPERIENCE"] == [blocks[3]]
    assert sections["EDUCATION"] == [blocks[5]]


def test_section_detection_two_column_resume():
    blocks = [
        _make_block("Professional Experience"),
        _make_block("Company A"),
        _make_block("Skills"),
        _make_block("Python"),
        _make_block("Projects"),
        _make_block("Resume Parser"),
    ]

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert sections["EXPERIENCE"] == [blocks[1]]
    assert sections["SKILLS"] == [blocks[3]]
    assert sections["PROJECTS"] == [blocks[5]]


def test_section_detection_nonstandard_header_aliases():
    blocks = [
        _make_block("Career Summary"),
        _make_block("Skilled at APIs"),
        _make_block("Work Experience"),
        _make_block("Company B"),
    ]

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert sections["SUMMARY"] == [blocks[1]]
    assert sections["EXPERIENCE"] == [blocks[3]]


def test_section_detection_missing_sections():
    blocks = [
        _make_block("Skills"),
        _make_block("Python"),
        _make_block("Certifications"),
        _make_block("AWS Certified"),
    ]

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert sections["SKILLS"] == [blocks[1]]
    assert sections["CERTIFICATIONS"] == [blocks[3]]
    assert sections["EXPERIENCE"] == []
    assert sections["EDUCATION"] == []


def test_section_detection_spaced_letter_headings_and_known_aliases():
    detector = SectionDetector()

    assert detector._find_section_header("P R O F I L E") == "SUMMARY"
    assert detector._find_section_header("E D U C A T I O N") == "EDUCATION"
    assert detector._find_section_header("K E Y S K I L L S") == "SKILLS"
    assert detector._find_section_header("PROFESSIONAL EXPERIENCE") == "EXPERIENCE"
    assert detector._find_section_header("CAREER OBJECTIVE") == "SUMMARY"
    assert detector._find_section_header("SKILL HIGHLIGHTS") == "SKILLS"
    assert detector._find_section_header("SOFT SKILLS") == "SKILLS"
    assert detector._find_section_header("HARD SKILLS") == "SKILLS"
    assert detector._find_section_header("SUMMARY") == "SUMMARY"
    assert detector._find_section_header("EXPERIENCE") == "EXPERIENCE"
    assert detector._find_section_header("EDUCATION") == "EDUCATION"
    assert detector._find_section_header("PROJECTS") == "PROJECTS"
    assert detector._find_section_header("PROFESSIONAL  EXPERIENCE") == "EXPERIENCE"
    assert detector._find_section_header("SKILL  HIGHLIGHTS") == "SKILLS"


def test_section_detection_fresher_hr_resume_fixture():
    from pathlib import Path
    from app.pipeline.stages.text_extraction import PDFExtractor

    path = Path("tests/fixtures/fresher_hr_resume.pdf")
    raw_pdf = path.read_bytes()
    blocks = PDFExtractor.extract(raw_pdf)

    detector = SectionDetector()
    sections = detector.detect(blocks)

    assert any("MBA (Human Resource Management)" in block.text for block in sections["EDUCATION"])
    assert any("Bachelor of Arts (General)" in block.text for block in sections["EDUCATION"])
    assert not any(
        "An ambitious MBA student specializing in Human Resource Management" in block.text
        for block in sections["EDUCATION"]
    )
    assert sections["EXPERIENCE"] == []


def test_section_detection_fresher_hr_resume_keeps_date_ranges_in_education():
    from app.pipeline.stages.text_extraction import PDFExtractor

    blocks = PDFExtractor.extract(Path("tests/fixtures/fresher_hr_resume.pdf").read_bytes())
    detector = SectionDetector()
    sections = detector.detect(blocks)

    education_texts = {block.text.strip() for block in sections["EDUCATION"]}
    skill_texts = {block.text.strip() for block in sections["SKILLS"]}

    assert "2024 – 2026" in education_texts
    assert "2020 – 2023" in education_texts
    assert "2024 – 2026" not in skill_texts
    assert "2020 – 2023" not in skill_texts


def test_section_detection_fresher_hr_resume_keeps_dates_after_reading_order():
    from app.pipeline.stages.normalization import TextNormalizer
    from app.pipeline.stages.reading_order import ReadingOrder
    from app.pipeline.stages.text_extraction import PDFExtractor

    blocks = TextNormalizer.normalize_blocks(
        ReadingOrder.reorder(PDFExtractor.extract(Path("tests/fixtures/fresher_hr_resume.pdf").read_bytes()))
    )
    sections = SectionDetector().detect(blocks)

    education_texts = {block.text.strip() for block in sections["EDUCATION"]}
    skill_texts = {block.text.strip() for block in sections["SKILLS"]}

    assert {"MBA (Human Resource Management) | School Of Open Learning , Delhi University (DU-SOL)", "Bachelor of Arts (General) | BIR Tikendrajit University"} <= education_texts
    assert {"2024 – 2026", "2020 – 2023"} <= education_texts
    assert not {"2024 – 2026", "2020 – 2023"} & skill_texts


def test_section_detection_fresher_hr_resume_separates_skill_subsections():
    from app.pipeline.stages.normalization import TextNormalizer
    from app.pipeline.stages.reading_order import ReadingOrder
    from app.pipeline.stages.text_extraction import PDFExtractor

    blocks = TextNormalizer.normalize_blocks(
        ReadingOrder.reorder(PDFExtractor.extract(Path("tests/fixtures/fresher_hr_resume.pdf").read_bytes()))
    )
    sections = SectionDetector().detect(blocks)

    education_texts = {block.text.strip() for block in sections["EDUCATION"]}
    skill_texts = {block.text.strip() for block in sections["SKILLS"]}

    assert {"MBA (Human Resource Management) | School Of Open Learning , Delhi University (DU-SOL)", "2024 – 2026", "Bachelor of Arts (General) | BIR Tikendrajit University", "2020 – 2023"} <= education_texts
    assert {"Recruitment & Selection Basics", "Employee Engagement Concepts", "Onboarding Process Understanding", "HR Policy Awareness", "Training & Development Support", "Basic Labor Law Knowledge", "MS Excel (VLOOKUP, Pivot Tables, Filters)", "Google Sheets & Docs", "HRMS (Basic understanding)", "Email & Calendar Management"} <= skill_texts
    assert "Human Resource Skills" not in education_texts
    assert "Tools & Technical Skills" not in education_texts


def test_section_detection_resume_1_objective_education_certification_boundaries():
    from app.pipeline.stages.normalization import TextNormalizer
    from app.pipeline.stages.reading_order import ReadingOrder
    from app.pipeline.stages.text_extraction import PDFExtractor

    blocks = TextNormalizer.normalize_blocks(
        ReadingOrder.reorder(PDFExtractor.extract(Path("tests/fixtures/resume_1.pdf").read_bytes()))
    )
    sections = SectionDetector().detect(blocks)

    summary_text = "\n".join(block.text for block in sections["SUMMARY"])
    assert "RESUME OBJECTIVE" in summary_text
    assert "Administrative Assistant with 6+ years of experience organizing" in summary_text
    assert not any("AWARD RECEIVED" in block.text for block in sections["SUMMARY"])

    skill_texts = {block.text.strip() for block in sections["SKILLS"]}
    assert {"Problem Solving", "Adaptability", "Collaboration", "Strong Work Ethic", "Time Management", "Critical Thinking", "Handling Pressure"} <= skill_texts
    assert not any("EDUCATIO" in block.text or "DEGREE NAME" in block.text or "CERTIFICATION #1" in block.text for block in sections["SKILLS"])

    education_texts = {block.text.strip() for block in sections["EDUCATION"]}
    assert any("University" in text for text in education_texts)
    assert any("Location 2007" in text for text in education_texts)

    certification_texts = {block.text.strip() for block in sections["CERTIFICATIONS"]}
    assert any("CERTIFICATION #1" in text for text in certification_texts)
    assert any("University, Location" in text for text in certification_texts)

    achievement_texts = {block.text.strip() for block in sections["ACHIEVEMENTS"]}
    assert any("AWARD RECEIVED" in text for text in achievement_texts)
    assert not any("RESUME OBJECTIVE" in text for text in achievement_texts)


def test_section_detection_resume_7_parallel_skill_column():
    from pathlib import Path
    from app.pipeline.stages.text_extraction import PDFExtractor

    path = Path("tests/fixtures/resume_7.pdf")
    blocks = PDFExtractor.extract(path.read_bytes())

    detector = SectionDetector()
    sections = detector.detect(blocks)

    skill_texts = {block.text.strip() for block in sections["SKILLS"]}
    certification_texts = {
        block.text.strip()
        for block in sections["CERTIFICATIONS"]
    }

    assert {
        "Project management",
        "Strong decision maker",
        "Complex problem solver",
    } <= skill_texts

    assert {
        "Creative design",
        "Innovative",
        "Service-focused",
    } <= skill_texts

    assert not {
        "Creative design",
        "Innovative",
        "Service-focused",
    } & certification_texts
