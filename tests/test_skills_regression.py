from pathlib import Path
from app.pipeline.parser import ResumeParser


def test_fixture_global_skills_include_expected():
    p = Path('tests/fixtures/swe_experienced_resume.pdf')
    pdf_bytes = p.read_bytes()

    parser = ResumeParser()
    resume = parser.parse(pdf_bytes)

    skills = set(resume.skills)

    expected_subset = {
        'Apache Kafka',
        'Weblogic JMS',
        'API Composition',
        'Config Server',
        'Apache Ignite',
        'OAuth2',
        'ELK Stack',
        'SLF4J',
        'MongoDB',
        'MySQL',
        'Spring Boot'
    }

    assert expected_subset.issubset(skills)


def test_fixture_project_technologies_extracted():
    p = Path('tests/fixtures/swe_experienced_resume.pdf')
    pdf_bytes = p.read_bytes()

    parser = ResumeParser()
    resume = parser.parse(pdf_bytes)

    # Find a project that mentions Kafka and some form of Weblogic/JMS in its description
    found = False
    for proj in resume.projects:
        techs = set(proj.technologies or [])
        if 'Apache Kafka' in techs and ('Weblogic JMS' in techs or 'Weblogic' in techs):
            found = True
            break

    assert found, 'Expected a project with Apache Kafka and Weblogic-related technologies'


def test_fixture_experience_skills_extracted():
    p = Path('tests/fixtures/swe_experienced_resume.pdf')
    pdf_bytes = p.read_bytes()

    parser = ResumeParser()
    resume = parser.parse(pdf_bytes)

    # Check that at least one experience entry mentions Kafka and Apache Ignite
    exp_found = False
    for exp in resume.experience:
        s = set(exp.skills or [])
        if {'Apache Kafka', 'Apache Ignite'}.issubset(s):
            exp_found = True
            break

    assert exp_found, 'Expected experience entry to include Apache Kafka and Apache Ignite'
