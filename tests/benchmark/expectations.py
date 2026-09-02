"""Objective expectations and structural anomaly detection for resume parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.domain.resume import Resume

# Known document titles / form labels that should NEVER be extracted as candidate names
INVALID_NAME_PATTERNS = [
    re.compile(r"^application\s+form\b", re.I),
    re.compile(r"^curriculum\s+vitae\b", re.I),
    re.compile(r"^seafarer\s+profile\b", re.I),
    re.compile(r"^surname\b", re.I),
    re.compile(r"^resume\b", re.I),
    re.compile(r"^biodata\b", re.I),
]

# Section headers or form labels that falsely leak into personal.location
LEAKED_LOCATION_PATTERNS = [
    re.compile(r"^qualification\b", re.I),
    re.compile(r"^references?\b", re.I),
    re.compile(r"^position\b", re.I),
    re.compile(r"^discipline\b", re.I),
    re.compile(r"^declaration\b", re.I),
]

# Table header strings that contaminate Experience records
EXPERIENCE_TABLE_HEADER_PATTERNS = [
    re.compile(r"\bship\s+name\b", re.I),
    re.compile(r"\bs\.?\s*no\.?\b", re.I),
    re.compile(r"\bvessel\s+(?:name|type)\b", re.I),
    re.compile(r"\bdocuments?\s+details?\b", re.I),
    re.compile(r"\bperiod\b", re.I),
    re.compile(r"\bpoi\b", re.I),
    re.compile(r"\bdoi\b", re.I),
]

# Table header strings / vessel names that contaminate Education records
EDUCATION_TABLE_HEADER_PATTERNS = [
    re.compile(r"\bsea\s+service\b", re.I),
    re.compile(r"\bdocuments?\b", re.I),
    re.compile(r"\bendorsements?\b", re.I),
    re.compile(r"\breadiness\s+date\b", re.I),
    re.compile(r"\bcovid\b", re.I),
    re.compile(r"\bithacki\b", re.I),
    re.compile(r"\bdeclaration\b", re.I),
    re.compile(r"\bvaccinations?\b", re.I),
]


@dataclass
class GoldenExpectations:
    expected_name: str | None = None
    expected_email: str | None = None
    expected_phone: str | None = None
    expected_location: str | None = None
    min_skills: int = 0
    max_skills: int | None = None
    exact_skills_count: int | None = None
    min_experience: int = 0
    exact_experience_count: int | None = None
    min_education: int = 0
    exact_education_count: int | None = None
    min_projects: int = 0
    exact_projects_count: int | None = None


# Established golden expectations for verified ground-truth resumes
GOLDEN_EXPECTATIONS: dict[str, GoldenExpectations] = {
    "AditCV_SOL.pdf": GoldenExpectations(
        expected_name="ADITI ANAND",
        expected_email="aditianand136@gmail.com",
        expected_phone="+91 9241242699",
        expected_location=None,
        exact_skills_count=19,
        exact_experience_count=1,
        exact_education_count=1,
        exact_projects_count=0,
    ),
    "fresher_hr_resume.pdf": GoldenExpectations(
        expected_name="Aditi Anand",
        expected_email="aditianand136@gmail.com",
        expected_phone="9241242699",
        expected_location=None,
        exact_skills_count=16,
        exact_experience_count=0,
        exact_education_count=2,
        exact_projects_count=0,
    ),
    "swe_experienced_resume.pdf": GoldenExpectations(
        expected_name="ANKIT JHA",
        expected_email="ankitjha6035@gmail.com",
        expected_phone="9570716035",
        expected_location="Noida, U.P, India",
        exact_skills_count=39,
        exact_experience_count=2,
        exact_education_count=2,
        exact_projects_count=3,
    ),
    "Résume_Shubham.pdf": GoldenExpectations(
        expected_name="SHUBHAM BAJAJ",
        expected_email="shubhambajaj204@gmail.com",
        expected_phone="6354511915",
        expected_location=None,
        exact_skills_count=13,
        exact_experience_count=3,
        exact_education_count=1,
        exact_projects_count=2,
    ),
}


def detect_structural_anomalies(resume: Resume) -> list[str]:
    """Detect objective structural anomalies and failure signatures in a parsed resume."""
    anomalies: list[str] = []

    # 1. Personal Name Anomaly
    name = (resume.personal.name or "").strip()
    for pat in INVALID_NAME_PATTERNS:
        if pat.search(name):
            anomalies.append(f"NAME_IS_FORM_OR_DOC_TITLE: {name!r}")
            break

    # 2. Location Leakage Anomaly
    loc = (resume.personal.location or "").strip()
    for pat in LEAKED_LOCATION_PATTERNS:
        if pat.search(loc):
            anomalies.append(f"SECTION_HEADER_IN_LOCATION: {loc!r}")
            break

    # Check for name-split artifact into location (e.g. MAYUR / AGARWAL)
    if name and loc and loc.upper() in ["AGARWAL", "ALAM"]:
        anomalies.append(f"NAME_SPLIT_INTO_LOCATION: {loc!r}")

    # 3. Table Headers Contaminating Experience
    for i, exp in enumerate(resume.experience):
        comp = (exp.company or "").strip()
        desig = (exp.designation or "").strip()
        for pat in EXPERIENCE_TABLE_HEADER_PATTERNS:
            if pat.search(comp) or pat.search(desig):
                anomalies.append(f"TABLE_HEADER_IN_EXPERIENCE[{i}]: comp={comp!r}, desig={desig!r}")
                break

    # 4. Table Metadata Contaminating Education
    for i, edu in enumerate(resume.education):
        inst = (edu.institution or "").strip()
        deg = (edu.degree or "").strip()
        for pat in EDUCATION_TABLE_HEADER_PATTERNS:
            if pat.search(inst) or pat.search(deg):
                anomalies.append(f"TABLE_METADATA_IN_EDUCATION[{i}]: inst={inst!r}, deg={deg!r}")
                break

    # 5. Extreme Education Count Skew (indicating table cell row explosion)
    if len(resume.education) >= 8:
        anomalies.append(f"EXTREME_EDUCATION_COUNT_SKEW: {len(resume.education)} entries")

    # 6. Massive Experience Contamination (e.g. references parsed as jobs)
    if len(resume.experience) >= 12:
        anomalies.append(f"EXCESSIVE_EXPERIENCE_COUNT: {len(resume.experience)} entries")

    # 7. Projects with Missing Titles (grouping/header association failures)
    nameless_projects = [i for i, prj in enumerate(resume.projects) if not prj.name]
    if nameless_projects:
        anomalies.append(f"PROJECTS_WITHOUT_NAMES: {len(nameless_projects)} items at indices {nameless_projects}")

    return anomalies


def evaluate_status(resume: Resume, filename: str) -> tuple[str, list[str]]:
    """Determine PASS, PARTIAL, or FAIL based on objective criteria."""
    anomalies = detect_structural_anomalies(resume)
    golden = GOLDEN_EXPECTATIONS.get(filename)

    if golden is not None:
        # Check against locked golden expectations
        discrepancies: list[str] = []
        if golden.expected_name and resume.personal.name != golden.expected_name:
            discrepancies.append(f"name mismatch: got {resume.personal.name!r}, expected {golden.expected_name!r}")
        if golden.expected_email and resume.personal.email != golden.expected_email:
            discrepancies.append(f"email mismatch: got {resume.personal.email!r}, expected {golden.expected_email!r}")
        if golden.exact_experience_count is not None and len(resume.experience) != golden.exact_experience_count:
            discrepancies.append(f"exp count mismatch: got {len(resume.experience)}, expected {golden.exact_experience_count}")
        if golden.exact_education_count is not None and len(resume.education) != golden.exact_education_count:
            discrepancies.append(f"edu count mismatch: got {len(resume.education)}, expected {golden.exact_education_count}")
        if golden.exact_skills_count is not None and len(resume.skills) != golden.exact_skills_count:
            discrepancies.append(f"skills count mismatch: got {len(resume.skills)}, expected {golden.exact_skills_count}")

        if discrepancies:
            return "FAIL", discrepancies + anomalies
        if anomalies:
            return "PARTIAL", anomalies
        return "PASS", []

    # For exploratory resumes without locked golden ground truth
    # FAIL if severe structural failures (document title as name, table contamination, extreme skew)
    severe_markers = [
        "NAME_IS_FORM_OR_DOC_TITLE",
        "SECTION_HEADER_IN_LOCATION",
        "NAME_SPLIT_INTO_LOCATION",
        "EXTREME_EDUCATION_COUNT_SKEW",
        "EXCESSIVE_EXPERIENCE_COUNT",
        "TABLE_HEADER_IN_EXPERIENCE",
    ]
    has_severe = any(any(m in a for m in severe_markers) for a in anomalies)

    if has_severe:
        return "FAIL", anomalies
    elif anomalies:
        return "PARTIAL", anomalies
    else:
        return "PASS", []
