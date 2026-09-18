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
    expected_skills: set[str] | None = None
    min_experience: int = 0
    exact_experience_count: int | None = None
    min_education: int = 0
    exact_education_count: int | None = None
    min_projects: int = 0
    exact_projects_count: int | None = None
    mandatory_entities: list[str] | None = None


@dataclass
class EvaluationResult:
    """Comprehensive evaluation separating hard correctness from completeness."""
    status: str  # "PASS", "PARTIAL", "VALIDATION_FAILED", "ERROR"
    hard_correctness_passed: bool
    hard_correctness_violations: list[str] = field(default_factory=list)
    entity_completeness_pct: float = 100.0
    entity_completeness_details: dict[str, Any] = field(default_factory=dict)
    field_completeness_pct: float = 100.0
    field_completeness_details: dict[str, Any] = field(default_factory=dict)
    skills_metrics: dict[str, Any] = field(default_factory=dict)
    discrepancies: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


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
        expected_skills={
            "Recruitment & Selection Basics",
            "Employee Engagement Concepts",
            "Onboarding Process Understanding",
            "HR Policy Awareness",
            "Training & Development Support",
            "Basic Labor Law Knowledge",
            "MS Excel (VLOOKUP, Pivot Tables, Filters)",
            "Google Sheets & Docs",
            "HRMS (Basic understanding)",
            "Email & Calendar Management",
            "Strong Verbal & Written Communication",
            "Team Collaboration",
            "Time Management",
            "Adaptability & Willingness to Learn",
        },
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


# Markers for objective hard correctness failures
SEVERE_ANOMALY_MARKERS = [
    "NAME_IS_FORM_OR_DOC_TITLE",
    "SECTION_HEADER_IN_LOCATION",
    "NAME_SPLIT_INTO_LOCATION",
    "EXTREME_EDUCATION_COUNT_SKEW",
    "EXCESSIVE_EXPERIENCE_COUNT",
    "TABLE_HEADER_IN_EXPERIENCE",
    "TABLE_METADATA_IN_EDUCATION",
    "PROJECTS_WITHOUT_NAMES",
]


def evaluate_resume_comprehensive(
    resume: Resume,
    filename: str,
    validation_violations: list[str] | None = None,
) -> EvaluationResult:
    """Evaluate a parsed resume with clear separation between Hard Correctness and Completeness."""
    violations = list(validation_violations or [])
    anomalies = detect_structural_anomalies(resume)
    golden = GOLDEN_EXPECTATIONS.get(filename)

    # 1. HARD CORRECTNESS
    hard_violations: list[str] = []

    # Any parser / pipeline validation violation fails hard correctness
    for v in violations:
        hard_violations.append(f"VALIDATION_VIOLATION: {v}")

    # Severe structural anomalies (form/doc title name, table header contamination, skew)
    for a in anomalies:
        if any(marker in a for marker in SEVERE_ANOMALY_MARKERS):
            hard_violations.append(f"ANOMALY_VIOLATION: {a}")

    # Check candidate name if golden expectation exists
    if golden is not None and golden.expected_name:
        extracted_name = (resume.personal.name or "").strip()
        expected_name = golden.expected_name.strip()
        if not extracted_name:
            hard_violations.append(f"MISSING_EXPECTED_NAME: expected {expected_name!r}")
        elif extracted_name.lower() != expected_name.lower():
            # If name is present but slightly different in casing/formatting, note as discrepancy
            pass

    hard_correctness_passed = (len(hard_violations) == 0)

    # 2. ENTITY COMPLETENESS
    # Entities: experience, education, projects, certifications, languages, achievements
    entity_counts = {
        "experience": len(resume.experience),
        "education": len(resume.education),
        "projects": len(resume.projects),
        "certifications": len(resume.certifications),
        "languages": len(resume.languages),
        "achievements": len(resume.achievements),
        "skills": len(resume.skills),
    }

    entity_scores: dict[str, float] = {}
    if golden is not None:
        if golden.exact_experience_count is not None:
            if golden.exact_experience_count == 0:
                entity_scores["experience"] = 1.0 if len(resume.experience) == 0 else 0.8
            else:
                entity_scores["experience"] = min(1.0, len(resume.experience) / golden.exact_experience_count)
        else:
            entity_scores["experience"] = 1.0 if len(resume.experience) > 0 else 0.5

        if golden.exact_education_count is not None:
            if golden.exact_education_count == 0:
                entity_scores["education"] = 1.0 if len(resume.education) == 0 else 0.8
            else:
                entity_scores["education"] = min(1.0, len(resume.education) / golden.exact_education_count)
        else:
            entity_scores["education"] = 1.0 if len(resume.education) > 0 else 0.5

        if golden.exact_projects_count is not None:
            if golden.exact_projects_count == 0:
                entity_scores["projects"] = 1.0
            else:
                entity_scores["projects"] = min(1.0, len(resume.projects) / golden.exact_projects_count)
    else:
        # Exploratory resumes: reward recovery of present entities
        core_entities = ["experience", "education", "skills"]
        for ent in core_entities:
            entity_scores[ent] = 1.0 if entity_counts[ent] > 0 else 0.5

    entity_completeness_pct = (
        round((sum(entity_scores.values()) / len(entity_scores)) * 100.0, 1)
        if entity_scores
        else 100.0
    )

    # 3. FIELD COMPLETENESS
    # Core fields: name, email, phone, location, summary, skills
    field_checks: dict[str, float] = {
        "name": 1.0 if (resume.personal.name or "").strip() else 0.0,
        "email": 1.0 if (resume.personal.email or "").strip() else 0.0,
        "phone": 1.0 if (resume.personal.phone or "").strip() else 0.0,
        "location": 1.0 if (resume.personal.location or "").strip() else 0.0,
        "skills": 1.0 if len(resume.skills) > 0 else 0.0,
    }

    if resume.experience:
        has_comp = sum(1 for e in resume.experience if (e.company or "").strip()) / len(resume.experience)
        has_desig = sum(1 for e in resume.experience if (e.designation or "").strip()) / len(resume.experience)
        has_dates = sum(1 for e in resume.experience if e.startDate or e.endDate) / len(resume.experience)
        field_checks["experience.company"] = round(has_comp, 2)
        field_checks["experience.designation"] = round(has_desig, 2)
        field_checks["experience.dates"] = round(has_dates, 2)

    if resume.education:
        has_inst = sum(1 for e in resume.education if (e.institution or "").strip()) / len(resume.education)
        has_deg = sum(1 for e in resume.education if (e.degree or "").strip()) / len(resume.education)
        field_checks["education.institution"] = round(has_inst, 2)
        field_checks["education.degree"] = round(has_deg, 2)

    if resume.projects:
        has_pname = sum(1 for p in resume.projects if (p.name or "").strip()) / len(resume.projects)
        field_checks["projects.name"] = round(has_pname, 2)

    field_completeness_pct = round((sum(field_checks.values()) / len(field_checks)) * 100.0, 1)

    # 4. SKILLS EVALUATION
    extracted_skills_count = len(resume.skills)
    expected_skills_count = golden.exact_skills_count if golden else None
    grounded_status = not any("skills" in v.lower() for v in violations)

    if golden is not None and golden.expected_skills:
        actual_norm = {s.strip().lower() for s in resume.skills}
        matched_skills = [s for s in golden.expected_skills if s.strip().lower() in actual_norm]
        skills_recall = round((len(matched_skills) / len(golden.expected_skills)) * 100.0, 1)
    elif expected_skills_count is not None and expected_skills_count > 0:
        skills_recall = round(min(1.0, extracted_skills_count / expected_skills_count) * 100.0, 1)
    else:
        skills_recall = 100.0 if extracted_skills_count > 0 else 0.0

    skills_metrics = {
        "extracted_count": extracted_skills_count,
        "expected_count": expected_skills_count,
        "recall_pct": skills_recall,
        "grounded_status": grounded_status,
    }

    # 5. DISCREPANCIES (Completeness / Exact Reference Comparisons)
    discrepancies: list[str] = []
    if golden is not None:
        if golden.expected_name and (resume.personal.name or "").strip().lower() != golden.expected_name.strip().lower():
            discrepancies.append(f"name mismatch: got {resume.personal.name!r}, expected {golden.expected_name!r}")
        if golden.expected_email and (resume.personal.email or "").strip().lower() != golden.expected_email.strip().lower():
            discrepancies.append(f"email mismatch: got {resume.personal.email!r}, expected {golden.expected_email!r}")
        if golden.exact_experience_count is not None and len(resume.experience) != golden.exact_experience_count:
            discrepancies.append(f"exp count mismatch: got {len(resume.experience)}, expected {golden.exact_experience_count}")
        if golden.exact_education_count is not None and len(resume.education) != golden.exact_education_count:
            discrepancies.append(f"edu count mismatch: got {len(resume.education)}, expected {golden.exact_education_count}")
        if golden.exact_skills_count is not None and len(resume.skills) != golden.exact_skills_count:
            discrepancies.append(f"skills count mismatch: got {len(resume.skills)}, expected {golden.exact_skills_count} (recall: {skills_recall}%)")
        if golden.expected_skills is not None:
            actual_skills_normalized = {s.strip().lower() for s in resume.skills}
            missing = [s for s in sorted(golden.expected_skills) if s.strip().lower() not in actual_skills_normalized]
            if missing:
                discrepancies.append(f"missing expected skills: {missing}")

    # Non-severe anomalies (e.g. minor formatting flags)
    non_severe_anomalies = [a for a in anomalies if not any(m in a for m in SEVERE_ANOMALY_MARKERS)]

    # 6. QUALITY STATUS SEMANTICS
    if not hard_correctness_passed:
        status = "VALIDATION_FAILED"
    elif golden is not None:
        # Hard correctness passed. Check whether completeness is partial or full.
        has_completeness_gap = (
            bool(discrepancies)
            or bool(non_severe_anomalies)
            or (skills_recall < 100.0 and expected_skills_count is not None)
            or (entity_completeness_pct < 85.0)
        )
        if has_completeness_gap:
            status = "PARTIAL"
        else:
            status = "PASS"
    else:
        # For exploratory resumes without locked golden ground truth
        if non_severe_anomalies:
            status = "PARTIAL"
        else:
            status = "PASS"

    return EvaluationResult(
        status=status,
        hard_correctness_passed=hard_correctness_passed,
        hard_correctness_violations=hard_violations,
        entity_completeness_pct=entity_completeness_pct,
        entity_completeness_details=entity_scores,
        field_completeness_pct=field_completeness_pct,
        field_completeness_details=field_checks,
        skills_metrics=skills_metrics,
        discrepancies=discrepancies,
        notes=non_severe_anomalies,
    )


def evaluate_status(
    resume: Resume,
    filename: str,
    validation_violations: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Determine PASS, PARTIAL, or VALIDATION_FAILED based on objective criteria.

    Backward compatible helper returning (status, diagnostics).
    """
    res = evaluate_resume_comprehensive(resume, filename, validation_violations)
    diagnostics = res.hard_correctness_violations + res.discrepancies + res.notes
    return res.status, diagnostics

