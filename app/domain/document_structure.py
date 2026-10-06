"""Generic Document Structure v1 domain models and projection helpers."""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.domain.resume import (
    CertificationItem,
    EducationItem,
    ExperienceItem,
    PersonalInfo,
    ProjectItem,
    Resume,
)


class DocumentBlock(BaseModel):
    """Generic atomic content block within a document section."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    type: str = "paragraph"  # paragraph, list_item, table, heading, text
    text: str = ""
    source_block_ids: list[str] = Field(default_factory=list)
    page_number: int | None = None
    reading_order: int | None = None
    table_data: dict[str, Any] | None = None  # headers, rows, caption
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    """Generic hierarchical section in a document."""

    model_config = ConfigDict(extra="ignore")

    heading: str | None = None  # Exact source heading, or None for unheaded/pre-heading content
    level: int = 1
    blocks: list[DocumentBlock] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)
    subsections: list[DocumentSection] = Field(default_factory=list)


class DocumentStructure(BaseModel):
    """Root model for generic document reconstruction."""

    model_config = ConfigDict(extra="ignore")

    sections: list[DocumentSection] = Field(default_factory=list)
    page_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


def document_structure_to_resume(doc: DocumentStructure) -> Resume:
    """Project generic document structure to legacy Resume schema for backward compatibility."""
    personal = PersonalInfo()
    summary: str | None = None
    skills: list[str] = []
    experience: list[ExperienceItem] = []
    education: list[EducationItem] = []
    projects: list[ProjectItem] = []
    certifications: list[CertificationItem] = []
    achievements: list[str] = []
    languages: list[str] = []

    email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    phone_re = re.compile(r"(\+?\d{1,4}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b")
    linkedin_re = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+", re.I)
    github_re = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+", re.I)

    # Flatten sections recursively for scanning
    def _collect_sections(sections: list[DocumentSection]) -> list[DocumentSection]:
        collected = []
        for s in sections:
            collected.append(s)
            if s.subsections:
                collected.extend(_collect_sections(s.subsections))
        return collected

    all_sections = _collect_sections(doc.sections)

    for section in all_sections:
        h_lower = (section.heading or "").strip().lower()

        # Unheaded or contact-related section
        if section.heading is None or any(k in h_lower for k in ("contact", "personal", "profile", "details", "info")):
            for b in section.blocks:
                txt = b.text.strip()
                if not txt:
                    continue
                if not personal.email:
                    m = email_re.search(txt)
                    if m:
                        personal.email = m.group(0)
                if not personal.phone:
                    m = phone_re.search(txt)
                    if m and len(re.sub(r"\D", "", m.group(0))) >= 7:
                        personal.phone = m.group(0).strip()
                if not personal.linkedin:
                    m = linkedin_re.search(txt)
                    if m:
                        personal.linkedin = m.group(0)
                if not personal.github:
                    m = github_re.search(txt)
                    if m:
                        personal.github = m.group(0)
                if not personal.name and not email_re.search(txt) and not phone_re.search(txt):
                    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                    if lines and len(lines[0].split()) <= 4:
                        personal.name = lines[0]

        # Summary
        if any(k in h_lower for k in ("summary", "about", "objective", "biography", "overview")):
            texts = [b.text.strip() for b in section.blocks if b.text.strip()]
            if texts and not summary:
                summary = " ".join(texts)

        # Skills
        elif any(k in h_lower for k in ("skill", "competenc", "technolog", "proficienc", "expertise")):
            for b in section.blocks:
                txt = b.text.strip()
                if not txt:
                    continue
                if b.type == "list_item" or "\n" in txt or "," in txt:
                    parts = [p.strip(" •-\t\r\n*") for p in re.split(r"[,;\n•]+", txt) if p.strip(" •-\t\r\n*")]
                    for p in parts:
                        if p and p not in skills:
                            skills.append(p)
                else:
                    if txt not in skills:
                        skills.append(txt)

        # Experience / Employment / Career History
        elif any(k in h_lower for k in ("experience", "employment", "career", "work history", "job")):
            for b in section.blocks:
                if b.table_data and "rows" in b.table_data:
                    for row in b.table_data["rows"]:
                        if row:
                            experience.append(ExperienceItem(
                                company=row[0] if len(row) > 0 else None,
                                designation=row[1] if len(row) > 1 else None,
                                description=" ".join(str(c) for c in row[2:]) if len(row) > 2 else None,
                            ))
                elif b.text.strip():
                    experience.append(ExperienceItem(description=b.text.strip()))

        # Education
        elif any(k in h_lower for k in ("education", "academic", "qualification", "degree", "university", "college")):
            for b in section.blocks:
                if b.table_data and "rows" in b.table_data:
                    for row in b.table_data["rows"]:
                        if row:
                            education.append(EducationItem(
                                institution=row[0] if len(row) > 0 else None,
                                degree=row[1] if len(row) > 1 else None,
                            ))
                elif b.text.strip():
                    education.append(EducationItem(institution=b.text.strip()))

        # Projects
        elif "project" in h_lower:
            for b in section.blocks:
                if b.text.strip():
                    projects.append(ProjectItem(name=b.text.strip()[:60], description=b.text.strip()))

        # Certifications / Trainings
        elif any(k in h_lower for k in ("certif", "license", "course", "training", "accredit")):
            for b in section.blocks:
                if b.text.strip():
                    certifications.append(CertificationItem(name=b.text.strip()))

        # Languages
        elif "language" in h_lower:
            for b in section.blocks:
                txt = b.text.strip()
                if txt:
                    parts = [p.strip(" •-\t\r\n*") for p in re.split(r"[,;\n•]+", txt) if p.strip(" •-\t\r\n*")]
                    for p in parts:
                        if p and p not in languages:
                            languages.append(p)

        # Achievements / Honors / Awards
        elif any(k in h_lower for k in ("achievement", "award", "honor", "publication")):
            for b in section.blocks:
                if b.text.strip() and b.text.strip() not in achievements:
                    achievements.append(b.text.strip())

    return Resume(
        schemaVersion="1.0",
        parserVersion="1.0.0",
        personal=personal,
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
        certifications=certifications,
        achievements=achievements,
        languages=languages,
        metadata={"pageCount": doc.page_count},
    )
