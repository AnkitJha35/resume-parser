"""Resume export service providing deterministic formatting for JSON, text, and CSV."""

from __future__ import annotations

import csv
import io
import json
from typing import Any


def sanitize_filename(resume_id: str, extension: str) -> str:
    """Produce a safe ASCII filename for Content-Disposition headers."""
    safe_id = "".join(c for c in resume_id if c.isalnum() or c in ("-", "_"))
    if not safe_id:
        safe_id = "export"
    ext_clean = extension.lstrip(".")
    return f"resume_{safe_id}.{ext_clean}"


def export_as_json(resume_data: dict[str, Any]) -> str:
    """Return canonical persisted Resume JSON with deterministic formatting."""
    return json.dumps(resume_data, indent=2, ensure_ascii=False)


def export_as_text(resume_data: dict[str, Any]) -> str:
    """Produce a deterministic human-readable text representation of the canonical Resume."""
    sections: list[str] = []

    # 1. Name & Contact
    personal = resume_data.get("personal") or {}
    contact_lines: list[str] = []
    name = personal.get("name")
    if name and str(name).strip():
        contact_lines.append(str(name).strip())

    details: list[str] = []
    for k in ("email", "phone", "location"):
        val = personal.get(k)
        if val and str(val).strip():
            details.append(str(val).strip())
    if details:
        contact_lines.append(" | ".join(details))

    links: list[str] = []
    for k in ("linkedin", "github", "portfolio"):
        val = personal.get(k)
        if val and str(val).strip():
            links.append(str(val).strip())
    if links:
        contact_lines.append(" | ".join(links))

    if contact_lines:
        sections.append("\n".join(contact_lines))

    # 2. Summary
    summary = resume_data.get("summary")
    if summary and str(summary).strip():
        sections.append(f"SUMMARY\n-------\n{str(summary).strip()}")

    # 3. Skills
    skills = resume_data.get("skills") or []
    if skills:
        cleaned_skills = [str(s).strip() for s in skills if s and str(s).strip()]
        if cleaned_skills:
            sections.append(f"SKILLS\n------\n" + ", ".join(cleaned_skills))

    # 4. Experience
    experience = resume_data.get("experience") or []
    if experience:
        exp_items: list[str] = []
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            item_lines: list[str] = []
            title_parts = [p for p in (exp.get("designation"), exp.get("company")) if p and str(p).strip()]
            header = " at ".join(str(p).strip() for p in title_parts) if title_parts else "Experience"

            date_parts: list[str] = []
            if exp.get("startDate"):
                date_parts.append(str(exp["startDate"]).strip())
            if exp.get("current"):
                date_parts.append("Present")
            elif exp.get("endDate"):
                date_parts.append(str(exp["endDate"]).strip())
            date_str = " - ".join(date_parts) if date_parts else ""

            loc = exp.get("location")
            loc_str = str(loc).strip() if loc and str(loc).strip() else ""

            meta_line_parts = [p for p in (header, date_str, loc_str) if p]
            item_lines.append(" | ".join(meta_line_parts))

            desc = exp.get("description")
            if desc and str(desc).strip():
                item_lines.append(str(desc).strip())

            techs = exp.get("technologies") or exp.get("skills") or []
            if techs:
                cleaned_techs = [str(t).strip() for t in techs if t and str(t).strip()]
                if cleaned_techs:
                    item_lines.append("Technologies: " + ", ".join(cleaned_techs))

            exp_items.append("\n".join(item_lines))
        if exp_items:
            sections.append("EXPERIENCE\n----------\n" + "\n\n".join(exp_items))

    # 5. Education
    education = resume_data.get("education") or []
    if education:
        edu_items: list[str] = []
        for edu in education:
            if not isinstance(edu, dict):
                continue
            item_lines = []
            deg = edu.get("degree")
            field = edu.get("fieldOfStudy")
            deg_field = f"{deg} in {field}" if deg and field else (deg or field or "Degree")
            inst = edu.get("institution")
            inst_str = str(inst).strip() if inst and str(inst).strip() else ""
            header = f"{deg_field}, {inst_str}" if inst_str else str(deg_field)

            date_parts = []
            if edu.get("startDate"):
                date_parts.append(str(edu["startDate"]).strip())
            if edu.get("endDate"):
                date_parts.append(str(edu["endDate"]).strip())
            date_str = " - ".join(date_parts) if date_parts else ""

            grade = edu.get("grade")
            grade_str = f"Grade: {grade}" if grade and str(grade).strip() else ""

            meta_parts = [p for p in (header, date_str, grade_str) if p]
            item_lines.append(" | ".join(meta_parts))
            edu_items.append("\n".join(item_lines))
        if edu_items:
            sections.append("EDUCATION\n---------\n" + "\n\n".join(edu_items))

    # 6. Projects
    projects = resume_data.get("projects") or []
    if projects:
        prj_items: list[str] = []
        for prj in projects:
            if not isinstance(prj, dict):
                continue
            item_lines = []
            name = prj.get("name") or "Project"
            date_parts = []
            if prj.get("startDate"):
                date_parts.append(str(prj["startDate"]).strip())
            if prj.get("current"):
                date_parts.append("Present")
            elif prj.get("endDate"):
                date_parts.append(str(prj["endDate"]).strip())
            date_str = " - ".join(date_parts) if date_parts else ""

            url = prj.get("url")
            url_str = str(url).strip() if url and str(url).strip() else ""

            header_parts = [p for p in (str(name).strip(), date_str, url_str) if p]
            item_lines.append(" | ".join(header_parts))

            desc = prj.get("description")
            if desc and str(desc).strip():
                item_lines.append(str(desc).strip())

            techs = prj.get("technologies") or []
            if techs:
                cleaned_techs = [str(t).strip() for t in techs if t and str(t).strip()]
                if cleaned_techs:
                    item_lines.append("Technologies: " + ", ".join(cleaned_techs))
            prj_items.append("\n".join(item_lines))
        if prj_items:
            sections.append("PROJECTS\n--------\n" + "\n\n".join(prj_items))

    # 7. Certifications
    certifications = resume_data.get("certifications") or []
    if certifications:
        cert_blocks: list[str] = ["CERTIFICATIONS\n--------------"]
        for cert in certifications:
            if isinstance(cert, dict):
                c_name = cert.get("name") or "Certification"
                issuer = cert.get("issuingOrganization")
                header = f"{c_name} - {issuer}" if issuer else str(c_name)
                parts = [header]
                if cert.get("issueDate"):
                    parts.append(f"Issued: {cert['issueDate']}")
                if cert.get("expiryDate"):
                    parts.append(f"Expires: {cert['expiryDate']}")
                if cert.get("credentialId"):
                    parts.append(f"ID: {cert['credentialId']}")
                if cert.get("credentialUrl"):
                    parts.append(f"URL: {cert['credentialUrl']}")
                cert_blocks.append(" | ".join(parts))
            elif isinstance(cert, str) and cert.strip():
                cert_blocks.append(cert.strip())
        sections.append("\n".join(cert_blocks))

    # 8. Achievements
    achievements = resume_data.get("achievements") or []
    if achievements:
        cleaned_ach = [f"- {str(a).strip()}" for a in achievements if a and str(a).strip()]
        if cleaned_ach:
            sections.append("ACHIEVEMENTS\n------------\n" + "\n".join(cleaned_ach))

    # 9. Languages
    languages = resume_data.get("languages") or []
    if languages:
        cleaned_lang = [str(l).strip() for l in languages if l and str(l).strip()]
        if cleaned_lang:
            sections.append("LANGUAGES\n---------\n" + ", ".join(cleaned_lang))

    return "\n\n".join(sections) + "\n"


def export_as_csv(resume_id: str, resume_data: dict[str, Any]) -> str:
    """Flatten canonical Resume into a deterministic 5-column RFC 4180 CSV document.

    Columns:
        resume_id: str
        section: str
        item_index: int (preserves collection ordering)
        field: str
        value: str
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\r\n")

    # Header
    writer.writerow(["resume_id", "section", "item_index", "field", "value"])

    def _write_field(section: str, item_index: int, field: str, value: Any) -> None:
        if value is None:
            return
        val_str = str(value).strip()
        if not val_str:
            return
        writer.writerow([resume_id, section, item_index, field, val_str])

    # 1. Personal
    personal = resume_data.get("personal") or {}
    for f in ("name", "email", "phone", "location", "linkedin", "github", "portfolio"):
        _write_field("personal", 0, f, personal.get(f))

    # 2. Summary
    _write_field("summary", 0, "summary", resume_data.get("summary"))

    # 3. Skills
    skills = resume_data.get("skills") or []
    for idx, sk in enumerate(skills):
        _write_field("skills", idx, "skill", sk)

    # 4. Experience
    experience = resume_data.get("experience") or []
    for idx, exp in enumerate(experience):
        if not isinstance(exp, dict):
            continue
        for f in ("company", "designation", "location", "startDate", "endDate", "current", "description"):
            _write_field("experience", idx, f, exp.get(f))
        techs = exp.get("technologies") or exp.get("skills") or []
        for t_idx, tech in enumerate(techs):
            _write_field("experience", idx, f"technology[{t_idx}]", tech)

    # 5. Education
    education = resume_data.get("education") or []
    for idx, edu in enumerate(education):
        if not isinstance(edu, dict):
            continue
        for f in ("institution", "degree", "fieldOfStudy", "startDate", "endDate", "grade"):
            _write_field("education", idx, f, edu.get(f))

    # 6. Projects
    projects = resume_data.get("projects") or []
    for idx, prj in enumerate(projects):
        if not isinstance(prj, dict):
            continue
        for f in ("name", "description", "startDate", "endDate", "current", "url"):
            _write_field("projects", idx, f, prj.get(f))
        techs = prj.get("technologies") or []
        for t_idx, tech in enumerate(techs):
            _write_field("projects", idx, f"technology[{t_idx}]", tech)

    # 7. Certifications
    certifications = resume_data.get("certifications") or []
    for idx, cert in enumerate(certifications):
        if isinstance(cert, dict):
            for f in ("name", "issuingOrganization", "issueDate", "expiryDate", "credentialId", "credentialUrl"):
                _write_field("certifications", idx, f, cert.get(f))
        elif isinstance(cert, str):
            _write_field("certifications", idx, "name", cert)

    # 8. Achievements
    achievements = resume_data.get("achievements") or []
    for idx, ach in enumerate(achievements):
        _write_field("achievements", idx, "achievement", ach)

    # 9. Languages
    languages = resume_data.get("languages") or []
    for idx, lang in enumerate(languages):
        _write_field("languages", idx, "language", lang)

    return output.getvalue()
