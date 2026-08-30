from typing import Any

from pydantic import BaseModel, ValidationError


class PersonalInfo(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    portfolio: str | None = None


class ExperienceItem(BaseModel):
    company: str | None = None
    designation: str | None = None
    location: str | None = None
    startDate: str | None = None
    endDate: str | None = None
    current: bool | None = None
    description: str | None = None
    skills: list[str] | None = None
    confidence: float | None = None


class EducationItem(BaseModel):
    institution: str | None = None
    degree: str | None = None
    fieldOfStudy: str | None = None
    startDate: str | None = None
    endDate: str | None = None
    grade: str | None = None
    confidence: float | None = None


class ProjectItem(BaseModel):
    name: str | None = None
    description: str | None = None
    technologies: list[str] | None = None
    startDate: str | None = None
    endDate: str | None = None
    current: bool | None = None
    url: str | None = None
    confidence: float | None = None


class CertificationItem(BaseModel):
    name: str | None = None
    issuingOrganization: str | None = None
    issueDate: str | None = None
    expiryDate: str | None = None
    credentialId: str | None = None
    credentialUrl: str | None = None
    confidence: float | None = None


class Resume(BaseModel):
    schemaVersion: str = "1.0"
    parserVersion: str
    personal: PersonalInfo
    summary: str | None = None
    skills: list[str] = []
    experience: list[ExperienceItem] = []
    education: list[EducationItem] = []
    projects: list[ProjectItem] = []
    certifications: list[CertificationItem] = []
    achievements: list[str] = []
    languages: list[str] = []
    metadata: dict[str, Any] = {}


def _extract_field(field: Any) -> str | None:
    if isinstance(field, dict):
        return field.get("value")
    return field


def build_personal_info(personal_data: dict[str, Any]) -> PersonalInfo:
    return PersonalInfo(
        name=_extract_field(personal_data.get("name")),
        email=_extract_field(personal_data.get("email")),
        phone=_extract_field(personal_data.get("phone")),
        location=_extract_field(personal_data.get("location")),
        linkedin=_extract_field(personal_data.get("linkedin")),
        github=_extract_field(personal_data.get("github")),
        portfolio=_extract_field(personal_data.get("portfolio")),
    )


def validate_resume(partial_result: dict[str, Any]) -> Resume:
    return Resume(
        parserVersion=partial_result["parserVersion"],
        personal=build_personal_info(partial_result["personal"]),
        summary=partial_result.get("summary"),
        skills=partial_result.get("skills", []),
        experience=partial_result.get("experience", []),
        education=partial_result.get("education", []),
        projects=partial_result.get("projects", []),
        certifications=partial_result.get("certifications", []),
        achievements=partial_result.get("achievements", []),
        languages=partial_result.get("languages", []),
        metadata=partial_result.get("metadata", {}),
    )
