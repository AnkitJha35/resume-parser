from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from app.extractors.date_parser import DateRangeParser
from app.pipeline.stages.semantic_resources import SemanticResources


SEMANTIC_SECTIONS = (
    "SUMMARY",
    "SKILLS",
    "ACHIEVEMENTS",
    "EXPERIENCE",
    "EDUCATION",
    "CERTIFICATIONS",
    "PROJECTS",
    "LANGUAGES",
)
MIN_CONFIDENCE = 0.70
MIN_MARGIN = 0.15


@dataclass(frozen=True)
class SectionInference:
    section: str
    confidence: float
    scores: dict[str, float]
    signals: dict[str, tuple[str, ...]]


@dataclass
class UnknownSectionCandidate:
    heading: object
    content: list[object] = field(default_factory=list)
    inference: SectionInference | None = None


def is_unknown_heading(text: str, font_size: float | None, bold: bool | None) -> bool:
    value = (text or "").strip()
    if not value or len(value.split()) > 5 or any(char.isdigit() for char in value):
        return False
    if value.lower() in {alias.lower() for alias in SemanticResources.skills()}:
        return False
    if value.endswith((".", ":", ";", ",")):
        return False
    words = value.split()
    upper_like = value.upper() == value and any(char.isalpha() for char in value)
    return bool(bold or upper_like or (font_size is not None and font_size >= 12))


def infer_section(heading: str, content: Iterable[str]) -> SectionInference:
    lines = [line.strip() for line in content if line and line.strip()]
    heading_tokens = set(re.findall(r"[a-z]+", (heading or "").lower()))
    scores = {section: 0.0 for section in SEMANTIC_SECTIONS}
    signals = {section: [] for section in SEMANTIC_SECTIONS}
    skill_vocab = SemanticResources.skills()
    skill_aliases = tuple(skill_vocab)
    has_language_content = any(
        re.search(r"\b(native|fluent|proficiency|spoken|written|language)\b", line, re.IGNORECASE)
        or re.search(r"\b(english|spanish|french|german|hindi|mandarin)\b", line, re.IGNORECASE)
        for line in lines
    )
    has_academic_content = any(
        re.search(r"\b(bachelor|master|associate|phd|degree|university|college|school|institute|graduat)\b", line, re.IGNORECASE)
        for line in lines
    )
    has_achievement_content = any(
        re.search(r"\b(award|awarded|recognition|recognized|employee\s+of|honor|winner)\b", line, re.IGNORECASE)
        or re.search(r"\b\d+(?:\.\d+)?%\b|\$\s?\d+(?:\.\d+)?[km]?\b|\b\d+(?:\.\d+)?x\b", line, re.IGNORECASE)
        or re.search(r"\b(?:increased|reduced|improved|saved|generated|achieved|accomplished)\b.+\b(?:by|to|from|through|resulting|leading)\b", line, re.IGNORECASE)
        for line in lines
    )
    has_employment_structure = any(
        DateRangeParser.parse(line) is not None
        or re.search(r"\b(inc|llc|ltd|corp|company|co\.)\b", line, re.IGNORECASE)
        for line in lines
    )

    for line in lines:
        lowered = line.lower()
        words = set(re.findall(r"[a-z]+", lowered))
        bullet = bool(re.match(r"^[\u2022\u2023\u25e6\-*\u00b7]\s+", line))
        language_like = bool(re.search(r"\b(native|fluent|proficiency|spoken|written|language)\b", lowered)) or bool(re.search(r"\b(english|spanish|french|german|hindi|mandarin)\b", lowered))
        credential_like = bool(re.search(r"\b(certif(?:icate|ication)|certified|license|credential|issued|expiry|expires)\b", lowered))
        if bullet:
            for section in SEMANTIC_SECTIONS:
                scores[section] += 0.15
                signals[section].append("list_item")

        if not language_like and not credential_like and any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in skill_aliases):
            scores["SKILLS"] += 2.0
            signals["SKILLS"].append("skill_vocabulary")
        if words & {"team", "player", "safety", "problem", "resourceful", "friendly", "helpful", "leadership", "communication", "competency", "proficient"}:
            scores["SKILLS"] += 1.0
            signals["SKILLS"].append("competency_phrase")

        recognition = re.search(r"\b(award|awarded|recognition|recognized|employee\s+of|honor|winner)\b", lowered)
        metric = re.search(r"\b\d+(?:\.\d+)?%\b|\$\s?\d+(?:\.\d+)?[km]?\b|\b\d+(?:\.\d+)?x\b", lowered)
        outcome = re.search(r"\b(?:increased|reduced|improved|saved|generated|achieved|accomplished)\b.+\b(?:by|to|from|through|resulting|leading)\b", lowered)
        if recognition or metric or outcome:
            scores["ACHIEVEMENTS"] += 2.5
            signals["ACHIEVEMENTS"].append("achievement_or_metric")

        if DateRangeParser.parse(line) is not None:
            scores["EXPERIENCE"] += 1.5
            scores["EDUCATION"] += 0.8
            scores["PROJECTS"] += 0.5
            signals["EXPERIENCE"].append("date_range")
        if (
            re.search(r"\b(inc|llc|ltd|corp|company|co\.)\b", lowered)
            or (
                not (has_profile_language := bool(re.search(r"\b(experienced|results-oriented|seeking|looking\s+to|career\s+objective|profile)\b", lowered)))
                and re.search(r"\b(manager|engineer|developer|assistant|analyst|director|supervisor|picker|packer)\b", lowered)
            )
        ):
            scores["EXPERIENCE"] += 1.5
            signals["EXPERIENCE"].append("role_or_company_pattern")
        if re.search(r"\b(managed|coordinated|worked|maintained|supervised|responsible|operated)\b", lowered) and (
            DateRangeParser.parse(line) is not None
            or re.search(r"\b(inc|llc|ltd|corp|company|co\.)\b", lowered)
            or re.search(r"\b(manager|engineer|developer|assistant|analyst|director|supervisor|picker|packer)\b", lowered)
        ):
            scores["EXPERIENCE"] += 0.8
            signals["EXPERIENCE"].append("employment_verb")

        if re.search(r"\b(experienced|results-oriented|seeking|looking\s+to|career\s+objective|profile)\b", lowered):
            scores["SUMMARY"] += 2.5
            signals["SUMMARY"].append("profile_language")

        if re.search(r"\b(bachelor|master|associate|phd|degree|university|college|school|institute|graduat)\b", lowered):
            scores["EDUCATION"] += 1.8
            signals["EDUCATION"].append("academic_pattern")
        elif re.search(r"\b(?:19|20)\d{2}\b", lowered):
            scores["EDUCATION"] += 0.3
            signals["EDUCATION"].append("graduation_year")

        if credential_like:
            scores["CERTIFICATIONS"] += 3.5
            signals["CERTIFICATIONS"].append("credential_pattern")

        if re.search(r"\b(built|developed|implemented|created|migrated|automated|application|system|project)\b", lowered) or re.search(r"https?://", lowered):
            scores["PROJECTS"] += 1.5
            signals["PROJECTS"].append("project_pattern")
        if language_like:
            scores["LANGUAGES"] += 2.0
            signals["LANGUAGES"].append("language_pattern")

    structure_ratio = sum(bool(re.match(r"^[\u2022\u2023\u25e6\-*\u00b7]\s+", line)) for line in lines) / max(len(lines), 1)
    if structure_ratio >= 0.5:
        for section in ("SKILLS", "ACHIEVEMENTS", "LANGUAGES"):
            scores[section] += 0.8
            signals[section].append("list_structure")
    elif sum(len(line.split()) for line in lines) / max(len(lines), 1) > 10:
        scores["SUMMARY"] = scores.get("SUMMARY", 0.0) + 0.5
        signals["SUMMARY"].append("paragraph_structure")

    heading_hints = {
        "SUMMARY": {"summary", "profile", "objective", "overview", "about"},
        "SKILLS": {"competencies", "competency", "expertise", "proficiencies", "proficiency", "strengths"},
        "ACHIEVEMENTS": {"achievements", "accomplishments", "honors", "recognition"},
        "EXPERIENCE": {"background", "employment", "career"},
        "EDUCATION": {"academic", "academics", "educational", "education"},
        "CERTIFICATIONS": {"credentials", "certification", "license"},
        "PROJECTS": {"work", "projects", "portfolio", "initiatives"},
        "LANGUAGES": {"language", "languages", "spoken"},
    }
    for section, hints in heading_hints.items():
        overlap = len(heading_tokens & hints)
        if section == "EXPERIENCE" and has_achievement_content:
            continue
        if section == "SUMMARY" and has_language_content and "language" in heading_tokens:
            continue
        if overlap and scores[section] > 0.0:
            scores[section] += min(2.0, float(overlap) * 2.0)
            signals[section].append("heading_hint")

    if has_language_content:
        scores["SKILLS"] = 0.0
        signals["SKILLS"] = []
        scores["SUMMARY"] = 0.0
        signals["SUMMARY"] = []

    project_pair = (
        len(lines) >= 2
        and len(lines[0].split()) <= 5
        and not has_academic_content
        and not any(re.search(r"\b(software engineer|developer|manager|analyst|assistant|supervisor)\b", line, re.IGNORECASE) for line in lines[:2])
        and bool(re.search(r"\b(built|created|developed|implemented|migrated|automated)\b", lines[1], re.IGNORECASE))
    )
    if project_pair:
        scores["PROJECTS"] += 2.5
        signals["PROJECTS"].append("project_entry_pair")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_section, top_score = ranked[0]
    second_score = ranked[1][1]
    total = sum(scores.values())
    confidence = top_score / total if total else 0.0
    section = top_section if confidence >= MIN_CONFIDENCE and confidence - (second_score / total if total else 0.0) >= MIN_MARGIN else "UNKNOWN"
    return SectionInference(
        section=section,
        confidence=confidence,
        scores=scores,
        signals={key: tuple(value) for key, value in signals.items()},
    )
