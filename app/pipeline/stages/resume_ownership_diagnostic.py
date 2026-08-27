from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.candidate_section import CandidateSection, SectionOrigin
from app.domain.document import Document
from app.domain.resume import Resume, validate_resume
from app.extractors.certifications import CertificationExtractor
from app.extractors.contact import ContactExtractor
from app.extractors.education import EducationExtractor
from app.extractors.experience import ExperienceExtractor
from app.extractors.projects import ProjectExtractor
from app.extractors.skills import SkillsExtractor
from app.pipeline.stages.block_classification import classify_block
from app.pipeline.stages.candidate_grouping import group_candidates
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.sections import SectionDetector
from app.pipeline.stages.semantic_compat import (
    candidate_header_to_text_blocks,
    candidate_sections_to_text_blocks,
    semantic_header_to_text_blocks,
    semantic_sections_to_text_blocks,
)
from app.pipeline.stages.semantic_ownership_comparison import (
    OwnershipComparison,
    compare_semantic_ownership,
)
from app.pipeline.stages.semantic_paths import detect_region_aware_sections
from app.pipeline.stages.structural_roles import build_structural_blocks
from app.pipeline.stages.text_extraction import TextBlock


FIELD_DIFFERENCE_CATEGORIES = (
    "FIELD_ADDED",
    "FIELD_REMOVED",
    "FIELD_CHANGED",
    "ENTRY_COUNT_DIFFERENCE",
    "DESCRIPTION_DIFFERENCE",
    "CONTACT_DIFFERENCE",
    "EDUCATION_DIFFERENCE",
    "SKILL_DIFFERENCE",
    "PROVENANCE_DIFFERENCE",
)


@dataclass(frozen=True)
class OwnershipCause:
    section_label: str | None
    section_origin: str | None
    line_ids: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    page: int | None
    region_id: str | None
    path_id: str | None
    heading: str | None
    text: str | None
    ownership_category: str | None


@dataclass(frozen=True)
class FieldDifference:
    category: str
    path: str
    current_value: Any
    candidate_value: Any
    cause: OwnershipCause | None


@dataclass(frozen=True)
class ResumeOwnershipDiagnostic:
    current_resume: Resume
    candidate_resume: Resume
    ownership: OwnershipComparison
    differences: tuple[FieldDifference, ...]
    first_meaningful_difference: FieldDifference | None


def diagnose_resume_ownership(
    document: Document,
    *,
    page_count: int | None = None,
    section_detector: SectionDetector | None = None,
) -> ResumeOwnershipDiagnostic:
    """Build Resume A (current semantic path) vs Resume B (CandidateSection ownership).

    Diagnostic only — not used by ResumeParser production assembly.
    """
    detector = section_detector or SectionDetector()
    pages = page_count if page_count is not None else max((page.page_number for page in document.pages), default=0)

    semantic = detect_region_aware_sections(document, detector)
    current_sections = semantic_sections_to_text_blocks(semantic)
    current_header = semantic_header_to_text_blocks(semantic)
    current_resume = _assemble_resume(current_sections, current_header, pages)

    structural_blocks = build_structural_blocks(document)
    candidate_sections = build_candidate_sections(structural_blocks, detector)
    candidate_stream = candidate_sections_to_text_blocks(candidate_sections, structural_blocks)
    _stamp_candidate_provenance(candidate_stream, candidate_sections)
    candidate_header = candidate_header_to_text_blocks(candidate_sections, structural_blocks)
    _stamp_header_provenance(candidate_header, structural_blocks)
    candidate_resume = _assemble_resume(candidate_stream, candidate_header, pages)

    ownership = compare_semantic_ownership(document, detector)
    differences = _diff_resumes(current_resume, candidate_resume, ownership, candidate_sections)
    first = _first_meaningful(differences)
    return ResumeOwnershipDiagnostic(
        current_resume=current_resume,
        candidate_resume=candidate_resume,
        ownership=ownership,
        differences=tuple(differences),
        first_meaningful_difference=first,
    )


def difference_categories(diagnostic: ResumeOwnershipDiagnostic) -> set[str]:
    return {item.category for item in diagnostic.differences}


def _assemble_resume(
    sections: dict[str, list[TextBlock]],
    header_blocks: list[TextBlock],
    page_count: int,
) -> Resume:
    contact = ContactExtractor()
    skills = SkillsExtractor()
    experience = ExperienceExtractor()
    education = EducationExtractor()
    projects = ProjectExtractor()
    certifications = CertificationExtractor()

    classified = {
        name: [classify_block(block) for block in blocks]
        for name, blocks in sections.items()
    }
    groups = {
        name: group_candidates(classified_blocks, name)
        for name, classified_blocks in classified.items()
        if name != "UNASSIGNED"
    }
    partial = {
        "parserVersion": "1.0.0",
        "personal": contact.extract(header_blocks),
        "skills": [
            skill["value"]
            for skill in skills.extract(sections.get("SKILLS", []), section_name="SKILLS")
        ],
        "experience": experience.extract(
            sections.get("EXPERIENCE", []), groups=groups.get("EXPERIENCE")
        ),
        "education": education.extract(
            [block.text for block in sections.get("EDUCATION", [])],
            groups=groups.get("EDUCATION"),
        ),
        "projects": projects.extract(sections.get("PROJECTS", []), groups=groups.get("PROJECTS")),
        "summary": " ".join(block.text.strip() for block in sections.get("SUMMARY", []) if block.text.strip()) or None,
        "certifications": certifications.extract(sections.get("CERTIFICATIONS", [])),
        "achievements": [block.text for block in sections.get("ACHIEVEMENTS", [])],
        "languages": [block.text for block in sections.get("LANGUAGES", [])],
        "metadata": {"pageCount": page_count, "ocrUsed": False},
    }
    return validate_resume(partial)


def _stamp_candidate_provenance(
    stream: dict[str, list[TextBlock]],
    sections: list[CandidateSection],
) -> None:
    by_first_line: dict[str, CandidateSection] = {}
    for section in sections:
        for block in section.content:
            if block.line_ids:
                by_first_line[block.line_ids[0]] = section
            for line_id in block.line_ids:
                by_first_line.setdefault(line_id, section)
    for label, blocks in stream.items():
        if label == "UNASSIGNED":
            continue
        for block in blocks:
            line_id = getattr(block, "source_line_id", None)
            section = by_first_line.get(str(line_id)) if line_id else None
            if section is None:
                continue
            member = next(
                (
                    item
                    for item in section.content
                    if line_id in item.line_ids or (item.line_ids and item.line_ids[0] == line_id)
                ),
                None,
            )
            block.path_id = section.path_id
            block.region_id = section.region_id
            if member is not None:
                block.source_line_ids = list(member.line_ids)
                block.source_span_ids = list(member.source_span_ids)
                block.reconstruction_method = member.reconstruction_method


def _stamp_header_provenance(blocks: list[TextBlock], structural_blocks) -> None:
    by_line = {}
    for item in structural_blocks:
        if item.line_ids:
            by_line[item.line_ids[0]] = item
    for block in blocks:
        line_id = getattr(block, "source_line_id", None)
        source = by_line.get(str(line_id)) if line_id else None
        if source is None:
            continue
        block.path_id = source.path_id
        block.region_id = source.region_id
        block.source_line_ids = list(source.line_ids)
        block.source_span_ids = list(source.source_span_ids)


def _diff_resumes(
    current: Resume,
    candidate: Resume,
    ownership: OwnershipComparison,
    candidate_sections: list[CandidateSection],
) -> list[FieldDifference]:
    diffs: list[FieldDifference] = []
    diffs.extend(_diff_personal(current, candidate, ownership, candidate_sections))
    diffs.extend(_diff_scalar("summary", current.summary, candidate.summary, ownership, "SUMMARY", candidate_sections))
    diffs.extend(_diff_string_list("skills", current.skills, candidate.skills, "SKILL_DIFFERENCE", ownership, "SKILLS", candidate_sections))
    diffs.extend(_diff_string_list("achievements", current.achievements, candidate.achievements, "FIELD_CHANGED", ownership, "ACHIEVEMENTS", candidate_sections))
    diffs.extend(_diff_string_list("languages", current.languages, candidate.languages, "FIELD_CHANGED", ownership, "LANGUAGES", candidate_sections))
    diffs.extend(_diff_entry_list("experience", current.experience, candidate.experience, ownership, "EXPERIENCE", candidate_sections))
    diffs.extend(_diff_entry_list("education", current.education, candidate.education, ownership, "EDUCATION", candidate_sections, category_override="EDUCATION_DIFFERENCE"))
    diffs.extend(_diff_entry_list("projects", current.projects, candidate.projects, ownership, "PROJECTS", candidate_sections))
    diffs.extend(_diff_entry_list("certifications", current.certifications, candidate.certifications, ownership, "CERTIFICATIONS", candidate_sections))
    diffs.extend(_provenance_diffs(ownership))
    return diffs


def _diff_personal(
    current: Resume,
    candidate: Resume,
    ownership: OwnershipComparison,
    candidate_sections: list[CandidateSection],
) -> list[FieldDifference]:
    diffs: list[FieldDifference] = []
    for name in ("name", "email", "phone", "location", "linkedin", "github", "portfolio"):
        left = getattr(current.personal, name)
        right = getattr(candidate.personal, name)
        if left == right:
            continue
        category = _scalar_category(left, right, contact=True)
        diffs.append(
            FieldDifference(
                category=category,
                path=f"personal.{name}",
                current_value=left,
                candidate_value=right,
                cause=_cause_for_label(ownership, candidate_sections, None),
            )
        )
    return diffs


def _diff_scalar(
    path: str,
    left: Any,
    right: Any,
    ownership: OwnershipComparison,
    label: str,
    candidate_sections: list[CandidateSection],
) -> list[FieldDifference]:
    if left == right:
        return []
    category = "DESCRIPTION_DIFFERENCE" if path == "summary" and left and right else _scalar_category(left, right)
    return [
        FieldDifference(
            category=category,
            path=path,
            current_value=left,
            candidate_value=right,
            cause=_cause_for_label(ownership, candidate_sections, label),
        )
    ]


def _diff_string_list(
    path: str,
    left: list[str],
    right: list[str],
    list_category: str,
    ownership: OwnershipComparison,
    label: str,
    candidate_sections: list[CandidateSection],
) -> list[FieldDifference]:
    if left == right:
        return []
    diffs: list[FieldDifference] = []
    if len(left) != len(right):
        diffs.append(
            FieldDifference(
                category="ENTRY_COUNT_DIFFERENCE",
                path=path,
                current_value=left,
                candidate_value=right,
                cause=_cause_for_label(ownership, candidate_sections, label),
            )
        )
    added = [item for item in right if item not in left]
    removed = [item for item in left if item not in right]
    if added:
        diffs.append(
            FieldDifference(
                category="FIELD_ADDED" if path != "skills" else list_category,
                path=f"{path}.added",
                current_value=None,
                candidate_value=added,
                cause=_cause_for_label(ownership, candidate_sections, label),
            )
        )
    if removed:
        diffs.append(
            FieldDifference(
                category="FIELD_REMOVED" if path != "skills" else list_category,
                path=f"{path}.removed",
                current_value=removed,
                candidate_value=None,
                cause=_cause_for_label(ownership, candidate_sections, label),
            )
        )
    if not added and not removed and left != right:
        diffs.append(
            FieldDifference(
                category=list_category,
                path=path,
                current_value=left,
                candidate_value=right,
                cause=_cause_for_label(ownership, candidate_sections, label),
            )
        )
    return diffs


def _diff_entry_list(
    path: str,
    left: list,
    right: list,
    ownership: OwnershipComparison,
    label: str,
    candidate_sections: list[CandidateSection],
    *,
    category_override: str | None = None,
) -> list[FieldDifference]:
    diffs: list[FieldDifference] = []
    cause = _cause_for_label(ownership, candidate_sections, label)
    if len(left) != len(right):
        diffs.append(
            FieldDifference(
                category="ENTRY_COUNT_DIFFERENCE",
                path=path,
                current_value=_public_entries(left),
                candidate_value=_public_entries(right),
                cause=cause,
            )
        )
    for index, (current_item, candidate_item) in enumerate(zip(left, right)):
        current_dump = current_item.model_dump()
        candidate_dump = candidate_item.model_dump()
        for key in current_dump:
            if key == "confidence":
                continue
            lv, rv = current_dump[key], candidate_dump[key]
            if lv == rv:
                continue
            if key == "description":
                category = "DESCRIPTION_DIFFERENCE"
            elif path == "education" or category_override == "EDUCATION_DIFFERENCE":
                category = "EDUCATION_DIFFERENCE"
            else:
                category = _scalar_category(lv, rv)
            diffs.append(
                FieldDifference(
                    category=category,
                    path=f"{path}[{index}].{key}",
                    current_value=lv,
                    candidate_value=rv,
                    cause=cause,
                )
            )
    if len(right) > len(left):
        for index, extra in enumerate(right[len(left) :], start=len(left)):
            diffs.append(
                FieldDifference(
                    category="FIELD_ADDED",
                    path=f"{path}[{index}]",
                    current_value=None,
                    candidate_value=extra.model_dump(),
                    cause=cause,
                )
            )
    if len(left) > len(right):
        for index, extra in enumerate(left[len(right) :], start=len(right)):
            diffs.append(
                FieldDifference(
                    category="FIELD_REMOVED",
                    path=f"{path}[{index}]",
                    current_value=extra.model_dump(),
                    candidate_value=None,
                    cause=cause,
                )
            )
    return diffs


def _public_entries(items: list) -> list[dict]:
    dumps = []
    for item in items:
        data = item.model_dump()
        data.pop("confidence", None)
        dumps.append(data)
    return dumps


def _scalar_category(left: Any, right: Any, *, contact: bool = False) -> str:
    if contact:
        if left in (None, "", []) and right not in (None, "", []):
            return "CONTACT_DIFFERENCE"
        if right in (None, "", []) and left not in (None, "", []):
            return "CONTACT_DIFFERENCE"
        return "CONTACT_DIFFERENCE"
    if left in (None, "", []) and right not in (None, "", []):
        return "FIELD_ADDED"
    if right in (None, "", []) and left not in (None, "", []):
        return "FIELD_REMOVED"
    return "FIELD_CHANGED"


def _cause_for_label(
    ownership: OwnershipComparison,
    candidate_sections: list[CandidateSection],
    label: str | None,
) -> OwnershipCause:
    section = next(
        (
            item
            for item in candidate_sections
            if label is None
            or item.semantic_label == label
            or (label is None and item.origin == SectionOrigin.UNLABELED)
        ),
        None,
    )
    unassigned = [
        record
        for record in ownership.line_records
        if (label is None or label in record.candidate_labels or label in record.current_labels)
        and (not record.current_labels or not record.candidate_labels or set(record.current_labels) != set(record.candidate_labels))
    ]
    sample = unassigned[0] if unassigned else (ownership.line_records[0] if ownership.line_records else None)
    heading_text = section.heading.text if section and section.heading is not None else None
    category = None
    if ownership.first_loss is not None:
        category = ownership.first_loss.category
    elif unassigned:
        category = "LINE_UNASSIGNED"
    return OwnershipCause(
        section_label=section.semantic_label if section else (sample.candidate_labels[0] if sample and sample.candidate_labels else None),
        section_origin=section.origin.value if section else (sample.candidate_origins[0] if sample and sample.candidate_origins else None),
        line_ids=tuple(sample.line_id for sample in unassigned[:12]) if unassigned else (section.line_ids if section else ()),
        source_span_ids=section.source_span_ids if section else (sample.source_span_ids if sample else ()),
        page=section.page_number if section else (sample.page_number if sample else None),
        region_id=section.region_id if section else (sample.region_id if sample else None),
        path_id=section.path_id if section else (sample.path_id if sample else None),
        heading=heading_text,
        text=sample.text if sample else heading_text,
        ownership_category=category,
    )


def _provenance_diffs(ownership: OwnershipComparison) -> list[FieldDifference]:
    diffs: list[FieldDifference] = []
    seen = False
    for item in ownership.disagreements:
        if item.category != "PROVENANCE_DIFFERENCE":
            continue
        if seen:
            break
        seen = True
        diffs.append(
            FieldDifference(
                category="PROVENANCE_DIFFERENCE",
                path="compat.TextBlock",
                current_value=item.detail,
                candidate_value=item.detail,
                cause=OwnershipCause(
                    section_label=item.semantic_label,
                    section_origin=item.origin,
                    line_ids=(item.line_id,) if item.line_id else (),
                    source_span_ids=(),
                    page=item.page,
                    region_id=item.region_id,
                    path_id=item.path_id,
                    heading=None,
                    text=item.text,
                    ownership_category="PROVENANCE_DIFFERENCE",
                ),
            )
        )
    return diffs


def _first_meaningful(differences: list[FieldDifference]) -> FieldDifference | None:
    preferred = (
        "ENTRY_COUNT_DIFFERENCE",
        "FIELD_ADDED",
        "FIELD_REMOVED",
        "CONTACT_DIFFERENCE",
        "SKILL_DIFFERENCE",
        "EDUCATION_DIFFERENCE",
        "DESCRIPTION_DIFFERENCE",
        "FIELD_CHANGED",
        "PROVENANCE_DIFFERENCE",
    )
    for category in preferred:
        for item in differences:
            if item.category == category and item.path != "compat.TextBlock":
                return item
    return differences[0] if differences else None
