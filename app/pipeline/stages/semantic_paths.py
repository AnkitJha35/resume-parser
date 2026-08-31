from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.document import Document, Line
from app.extractors.date_parser import DateRangeParser
from app.pipeline.stages.sections import SECTION_NAMES, SectionDetector
from app.pipeline.stages.semantic_inference import (
    UnknownSectionCandidate,
    infer_section,
    is_unknown_heading,
)


@dataclass(frozen=True)
class SemanticPath:
    path_id: str
    page_number: int
    region_id: str
    region_order: int
    lines: list[Line]


@dataclass
class SemanticSection:
    section: str
    page_number: int
    region_id: str
    path_id: str
    lines: list[Line] = field(default_factory=list)


@dataclass
class SemanticDocument:
    paths: list[SemanticPath] = field(default_factory=list)
    sections: dict[str, list[SemanticSection]] = field(
        default_factory=lambda: {name: [] for name in SECTION_NAMES}
    )
    unassigned_lines: list[Line] = field(default_factory=list)
    unknown_candidates: list[UnknownSectionCandidate] = field(default_factory=list)


def build_semantic_paths(document: Document) -> list[SemanticPath]:
    """Create deterministic page-local paths without flattening region identity."""
    paths: list[SemanticPath] = []
    for page in document.pages:
        regions = sorted(
            page.regions,
            key=lambda region: (
                region.reading_order if region.reading_order is not None else 10**9,
                region.region_id,
            ),
        )
        for region_order, region in enumerate(regions):
            paths.append(
                SemanticPath(
                    path_id=f"page-{page.page_number}-path-{region_order}",
                    page_number=page.page_number,
                    region_id=region.region_id,
                    region_order=region_order,
                    lines=list(region.lines),
                )
            )
    return paths


def detect_region_aware_sections(
    document: Document,
    section_detector: SectionDetector | None = None,
) -> SemanticDocument:
    """Detect sections independently inside each physical path."""
    detector = section_detector or SectionDetector()
    paths = build_semantic_paths(document)
    region_kind_by_id = {
        region.region_id: region.kind
        for page in document.pages
        for region in page.regions
    }
    result = SemanticDocument(paths=paths)
    previous_paths: list[tuple[SemanticPath, str]] = []

    pages: dict[int, list[SemanticPath]] = {}
    for path in paths:
        pages.setdefault(path.page_number, []).append(path)

    for page_number in sorted(pages):
        current_page_paths = pages[page_number]
        page_sections: list[tuple[SemanticPath, str]] = []
        for path in current_page_paths:
            heading = _path_heading(path, detector)
            current_section = _continuation_section(path, previous_paths)
            related_heading = _related_heading(path, current_page_paths, detector)
            if related_heading is not None:
                current_section = related_heading
            if heading is not None and related_heading is None:
                current_section = None
            section_lines: dict[str, list[Line]] = {}
            index = 0
            region_kind = region_kind_by_id.get(path.region_id)
            lead_end = _leading_headingless_summary_end(
                path.lines, detector, region_kind, current_section
            )
            if lead_end:
                section_lines.setdefault("SUMMARY", []).extend(path.lines[:lead_end])
                index = lead_end
            while index < len(path.lines):
                line = path.lines[index]
                matched_section = detector._find_section_header(line.text)
                content_line = False
                if matched_section is None and _is_unknown_heading_line(line, path.lines, index, detector, current_section):
                    end = _next_section_boundary(
                        path.lines, index + 1, detector, current_section
                    )
                    content = path.lines[index + 1 : end]
                    inference = infer_section(line.text, [item.text for item in content])
                    result.unknown_candidates.append(
                        UnknownSectionCandidate(heading=line, content=list(content), inference=inference)
                    )
                    if inference.section != "UNKNOWN":
                        section_lines.setdefault(inference.section, []).extend(content)
                        current_section = inference.section
                    elif _is_skills_like_unknown_section(line, content):
                        # Narrow bypass of UNKNOWN+active keep: skills token in
                        # the heading plus short chip/grid content evidence.
                        section_lines.setdefault("SKILLS", []).extend(content)
                        current_section = "SKILLS"
                    elif current_section is not None:
                        # False unknown heading inside an active section: keep
                        # ownership. Do not drop the title line or the span.
                        section_lines.setdefault(current_section, []).append(line)
                        section_lines.setdefault(current_section, []).extend(content)
                    else:
                        current_section = None
                    index = end
                    continue
                if matched_section is None and detector._looks_like_education_header(line.text):
                    matched_section = "EDUCATION"
                    content_line = True
                if matched_section is None and DateRangeParser.parse(line.text) is not None:
                    if _is_aligned_education_date(line, current_page_paths, detector):
                        matched_section = "EDUCATION"
                        content_line = True
                if matched_section is not None:
                    current_section = matched_section
                    if content_line:
                        section_lines.setdefault(matched_section, []).append(line)
                    index += 1
                    continue
                if current_section is not None:
                    section_lines.setdefault(current_section, []).append(line)
                index += 1

            for section, lines in section_lines.items():
                result.sections[section].append(
                    SemanticSection(
                        section=section,
                        page_number=path.page_number,
                        region_id=path.region_id,
                        path_id=path.path_id,
                        lines=lines,
                    )
                )
            if current_section is not None:
                page_sections.append((path, current_section))
        previous_paths = page_sections

    assigned_line_ids = {
        line.line_id
        for sections in result.sections.values()
        for section in sections
        for line in section.lines
    }
    result.unassigned_lines = [
        line
        for path in paths
        for line in path.lines
        if line.line_id not in assigned_line_ids
    ]

    return result


_BODY_REGION_KINDS = frozenset({"column", "physical_region"})
_MIN_HEADINGLESS_SUMMARY_LINES = 3
_MIN_HEADINGLESS_SUMMARY_WORDS = 30
_MIN_PROSE_LINE_WORDS = 8


def _leading_headingless_summary_end(
    lines: list[Line],
    detector: SectionDetector,
    region_kind: str | None,
    current_section: str | None,
) -> int | None:
    """Return exclusive end index of a leading body paragraph, or None.

    Ownership is structural: a wrapped multi-line prose run at the start of a
    body path, before a later section opener. Does not call infer_section.
    """
    if current_section is not None or region_kind not in _BODY_REGION_KINDS or not lines:
        return None
    if detector._find_section_header(lines[0].text) is not None:
        return None

    end = 0
    while end < len(lines) and _is_paragraph_prose_line(lines[end], detector):
        if end > 0 and not _is_paragraph_continuation(lines[end - 1], lines[end]):
            break
        end += 1
    if end < _MIN_HEADINGLESS_SUMMARY_LINES:
        return None
    word_count = sum(len((line.text or "").split()) for line in lines[:end])
    if word_count < _MIN_HEADINGLESS_SUMMARY_WORDS:
        return None
    if end >= len(lines) or not _is_strong_section_opener(lines[end], detector):
        return None
    return end


def _is_paragraph_prose_line(line: Line, detector: SectionDetector) -> bool:
    text = (line.text or "").strip()
    words = text.split()
    if len(words) < _MIN_PROSE_LINE_WORDS:
        return False
    if _is_list_marker(text):
        return False
    if detector._find_section_header(text) is not None:
        return False
    if detector._looks_like_education_header(text):
        return False
    if DateRangeParser.parse(text) is not None:
        return False
    return True


def _is_paragraph_continuation(previous: Line, current: Line) -> bool:
    same_indent = abs(previous.bbox.x0 - current.bbox.x0) <= max(
        previous.bbox.y1 - previous.bbox.y0,
        current.bbox.y1 - current.bbox.y0,
        8.0,
    )
    gap = current.bbox.y0 - previous.bbox.y1
    line_height = max(
        previous.bbox.y1 - previous.bbox.y0,
        current.bbox.y1 - current.bbox.y0,
        1.0,
    )
    close_vertical = gap <= line_height * 1.75
    same_size = abs((previous.style.font_size or 0.0) - (current.style.font_size or 0.0)) <= 1.5
    return same_indent and close_vertical and same_size


def _is_strong_section_opener(line: Line, detector: SectionDetector) -> bool:
    text = (line.text or "").strip()
    if detector._find_section_header(text) is not None:
        return True
    if detector._looks_like_education_header(text):
        return True
    if DateRangeParser.parse(text) is not None:
        return True
    return False


_SKILLS_HEADING_TOKENS = frozenset({"skill", "skills"})
_MAX_SKILL_CHIP_WORDS = 4
_MAX_SKILL_CHIP_WORDS_HARD = 6
_MIN_SKILL_CHIP_ITEMS = 2
_EDU_OR_JOB_CONTENT_RE = re.compile(
    r"\b(?:bachelor|master|mba|phd|b\.?s\.?|m\.?s\.?|university|college|"
    r"school|institute|academy|degree|inc|llc|ltd|corp|company)\b",
    re.IGNORECASE,
)
_YEAR_TOKEN_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _heading_has_skills_signal(text: str) -> bool:
    """True when a normalized heading contains a skills token (skill/skills)."""
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    return bool(tokens & _SKILLS_HEADING_TOKENS)


def _content_looks_like_skill_list(content: list[Line]) -> bool:
    """True for short chip/grid spans without education/experience structure."""
    texts = [
        (line.text or "").strip()
        for line in content
        if (line.text or "").strip() and not _is_list_marker(line.text)
    ]
    if len(texts) < _MIN_SKILL_CHIP_ITEMS:
        return False
    if any(DateRangeParser.parse(text) is not None for text in texts):
        return False
    if any(_YEAR_TOKEN_RE.search(text) for text in texts):
        return False
    if any(_EDU_OR_JOB_CONTENT_RE.search(text) for text in texts):
        return False
    word_counts = [len(text.split()) for text in texts]
    avg_words = sum(word_counts) / len(word_counts)
    if avg_words > _MAX_SKILL_CHIP_WORDS:
        return False
    if any(count > _MAX_SKILL_CHIP_WORDS_HARD for count in word_counts):
        return False
    # Reject prose paragraphs mistaken for chips.
    if any(len(text) > 60 or (text.endswith(".") and len(text.split()) > 8) for text in texts):
        return False
    short_items = sum(1 for count in word_counts if 1 <= count <= _MAX_SKILL_CHIP_WORDS)
    return short_items >= _MIN_SKILL_CHIP_ITEMS


def _is_skills_like_unknown_section(heading: Line, content: list[Line]) -> bool:
    """Open SKILLS only when heading and following content both evidence skills."""
    return _heading_has_skills_signal(heading.text) and _content_looks_like_skill_list(content)


def _is_unknown_heading_line(
    line: Line,
    lines: list[Line],
    index: int,
    detector: SectionDetector,
    current_section: str | None = None,
) -> bool:
    if detector._find_section_header(line.text) is not None:
        return False
    if (
        _is_list_marker(line.text)
        or (index > 0 and _is_list_marker(lines[index - 1].text))
    ):
        return False
    if (
        index + 1 < len(lines)
        and _is_list_marker(lines[index + 1].text)
        and index > 0
        and not _is_list_marker(lines[index - 1].text)
        and (lines[index - 1].style.font_size or 0.0) > (line.style.font_size or 0.0) + 1.5
    ):
        return False
    if index > 0 and _is_wrapped_content_line(lines[index - 1], line):
        return False
    if index > 0 and _is_same_visual_row(lines[index - 1], line):
        return False
    if not is_unknown_heading(line.text, line.style.font_size, line.style.bold):
        return False
    if current_section and _looks_like_item_heading(line, lines, index, current_section):
        return False
    return index + 1 < len(lines)


def _next_section_boundary(
    lines: list[Line],
    start: int,
    detector: SectionDetector,
    current_section: str | None = None,
) -> int:
    for index in range(start, len(lines)):
        line = lines[index]
        if detector._find_section_header(line.text) is not None:
            return index
        if index > start and _is_unknown_heading_line(
            line, lines, index, detector, current_section
        ):
            return index
    return len(lines)


def _is_list_marker(text: str) -> bool:
    value = (text or "").strip()
    return bool(value) and len(value) <= 3 and not any(char.isalnum() for char in value)


def _is_wrapped_content_line(previous: Line, current: Line) -> bool:
    previous_text = (previous.text or "").strip()
    current_text = (current.text or "").strip()
    if not previous_text or not current_text:
        return False
    same_style = (
        abs((previous.style.font_size or 0.0) - (current.style.font_size or 0.0)) <= 1.5
        and previous.style.bold == current.style.bold
    )
    same_indent = abs(previous.bbox.x0 - current.bbox.x0) <= max(
        previous.bbox.y1 - previous.bbox.y0,
        current.bbox.y1 - current.bbox.y0,
        1.0,
    )
    close_vertical = current.bbox.y0 - previous.bbox.y1 <= max(
        previous.bbox.y1 - previous.bbox.y0,
        current.bbox.y1 - current.bbox.y0,
        4.0,
    )
    previous_continues = not previous_text.endswith((".", "!", "?", ":", ";"))
    current_continues = current_text[:1].islower() or len(previous_text.split()) >= 5
    return same_style and same_indent and close_vertical and previous_continues and current_continues


def _is_same_visual_row(previous: Line, current: Line) -> bool:
    return abs(previous.bbox.y0 - current.bbox.y0) <= 1.0


_ITEM_ROLE_WORDS = {
    "assistant",
    "secretary",
    "coordinator",
    "manager",
    "analyst",
    "specialist",
    "developer",
    "engineer",
    "supervisor",
    "administrator",
    "associate",
    "consultant",
    "director",
    "executive",
    "clerk",
    "intern",
    "officer",
    "lead",
    "picker",
    "packer",
    "teacher",
    "professor",
}


def _looks_like_item_heading(line: Line, lines: list[Line], index: int, section: str) -> bool:
    following = [item for item in lines[index + 1 : index + 4] if not _is_list_marker(item.text)]
    texts = [item.text.strip() for item in following if item.text.strip()]
    if not texts:
        return False
    has_date = any(DateRangeParser.parse(text) is not None or re.search(r"\b(?:19|20)\d{2}\b", text) for text in texts)
    has_company_or_location = any(_looks_like_org_or_location(text) for text in texts)
    title_like = _looks_like_entry_title_line(line)
    if section == "EXPERIENCE":
        # Combined org/date line still wins. Split stacks (title / unsuffixed
        # org / date) are also in-section entries, not new headings.
        if has_date and has_company_or_location:
            return True
        return title_like and (has_date or has_company_or_location)
    if section == "EDUCATION":
        return has_date and any(re.search(r"\b(?:university|college|school|institute|academy)\b", text, re.IGNORECASE) for text in texts)
    if section == "CERTIFICATIONS":
        return any(re.search(r"https?://|\b(?:credential|cert(?:ificate|ification)?\s*(?:id)?|issued|expires?)\b", text, re.IGNORECASE) for text in texts)
    if section == "PROJECTS":
        return any(re.search(r"\b(?:built|created|developed|implemented|designed)\b", text, re.IGNORECASE) or re.search(r"https?://", text, re.IGNORECASE) for text in texts)
    if section == "ACHIEVEMENTS":
        recognition = any(
            re.search(
                r"\b(?:award|awarded|recognized|recognition|employee\s+of|increased|reduced|improved|achieved)\b",
                text,
                re.IGNORECASE,
            )
            or re.search(r"\b\d+(?:\.\d+)?%\b", text)
            for text in texts
        )
        if recognition:
            return True
        # Title-like award row + year/org is an entry, not a new section.
        # A lone year must not retarget EDUCATION via unknown-heading inference.
        if title_like and (has_date or has_company_or_location):
            return True
        return False
    return False


def _looks_like_entry_title_line(line: Line) -> bool:
    value = (line.text or "").strip()
    words = value.split()
    if not value or len(words) > 6 or any(char.isdigit() for char in value):
        return False
    if value.endswith((".", ":", ";", ",")):
        return False
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]+", value)}
    if tokens & _ITEM_ROLE_WORDS:
        return True
    upper_like = value.upper() == value and any(char.isalpha() for char in value)
    title_case = all(word[:1].isupper() for word in words if word)
    emphasized = bool(line.style.bold) or upper_like or title_case
    return emphasized and len(words) <= 5


def _looks_like_org_or_location(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if re.search(r"\b(?:inc|llc|ltd|corp|company|co\.)\b", value, re.IGNORECASE):
        return True
    if "," in value or " / " in value:
        return True
    if "&" in value and 1 < len(value.split()) <= 8:
        return True
    return False


def _path_heading(path: SemanticPath, detector: SectionDetector) -> str | None:
    heading = None
    for line in path.lines:
        section = detector._find_section_header(line.text)
        if section is not None:
            heading = section
    return heading


def _related_heading(
    path: SemanticPath,
    page_paths: list[SemanticPath],
    detector: SectionDetector,
) -> str | None:
    candidates: list[tuple[float, str]] = []
    for other in page_paths:
        section = _path_heading(other, detector)
        if section is None or not other.lines or not path.lines:
            continue
        heading_line = next(
            line
            for line in reversed(other.lines)
            if detector._find_section_header(line.text) is not None
        )
        if heading_line.bbox.y1 > path.lines[0].bbox.y0:
            continue
        if not (_x_overlaps(path, other) or _horizontally_adjacent(path, other)):
            continue
        candidates.append((path.lines[0].bbox.y0 - heading_line.bbox.y1, section))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _x_overlaps(first: SemanticPath, second: SemanticPath) -> bool:
    first_x0 = min(line.bbox.x0 for line in first.lines)
    first_x1 = max(line.bbox.x1 for line in first.lines)
    second_x0 = min(line.bbox.x0 for line in second.lines)
    second_x1 = max(line.bbox.x1 for line in second.lines)
    return first_x0 <= second_x1 and second_x0 <= first_x1


def _horizontally_adjacent(first: SemanticPath, second: SemanticPath) -> bool:
    first_x0 = min(line.bbox.x0 for line in first.lines)
    first_x1 = max(line.bbox.x1 for line in first.lines)
    second_x0 = min(line.bbox.x0 for line in second.lines)
    second_x1 = max(line.bbox.x1 for line in second.lines)
    return first_x1 <= second_x0 or second_x1 <= first_x0


def _is_aligned_education_date(
    line: Line,
    page_paths: list[SemanticPath],
    detector: SectionDetector,
) -> bool:
    for path in page_paths:
        for candidate in path.lines:
            if not detector._looks_like_education_header(candidate.text):
                continue
            if (
                candidate.page_number == line.page_number
                and abs(candidate.bbox.y0 - line.bbox.y0) <= 1.0
                and line.bbox.x0 > candidate.bbox.x1
            ):
                return True
    return False


def _continuation_section(path: SemanticPath, previous_paths: list[tuple[SemanticPath, str]]) -> str | None:
    if not previous_paths or not path.lines:
        return None

    first_line = path.lines[0]
    _, section = min(
        previous_paths,
        key=lambda item: (
            abs(_path_center(item[0]) - _line_center(first_line)),
            item[0].path_id,
        ),
    )
    return section


def _path_center(path: SemanticPath) -> float:
    return sum(_line_center(line) for line in path.lines) / len(path.lines)


def _line_center(line: Line) -> float:
    return (line.bbox.x0 + line.bbox.x1) / 2.0
