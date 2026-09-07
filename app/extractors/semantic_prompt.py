"""Provider-neutral serialization, prompt definitions, and structured output parsing for semantic LLM extraction."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from app.domain.semantic_contract import (
    BodySemanticOutput,
    DocumentArchetype,
    PersonalSemanticOutput,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    summarize_body_evidence,
)
from app.extractors.semantic_extractor import SemanticExtractionError

SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """You are a precise, layout-aware resume data extractor.
Your task is to analyze the provided structured document blocks and extract canonical resume entities strictly grounded in the supplied text blocks.

CRITICAL GROUNDING AND PROVENANCE RULES:
1. Every non-null grounded value MUST reference the exact `source_block_ids` from which it was extracted.
2. EXTRACT ONLY supported information. DO NOT invent, hallucinate, or infer missing values.
3. DO NOT perform semantic renaming or enrichment (e.g., do not rename companies, do not expand job titles).
4. Use exact verbatim text from source blocks in `raw_value` when available. In `value`, only safe canonical normalizations are permitted (e.g., ISO dates 'YYYY-MM-DD'/'YYYY-MM'/'YYYY', phone digits, whitespace/case cleanup). When source evidence contains only a year (for example '2020' or a range such as '2020 - 2024'), preserve year precision and return '2020' and '2024'. NEVER pad a year-only source with '-01' or '-01-01'. Only include month/day when those components are explicitly present in the source evidence.
5. DO NOT treat document headers, form titles (e.g., 'APPLICATION FORM', 'Curriculum Vitae', 'Surname'), or section labels as personal names.
6. DO NOT treat table column headers (e.g., 'Ship Name', 'Period', 'S.No.', 'Documents Details') as actual company, designation, or degree values.
7. DO NOT classify referee or reference contacts as employment experience. Mark referee and boilerplate blocks explicitly under block_classifications.
8. Every boolean field (e.g., `current`) MUST reference the source block IDs providing explicit evidence (e.g., blocks explicitly containing 'Present', 'Current', 'Currently', 'Ongoing', 'Till Date', 'Now'). An end date or year alone (including future years, current calendar years, or date ranges without explicit current wording) DOES NOT establish `current=true`. When explicit current wording is absent, set `current: null` (or omit).
9. If evidence for a field is absent or ambiguous, return null or empty list rather than guessing.
10. Return a single valid JSON object adhering strictly to the SemanticOutput schema.

STRUCTURED FORMS AND TABLE EXTRACTION GUIDANCE:
1. Form & Archetype Completeness:
   * A document classification of `structured_form`, `maritime_tabular`, or `maritime_cv` indicates layout structure (application form or tabular seafarer profile) and does NOT mean the document should produce empty canonical fields.
   * You must inspect EVERY table data row and labeled section across the document and map all supported data to canonical entities when column headers or field labels provide the semantic mapping.

2. Table Interpretation:
   * `table`, `row`, `col`, and `cell_role` identify physical grid relationships.
   * HEADER cells (e.g., `cell_role: "HEADER"` or row 0 headers) describe the meaning/attribute of their column. Never treat a HEADER cell itself as an extracted entity value.
   * Non-header data cells in the same row form one logical record.
   * Multiple blocks sharing the same `table` + `row` + `col` belong to the same physical cell and may be combined in reading order.

3. Grounded Semantic Interpretation (Allowed vs. Prohibited):
   * Allowed (Semantic interpretation supported by explicit source relationships):
     - Removing field labels such as `Email:`, `Phone:`, `Name:` to extract the underlying value.
     - Mapping a data value to the canonical field whose table column or header label explicitly describes it.
     - Combining adjacent blocks belonging to the same table cell in reading order.
     - Mapping explicit table row relationships to canonical schema fields.
     - Normalizing dates to ISO format when the source contains the corresponding date components.
     - Normalizing phone numbers.
     - Extracting an email address from an explicitly labeled email field or contact block.
   * Still Prohibited (Invention / Fabrication):
     - Inventing facts or adding ungrounded details.
     - Renaming companies (e.g., 'Darya Shaan' -> 'Darya Shipping').
     - Expanding job titles (e.g., 'Cadet' -> 'Chief Officer').
     - Hallucinating duties or technologies not present in the text.
     - Inventing dates or extrapolating timelines.
     - Creating unsupported entities.
     - Fabricating source block IDs.

SPECIFIC SECTION EXTRACTION MAPPINGS:
1. Personal Labeled Fields & Contact Information:
   * Inspect labeled form fields and contact blocks to populate `personal`.
   * `Email: user@example.com` populates `personal.email` with `user@example.com`, grounded to the source block.
   * `Phone: ...` populates `personal.phone`. Copy the phone text exactly as written in the source block (e.g. `+1 (415) 555-0188`). Do NOT manually retype, reconstruct, infer, or normalize phone digits in your head. Preserve country code, area code, exchange, subscriber digits, and punctuation exactly as present in the source. Downstream deterministic validation/normalization handles canonical phone representation. If uncertain, return null rather than inventing or reconstructing digits.
   * Assemble personal name tokens from explicit name fields (e.g., Surname, First Name, Middle Name) while discarding descriptor labels.

2. Education Tables:
   * Inspect every data row in educational qualification tables. Each non-header data row populates one `GroundedEducationItem`.
   * `Name of Institute / College` / `School / College / University` -> `institution`
   * `Type of Degree` / `Degree / Certificate` / `Examination Passed` -> `degree` or `fieldOfStudy` (choose the canonical field that best matches the explicit header without inventing terminology)
   * `From` / `Commenced` -> `startDate`
   * `To` / `Completed` / `Passed` -> `endDate`
   * `Grade` / `Class` / `Division` -> `grade`

3. Maritime Sea-Service & Employment Tables:
   * Inspect every data row in tables titled `Previous Sea Service`, `Sea Service`, `Experience`, or `Employment History`. Each non-header data row represents an `ExperienceItem`.
   * `Owners / Manager` / `Company Name` / `Employer` -> `experience.company`
   * `Rank` / `Position` / `Capacity` -> `experience.designation`. For tabular experience, designation must be taken strictly from the Rank/Position column. Do not combine it with serial numbers, vessel specifications, engine models, BHP, DWT, GRT, flags, or other adjacent columns. Technical specifications belong in contextual description, when supported.
   * `From` / `Sign On` / `Date Commencing` -> `startDate`
   * `To` / `Sign Off` / `Date of S/OFF` -> `endDate`
   * `Vessel Name` is NOT the company; preserve the vessel name as grounded contextual information in an allowed existing field such as `experience.description`. Never convert vessel names into companies.
   * Do not invent an employer if `Owners / Manager` is absent.

4. Certification, Course, & Endorsement Tables:
   * Inspect every data row in tables describing `Courses & Certificates`, `STCW Courses`, `Dangerous Cargo Endorsements`, `Vaccinations`, or `Trainings`.
   * When a data row contains a specific certificate, course, or endorsement name, extract it as an entry in `certifications`. Each logical certification data row represents at most one certification record. Do not repeat a certification across rows. All source_block_ids for a certification must come from that same logical table row.
   * Column headers themselves must never become certification values.

5. Skills Sections & Explicit Inventories:
   * Extract skills only from an explicit skills section, technical-skills section, competencies section, technologies section, or clearly enumerated skill list.
   * Do NOT infer or synthesize skills from:
     - narrative summaries
     - job titles
     - experience descriptions
     - project descriptions
     - table rows
     - employer/company names
     - client/industry labels
   * If the resume contains no explicit skills inventory/list, return `skills: []`.
   * Every extracted skill must be grounded in the source block(s) containing that skill.
   * Do not convert concepts mentioned in prose into skills merely because they are plausible resume skills.

6. Academic CV Experience & Appointment Sections:
   * In academic experience sections (e.g., research experience, teaching experience, academic appointments), each distinct position or appointment must be emitted as a separate experience item.
   * Do NOT aggregate multiple appointments or roles under one section into a single experience object.
   * For example, a Postdoctoral Fellow role, Graduate Research Assistant role, and Teaching Assistant roles within academic sections are distinct positions and must each populate a separate `GroundedExperienceItem`.
   * Each experience item's `source_block_ids` must cover only that appointment's blocks.
   * A new `ENTRY_TITLE` represents a new position when it occurs within an experience-target section.
   * Preserve all existing provenance and grounding rules.

PROVENANCE PRECISION AND ENTITY LINKING GUIDELINES:
1. Structural Roles as Authoritative Provenance Hints:
   * ENTRY_TITLE contains the job/role title and should be cited for experience.designation.
   * ORGANIZATION contains employer/company information and should be cited for experience.company.
   * LOCATION contains location information and should be cited for experience.location.
   * DATE contains employment/education date information and should be cited for the corresponding date fields.
   * DESCRIPTION/BULLET contains narrative description or achievement text.

2. Precise Field-to-Block Grounding:
   * For every grounded field, source_block_ids MUST identify the block(s) that actually contain the emitted value.
   * Never cite an adjacent block merely because it belongs to the same experience/education/project entity.
   * Do not reuse another field's block ID as a convenient citation.

3. Entity Headers with Adjacent Header Blocks:
   * For an entity header containing adjacent title/company/location/date blocks, map each field to the block that actually contains that field.
   * designation must not cite DATE.
   * location must not cite ENTRY_TITLE.
   * company must not cite DATE.
   * dates must not cite ENTRY_TITLE.

4. Multi-Block Text:
   * If the emitted value combines text from multiple source blocks, include ALL contributing block IDs in reading order.
   * Do not cite only the first block when continuation blocks contribute text.

5. Entity Boundaries:
   * Do not assign the final description/bullet of one entry to the following entry.
   * When a new ENTRY_TITLE begins, subsequent fields belong to that new entity unless structural evidence clearly indicates otherwise.

6. Block Classifications Integrity:
   * TABLE_HEADER is only valid for an actual table/grid header represented by table metadata or unmistakable table-header structure.
   * Do not classify ENTRY_TITLE blocks as TABLE_HEADER merely because surrounding content is multi-column.
   * Preserve the existing structural role information as the primary interpretation.

IMPORTANT VALIDATION REMINDER:
These guidelines are extraction mappings grounded in explicit source relationships, not new domain facts. Every resulting value must strictly satisfy deterministic provenance and grounding validators."""


PERSONAL_EXTRACTION_SYSTEM_PROMPT = """You are a precise, layout-aware resume data extractor.
Your task is to analyze the provided structured document blocks and extract candidate personal contact and identity information strictly grounded in the supplied text blocks.

CRITICAL GROUNDING AND PROVENANCE RULES:
1. Every non-null grounded value MUST reference the exact `source_block_ids` from which it was extracted.
2. EXTRACT ONLY supported information. DO NOT invent, hallucinate, or infer missing values.
3. DO NOT perform semantic renaming or enrichment.
4. Use exact verbatim text from source blocks in `raw_value` when available. In `value`, only safe canonical normalizations are permitted (e.g., whitespace/case cleanup). For phone numbers, copy the text verbatim as written in the source block.
5. DO NOT treat document headers, form titles (e.g., 'APPLICATION FORM', 'Curriculum Vitae', 'Surname'), or section labels as personal names.
6. Return a single valid JSON object adhering strictly to the PersonalSemanticOutput schema.

PERSONAL CONTACT & IDENTITY EXTRACTION GUIDELINES:
1. Personal Labeled Fields & Contact Information:
   * Inspect labeled form fields and contact blocks across the document to populate `personal`.
   * `Email: user@example.com` populates `personal.email` with `user@example.com`, grounded to the source block.
   * `Phone: ...` populates `personal.phone`. Copy the phone text exactly as written in the source block (e.g. `+1 (415) 555-0188`). Do NOT manually retype, reconstruct, infer, or normalize phone digits in your head. Preserve country code, area code, exchange, subscriber digits, and punctuation exactly as present in the source. Downstream deterministic validation/normalization handles canonical phone representation. If uncertain, return null rather than inventing or reconstructing digits.
   * `Location: ...` populates `personal.location` strictly from header or contact blocks on page 1. Do NOT extract personal location from body employment, academic appointment, or education entries.
   * Assemble personal name tokens from explicit name fields (e.g., Surname, First Name, Middle Name) while discarding descriptor labels.
   * Classify personal identity and contact blocks explicitly under `block_classifications` with category `PERSONAL`.

IMPORTANT VALIDATION REMINDER:
Every resulting value must strictly satisfy deterministic provenance and grounding validators."""


BODY_EXTRACTION_SYSTEM_PROMPT = """You are a precise, layout-aware resume data extractor.
Your task is to analyze the provided structured document blocks and extract resume body entities (experience, education, certifications, skills, projects, languages, achievements, summary) strictly grounded in the supplied text blocks.

CRITICAL GROUNDING AND PROVENANCE RULES:
1. Every non-null grounded value MUST reference the exact `source_block_ids` from which it was extracted.
2. EXTRACT ONLY supported information. DO NOT invent, hallucinate, or infer missing values.
3. DO NOT perform semantic renaming or enrichment (e.g., do not rename companies, do not expand job titles).
4. Use exact verbatim text from source blocks in `raw_value` when available. In `value`, only safe canonical normalizations are permitted (e.g., ISO dates 'YYYY-MM-DD'/'YYYY-MM'/'YYYY', whitespace/case cleanup). When source evidence contains only a year (for example '2020' or a range such as '2020 - 2024'), preserve year precision and return '2020' and '2024'. NEVER pad a year-only source with '-01' or '-01-01'. Only include month/day when those components are explicitly present in the source evidence.
5. DO NOT treat table column headers (e.g., 'Ship Name', 'Period', 'S.No.', 'Documents Details') as actual company, designation, or degree values.
6. DO NOT classify referee or reference contacts as employment experience. Mark referee and boilerplate blocks explicitly under block_classifications.
7. Every boolean field (e.g., `current`) MUST reference the source block IDs providing explicit evidence (e.g., blocks explicitly containing 'Present', 'Current', 'Currently', 'Ongoing', 'Till Date', 'Now'). An end date or year alone (including future years, current calendar years, or date ranges without explicit current wording like 'Present' or 'Current') DOES NOT establish `current=true`. When explicit current wording is absent, set `current: null`.
8. If evidence for a field is absent or ambiguous, return null or empty list rather than guessing.
9. Return a single valid JSON object adhering strictly to the BodySemanticOutput schema.

OUTPUT COMPLETENESS & STRUCTURE REQUIREMENTS:
1. The response JSON MUST explicitly include every field defined by BodySemanticOutput: `document_archetype`, `block_classifications`, `summary`, `skills`, `experience`, `education`, `projects`, `certifications`, `languages`, `achievements`.
2. NEVER omit optional body fields from the output JSON. Use `[]` for empty collections and `null` for unavailable scalar/object values.
3. EXTRACT EVERY applicable body collection supported by the supplied evidence (including all experience, education, certifications, and skills present in document tables or sections).
4. DO NOT return only summary or document_archetype when body evidence exists across the document.

STRUCTURED FORMS AND TABLE EXTRACTION GUIDANCE:
1. Form & Archetype Completeness:
   * A document classification of `structured_form`, `maritime_tabular`, or `maritime_cv` indicates layout structure (application form or tabular seafarer profile) and does NOT mean the document should produce empty canonical fields.
   * You must inspect EVERY table data row and labeled section across the document and map all supported data to canonical entities when column headers or field labels provide the semantic mapping.

2. Table Interpretation:
   * `table`, `row`, `col`, and `cell_role` identify physical grid relationships.
   * HEADER cells (e.g., `cell_role: "HEADER"` or row 0 headers) describe the meaning/attribute of their column. Never treat a HEADER cell itself as an extracted entity value.
   * Non-header data cells in the same row form one logical record.
   * Multiple blocks sharing the same `table` + `row` + `col` belong to the same physical cell and may be combined in reading order.

3. Grounded Semantic Interpretation (Allowed vs. Prohibited):
   * Allowed (Semantic interpretation supported by explicit source relationships):
     - Mapping a data value to the canonical field whose table column or header label explicitly describes it.
     - Combining adjacent blocks belonging to the same table cell in reading order.
     - Mapping explicit table row relationships to canonical schema fields.
     - Normalizing dates to ISO format when the source contains the corresponding date components.
   * Still Prohibited (Invention / Fabrication):
     - Inventing facts or adding ungrounded details.
     - Renaming companies (e.g., 'Darya Shaan' -> 'Darya Shipping').
     - Expanding job titles (e.g., 'Cadet' -> 'Chief Officer').
     - Hallucinating duties or technologies not present in the text.
     - Inventing dates or extrapolating timelines.
     - Creating unsupported entities.
     - Fabricating source block IDs.

SPECIFIC SECTION EXTRACTION MAPPINGS:
1. Education Tables:
   * Inspect every data row in educational qualification tables. Each non-header data row populates one `GroundedEducationItem`.
   * `Name of Institute / College` / `School / College / University` -> `institution`
   * `Type of Degree` / `Degree / Certificate` / `Examination Passed` -> `degree` or `fieldOfStudy` (choose the canonical field that best matches the explicit header without inventing terminology)
   * `From` / `Commenced` -> `startDate`
   * `To` / `Completed` / `Passed` -> `endDate`
   * `Grade` / `Class` / `Division` -> `grade`

2. Maritime Sea-Service & Employment Tables:
   * Inspect every data row in tables titled `Previous Sea Service`, `Sea Service`, `Experience`, or `Employment History`. Each non-header data row represents an `ExperienceItem`.
   * `Owners / Manager` / `Company Name` / `Employer` -> `experience.company`
   * `Rank` / `Position` / `Capacity` -> `experience.designation`. For tabular experience, designation must be taken strictly from the Rank/Position column. Do not combine it with serial numbers, vessel specifications, engine models, BHP, DWT, GRT, flags, or other adjacent columns. Technical specifications belong in contextual description, when supported.
   * `From` / `Sign On` / `Date Commencing` -> `startDate`
   * `To` / `Sign Off` / `Date of S/OFF` -> `endDate`
   * `Vessel Name` is NOT the company; preserve the vessel name as grounded contextual information in an allowed existing field such as `experience.description`. Never convert vessel names into companies.
   * Do not invent an employer if `Owners / Manager` is absent.

3. Certification, Course, & Endorsement Tables:
   * Inspect every data row in tables describing `Courses & Certificates`, `STCW Courses`, `Dangerous Cargo Endorsements`, `Vaccinations`, or `Trainings`.
   * When a data row contains a specific certificate, course, or endorsement name, extract it as an entry in `certifications`. Each logical certification data row represents at most one certification record. Do not repeat a certification across rows. All source_block_ids for a certification must come from that same logical table row.
   * Column headers themselves must never become certification values.

4. Skills Sections & Explicit Inventories:
   * Extract skills only from an explicit skills section, technical-skills section, competencies section, technologies section, or clearly enumerated skill list.
   * Do NOT infer or synthesize skills from:
     - narrative summaries
     - job titles
     - experience descriptions
     - project descriptions
     - table rows
     - employer/company names
     - client/industry labels
   * If the resume contains no explicit skills inventory/list, return `skills: []`.
   * Every extracted skill must be grounded in the source block(s) containing that skill.
   * Do not convert concepts mentioned in prose into skills merely because they are plausible resume skills.

5. Academic CV Experience & Appointment Sections:
   * In academic experience sections (e.g., research experience, teaching experience, academic appointments), each distinct position or appointment must be emitted as a separate experience item.
   * Do NOT aggregate multiple appointments or roles under one section into a single experience object.
   * For example, a Postdoctoral Fellow role, Graduate Research Assistant role, and Teaching Assistant roles within academic sections are distinct positions and must each populate a separate `GroundedExperienceItem`.
   * Each experience item's `source_block_ids` must cover only that appointment's blocks.
   * A new `ENTRY_TITLE` represents a new position when it occurs within an experience-target section.
   * Preserve all existing provenance and grounding rules.

PROVENANCE PRECISION AND ENTITY LINKING GUIDELINES:
1. Structural Roles as Authoritative Provenance Hints:
   * ENTRY_TITLE contains the job/role title and should be cited for experience.designation.
   * ORGANIZATION contains employer/company information and should be cited for experience.company.
   * LOCATION contains location information and should be cited for experience.location.
   * DATE contains employment/education date information and should be cited for the corresponding date fields.
   * DESCRIPTION/BULLET contains narrative description or achievement text.

2. Precise Field-to-Block Grounding:
   * For every grounded field, source_block_ids MUST identify the block(s) that actually contain the emitted value.
   * Never cite an adjacent block merely because it belongs to the same experience/education/project entity.
   * Do not reuse another field's block ID as a convenient citation.

3. Entity Headers with Adjacent Header Blocks:
   * For an entity header containing adjacent title/company/location/date blocks, map each field to the block that actually contains that field.
   * designation must not cite DATE.
   * location must not cite ENTRY_TITLE.
   * company must not cite DATE.
   * dates must not cite ENTRY_TITLE.

4. Multi-Block Text:
   * If the emitted value combines text from multiple source blocks, include ALL contributing block IDs in reading order.
   * Do not cite only the first block when continuation blocks contribute text.

5. Entity Boundaries:
   * Do not assign the final description/bullet of one entry to the following entry.
   * When a new ENTRY_TITLE begins, subsequent fields belong to that new entity unless structural evidence clearly indicates otherwise.

6. Block Classifications Integrity:
   * TABLE_HEADER is only valid for an actual table/grid header represented by table metadata or unmistakable table-header structure.
   * Do not classify ENTRY_TITLE blocks as TABLE_HEADER merely because surrounding content is multi-column.
   * Preserve the existing structural role information as the primary interpretation.

IMPORTANT VALIDATION REMINDER:
These guidelines are extraction mappings grounded in explicit source relationships, not new domain facts. Every resulting value must strictly satisfy deterministic provenance and grounding validators."""


def serialize_semantic_input_full(semantic_input: SemanticInput) -> str:
    """Reference / debug serialization preserving all layout, coordinates, and full metadata.

    Preserves exact block IDs, coordinates, style hints, reading order, suggested
    roles, and table metadata without redundant block copies.
    """
    # Sort blocks deterministically by page, reading_order, and block_id
    sorted_blocks = sorted(
        semantic_input.blocks,
        key=lambda b: (b.page, b.reading_order, b.block_id),
    )

    payload: dict[str, Any] = {
        "document_id": semantic_input.document_id,
        "page_count": semantic_input.page_count,
        "pages": [
            {
                "page_number": p.page_number,
                "width": p.width,
                "height": p.height,
            }
            for p in sorted(semantic_input.pages, key=lambda p: p.page_number)
        ],
        "blocks": [
            {
                "block_id": b.block_id,
                "text": b.text,
                "page": b.page,
                "bbox": b.bbox,
                "region_id": b.region_id,
                "region_kind": b.region_kind,
                "reading_order": b.reading_order,
                "column_id": b.column_id,
                "is_bold": b.is_bold,
                "font_size": b.font_size,
                "suggested_role": b.suggested_role,
                "table_id": b.table_id,
                "row_index": b.row_index,
                "column_index": b.column_index,
                "cell_role": b.cell_role,
            }
            for b in sorted_blocks
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# Backward-compatible alias for existing tests and debug inspection
serialize_semantic_input = serialize_semantic_input_full


def build_full_extraction_prompt(semantic_input: SemanticInput) -> str:
    """Construct reference extraction prompt combining instructions and full serialized input."""
    serialized_input = serialize_semantic_input_full(semantic_input)
    return (
        f"{SEMANTIC_EXTRACTION_SYSTEM_PROMPT}\n\n"
        f"DOCUMENT BLOCKS (JSON):\n"
        f"```json\n{serialized_input}\n```\n\n"
        f"Extract the resume data as a JSON object adhering strictly to the SemanticOutput schema."
    )


def _strip_markdown_code_fences(text: str) -> str:
    """Strip ```json ... ``` code fences if model enclosed JSON response at the boundary.

    Does NOT accept arbitrary conversational prose surrounding the JSON.
    """
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def parse_semantic_output(raw_json_or_text: str | dict[str, Any]) -> SemanticOutput:
    """Parse raw model JSON response into the constrained SemanticOutput domain model.

    Raises:
        SemanticExtractionError: If JSON is malformed or structure violates SemanticOutput schema.
    """
    if isinstance(raw_json_or_text, str):
        cleaned_text = _strip_markdown_code_fences(raw_json_or_text)
        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as err:
            raise SemanticExtractionError(f"Malformed JSON response from LLM: {err}") from err
    elif isinstance(raw_json_or_text, dict):
        parsed = raw_json_or_text
    else:
        raise SemanticExtractionError(
            f"Expected JSON string or dict, got {type(raw_json_or_text).__name__}"
        )

    if not isinstance(parsed, dict):
        raise SemanticExtractionError(
            f"Expected JSON object at root of LLM response, got {type(parsed).__name__}"
        )

    try:
        return SemanticOutput.model_validate(parsed)
    except ValidationError as err:
        raise SemanticExtractionError(f"Invalid SemanticOutput structure from LLM: {err}") from err


def parse_personal_output(raw_json_or_text: str | dict[str, Any]) -> PersonalSemanticOutput:
    """Parse raw model JSON response into the constrained PersonalSemanticOutput domain model."""
    if isinstance(raw_json_or_text, str):
        cleaned_text = _strip_markdown_code_fences(raw_json_or_text)
        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as err:
            raise SemanticExtractionError(f"Malformed JSON response from LLM (personal pass): {err}") from err
    elif isinstance(raw_json_or_text, dict):
        parsed = raw_json_or_text
    else:
        raise SemanticExtractionError(
            f"Expected JSON string or dict, got {type(raw_json_or_text).__name__}"
        )

    if not isinstance(parsed, dict):
        raise SemanticExtractionError(
            f"Expected JSON object at root of LLM response, got {type(parsed).__name__}"
        )

    try:
        return PersonalSemanticOutput.model_validate(parsed)
    except ValidationError as err:
        raise SemanticExtractionError(f"Invalid PersonalSemanticOutput structure from LLM: {err}") from err


def parse_body_output(raw_json_or_text: str | dict[str, Any]) -> BodySemanticOutput:
    """Parse raw model JSON response into the constrained BodySemanticOutput domain model."""
    if isinstance(raw_json_or_text, str):
        cleaned_text = _strip_markdown_code_fences(raw_json_or_text)
        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as err:
            raise SemanticExtractionError(f"Malformed JSON response from LLM (body pass): {err}") from err
    elif isinstance(raw_json_or_text, dict):
        parsed = raw_json_or_text
    else:
        raise SemanticExtractionError(
            f"Expected JSON string or dict, got {type(raw_json_or_text).__name__}"
        )

    if not isinstance(parsed, dict):
        raise SemanticExtractionError(
            f"Expected JSON object at root of LLM response, got {type(parsed).__name__}"
        )

    try:
        return BodySemanticOutput.model_validate(parsed)
    except ValidationError as err:
        raise SemanticExtractionError(f"Invalid BodySemanticOutput structure from LLM: {err}") from err


def resolve_schema_defs(pydantic_model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model's JSON schema to a self-contained schema with all $defs inlined.

    Useful for providers (Ollama, Gemini) that support structured JSON output with an inlined schema.
    """
    raw_schema = pydantic_model.model_json_schema()
    defs = raw_schema.pop("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                if ref_key in defs:
                    resolved = _resolve(defs[ref_key])
                    merged = dict(resolved)
                    for k, v in node.items():
                        if k != "$ref":
                            merged[k] = _resolve(v)
                    return merged
            res = {}
            for k, v in node.items():
                if k in ("title", "$defs"):
                    continue
                res[k] = _resolve(v)
            return res
        elif isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(raw_schema)


def serialize_compact_semantic_input(semantic_input: SemanticInput) -> str:
    """Serialize SemanticInput into Candidate B compact JSON representation.

    For STANDARD_CV:
      Preserves deterministic block IDs, verbatim text, structural roles, multi-page indicators,
      and tabular grid coordinates without redundant bounding boxes, style metadata, or explicit nulls.

    For complex archetypes (MARITIME_CV, MARITIME_TABULAR, STRUCTURED_FORM, ACADEMIC_CV, etc.):
      Preserves deterministic region/column structural context (region_id, region_kind, column_id)
      and region-aware deterministic ordering so multi-column form tables and bio-data fields
      retain layout grouping without fabricating synthetic table rows or cells.
    """
    is_standard = semantic_input.archetype == DocumentArchetype.STANDARD_CV

    if is_standard:
        sorted_blocks = sorted(
            semantic_input.blocks,
            key=lambda b: (b.page, b.reading_order, b.block_id),
        )

        compact_blocks: list[dict[str, Any]] = []
        for b in sorted_blocks:
            block_dict: dict[str, Any] = {
                "id": b.block_id,
                "text": b.text,
            }
            if b.suggested_role and b.suggested_role != "UNKNOWN":
                block_dict["role"] = b.suggested_role
            if b.page > 1:
                block_dict["page"] = b.page
            if b.is_bold is True:
                block_dict["bold"] = True
            if b.table_id is not None:
                block_dict["table"] = b.table_id
                if b.row_index is not None:
                    block_dict["row"] = b.row_index
                if b.column_index is not None:
                    block_dict["col"] = b.column_index
                if b.cell_role and b.cell_role != "DATA":
                    block_dict["cell_role"] = b.cell_role
            compact_blocks.append(block_dict)

        payload: dict[str, Any] = {
            "doc_id": semantic_input.document_id,
            "blocks": compact_blocks,
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    # Complex archetypes (MARITIME_CV, MARITIME_TABULAR, STRUCTURED_FORM, ACADEMIC_CV, etc.)
    # Compute table anchors on each page so tables appear in deterministic page reading order,
    # while table internal blocks are grouped row-major: table_id -> row_index -> col_index -> reading_order -> block_id
    table_anchors: dict[tuple[int, str], int] = {}
    for b in semantic_input.blocks:
        if b.table_id is not None:
            key = (b.page, b.table_id)
            if key not in table_anchors:
                table_anchors[key] = b.reading_order
            else:
                table_anchors[key] = min(table_anchors[key], b.reading_order)

    def _complex_block_sort_key(b: SemanticBlockInput) -> tuple[Any, ...]:
        if b.table_id is not None:
            anchor_order = table_anchors.get((b.page, b.table_id), b.reading_order)
            return (
                b.page,
                anchor_order,
                b.table_id,
                1,  # Table block
                b.row_index if b.row_index is not None else 0,
                b.column_index if b.column_index is not None else 0,
                b.reading_order,
                b.block_id,
            )
        else:
            return (
                b.page,
                b.reading_order,
                b.region_id or "",
                0,  # Non-table block
                b.column_id if b.column_id is not None else -1,
                0,
                b.reading_order,
                b.block_id,
            )

    sorted_blocks = sorted(semantic_input.blocks, key=_complex_block_sort_key)

    compact_blocks = []
    for b in sorted_blocks:
        block_dict = {
            "id": b.block_id,
            "text": b.text,
        }
        if b.suggested_role and b.suggested_role != "UNKNOWN":
            block_dict["role"] = b.suggested_role
        if b.page > 1 or semantic_input.page_count > 1:
            block_dict["page"] = b.page
        if b.region_id:
            block_dict["region"] = b.region_id
        if b.region_kind and b.region_kind != "physical_region":
            block_dict["kind"] = b.region_kind
        if b.column_id is not None:
            block_dict["col"] = b.column_id
        if b.is_bold is True:
            block_dict["bold"] = True
        if b.table_id is not None:
            block_dict["table"] = b.table_id
            if b.row_index is not None:
                block_dict["row"] = b.row_index
            if b.column_index is not None:
                block_dict["col"] = b.column_index
            if b.cell_role and b.cell_role != "DATA":
                block_dict["cell_role"] = b.cell_role
        compact_blocks.append(block_dict)

    payload = {
        "doc_id": semantic_input.document_id,
        "archetype": semantic_input.archetype.value,
        "blocks": compact_blocks,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def get_compact_schema(pydantic_model: type[BaseModel] = SemanticOutput) -> dict[str, Any]:
    """Generate a compact JSON schema with descriptions and non-structural metadata stripped."""
    raw_schema = resolve_schema_defs(pydantic_model)

    def _strip_metadata(node: Any) -> Any:
        if isinstance(node, dict):
            res: dict[str, Any] = {}
            for k, v in node.items():
                if k in ("title", "description", "default", "examples", "format"):
                    continue
                res[k] = _strip_metadata(v)
            return res
        elif isinstance(node, list):
            return [_strip_metadata(item) for item in node]
        return node

    return _strip_metadata(raw_schema)


def build_compact_extraction_prompt(semantic_input: SemanticInput) -> str:
    """Construct extraction prompt using Candidate B compact input representation."""
    serialized_input = serialize_compact_semantic_input(semantic_input)
    return (
        f"{SEMANTIC_EXTRACTION_SYSTEM_PROMPT}\n\n"
        f"DOCUMENT BLOCKS (JSON):\n"
        f"```json\n{serialized_input}\n```\n\n"
        f"Extract the resume data as a JSON object adhering strictly to the SemanticOutput schema."
    )


# Production default prompt builder (Candidate B compact representation)
build_extraction_prompt = build_compact_extraction_prompt


def build_personal_extraction_prompt(semantic_input: SemanticInput) -> str:
    """Construct focused Pass 1 extraction prompt for personal contact and identity data."""
    serialized_input = serialize_compact_semantic_input(semantic_input)
    return (
        f"{PERSONAL_EXTRACTION_SYSTEM_PROMPT}\n\n"
        f"DOCUMENT BLOCKS (JSON):\n"
        f"```json\n{serialized_input}\n```\n\n"
        f"Extract the personal data as a JSON object adhering strictly to the PersonalSemanticOutput schema."
    )


def build_body_extraction_prompt(semantic_input: SemanticInput) -> str:
    """Construct focused Pass 2 extraction prompt for resume body entities (exp, edu, certs, skills, etc.)."""
    serialized_input = serialize_compact_semantic_input(semantic_input)
    return (
        f"{BODY_EXTRACTION_SYSTEM_PROMPT}\n\n"
        f"DOCUMENT BLOCKS (JSON):\n"
        f"```json\n{serialized_input}\n```\n\n"
        f"Extract the resume body data as a JSON object adhering strictly to the BodySemanticOutput schema."
    )


def build_body_recovery_prompt(semantic_input: SemanticInput) -> str:
    """Construct focused recovery prompt for Pass 2 when initial body output was suspiciously empty.

    Surfaces deterministic structural evidence summary and instructs the model to inspect
    evidence blocks individually without omitting body collections.
    """
    base_prompt = build_body_extraction_prompt(semantic_input)
    evidence = summarize_body_evidence(semantic_input)

    lines = [
        base_prompt,
        "",
        "CRITICAL RECOVERY INSTRUCTION (EVIDENCE-DIRECTED):",
        "The previous extraction returned empty lists for all body collections (skills, experience, education, projects, certifications, etc.) despite rich body content in the supplied blocks.",
        f"Deterministic structural analysis identified {evidence['total_body_blocks']} body evidence blocks in this document:",
    ]

    if evidence["section_headings"]:
        lines.append(f"- Section Heading Blocks: {', '.join(evidence['section_headings'])}")

    if evidence["roles"]:
        lines.append("- Evidence by Structural Role:")
        for role, b_ids in evidence["roles"].items():
            if role == "SECTION_HEADING":
                continue
            sample_ids = b_ids if len(b_ids) <= 12 else b_ids[:12]
            suffix = f" (showing first 12 of {len(b_ids)})" if len(b_ids) > 12 else ""
            lines.append(f"  * {role} ({len(b_ids)} blocks){suffix}: {', '.join(sample_ids)}")

    if evidence["table_blocks"]:
        sample_tbl = evidence["table_blocks"] if len(evidence["table_blocks"]) <= 12 else evidence["table_blocks"][:12]
        suffix = f" (showing first 12 of {len(evidence['table_blocks'])})" if len(evidence['table_blocks']) > 12 else ""
        lines.append(f"- Table Data Blocks ({len(evidence['table_blocks'])} blocks){suffix}: {', '.join(sample_tbl)}")

    lines.extend([
        "",
        "MANDATORY EXTRACTION REQUIREMENTS FOR RECOVERY:",
        "1. Inspect the identified evidence blocks individually and extract all applicable entities (experience, education, skills, projects, certifications, languages, achievements) supported by the source text.",
        "2. DO NOT return only a summary or document archetype when body evidence exists.",
        "3. Every non-null extracted value MUST cite the exact `source_block_ids` from which it was extracted.",
        "4. DO NOT perform semantic renaming, title expansion, or ungrounded inference.",
        "5. Set `current: true` ONLY when source blocks contain explicit wording such as 'Present', 'Current', 'Currently', 'Ongoing', 'Till Date', or 'Now'. An end date/year alone (e.g. '2021 - 2026') does NOT support `current=true`; use `current: null`.",
        "6. Use `[]` for collections where no source evidence exists and `null` for missing scalar/object fields.",
        "7. Return a single valid JSON object adhering strictly to the BodySemanticOutput schema.",
    ])

    return "\n".join(lines)


def get_personal_schema() -> dict[str, Any]:
    """Return inlined JSON schema for PersonalSemanticOutput."""
    return get_compact_schema(PersonalSemanticOutput)


def get_body_schema() -> dict[str, Any]:
    """Return inlined JSON schema for BodySemanticOutput."""
    return get_compact_schema(BodySemanticOutput)
