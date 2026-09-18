#!/usr/bin/env python3
"""Temporary diagnostic script: Two-call schema split extraction with Gemini 3.6 Flash on AKIBUL ALAM Candidate B input."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.domain.document import document_from_text_blocks
from app.domain.semantic_contract import build_semantic_input
from app.extractors.semantic_prompt import (
    resolve_schema_defs,
    serialize_compact_semantic_input,
)
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.text_extraction import PDFExtractor

load_dotenv()


# =====================================================================
# 1. Pydantic Schemas for Split Calls
# =====================================================================

class GroundedString(BaseModel):
    value: str
    raw_value: str | None = None
    source_block_ids: list[str] = Field(default_factory=list)


class GroundedBool(BaseModel):
    value: bool
    source_block_ids: list[str] = Field(default_factory=list)


# Call 1 Schema: Personal Details Only
class SplitPersonalOutput(BaseModel):
    name: GroundedString | None = None
    email: GroundedString | None = None
    phone: GroundedString | None = None
    location: GroundedString | None = None
    linkedin: GroundedString | None = None
    github: GroundedString | None = None
    portfolio: GroundedString | None = None


# Call 2 Schema: Resume Body Entities Only
class GroundedExperienceItem(BaseModel):
    company: GroundedString | None = None
    designation: GroundedString | None = None
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    current: GroundedBool | None = None
    location: GroundedString | None = None
    description: GroundedString | None = None
    technologies: list[GroundedString] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)


class GroundedEducationItem(BaseModel):
    institution: GroundedString | None = None
    degree: GroundedString | None = None
    fieldOfStudy: GroundedString | None = None
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    grade: GroundedString | None = None
    source_block_ids: list[str] = Field(default_factory=list)


class GroundedProjectItem(BaseModel):
    name: GroundedString | None = None
    description: GroundedString | None = None
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    current: GroundedBool | None = None
    technologies: list[GroundedString] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)


class SplitBodyOutput(BaseModel):
    summary: GroundedString | None = None
    skills: list[GroundedString] = Field(default_factory=list)
    experience: list[GroundedExperienceItem] = Field(default_factory=list)
    education: list[GroundedEducationItem] = Field(default_factory=list)
    projects: list[GroundedProjectItem] = Field(default_factory=list)
    certifications: list[GroundedString] = Field(default_factory=list)
    languages: list[GroundedString] = Field(default_factory=list)
    achievements: list[GroundedString] = Field(default_factory=list)


# =====================================================================
# 2. Prompts for Split Calls
# =====================================================================

PERSONAL_PROMPT_TEMPLATE = """You are a precise, layout-aware resume data extractor.
Your task is to analyze the provided structured document blocks and extract candidate personal contact and identity information strictly grounded in the supplied text blocks.

CRITICAL GROUNDING AND PROVENANCE RULES:
1. Every non-null grounded value MUST reference the exact `source_block_ids` from which it was extracted.
2. EXTRACT ONLY supported information. DO NOT invent, hallucinate, or infer missing values.
3. DO NOT perform semantic renaming or enrichment.
4. Use exact verbatim text from source blocks in `raw_value` when available. In `value`, only safe canonical normalizations are permitted (e.g., phone digits, whitespace/case cleanup).
5. DO NOT treat document headers, form titles (e.g., 'APPLICATION FORM', 'Curriculum Vitae', 'Surname'), or section labels as personal names.
6. Return a single valid JSON object adhering strictly to the SplitPersonalOutput schema.

PERSONAL CONTACT & IDENTITY EXTRACTION GUIDELINES:
- Inspect labeled form fields and contact blocks across the document to populate personal information.
- `Email: user@example.com` populates `email` with `user@example.com`, grounded to the source block.
- `Phone: ...` populates `phone` with normalized phone digits grounded to the source blocks.
- Assemble personal name tokens from explicit name fields (e.g., Surname, First Name, Middle Name) while discarding descriptor labels.

DOCUMENT BLOCKS (JSON):
```json
{serialized_input}
```

Extract the personal data as a JSON object adhering strictly to the SplitPersonalOutput schema."""


BODY_PROMPT_TEMPLATE = """You are a precise, layout-aware resume data extractor.
Your task is to analyze the provided structured document blocks and extract resume body entities (experience, education, certifications, skills, projects, languages, achievements, summary) strictly grounded in the supplied text blocks.

CRITICAL GROUNDING AND PROVENANCE RULES:
1. Every non-null grounded value MUST reference the exact `source_block_ids` from which it was extracted.
2. EXTRACT ONLY supported information. DO NOT invent, hallucinate, or infer missing values.
3. DO NOT perform semantic renaming or enrichment (e.g., do not rename companies, do not expand job titles).
4. Use exact verbatim text from source blocks in `raw_value` when available. In `value`, only safe canonical normalizations are permitted (e.g., ISO dates 'YYYY-MM-DD'/'YYYY-MM'/'YYYY', whitespace/case cleanup).
5. DO NOT treat table column headers (e.g., 'Ship Name', 'Period', 'S.No.', 'Documents Details') as actual company, designation, or degree values.
6. DO NOT classify referee or reference contacts as employment experience.
7. Return a single valid JSON object adhering strictly to the SplitBodyOutput schema.

STRUCTURED FORMS AND TABLE EXTRACTION GUIDANCE:
1. Form & Archetype Completeness:
   * A document classification of `structured_form`, `maritime_tabular`, or `maritime_cv` indicates layout structure (application form or tabular seafarer profile) and does NOT mean the document should produce empty canonical fields.
   * You must inspect EVERY table data row across the document and map all supported data to canonical entities when column headers provide the semantic mapping.
2. Table Interpretation:
   * `table`, `row`, `col`, and `cell_role` identify physical grid relationships.
   * HEADER cells (e.g., `cell_role: "HEADER"` or row 0 headers) describe the meaning/attribute of their column. Never treat a HEADER cell itself as an extracted entity value.
   * Non-header data cells in the same row form one logical record.
   * Multiple blocks sharing the same `table` + `row` + `col` belong to the same physical cell and may be combined in reading order.
3. Education Tables:
   * Inspect every data row in educational qualification tables. Each non-header data row populates one `GroundedEducationItem`.
   * `Name of Institute / College` / `School / College / University` -> `institution`
   * `Type of Degree` / `Degree / Certificate` / `Examination Passed` -> `degree` or `fieldOfStudy`
   * `From` / `Commenced` -> `startDate`
   * `To` / `Completed` / `Passed` -> `endDate`
   * `Grade` / `Class` / `Division` -> `grade`
4. Maritime Sea-Service & Employment Tables:
   * Inspect every data row in tables titled `Previous Sea Service`, `Sea Service`, `Experience`, or `Employment History`. Each non-header data row represents an `ExperienceItem`.
   * `Owners / Manager` / `Company Name` / `Employer` -> `experience.company`
   * `Rank` / `Position` / `Capacity` -> `experience.designation`
   * `From` / `Sign On` / `Date Commencing` -> `startDate`
   * `To` / `Sign Off` / `Date of S/OFF` -> `endDate`
   * `Vessel Name` is NOT the company; preserve the vessel name as grounded contextual information in an allowed existing field such as `experience.description`. Never convert vessel names into companies.
   * Do not invent an employer if `Owners / Manager` is absent.
5. Certification, Course, & Endorsement Tables:
   * Inspect every data row in tables describing `Courses & Certificates`, `STCW Courses`, `Dangerous Cargo Endorsements`, `Vaccinations`, or `Trainings`.
   * When a data row contains a specific certificate, course, or endorsement name, extract it as an entry in `certifications`.
   * Column headers themselves must never become certification values.

DOCUMENT BLOCKS (JSON):
```json
{serialized_input}
```

Extract the resume body data as a JSON object adhering strictly to the SplitBodyOutput schema."""


# =====================================================================
# 3. Main Execution Function
# =====================================================================

def run_diagnostic() -> None:
    # 1. Resolve model & API key
    model = os.environ.get("GEMINI_MODEL") or "gemini-3.6-flash"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            settings = Settings()
            api_key = settings.gemini_api_key
        except Exception:
            pass

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment or settings.")

    # 2. Load and build exact Candidate B input
    fixture_path = Path("tests/fixtures/AKIBUL ALAM CV(JO).pdf")
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    raw_bytes = fixture_path.read_bytes()
    extracted = PDFExtractor.extract(raw_bytes)
    doc = document_from_text_blocks(extracted)
    reconstructed = reconstruct_document(doc)
    layout = interpret_layout(reconstructed)
    sem_input = build_semantic_input(layout, document_id=fixture_path.name)

    serialized_input = serialize_compact_semantic_input(sem_input)
    payload_char_count = len(serialized_input)
    block_count = len(sem_input.blocks)

    # Count distinct tables
    table_ids = {b.table_id for b in sem_input.blocks if b.table_id}
    number_of_tables = len(table_ids)

    # Print initial metrics
    print("=" * 70)
    print("GEMINI SCHEMA SPLIT DIAGNOSTIC (AKIBUL ALAM CV)")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Serialized Payload Character Count: {payload_char_count}")
    print(f"Block Count: {block_count}")
    print(f"Number of Tables: {number_of_tables} ({', '.join(sorted(table_ids))})")
    print("=" * 70)

    endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    # =================================================================
    # CALL 1: PERSONAL
    # =================================================================
    print("\n" + "#" * 70)
    print("CALL 1 — PERSONAL (name, email, phone, location, links)")
    print("#" * 70)

    personal_prompt = PERSONAL_PROMPT_TEMPLATE.format(serialized_input=serialized_input)
    personal_schema = resolve_schema_defs(SplitPersonalOutput)

    personal_payload = {
        "contents": [{"role": "user", "parts": [{"text": personal_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": personal_schema,
            "temperature": 0.0,
        },
    }

    t0 = time.monotonic()
    with httpx.Client(timeout=90.0) as client:
        resp1 = client.post(endpoint_url, headers=headers, json=personal_payload)
        resp1.raise_for_status()
        res1_json = resp1.json()
    t1_ms = (time.monotonic() - t0) * 1000.0

    usage1 = res1_json.get("usageMetadata", {})
    candidates1 = res1_json.get("candidates", [])
    raw_text1 = candidates1[0]["content"]["parts"][0]["text"] if candidates1 else ""

    print(f"Call 1 Latency: {t1_ms:.2f} ms")
    print(f"Call 1 Usage: prompt_tokens={usage1.get('promptTokenCount')}, output_tokens={usage1.get('candidatesTokenCount')}, total_tokens={usage1.get('totalTokenCount')}")
    print("\nRaw JSON Response (CALL 1 - PERSONAL):")
    try:
        print(json.dumps(json.loads(raw_text1), indent=2))
    except Exception:
        print(raw_text1)

    # =================================================================
    # CALL 2: RESUME BODY
    # =================================================================
    print("\n" + "#" * 70)
    print("CALL 2 — RESUME BODY (experience, education, certifications, skills, etc.)")
    print("#" * 70)

    body_prompt = BODY_PROMPT_TEMPLATE.format(serialized_input=serialized_input)
    body_schema = resolve_schema_defs(SplitBodyOutput)

    body_payload = {
        "contents": [{"role": "user", "parts": [{"text": body_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": body_schema,
            "temperature": 0.0,
        },
    }

    t2_start = time.monotonic()
    with httpx.Client(timeout=90.0) as client:
        resp2 = client.post(endpoint_url, headers=headers, json=body_payload)
        resp2.raise_for_status()
        res2_json = resp2.json()
    t2_ms = (time.monotonic() - t2_start) * 1000.0

    usage2 = res2_json.get("usageMetadata", {})
    candidates2 = res2_json.get("candidates", [])
    raw_text2 = candidates2[0]["content"]["parts"][0]["text"] if candidates2 else ""

    print(f"Call 2 Latency: {t2_ms:.2f} ms")
    print(f"Call 2 Usage: prompt_tokens={usage2.get('promptTokenCount')}, output_tokens={usage2.get('candidatesTokenCount')}, total_tokens={usage2.get('totalTokenCount')}")
    print("\nRaw JSON Response (CALL 2 - RESUME BODY):")
    try:
        print(json.dumps(json.loads(raw_text2), indent=2))
    except Exception:
        print(raw_text2)

    # =================================================================
    # SUMMARY
    # =================================================================
    total_latency_ms = t1_ms + t2_ms
    p_tokens1 = usage1.get("promptTokenCount", 0) or 0
    p_tokens2 = usage2.get("promptTokenCount", 0) or 0
    o_tokens1 = usage1.get("candidatesTokenCount", 0) or 0
    o_tokens2 = usage2.get("candidatesTokenCount", 0) or 0
    tot_tokens1 = usage1.get("totalTokenCount", 0) or 0
    tot_tokens2 = usage2.get("totalTokenCount", 0) or 0

    print("\n" + "=" * 70)
    print("CUMULATIVE PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Call 1 (Personal):  latency={t1_ms:.2f} ms | prompt={p_tokens1} | output={o_tokens1} | total={tot_tokens1}")
    print(f"Call 2 (Body):      latency={t2_ms:.2f} ms | prompt={p_tokens2} | output={o_tokens2} | total={tot_tokens2}")
    print(f"Total:              latency={total_latency_ms:.2f} ms | prompt={p_tokens1 + p_tokens2} | output={o_tokens1 + o_tokens2} | total={tot_tokens1 + tot_tokens2}")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()
