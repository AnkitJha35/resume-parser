# resume-parser — Copilot Phase Prompts

Use these prompts one at a time, in order, in the `resume-parser` repo.
Each phase assumes all previous phases in this list are already implemented.
Pair with `.github/copilot-instructions.md` (or `copilot-instructions.md`
in this repo root) for background context Copilot should always have.

Do not combine phases in a single Copilot request.

---

## PHASE 6 — Python Foundation

Implement only the Python project foundation for the resume parser service.

Set up:
- FastAPI project structure per: app/main.py, app/core/{config.py, logging.py, exceptions.py}, app/domain/, app/pipeline/, app/extractors/, app/resources/, app/infrastructure/{kafka/, storage/}
- core/config.py: load settings (Kafka brokers, MinIO endpoint/credentials, topic names), and include a LOW_CONFIDENCE_THRESHOLD = 0.70 constant
- Base Pydantic models for the Resume domain (empty/skeleton is fine, will be filled in later phases)
- Kafka consumer skeleton that can consume from resume.parse.requested (no processing logic yet, just log the received event)
- MinIO client wrapper (download PDF by storage key)
- GET /health endpoint

Add basic tests: config loads correctly, health endpoint responds, Kafka consumer can be instantiated.

Do not implement PDF parsing.
Do not implement any extractors.
Do not implement Kafka result publishing yet.
Do not add spaCy, OCR, or LLM libraries.
Do not implement any public resume API — this service is Kafka + /health only.

---

## PHASE 7 — PDF Extraction

Implement only PyMuPDF-based PDF extraction.

Create PDFExtractor in app/pipeline/stages/text_extraction.py (and pdf_detection.py for type detection).

Behavior:
- pdf_detection: validate the file is a real PDF, detect whether it has enough embedded text vs. being image-only/scanned (do not implement OCR — just flag insufficient text for later)
- text_extraction: extract text blocks using PyMuPDF, each containing: text, page number, x0, y0, x1, y1, font size (when available), bold flag (when available)
- Do not flatten text into a single string — preserve the block/layout structure

Add unit tests using a few sample PDF fixtures (single-column, two-column, and a mostly-image PDF for the insufficient-text case).

Do not implement OCR.
Do not implement text normalization.
Do not implement section detection or any resume-specific extraction.
Do not modify Phase 6 foundation files beyond adding these two stage modules.

---

## PHASE 8 — Text Normalization

Implement only TextNormalizer in app/pipeline/stages/normalization.py.

Input: raw text blocks from Phase 7's PDFExtractor.

Behavior:
- Normalize line endings
- Collapse unnecessary whitespace
- Remove common PDF extraction artifacts (e.g. stray control characters, ligature glitches)
- Preserve meaningful line boundaries
- Preserve emails, URLs, phone numbers, dates, and technical terms exactly as extracted — do not alter their formatting
- Do not aggressively rewrite or reflow content

Add unit tests covering whitespace collapsing, artifact removal, and preservation of emails/URLs/phone numbers/dates through normalization.

Do not implement section detection.
Do not implement any extractors.
Do not modify PDFExtractor from Phase 7.

---

## PHASE 9 — Section Detection

Implement only SectionDetector in app/pipeline/stages/sections.py.

Input: normalized text blocks from Phase 8.

Behavior:
- Deterministically detect these sections: SUMMARY, EXPERIENCE, EDUCATION, SKILLS, PROJECTS, CERTIFICATIONS, ACHIEVEMENTS, LANGUAGES
- Load section header aliases from app/resources/section_aliases.json (create this file with the alias lists for each section — do not hard-code aliases in code)
- Use layout information (font size, bold, position) plus alias matching to identify section boundaries
- Return a mapping of section name → list of text blocks belonging to that section

Add unit tests covering: single-column resume, two-column resume, a resume using nonstandard section headers (e.g. "Professional Experience" instead of "Experience"), and a resume missing some sections entirely.

Do not implement contact, skills, experience, education, or any other extractor.
Do not modify TextNormalizer or PDFExtractor.
Do not hard-code aliases anywhere outside section_aliases.json.

---

## PHASE 10 — Contact Extraction

Implement only ContactExtractor in app/extractors/contact.py.

Use the existing Resume domain model. Use section-detected blocks from Phase 9 (contact info is usually near the top, outside/before detected sections).

Extract: name, email, phone, location, LinkedIn, GitHub, portfolio.

Use deterministic techniques only: regex for email/phone/URLs, layout information (top-of-page position, font size) for name detection, simple heuristics for location.

Apply confidence values per the spec: email regex 0.99, phone regex 0.95, header name 0.90. Return each field as {value, confidence, source} — a centralized ConfidenceScorer will be added later in Phase 17, so for now just attach these fixed confidence values directly.

Add unit tests covering: resume with all contact fields, resume without email, resume without phone, resume with LinkedIn/GitHub/portfolio links.

Do not modify SkillsExtractor, ExperienceExtractor, or any other extractor (not yet created).
Do not add OCR.
Do not add an LLM.
Do not add external APIs.
Do not modify SectionDetector or TextNormalizer.

---

## PHASE 11 — Skills Extraction

Implement only SkillsExtractor in app/extractors/skills.py, plus app/resources/skills.json.

skills.json: a dictionary mapping lowercase aliases to canonical skill names (e.g. "spring boot": "Spring Boot", "springboot": "Spring Boot", "postgres": "PostgreSQL"). Seed it with a reasonably broad starter set of common tech skills.

SkillsExtractor must:
1. Normalize input text
2. Find known skills via dictionary lookup against skills.json
3. Normalize aliases to canonical names
4. Remove duplicates
5. Return canonical skill names with confidence 0.90 (skill dictionary match) and source "skills_section" or "full_text" depending on where found

Add unit tests covering alias normalization, duplicate removal, and skills found outside the SKILLS section (e.g. mentioned in experience descriptions).

Do not use external APIs or an LLM.
Do not modify ContactExtractor.
Do not implement ExperienceExtractor, EducationExtractor, or ProjectExtractor yet.

---

## PHASE 12 — Date Range Parsing

Implement only DateRangeParser in a shared/reusable location (e.g. app/extractors/date_parser.py or an appropriate module in app/pipeline/stages — place it where ExperienceExtractor and EducationExtractor can both import it without duplication).

Must support parsing these formats into structured start/end dates plus a "current" flag:
- Jan 2022 - Present
- January 2022 - March 2024
- 2022 - 2024
- 06/2022 - Present
- Jun 2022 - Current

Return a Pydantic model with startDate, endDate (nullable), current (bool).

Add unit tests for every format above, plus at least one malformed/unparseable input to confirm graceful failure (return None or low confidence rather than throwing).

Do not implement ExperienceExtractor or EducationExtractor yet — this is a standalone reusable component only.
Do not duplicate this logic elsewhere.
Do not modify SkillsExtractor or ContactExtractor.

---

## PHASE 13 — Experience Extraction

Implement ExperienceExtractor in app/extractors/experience.py, using these reusable components (create if not already present): CompanyDetector, JobTitleDetector, DescriptionExtractor.

Use the existing DateRangeParser from Phase 12 — do not duplicate date parsing logic.
Reuse SkillsExtractor from Phase 11 to tag skills mentioned within each experience entry.

Extract per entry: company, designation, location, startDate, endDate, current, description, skills, confidence.

Use section-detected EXPERIENCE blocks from Phase 9 as input. Use job_titles.json (create this resource file) for JobTitleDetector's known-title matching.

Add unit tests covering: resume with a single job, resume with multiple jobs, current/ongoing role, missing location, and a description spanning multiple lines/bullets.

Do not modify SkillsExtractor, ContactExtractor, or DateRangeParser — only consume them.
Do not implement EducationExtractor, ProjectExtractor, or CertificationExtractor yet.
Do not add OCR or an LLM.

---

## PHASE 14 — Education Extraction

Implement only EducationExtractor in app/extractors/education.py, plus app/resources/degrees.json.

degrees.json: dictionary/list of known degree names and common abbreviations (e.g. "B.Tech", "Bachelor of Technology", "M.S.", "Master of Science") for normalization — do not hard-code degree names in code.

Use DateRangeParser from Phase 12 for start/end dates — do not duplicate date logic.

Extract per entry: institution, degree, fieldOfStudy, startDate, endDate, grade, confidence.

Use section-detected EDUCATION blocks from Phase 9.

Add unit tests covering: single degree, multiple degrees, missing grade, and degree name variants normalized via degrees.json.

Do not modify ExperienceExtractor, SkillsExtractor, ContactExtractor, or DateRangeParser.
Do not implement ProjectExtractor or CertificationExtractor yet.

---

## PHASE 15 — Project Extraction

Implement only ProjectExtractor in app/extractors/projects.py.

Reuse SkillsExtractor from Phase 11 to detect technologies used in each project — do not duplicate skill detection logic.
Reuse DateRangeParser from Phase 12 if projects include date ranges.

Extract per entry: project name, description, technologies, dates, URL.

Use section-detected PROJECTS blocks from Phase 9.

Add unit tests covering: project with a URL, project without dates, project with multiple technologies detected via SkillsExtractor.

Do not modify SkillsExtractor, ExperienceExtractor, EducationExtractor, or DateRangeParser.
Do not implement CertificationExtractor yet.

---

## PHASE 16 — Certification Extraction

Implement only CertificationExtractor in app/extractors/certifications.py.

Extract per entry: certification name, issuing organization, issue date, expiry date, credential ID, credential URL.

Use DateRangeParser from Phase 12 where applicable for issue/expiry dates — do not duplicate date logic.
Use section-detected CERTIFICATIONS blocks from Phase 9.

Add unit tests covering: certification with expiry, certification without expiry, certification with credential URL, certification with just a name and org (minimal fields).

Do not modify any other extractor.
Do not implement ConfidenceScorer yet (Phase 17) — continue using inline confidence values as done in prior extractors.

---

## PHASE 17 — Confidence Scoring

Implement only ConfidenceScorer in app/pipeline/stages/confidence.py.

Centralize the confidence values currently scattered inline across ContactExtractor, SkillsExtractor, ExperienceExtractor, EducationExtractor, ProjectExtractor, and CertificationExtractor:
- Email regex: 0.99
- Phone regex: 0.95
- Skill dictionary: 0.90
- Header name: 0.90
- Section extraction: 0.85
- Ambiguous detection: 0.55

Reference LOW_CONFIDENCE_THRESHOLD = 0.70 from core/config.py (already defined in Phase 6) — do not redefine it here.

Refactor the existing extractors to call ConfidenceScorer instead of hard-coding confidence values inline. This is a refactor of existing code, not new extraction logic — do not change what each extractor extracts, only how confidence is assigned.

Add unit tests for ConfidenceScorer directly, and update existing extractor tests only where confidence-value assertions need adjusting due to the refactor.

Do not change extraction logic/behavior in any extractor beyond swapping in ConfidenceScorer calls.
Do not implement Pydantic validation orchestration yet (Phase 18).
Do not implement the LLM fallback — LOW_CONFIDENCE_THRESHOLD is only used for the low_confidence_count metric in V1.

---

## PHASE 18 — Pydantic Validation

Implement only the final Pydantic validation layer for the assembled Resume model in app/domain/resume.py (and extraction.py as needed).

Behavior:
- Define/finalize Pydantic models for the full Resume schema (personal, summary, skills, experience, education, projects, certifications, achievements, languages) matching the spec's field definitions
- A validation function that takes assembled extractor output and returns either a validated Resume or a validation error
- On validation failure: this must be treated as a parse failure with error code VALIDATION_ERROR (the actual publishing to resume.parse.failed happens in Phase 20 — for now, just ensure the validation function raises/returns a clear, structured error that Phase 20 can consume)
- The result must never be usable/returned if validation fails

Add unit tests covering: fully valid resume passes, resume with an invalid/missing required field fails with VALIDATION_ERROR, and confirm no partial/invalid data is returned on failure.

Do not implement Kafka publishing (Phase 20).
Do not implement pipeline orchestration wiring (Phase 19) beyond what's needed to test validation in isolation.
Do not modify individual extractors.

---

## PHASE 19 — Parser Orchestration

Implement only the pipeline orchestrator in app/pipeline/parser.py and context.py, wiring together all existing modules in order: pdf_detection → text_extraction → normalization → sections → extraction (contact, skills, experience, education, projects, certifications) → confidence (ConfidenceScorer) → validation (Pydantic, Phase 18).

Create a PipelineContext object that carries state (raw bytes, text blocks, sections, partial results, errors) between stages.

The orchestrator should:
- Run stages in sequence
- Stop and produce a structured error if a stage fails unrecoverably
- Produce a final validated Resume object (or a validation failure) as output
- Not perform OCR — if pdf_detection flags insufficient text, produce a structured error/flag for now (OCR wiring comes in Phase 24)

Add integration tests running the full pipeline end-to-end against 2–3 sample resume fixtures (single-column, two-column, fresher with minimal sections).

Do not add OCR.
Do not modify individual stage/extractor implementations — only wire them together.
Do not implement Kafka consumption/publishing yet.

---

## PHASE 20 — Kafka Result Publishing

Implement only Kafka result publishing: resume.parse.completed and resume.parse.failed.

Wire the Phase 6 Kafka consumer (currently just logging) to:
1. On receiving a resume.parse.requested event, download the PDF via MinIO client (Phase 6) using storageKey
2. Run it through the Phase 19 orchestrator
3. On success: publish to resume.parse.completed with jobId, resumeId, status COMPLETED, parserVersion, result, and metadata (pageCount, ocrUsed=false, processingTimeMs)
4. On Pydantic validation failure (Phase 18): publish to resume.parse.failed with error code VALIDATION_ERROR
5. On other pipeline failures (e.g. PDF extraction failure): publish to resume.parse.failed with the appropriate error code (PDF_EXTRACTION_FAILED, etc.)
6. For transient I/O errors only (e.g. a single MinIO timeout), retry in-process up to 2 times with short backoff before treating as a failure — this is not the same as the job-level retry owned by `resume-platform` (a separate repo), and this service must not re-publish to resume.parse.requested itself

Add integration tests (using embedded/test Kafka) covering: successful parse → completed event published, validation failure → failed event with VALIDATION_ERROR, extraction failure → failed event with appropriate code.

Do not implement job-level retry/requeue logic here — that belongs to `resume-platform`.
Do not modify the orchestrator's internal stage logic.
Do not implement anything in the `resume-platform` repo.

---

## PHASE 24 — OCR (only after real scanned resumes are identified)

Implement only OCR as a fallback within the existing PDF extraction pipeline (Phase 7/19).

Wire OCR to trigger only when pdf_detection (Phase 7) determines insufficient embedded text exists — do not run OCR on every PDF.

Update the orchestrator (Phase 19) to call OCR only in that fallback branch, and set ocrUsed=true in the completed event metadata (Phase 20) when it's used.

Add unit tests covering: text-based PDF (OCR not triggered), scanned/image PDF (OCR triggered and produces usable text blocks).

Do not change extraction logic for non-scanned PDFs.
Do not modify any extractor beyond the pdf_detection/text_extraction stages.
Do not add this phase until real scanned resume samples have been collected and confirmed necessary.

---

## PHASE 25 — LLM Fallback (only after real low-confidence data is measured)

Implement only AIExtractionStrategy as a field-level fallback, per the existing ResumeExtractionStrategy interface.

Behavior:
- After RuleBasedExtractionStrategy runs and ConfidenceScorer (Phase 17) tags fields, identify fields with confidence below LOW_CONFIDENCE_THRESHOLD
- Send only those specific low-confidence fields (with minimal surrounding context) to the LLM — never the full resume, never high-confidence fields
- Validate LLM output through the existing Pydantic validation layer (Phase 18) before accepting it
- Keep RuleBasedExtractionStrategy as the default; AIExtractionStrategy only activates for the flagged low-confidence fields
- Add a HybridExtractionStrategy that combines rule-based output with LLM-corrected low-confidence fields

Add unit tests covering: high-confidence resume (LLM never called), low-confidence field triggers LLM call and gets validated, LLM output that fails Pydantic validation is rejected and falls back to the rule-based value.

Do not send full resumes to the LLM under any circumstance.
Do not modify RuleBasedExtractionStrategy's existing behavior.
Do not implement this phase until real low_confidence_count metrics from production/test data justify it.
