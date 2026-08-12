# Copilot Instructions — resume-parser (Python)

This repo is the **Python document-intelligence service** for the Resume
PDF Parser system. It owns all PDF parsing and resume field extraction. All
user management, APIs, database, retry, and rate limiting live in the
separate `resume-platform` (Spring Boot) repo. The two services communicate
only through Kafka and shared object storage — never call each other
directly, never share a database.

V1 must work WITHOUT an LLM. The system must be designed so an LLM can be
added later as a field-level fallback without redesigning the parser.

---

## TECHNOLOGY

- Python 3.12+
- FastAPI (health endpoint + service scaffolding only — no public resume API)
- PyMuPDF
- Pydantic
- Regex, dictionaries, heuristics
- Optional OCR later

Do NOT add spaCy initially — only if real resume samples show deterministic
extraction is insufficient.

Do NOT use an LLM in V1.

Do not add Redis, Elasticsearch, Kubernetes, or other infra.

---

## THIS REPO'S RESPONSIBILITIES

Owns:

- PDF validation
- PDF type detection
- PDF text extraction
- PDF layout extraction
- OCR later
- Text normalization
- Section detection
- Contact extraction
- Skill extraction
- Experience extraction
- Education extraction
- Project extraction
- Certification extraction
- Confidence scoring
- Pydantic validation

Does NOT own (this belongs to the `resume-platform` repo):

- Authentication
- Users
- Application database (Postgres)
- Resume lifecycle / status transitions
- Public APIs
- Retry management (job-level)
- Rate limiting

Never implement a public REST API, user auth, or a persistent application
database in this repo. This service is a Kafka consumer/producer plus a
`/health` endpoint — nothing else is externally reachable.

---

## PROJECT STRUCTURE

```
resume-parser/
  app/
    main.py
    core/
      config.py
      logging.py
      exceptions.py
    domain/
      resume.py
      document.py
      extraction.py
    pipeline/
      parser.py
      context.py
      stages/
        pdf_detection.py
        text_extraction.py
        normalization.py
        sections.py
        extraction.py
        validation.py
        confidence.py
    extractors/
      contact.py
      skills.py
      experience.py
      education.py
      projects.py
      certifications.py
    resources/
      skills.json
      degrees.json
      job_titles.json
      section_aliases.json
    infrastructure/
      kafka/
      storage/
  tests/
```

Do not create files until they are actually needed.

---

## PDF PROCESSING

Use PyMuPDF. First attempt normal text extraction; determine whether
meaningful text exists. If insufficient, fall back to OCR — but OCR must
never run for every PDF, only as a fallback, and must not be implemented
until the basic parser works (see Phase 24).

Preserve layout, do not immediately flatten text: keep text, page number,
x0/y0/x1/y1, font size (when available), bold flag (when available). This
supports two-column resumes, headings, sidebars, reading order, name
detection, section detection.

---

## TEXT NORMALIZATION

`TextNormalizer`: normalize line endings, collapse unnecessary whitespace,
remove extraction artifacts, preserve meaningful line boundaries, preserve
emails/URLs/phone numbers/dates/technical terms exactly. Do not aggressively
rewrite content.

---

## SECTION DETECTION

Deterministic detection for: `SUMMARY, EXPERIENCE, EDUCATION, SKILLS,
PROJECTS, CERTIFICATIONS, ACHIEVEMENTS, LANGUAGES`. Store all header
aliases in `resources/section_aliases.json` — never hard-code aliases in
code.

---

## EXTRACTORS

- **Contact**: name, email, phone, location, LinkedIn, GitHub, portfolio.
  Regex + layout + header heuristics. No LLM, no external APIs.
- **Skills**: configurable dictionary (`resources/skills.json`), normalize →
  dedupe → canonical names → confidence. No LLM, no external APIs.
- **Experience**: company, designation, location, dates, current, description,
  skills. Reusable components: `DateRangeParser`, `JobTitleDetector`,
  `CompanyDetector`, `DescriptionExtractor`. Reuse `SkillsExtractor` — do not
  duplicate skill detection. Reuse `DateRangeParser` — do not duplicate date
  parsing logic anywhere.
- **Education**: institution, degree, field of study, dates, grade. Use
  `resources/degrees.json` — do not hard-code degree names in code.
- **Projects**: name, description, technologies, dates, URL. Reuse
  `SkillsExtractor` and `DateRangeParser`.
- **Certifications**: name, issuing org, issue/expiry date, credential ID,
  credential URL.

`DateRangeParser` must support: `Jan 2022 - Present`, `January 2022 - March
2024`, `2022 - 2024`, `06/2022 - Present`, `Jun 2022 - Current`.

---

## RESUME MODEL

```
Resume: personal, summary, skills, experience, education, projects,
        certifications, achievements, languages

Personal: name, email, phone, location, linkedin, github, portfolio

Experience: company, designation, location, startDate, endDate, current,
            description, skills, confidence

Education: institution, degree, fieldOfStudy, startDate, endDate, grade,
           confidence
```

---

## CONFIDENCE SCORING

Every extracted field carries `{value, confidence, source}`.

Fixed values:
```
Email regex:        0.99
Phone regex:        0.95
Skill dictionary:    0.90
Header name:         0.90
Section extraction:  0.85
Ambiguous detection: 0.55
```

Define `LOW_CONFIDENCE_THRESHOLD = 0.70` in `core/config.py`. A field below
this threshold counts toward the `low_confidence_count` metric and is the
unit later sent to the LLM fallback (Phase 25). Centralize all confidence
logic in `ConfidenceScorer` — do not scatter it across extractors.

---

## EXTRACTION STRATEGY / FUTURE LLM

Create a `ResumeExtractionStrategy` interface. V1 implements only
`RuleBasedExtractionStrategy`. Do NOT implement `AIExtractionStrategy` now —
the interface exists only to make future integration easy (Phase 25).

Future behavior (not V1): only fields below `LOW_CONFIDENCE_THRESHOLD` are
sent to the LLM, never the full resume, never high-confidence fields. LLM
output is validated through the same Pydantic layer. Prefer field-level
fallback.

---

## PYDANTIC VALIDATION

All extraction results must pass Pydantic validation before being
published. A validation failure is a parse failure — publish to
`resume.parse.failed` with error code `VALIDATION_ERROR`, never publish
partial/invalid data, never silently drop it. Log the full validation error
internally. This validation layer must remain even after adding an LLM.

---

## KAFKA — SHARED CONTRACT WITH resume-platform

Do not change these topic names or message shapes without coordinating a
change in the `resume-platform` repo too — they are the interface between
the two services.

Topics: `resume.parse.requested`, `resume.parse.completed`, `resume.parse.failed`

This repo **consumes** `resume.parse.requested` and **publishes**
`resume.parse.completed` / `resume.parse.failed`. It never publishes a new
`resume.parse.requested` message itself — job-level retry/requeue is owned
entirely by `resume-platform`.

**Request message** (consumed):
```json
{
  "jobId": "uuid",
  "resumeId": "uuid",
  "storageKey": "resumes/user-id/resume-id.pdf",
  "requestedAt": "timestamp"
}
```

**Completed message** (published):
```json
{
  "jobId": "uuid",
  "resumeId": "uuid",
  "status": "COMPLETED",
  "parserVersion": "1.0.0",
  "result": {},
  "metadata": {
    "pageCount": 2,
    "ocrUsed": false,
    "processingTimeMs": 1500
  }
}
```

**Failed message** (published):
```json
{
  "jobId": "uuid",
  "resumeId": "uuid",
  "status": "FAILED",
  "error": {
    "code": "PDF_EXTRACTION_FAILED",
    "message": "Unable to extract meaningful text"
  }
}
```

**Idempotency & retry**: `resume-platform` enforces idempotency via a DB
unique constraint on `(resume_id, job_id)` — this repo does not need its own
dedup logic, just correct `jobId` propagation. This repo MAY do small
bounded in-process retries (e.g. 2 attempts, short backoff) for transient
I/O errors (e.g. one MinIO timeout) before treating it as a failure — that
is a low-level I/O retry, not a job-level retry, and must not be confused
with `resume-platform`'s job retry.

---

## OBJECT STORAGE

Storage key format (shared contract — do not change without updating
`resume-platform` too):

```
resumes/{userId}/{resumeId}.pdf
```

This repo downloads the PDF using `storageKey` from the request message.
Never sends PDF binary data through Kafka.

---

## ERROR HANDLING

Structured error codes (shared with `resume-platform`):
`INVALID_PDF, PDF_TOO_LARGE, PDF_EXTRACTION_FAILED, OCR_FAILED,
RESUME_PARSE_FAILED, UNSUPPORTED_FILE_TYPE, STORAGE_ERROR, VALIDATION_ERROR`

Never leak stack traces into published Kafka messages. Log details
internally.

---

## MONITORING

Structured logs, `/health` endpoint, processing duration. Emit metrics via
Kafka message metadata: `ocr_usage`, `low_confidence_count`,
`processingTimeMs` — `resume-platform` aggregates these, this repo just
computes and reports them.

---

## VERSIONING

Every parsed resume must contain `schemaVersion` and `parserVersion`, set
by this repo. Never silently change the meaning of previously stored parsed
data — bump the version when the shape changes.

---

## DEPENDENCY OPTIMIZATION

Start with the smallest dependency set: PyMuPDF, Pydantic, Kafka client,
S3/MinIO client, FastAPI, pytest. Do not add spaCy, OCR, LLM libraries, or
AI SDKs until a real requirement is identified.

---

## TESTING

Cover per module: PDF extraction, PDF detection, text normalization,
section detection, contact extraction, skill extraction, date parsing,
experience extraction, education extraction, project extraction,
certification extraction, confidence scoring, Pydantic validation
(including the failure → `VALIDATION_ERROR` path).

Fixtures should include: single-column resume, two-column resume, scanned
resume, fresher resume, experienced resume, resume without email, resume
without phone, resume with projects, resume with certifications, resume
with multiple jobs.

---

## COPILOT CREDIT OPTIMIZATION — RULES FOR EVERY TASK

Never ask Copilot to build the whole parser at once. One focused module per
prompt, always stating:

1. Exact module to implement
2. Existing interface/class to use
3. Required behavior
4. Tests required
5. Explicit exclusions

For every task: inspect existing code first, reuse existing
classes/interfaces, modify only required files, do not rewrite working
code, do not introduce unnecessary dependencies or abstractions, add
focused tests, keep changes small, stop after completing the requested
task.

See `docs/copilot-phase-prompts.md` in this repo for the exact phase-by-phase
prompts to use, in order.
