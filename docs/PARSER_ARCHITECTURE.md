# Resume Parser Architecture

This document describes the **current implementation** of the resume/CV parser as of the layout-pipeline hardening work. It separates **observed behavior**, **architectural intent**, **known limitations**, and **proposed future improvements**.

---

## 1. System Overview

The service is a Kafka-driven FastAPI worker (`app/main.py`) that downloads PDFs from MinIO, parses them via `ResumeParser`, and publishes structured results to Kafka (`ResumeRequestService` in `app/services/resume_request_service.py`).

### Two pipeline entry points

| Entry | Module | Production default |
|-------|--------|-------------------|
| Legacy | `ResumeParser.parse()` | Only when `PARSER_MODE=legacy` |
| Layout-aware | `ResumeParser.parse_with_layout_pipeline()` | `PARSER_MODE=auto` and `PARSER_MODE=layout` |

`Settings.parser_mode` (`app/core/config.py`) accepts `legacy`, `layout`, or `auto`. In `ResumeRequestService`, anything other than `legacy` calls `parse_with_layout_pipeline`.

### Legacy pipeline (flat block stream)

```
PDF bytes
  → PDFDetector.detect()                    [app/pipeline/stages/pdf_detection.py]
  → PDFExtractor.extract()                  [app/pipeline/stages/text_extraction.py]  → list[TextBlock]
  → ReadingOrder.reorder()                  [app/pipeline/stages/reading_order.py]
  → TextNormalizer.normalize_blocks()     [app/pipeline/stages/normalization.py]
  → SectionDetector.detect()                [app/pipeline/stages/sections.py]       → dict[section, list[TextBlock]]
  → classify_block() per block              [app/pipeline/stages/block_classification.py]
  → group_candidates() per section          [app/pipeline/stages/candidate_grouping.py]
  → Field extractors + inline assembly      [app/extractors/*, app/pipeline/parser.py]
  → validate_resume()                       [app/domain/resume.py]                  → Resume
```

Header/contact blocks: `_header_blocks()` returns normalized blocks before the first `SECTION_NAMES` exact heading match.

### Layout-aware pipeline (document IR → semantic sections → compatibility layer)

```
PDF bytes
  → PDFDetector.detect()
  → PDFExtractor.extract()                  → list[TextBlock]
  → document_from_text_blocks()             [app/domain/document.py]                → Document (physical_page regions)
  → reconstruct_document()                  [app/pipeline/stages/reconstruction.py]
  → interpret_layout()                      [app/pipeline/stages/layout.py]
  → detect_region_aware_sections()          [app/pipeline/stages/semantic_paths.py] → SemanticDocument
  → semantic_sections_to_text_blocks()      [app/pipeline/stages/semantic_compat.py]  → dict[section, list[TextBlock]]
  → semantic_header_to_text_blocks()        (unassigned lines → contact/header input)
  → classify_block() + group_candidates()
  → Field extractors + inline assembly
  → validate_resume()                       → Resume
```

**Architectural intent:** The layout path preserves region/column identity, reconstructs fragmented PDF lines, infers unknown headings from content, then converts back to `TextBlock` views so existing classification, grouping, and extractors remain usable without a full rewrite.

**Current limitation:** Legacy and layout paths diverge before sectioning; behavior is not guaranteed to match between them.

### Final Resume schema

`Resume` (`app/domain/resume.py`): `personal`, `summary`, `skills`, `experience`, `education`, `projects`, `certifications`, `achievements`, `languages`, `metadata`. Pydantic validation on assembly; no persistence layer in-parser—output is Kafka event payload.

---

## 2. Module Responsibilities

### PDF detection & extraction

| Module | Responsibility | Input | Output | Invariants | Should NOT |
|--------|----------------|-------|--------|------------|------------|
| `PDFDetector` | Validate PDF, detect text presence, page count | `bytes` | `PDFDetectionResult` | Raises pipeline error upstream if not PDF / no text | Parse sections or fields |
| `PDFExtractor` | PyMuPDF line-level extraction with bbox + typography | `bytes` | `list[TextBlock]` | One `TextBlock` per visual PDF line (span-merged within line) | Reading order, normalization, semantics |
| `ReadingOrder` | Page-local column clustering + left-to-right region order | `TextBlock` iterable | ordered `TextBlock` list | Legacy path only; strongest x-gap split with adaptive threshold | Semantic section assignment |
| `TextNormalizer` | Control-char cleanup, ligatures, same-line block coalescing | `TextBlock` list | normalized/coalesced blocks | Legacy path only; conservative merge on same baseline | Change semantic meaning of headings |

### Document IR (layout path only)

| Module | Responsibility | Input | Output | Invariants | Should NOT |
|--------|----------------|-------|--------|------------|------------|
| `document_from_text_blocks` | Build `Document` with deterministic line/span IDs | `TextBlock` iterable | `Document` | One physical region per page initially; preserves bbox | Infer columns |
| `reconstruct_document` | Merge geometrically adjacent line fragments | `Document` | `Document` | Creates replacement `Line` objects; accumulates `source_span_ids`; sets `reconstruction_method` | Assign semantic sections |
| `interpret_layout` | Split pages into header + column regions | `Document` | `Document` | Region `kind`: `header`, `column`, or `physical_region` | Label section types |

### Semantic sectioning

| Module | Responsibility | Input | Output | Invariants | Should NOT |
|--------|----------------|-------|--------|------------|------------|
| `SectionDetector` | Known heading alias matching; legacy section transitions | `TextBlock` stream | `dict[SECTION_NAMES]` | Alias lookup is deterministic from `section_aliases.json` | Infer arbitrary unknown headings (layout path delegates unknowns) |
| `detect_region_aware_sections` | Per-path sectioning, continuation, unknown-heading inference | `Document` + `SectionDetector` | `SemanticDocument` | Paths preserve `region_id`; unknown headings require content evidence | Extract experience/education fields |
| `infer_section` | Score unknown section content | heading string + content lines | `SectionInference` | Heading alone cannot win; `UNKNOWN` is valid | Mutate document objects |
| `semantic_compat` | Convert semantic output to legacy `TextBlock` views | `SemanticDocument` | section dict + provenance fields on blocks | Adds `source_line_id`, `source_span_ids`, `reconstruction_method` on blocks | Re-run inference |

### Classification & grouping (within a known section)

| Module | Responsibility | Input | Output | Invariants | Should NOT |
|--------|----------------|-------|--------|------------|------------|
| `classify_block` | Label individual block **role** (DATE, JOB_TITLE, …) | block-like object | `ClassifiedBlock` | Conservative; defaults to UNKNOWN | Decide which **section** owns content |
| `group_candidates` | Cluster blocks into entry-level groups | `ClassifiedBlock` list + section name | `list[CandidateGroup]` | Section-scoped; preserves page boundaries | Parse final field values |

### Extractors & schema assembly

| Module | Responsibility | Input | Output | Should NOT |
|--------|----------------|-------|--------|------------|
| `ContactExtractor` | Header personal fields | header `TextBlock` list | dict of `{value, confidence, source}` | Assign sections |
| `SkillsExtractor` | Canonical skill tokens | blocks + optional section name | skill dicts | Own non-skill sections |
| `ExperienceExtractor` | Job entries | blocks or `CandidateGroup` list | experience dicts | Fix upstream section misassignment |
| `EducationExtractor` | Education entries | lines or groups | education dicts | Same |
| `ProjectExtractor` | Project entries | blocks or groups | project dicts | Same |
| `CertificationExtractor` | Certification entries | blocks | certification dicts | Same |
| Parser inline | `summary`, `achievements`, `languages` from section text | section blocks | strings / string lists | — |
| `validate_resume` | Pydantic assembly | partial dict | `Resume` | — |

### Boundary: semantic sectioning vs classification vs extractors

```
Semantic layer     →  Which SECTION bucket? (SUMMARY, EXPERIENCE, SKILLS, UNKNOWN, UNASSIGNED)
Classification     →  What ROLE does this block play? (DATE, JOB_TITLE, BULLET, …)
Grouping           →  Which blocks form one entry candidate?
Extractors         →  What FIELDS does this entry have? (company, degree, …)
Schema assembly    →  Validated Resume object
```

**Non-negotiable:** Extractors must not be used to correct wrong section ownership. Fixes belong at the earliest layer that lost the information.

---

## 3. Semantic Sectioning

### SectionDetector (`app/pipeline/stages/sections.py`)

**Known headings:** Loaded from `app/resources/section_aliases.json`. Matching strategies:

1. Normalized exact alias (`[^a-z0-9]` stripped)
2. Prefix match when alias is 1–2 chars longer than text
3. Regex full-line patterns per alias
4. Special case: `\bobjective\b` → `SUMMARY`
5. `_looks_like_education_header`: `degree | institution` pipe pattern with degree prefix from `degrees.json`

**Same-line multi-block headings:** Collects blocks on same page within `LINE_Y_TOLERANCE` (1.0pt) y-alignment before header lookup.

**Legacy-only section transitions** (`_infer_section_transition`): Heuristics when already inside a section—e.g. SKILLS → EDUCATION on aligned date + education content; EDUCATION → SKILLS for right-column short skill-like text (`x0 > 200`); ACHIEVEMENTS → SUMMARY on `objective`.

**SECTION_NAMES:** `SUMMARY`, `EXPERIENCE`, `EDUCATION`, `SKILLS`, `PROJECTS`, `CERTIFICATIONS`, `ACHIEVEMENTS`, `LANGUAGES`.

### Section aliases resource

`section_aliases.json` maps canonical section → list of heading strings. Includes variants like `skill highlights`, `professional background` (experience), `credentials` (certifications). **Note:** standalone `HIGHLIGHTS` is **not** a known alias; it is handled via unknown-heading inference when content looks like skills.

### Semantic paths (`app/pipeline/stages/semantic_paths.py`)

**`build_semantic_paths`:** One `SemanticPath` per region per page, ordered by `reading_order` / `region_id`.

**`detect_region_aware_sections`:** For each path on each page:

1. **Continuation:** `_continuation_section` picks nearest prior path's section by vertical center.
2. **Related heading:** `_related_heading` links column paths under a heading in an adjacent/overlapping path above.
3. **Line loop:** Known header → set `current_section`; unknown heading candidate → `infer_section` on following content until next boundary; education header / aligned education date heuristics mirror legacy detector.
4. **Output:** `SemanticDocument` with `sections`, `unassigned_lines`, `unknown_candidates`.

**Unknown heading detection** (`_is_unknown_heading_line` + `is_unknown_heading`):

- Rejects known headers, list markers, wrapped continuation lines, item headings inside active sections (role-title protection).
- Requires typography cue: bold, ALL CAPS, or `font_size >= 12`.
- Rejects skill-vocabulary tokens, digits, >5 words, trailing punctuation.
- Must have following content line.

**Wrapped-line protection:** `_is_wrapped_content_line` compares style, indent, vertical gap, continuation cues.

**Role-title protection:** `_looks_like_item_heading` prevents splitting experience/education/project entries when following lines have dates, companies, institutions, etc.

### Semantic inference (`app/pipeline/stages/semantic_inference.py`)

**`infer_section(heading, content)`** scores eight categories plus implicit `UNKNOWN`.

**Content signals (per line):**

| Signal | Sections boosted |
|--------|------------------|
| Bullet/list marker | All (+0.15) |
| Skill vocabulary (`SemanticResources.skills`) | SKILLS (+2.0) |
| Competency phrases (team player, leadership, …) | SKILLS (+1.0) |
| Awards, metrics, outcome verbs | ACHIEVEMENTS (+2.5) |
| Date ranges | EXPERIENCE (+1.5), EDUCATION (+0.8), PROJECTS (+0.5) |
| Company suffixes / role words (with profile-language guard) | EXPERIENCE (+1.5) |
| Employment verbs + corroboration | EXPERIENCE (+0.8) |
| Profile language (experienced, seeking, …) | SUMMARY (+2.5) |
| Academic patterns / graduation years | EDUCATION |
| Credential patterns | CERTIFICATIONS (+3.5) |
| Project verbs / URLs | PROJECTS (+1.5) |
| Language proficiency patterns | LANGUAGES (+2.0) |

**Structure bonuses:** High list ratio → SKILLS/ACHIEVEMENTS/LANGUAGES; long paragraphs → SUMMARY.

**Heading hints:** Token overlap with per-section hint sets adds up to +2.0 **only when `scores[section] > 0`** (content established eligibility first). Guards: skip EXPERIENCE hint when achievement content; skip SUMMARY when language content + `language` in heading.

**Cross-category:** `has_language_content` zeroes SKILLS and SUMMARY scores.

**Project entry pair:** Short title line + corroborating build/migrate verb on next line → PROJECTS (+2.5).

**Confidence & UNKNOWN:**

```python
confidence = top_score / total
section = top if confidence >= 0.70 and margin >= 0.15 else "UNKNOWN"
margin = confidence - (second_score / total)
```

**Architectural principle (implemented):** A heading must not be sufficient evidence alone. Heading hints require prior content score > 0 for that category.

### Section compatibility conversion (`semantic_compat.py`)

- `semantic_sections_to_text_blocks`: Flattens `SemanticSection.lines` to `TextBlock`, adds provenance attrs.
- `semantic_header_to_text_blocks`: Returns **all** `unassigned_lines` as header candidates (not “blocks before first section”).
- `semantic_sections_to_section_views`: Preserves path/region boundaries for tests.

---

## 4. Unknown Section Inference (detail)

### Detection pipeline

1. Line is not a known `SectionDetector` header.
2. Not a list marker; not wrapped continuation of previous line.
3. Passes `is_unknown_heading` typography/length rules.
4. Not an in-section item heading (`_looks_like_item_heading`).
5. Has at least one following line.
6. Content runs until next known header or next unknown-heading candidate (`_next_section_boundary`).

### Content considered

All `Line.text` values in the candidate content span (including bullet marker lines). Inference receives `[item.text for item in content]`.

### Scoring & confidence

Additive per-line scores into `scores` dict; parallel `signals` audit trail. Ranked descending; `UNKNOWN` when confidence < 0.70 or top-vs-second margin < 0.15.

### UNKNOWN fallback

When `UNKNOWN`, content is **not** added to any section; lines remain in `unassigned_lines` (unless also captured elsewhere). `unknown_candidates` retains heading, content, and full `SectionInference` for diagnostics/tests.

### Heading influence

Heading tokens matched against `heading_hints` dict. Influence is **conditional** on existing positive content score for that section.

### Category-local & cross-category

- Language content suppresses SKILLS/SUMMARY competition.
- Achievement content suppresses EXPERIENCE heading hint.
- Date ranges contribute to multiple categories with different weights (experience > education > projects).

---

## 5. Provenance

### Representation

| Layer | IDs / metadata |
|-------|----------------|
| `TextBlock` (extraction) | `page_number`, `x0,y0,x1,y1`, `font_size`, `bold` |
| `Span` | `span_id`, `bbox`, typography, optional `link` |
| `Line` | `line_id`, `page_number`, `bbox`, `spans`, `text`, `style`, `reading_order`, `source_span_ids`, `reconstruction_method` |
| `Region` | `region_id`, `kind`, `bbox`, `column_id`, `reading_order` |
| `SemanticPath` | `path_id`, `page_number`, `region_id`, `region_order` |
| `SemanticSection` | `section`, `path_id`, `region_id`, lines |
| Compatibility `TextBlock` | `source_line_id`, `source_span_ids`, `reconstruction_method` (dynamic attrs) |

### Mutation vs replacement

- **Reconstruction:** Creates new `Line` via `dataclasses.replace`; merges spans; updates bbox and `source_span_ids`; sets `reconstruction_method="adjacent_physical_fragments"`.
- **Layout:** Replaces region lists on `Page`; lines unchanged unless reconstruction already ran.
- **Semantic paths:** Lines referenced in sections are original/reconstructed `Line` objects—not copied.
- **semantic_compat:** New `TextBlock` wrappers; original `Line` objects untouched.
- **Normalization (legacy):** New `TextBlock` instances for merged runs.

**Gap:** Legacy `parse()` path does not attach `source_line_id` to `TextBlock`. Provenance is richest on the layout path after compat conversion.

---

## 6. Extractors

### ContactExtractor (`app/extractors/contact.py`)

| Aspect | Detail |
|--------|--------|
| Input | Ordered header `TextBlock` list |
| Strategy | Regex for email, phone, LinkedIn, GitHub, portfolio URL; heuristic name scoring (position, font size, bold); letter-spaced name assembly; multiline street+city address |
| Normalization | NFKC for LinkedIn; spaced-name collapse; phone on same line as email allowed |
| Fallback | `null` fields with confidence 0 |
| Resources | Uses `SectionDetector`, `SkillsExtractor` to reject skill-like location false positives |
| Weaknesses | Location heuristic can still pick summary fragments (`docs/known-issues.md`); geo token list is limited |

### SkillsExtractor (`app/extractors/skills.py`)

| Aspect | Detail |
|--------|--------|
| Input | `TextBlock` iterable; `section_name` affects `source` label |
| Strategy | Whole-text join; whitespace collapsed for matching; `\b` alias regex from `skills.json` keys (longest first) |
| Output | Deduped canonical values with confidence 0.90 |
| Fallback | Empty list |
| Weaknesses | Only vocabulary present in `skills.json`; no fuzzy/OCR tolerance beyond normalization |

### ExperienceExtractor (`app/extractors/experience.py`)

| Aspect | Detail |
|--------|--------|
| Input | EXPERIENCE blocks **or** `CandidateGroup` list (preferred when grouping available) |
| Strategy | DateRangeParser; `job_titles.json` exact match; combined title-date, company/location/date, parenthesized dates; label-aware group path; description cleanup preserving bullets |
| Skills sub-field | Re-runs SkillsExtractor on cleaned description |
| Fallback | Block-sequential buffer until date/title/company signal |
| Weaknesses | Heuristic title/company ordering varies by layout; `_is_strong_new_experience_job` in grouping has resume-1-oriented repeated-title guard |

### EducationExtractor (`app/extractors/education.py`)

| Aspect | Detail |
|--------|--------|
| Input | EDUCATION line strings **or** groups |
| Strategy | `degrees.json` prefix patterns; institution/grade/field heuristics; compact degree+institution split; trailing year on degree line |
| Normalization | Degree canonicalization via alias map; institution comma spacing |
| Weaknesses | `EducationEntry` type referenced in annotations but **not defined** in codebase (runtime uses dicts); complex same-line splits can mis-split field vs institution |

### ProjectExtractor (`app/extractors/projects.py`)

| Aspect | Detail |
|--------|--------|
| Input | PROJECTS blocks or groups |
| Group strategy | First DATE in group; title = nearest non-date block before; description = lines after DATE |
| Technologies | SkillsExtractor on cleaned description |
| Weaknesses | Groups without DATE fall back to naive block sequencing |

### CertificationExtractor (`app/extractors/certifications.py`)

| Aspect | Detail |
|--------|--------|
| Input | CERTIFICATIONS blocks |
| Strategy | Sequential fill: URL, credential ID, issue/expiry labels, name, org, description; skips `Programming languages:` lines |
| Weaknesses | Blank line closes entry; weak org/name disambiguation |

### Summary, Achievements, Languages

**No dedicated extractors.** Parser joins SUMMARY blocks into one string; ACHIEVEMENTS and LANGUAGES are raw `block.text` lists.

### DateRangeParser (`app/extractors/date_parser.py`)

Shared utility: `YYYY-MM` output, current terms, invisible Unicode stripping, optional `Date:` label. Used across sectioning, classification, and extractors.

---

## 7. Resources

| File | Purpose | Loaded by | Cached? |
|------|---------|-----------|---------|
| `section_aliases.json` | Known section headings | `SectionDetector` (per instance), `block_classification` (module global) | Per-instance / once per process |
| `skills.json` | Skill alias → canonical | `SkillsExtractor`, `SemanticResources` | `SemanticResources._cache`; extractor loads per instance |
| `degrees.json` | Degree alias → canonical | `EducationExtractor`, `SectionDetector`, `block_classification` | Per instance / module global |
| `job_titles.json` | Known job titles list | `ExperienceExtractor`, `block_classification` | Per instance / module global |

`SemanticResources` (`app/pipeline/stages/semantic_resources.py`): class-level cache for JSON vocabularies used in inference only (currently `skills`).

---

## 8. Cross-Category Ambiguity

| Collision | Current handling |
|-------------|------------------|
| **AWS:** skill vs project technology | SKILLS inference uses skill vocab; PROJECTS needs build/migrate/URL signals. In extractors, AWS in project description → `technologies` via SkillsExtractor; in skills section → `skills` list. No global disambiguation—depends on section assignment. |
| **engineer:** experience vs summary | SUMMARY needs profile language (`experienced`, `seeking`); EXPERIENCE needs role words without profile language or with company/date corroboration. `ContactExtractor._looks_like_name` rejects role words. |
| **professional:** summary vs prose | `professional summary` is known alias → SUMMARY. Standalone "professional" prose without profile signals → likely UNKNOWN in inference. |
| **project title vs job title** | Grouping uses section context; `classify_block` avoids PROJECT_TITLE; experience uses JOB_TITLE heuristics + grouping splits. Short title + date in PROJECTS section → project group. |
| **dates:** experience vs education vs certification | Date blocks classify as DATE everywhere. Inference weights dates toward EXPERIENCE highest; education date alignment heuristic in layout path; education grouping attaches right-column dates to degree lines. |
| **language names vs skills** | Inference: language proficiency patterns route to LANGUAGES and **zero** SKILLS/SUMMARY. Skill vocab match suppressed when `language_like` on same line. |

---

## 9. Known Generic Fixes (implemented)

These are **generic mechanisms** evidenced in code/tests—not fixture-specific hacks:

| Fix | Where |
|-----|-------|
| Letter-spaced names | `ContactExtractor._find_letter_spaced_name`, `_is_letter_spaced_name` |
| One-word / hyphenated / apostrophe names | `ContactExtractor._looks_like_name` Unicode token pattern; adversarial `test_contact_variations` |
| Unicode phone whitespace | `ContactExtractor._find_phone`; `test_ocr_damage` NFKC path |
| Multiline addresses | Street + city line pair in `_find_location` |
| Generic skill aliases | `skills.json` + longest-match regex |
| Unknown HIGHLIGHTS → SKILLS | `infer_section` competency/skill signals (not alias table) |
| Project-entry inference | `project_pair` / `project_entry_pair` in `infer_section` |
| Migration project inference | Corroborated title + "migrated" line (`test_semantic_inference`) |
| Achievement corroboration | Metrics, awards, outcome verbs without generic action prose |
| UNKNOWN fallback | `MIN_CONFIDENCE` / `MIN_MARGIN` thresholds |
| Provenance preservation | `source_span_ids`, reconstruction metadata, compat layer attrs; `test_provenance_invariants` |
| Role-title protection | `_looks_like_item_heading`, `_is_unknown_heading_line` guards |
| Wrapped-line protection | `_is_wrapped_content_line`, reconstruction `_is_wrapped_continuation` |
| Adjacent email fragment merge | `reconstruct_document` (resume_2 email split) |
| Right-column education dates stay in EDUCATION | `SectionDetector._infer_section_transition`, layout `_is_aligned_education_date` |
| Experience contact sidebar exclusion | `group_candidates._is_obvious_contact_sidebar_block` |
| Repeated uppercase role title same job | `group_candidates._is_strong_new_experience_job` duplicate-text guard |

**Fixture-specific contracts** (e.g. exact experience tuples for `resume_1.pdf`) exist in integration tests but are **regression locks**, not architecture—the fixes above are the generic layer underneath.

---

## 10. Testing Architecture

### Layers and protecting tests

| Layer | Example tests |
|-------|----------------|
| PDF extraction | `test_pdf_extraction.py` |
| Document IR | `test_document_ir.py` |
| Reconstruction | `test_reconstruction.py` |
| Layout / columns | `test_layout.py`, `test_reading_order.py` |
| Section detection (legacy) | `test_section_detection.py` |
| Semantic paths | `test_semantic_paths.py` |
| Semantic inference | `test_semantic_inference.py` |
| Semantic compat | `test_semantic_compat.py` |
| Block classification | `test_block_classification.py` |
| Candidate grouping | `test_candidate_grouping.py` |
| Extractors | `test_contact_extractor.py`, `test_experience_extractor.py`, `test_education_extractor.py`, `test_skills_extractor.py`, `test_projects_extractor.py`, `test_certifications_extractor.py` |
| Parser integration | `test_pipeline_parser.py`, `test_layout_parser_integration.py` |
| Service / config | `test_resume_request_service.py`, `test_core_config.py` |
| Adversarial (generic) | `tests/adversarial/test_section_inference_adversarial.py`, `test_contact_variations.py`, `test_field_variations.py`, `test_experience_layouts.py`, `test_ocr_damage.py`, `test_provenance_invariants.py` |
| Skills regression | `test_skills_regression.py` (fixture PDFs) |
| Real PDF checks | Fixture-driven tests across `tests/fixtures/*.pdf` |

### Protected fixture contracts

`resume_1`, `resume_2`, `resume_7`, `fresher_hr_resume` (referenced in tests; may not all be present in repo snapshot), `swe_experienced_resume`—exact field tuples lock end-to-end behavior.

### Adversarial vs integration

- **Adversarial:** Synthetic inputs probing generic rules (name shapes, ambiguous inference, layout variations).
- **Integration:** Real or synthetic PDFs through full or partial pipeline.

**Practice:** Run smallest relevant test file first; stop at first unrelated regression.

---

## 11. Current Known Risks

Not claimed solved unless tests prove it:

| Risk class | Status |
|------------|--------|
| Arbitrary section headings | Partially addressed via unknown inference; many still → UNKNOWN or wrong bucket |
| Multi-column resumes | Layout path improves; legacy path uses simple x-gap clustering |
| Wrapped text | Reconstruction + wrapped-line guards; fragile on odd fonts |
| Bullet fragmentation | Normalization/reconstruction; bullet markers excluded from merges |
| PDF font artifacts | Ligature/control-char normalization only |
| Reading-order problems | No global optimizer; region-local paths |
| Tables | Not modeled (`Page.tables` unused) |
| Sidebars | Partial (contact sidebar exclusion in experience grouping) |
| Headers/footers | Header band detection in layout; no footer suppression |
| Repeated sections | No deduplication |
| Ambiguous headings | UNKNOWN fallback; some false positives remain |
| Missing resource vocabulary | Skills/degrees/titles not extracted or inferred |
| OCR damage | No OCR pipeline (`ocrUsed` always false); light Unicode normalization only |
| Multilingual resumes | Limited language-name list in inference |
| Unusual names / phones | Improved but not exhaustive |
| Multiline addresses | Street+city pattern only |
| Project/experience ambiguity | Section assignment dependent |
| Education/experience ambiguity | Date + context heuristics only |
| `parser_mode=auto` | **Same as layout today**—no runtime auto-detection between pipelines |
| Missing fixtures in repo | Tests reference PDFs not in `tests/fixtures/` (e.g. `fresher_hr_resume.pdf`, `swe_experienced_resume.pdf`) |

---

## 12. Architectural Non-Negotiables

## NON-NEGOTIABLE RULES

1. Never hardcode a specific resume fixture.
2. Fix problems at the earliest correct architectural layer.
3. Never use a single keyword as decisive semantic evidence.
4. Heading text alone must never force an unknown section.
5. `UNKNOWN` is a valid and intentional outcome.
6. Preserve provenance.
7. Known aliases remain deterministic.
8. Extractors determine fields, not semantic ownership.
9. Prefer structural evidence over lexical coincidence.
10. Cross-category evidence must be handled explicitly.
11. Every generic production change requires regression coverage.
12. Do not weaken protected contracts without understanding them.
13. Do not solve upstream semantic failures inside downstream extractors.
14. Do not remove existing behavior merely to make one fixture pass.
15. Optimize for arbitrary real-world resumes, not the current fixture set.

---

## 13. Current Development State

### Being hardened (in-flight git changes)

- **Layout pipeline as default:** `parse_with_layout_pipeline` wired in `ResumeParser` and `ResumeRequestService`; `PARSER_MODE` setting (`auto`/`layout`/`legacy`).
- **Document IR stack:** `app/domain/document.py`, `reconstruction.py`, `layout.py`, `semantic_paths.py`, `semantic_inference.py`, `semantic_compat.py`, `semantic_resources.py`.
- **Extractor improvements:** Contact (names, phone, location, LinkedIn), education trailing-year, experience combined headers—with matching unit tests.
- **Classification/grouping tweaks:** Block classification and candidate grouping adjustments for education dates, experience splits, project title+date pairing.
- **Resources:** `section_aliases.json`, `skills.json` vocabulary expansion.
- **Test expansion:** Adversarial suite, semantic inference/compat/paths tests, layout integration tests.

### Remains unresolved

- Legacy vs layout behavioral parity not guaranteed.
- `EducationEntry` type annotation without definition.
- `parser_mode=auto` does not auto-select pipeline by PDF characteristics.
- No OCR / scanned PDF support.
- Summary/achievements/languages lack structured extractors.
- Confidence scores are mostly static constants (`ConfidenceScorer`).
- Table, footer, and global reading-order handling immature.
- Known contact location false-positive (`docs/known-issues.md`).
- Full fixture PDF set may be missing from repository for CI.

### Proposed future improvements (not implemented)

- Unified pipeline with layout stages optional but provenance always attached.
- True `auto` mode heuristics (e.g. multi-column detection → layout path).
- Structured language proficiency extraction.
- Table-aware region detection.
- Expand adversarial coverage to replace brittle fixture-only contracts over time.
- Centralize resource loading and caching.

---

*Generated from codebase inspection. Update this document when pipeline stages or boundaries change.*
