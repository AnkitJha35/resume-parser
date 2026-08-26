# Parser Baseline Checkpoint

**Date:** 2026-08-26  
**Branch:** `master` (checkpoint commit of layout-aware parser work)  
**Production entry:** `PARSER_MODE` ≠ `legacy` → `ResumeParser.parse_with_layout_pipeline`  
**Purpose:** Freeze current working behavior before structural refactors. No production behavior was changed for this document.

Related docs: [`PARSER_ARCHITECTURE.md`](PARSER_ARCHITECTURE.md), [`.cursor/rules/resume-parser.mdc`](../.cursor/rules/resume-parser.mdc).

---

## Pipeline under test

```text
PDF → TextBlock → Document IR → reconstruction → layout
  → semantic paths / unknown-heading inference → compatibility TextBlocks
  → classify → group → extractors → Resume
```

---

## Git snapshot (pre-commit working tree)

### Modified (tracked)

| Path | Nature |
|------|--------|
| `app/core/config.py` | `parser_mode` setting |
| `app/services/resume_request_service.py` | layout path as default |
| `app/pipeline/parser.py` | `parse_with_layout_pipeline` |
| `app/pipeline/stages/{sections,block_classification,candidate_grouping}.py` | incremental hardening |
| `app/extractors/{contact,education,experience}.py` | field robustness |
| `app/resources/{section_aliases,skills}.json` | vocabulary |
| Matching unit/integration tests | regression coverage |

### Untracked (included in checkpoint)

| Path | Nature |
|------|--------|
| `app/domain/document.py` | Document IR |
| `app/pipeline/stages/{layout,reconstruction,semantic_*}.py` | layout/semantic stack |
| `docs/PARSER_ARCHITECTURE.md` | architecture reference |
| `docs/PARSER_BASELINE.md` | this file |
| `.cursor/rules/resume-parser.mdc` | agent rules |
| `tests/adversarial/*` | adversarial suite |
| `tests/test_{document_ir,layout,layout_parser_integration,reconstruction,semantic_*}.py` | layer tests |

Approximate diff on previously tracked files: **+576 / −69** across 19 files (layout stack and docs are additional untracked volume).

---

## Test counts (2026-08-26)

Collected with `pytest` against the local `.venv`:

| Scope | Collected | Result |
|-------|-----------|--------|
| **Full suite** | **268** | **266 passed, 2 failed** |
| **Adversarial** (`tests/adversarial/`) | **62** | **62 passed** |
| Non-adversarial | 206 | (included in full suite) |
| Test modules (`tests/**/test_*.py`) | 38 files | — |
| Adversarial modules | 6 files | — |

### Adversarial modules

- `test_contact_variations.py` (26)
- `test_experience_layouts.py` (6)
- `test_field_variations.py` (2)
- `test_ocr_damage.py` (1)
- `test_provenance_invariants.py` (1)
- `test_section_inference_adversarial.py` (26)

### Known failing tests at checkpoint (not fixed here)

1. `tests/test_education_extractor.py::test_extracts_parenthesized_date_from_resume_2_layout_path`  
   - Observed: education degree becomes phone fragment `(212) 204-5342` instead of `Bachelor Of Arts in History,`.
2. `tests/test_layout_parser_integration.py::test_resume_2_layout_parser_exposes_correct_semantic_inputs`  
   - Observed: contact/header expectations fail against current unassigned/header routing for resume_2.

These failures are part of the baseline snapshot, not regressions introduced by this documentation-only checkpoint.

---

## Resume 1–7 outputs (`parse_with_layout_pipeline`)

Captured from `tests/fixtures/resume_{1..7}.pdf` on 2026-08-26. Values are **observed**, not target contracts.

### resume_1.pdf

| Field | Value |
|-------|--------|
| name | `ANGELA WILKINSO` (truncated) |
| email / phone | `youremail@gmail.com` / `895 555 555` |
| location | `Drive Harrisburg, PA` (incomplete) |
| linkedin | `linkedin.com/in/yourprofile` |
| summary | Present (admin assistant blurb) |
| skills (7) | Problem Solving, Adaptability, Collaboration, Strong Work Ethic, Time Management, Critical Thinking, Handling Pressure |
| experience | (Administrative Assistant, Redford & Sons, Boston, MA, 2018-09, None, True); (Secretary, Bright Spot LTD, Boston, MA, 2015-06, 2018-08, False) |
| education | Placeholder/noisy: `DEGREE NAME / MAJOR` / `University,`; second junk entry |
| projects / certs / achievements / languages | empty |
| pages | 1 |

### resume_2.pdf

| Field | Value |
|-------|--------|
| name | `DAVID PÉREZ` |
| email / phone / location / linkedin | **None** (contact lost to layout/unassigned routing) |
| summary | Present but letter-spaced title prefix noise |
| skills (4) | Microsoft Office, Spanish, Typing, Problem Solving |
| experience | 3 jobs: REDFORD & SONS (2019-09–present); BRIGHT SPOT LTD (2017-06–2019-08); SUNTRUST FINANCIAL (2015-06–2017-08) |
| education | Corrupted first entry `(212) 204-5342`; second Bachelor Of Arts in History / RIVER BROOK UNIVERSITY / 2015-05 |
| achievements | `AWARD TITLE / Brand`, `(May 2018)` |
| pages | 2 |

### resume_3.pdf

| Field | Value |
|-------|--------|
| name | `LILLIAN GA ADMINASSISTANT` (role bleed) |
| email | `youremail@gmail.co` (truncated) |
| location | Street + summary fragment bleed |
| experience | Same 2-job pattern as resume_1 (Redford / Bright Spot) |
| education | Placeholder degree line |
| certs | Noisy (`i 2007` / `CERTIFICATION #1`) |
| pages | 1 |

### resume_4.pdf

| Field | Value |
|-------|--------|
| name | `DAVID PEREZ` |
| email / phone / linkedin | Present |
| location | Multiline address largely recovered |
| experience | 3 entries; middle job location/description contamination |
| education | Degree/institution concatenated noise |
| pages | 3 |

### resume_5.pdf

| Field | Value |
|-------|--------|
| name | `MARGARET THOMASO` (truncated) |
| contact | Largely present |
| skills (8) | Soft skills + Microsoft Office, Typing, Spanish, QuickBooks |
| experience | Redford / Bright Spot (2 jobs) |
| education | Two placeholder degree rows with years 2011–2015 and 2007–2011 |
| pages | 1 |

### resume_6.pdf

| Field | Value |
|-------|--------|
| name | `Robert Richardson` |
| email / phone | Present |
| location | Incorrectly `Highlights` |
| skills (6) | Warehouse/soft competency set (HIGHLIGHTS→SKILLS inference working) |
| experience | Fragmented / empty designation entries mixed with real picker/packer jobs |
| education | Bachelor of Science: Shipping / Trade school |
| projects | Bullet glyph junk entries |
| certs | Noisy permission lines |
| pages | 1 |

### resume_7.pdf

| Field | Value |
|-------|--------|
| name | `MATTHEW ELIOT` |
| email / phone / linkedin | Present |
| summary | Present |
| skills (6) | Soft/executive-style skills (not tech stack from body) |
| experience | (Web Developer, Luna Web Design, New York, 2015-09, 2019-05, False) |
| education | Bachelor of Science: Computer Information Systems / Columbia University , NY / 2014 |
| certs | PHP Framework certificate line |
| pages | 1 |

---

## Intentional / known limitations (baseline)

Do not treat these as accidental omissions of this checkpoint:

1. **Legacy vs layout paths diverge** — only layout is production default; parity is not guaranteed.
2. **`PARSER_MODE=auto` behaves like layout** — no PDF-feature auto-selection.
3. **No OCR path** — `metadata.ocrUsed` is always `false`; image-only PDFs fail extraction.
4. **Tables not first-class** — `Page.tables` unused; table cells enter normal reading order.
5. **No cross-page header/footer suppression.**
6. **No-heading / creative-heading resumes** often become `UNKNOWN` / `UNASSIGNED`.
7. **Entry segmentation is weak** — `CandidateGroup` is a proto-entry; extractors still repair structure.
8. **Semantic inference confidence** (`≥0.70` / margin `≥0.15`) is score-mass share, not calibrated evidence.
9. **Public `Resume` schema drops provenance** — IDs survive internal IR/compat attrs only until extraction.
10. **Resource dictionaries** (`skills.json`, degrees, titles) are required for many matches; unbounded vocab growth is not the long-term strategy.
11. **Summary / achievements / languages** have no dedicated structured extractors.
12. **Fixture PDFs outside repo** (e.g. `fresher_hr_resume.pdf`, `swe_experienced_resume.pdf`) may be referenced by some tests but are not in `tests/fixtures/` in this tree (`resume_1`–`resume_7` only).
13. **Contact location / name truncation / section bleed** remain known quality gaps (see also `docs/known-issues.md`).

---

## How to re-verify this baseline

```bash
. .venv/bin/activate
python3 -m pytest -q
python3 -m pytest tests/adversarial -q
python3 -c "from pathlib import Path; from app.pipeline.parser import ResumeParser; p=ResumeParser();
[print(f, p.parse_with_layout_pipeline(Path(f'tests/fixtures/{f}').read_bytes()).personal.name)
 for f in [f'resume_{i}.pdf' for i in range(1,8)]]"
```

---

## Next work (out of scope for this checkpoint)

Structural hardening toward richer intermediate representations (typed layout regions, pre-semantic structural roles, first-class candidate entries) without expanding fixture-specific hacks. See architecture review notes and `PARSER_ARCHITECTURE.md` §13.
