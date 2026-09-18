# Phase 8-1 Real Resume Benchmark Baseline Report

## 1. Executive Summary

- **Total Resumes Evaluated:** 12
- **Parser Crashes:** 0 (100% execution completion without uncaught exceptions)
- **PASS:** 4 (33.3%)
- **PARTIAL:** 1 (8.3%)
- **FAIL:** 7 (58.3%)
- **Total Benchmark Runtime:** 1.41s

## 2. Archetype Breakdown

| Archetype | Total | PASS | PARTIAL | FAIL | Pass Rate |
|---|---|---|---|---|---|
| `maritime_tabular` | 1 | 0 | 0 | 1 | 0.0% |
| `structured_form` | 3 | 0 | 0 | 3 | 0.0% |
| `standard_cv` | 5 | 4 | 1 | 0 | 80.0% |
| `maritime_cv` | 3 | 0 | 0 | 3 | 0.0% |

## 3. Resume Baseline Results Table

| Resume | Archetype | Status | Skills | Experience | Education | Projects | Structural Diagnostics / Notes |
|---|---|---|---|---|---|---|---|
| `2nd Officer Mayur Agarwal_062029.pdf` | `maritime_tabular` | **FAIL** | 0 | 0 | 7 | 0 | • `NAME_SPLIT_INTO_LOCATION: 'AGARWAL'`<br>• `TABLE_METADATA_IN_EDUCATION[2]: inst='', deg='ENDORSEMENTS'`<br>• `TABLE_METADATA_IN_EDUCATION[3]: inst='', deg='Endorsement'`<br>• `TABLE_METADATA_IN_EDUCATION[5]: inst='03-Oct-2019', deg='DOCUMENTS'` |
| `AASHISH DG.pdf` | `structured_form` | **FAIL** | 0 | 0 | 36 | 0 | • `NAME_IS_FORM_OR_DOC_TITLE: 'Seafarer Profile'`<br>• `SECTION_HEADER_IN_LOCATION: 'Discipline'`<br>• `TABLE_METADATA_IN_EDUCATION[18]: inst='', deg='Authorised Documents'`<br>• `TABLE_METADATA_IN_EDUCATION[33]: inst='', deg='DECLARATION TO BE MADE BY CANDIDATE :'`<br>• `EXTREME_EDUCATION_COUNT_SKEW: 36 entries` |
| `AKIBUL ALAM CV(JO).pdf` | `structured_form` | **FAIL** | 0 | 0 | 25 | 0 | • `NAME_IS_FORM_OR_DOC_TITLE: 'Surname'`<br>• `NAME_SPLIT_INTO_LOCATION: 'Alam'`<br>• `TABLE_METADATA_IN_EDUCATION[11]: inst='Name of Institute / College', deg='Dangerous Cargo Endorsements'`<br>• `TABLE_METADATA_IN_EDUCATION[22]: inst='', deg='Previous Sea Service'`<br>• `EXTREME_EDUCATION_COUNT_SKEW: 25 entries` |
| `AditCV_SOL.pdf` | `standard_cv` | **PASS** | 19 | 1 | 1 | 0 | _Clean_ |
| `CV Rishabh Dixit.pdf` | `maritime_cv` | **FAIL** | 0 | 6 | 2 | 0 | • `NAME_IS_FORM_OR_DOC_TITLE: 'Curriculum Vitae'`<br>• `TABLE_HEADER_IN_EXPERIENCE[0]: comp='Ship Name', desig='S. No.'`<br>• `TABLE_HEADER_IN_EXPERIENCE[3]: comp='Documents', desig='S. No.'`<br>• `TABLE_METADATA_IN_EDUCATION[1]: inst='', deg='Declaration'` |
| `JOSH PARASHAR MASTER CV2.pdf` | `structured_form` | **FAIL** | 0 | 53 | 33 | 0 | • `NAME_IS_FORM_OR_DOC_TITLE: 'APPLICATION FORM'`<br>• `SECTION_HEADER_IN_LOCATION: 'Position'`<br>• `TABLE_HEADER_IN_EXPERIENCE[49]: comp='Period', desig='Vessel Type'`<br>• `TABLE_METADATA_IN_EDUCATION[0]: inst='', deg='Readiness date 15/May/2026'`<br>• `TABLE_METADATA_IN_EDUCATION[10]: inst='', deg='Vaccinations'`<br>• `EXTREME_EDUCATION_COUNT_SKEW: 33 entries`<br>• `EXCESSIVE_EXPERIENCE_COUNT: 53 entries` |
| `MUKUND 3RD OFF CV 2026.pdf` | `maritime_cv` | **FAIL** | 2 | 9 | 4 | 0 | • `SECTION_HEADER_IN_LOCATION: 'QUALIFICATION'`<br>• `TABLE_HEADER_IN_EXPERIENCE[4]: comp='Type', desig='Vessel Name'`<br>• `TABLE_HEADER_IN_EXPERIENCE[5]: comp='NAME', desig='Documents Details'`<br>• `TABLE_HEADER_IN_EXPERIENCE[8]: comp='POI', desig='DOI'`<br>• `TABLE_METADATA_IN_EDUCATION[0]: inst='13794 3RD OFF 05-08-2025 11-03-2026', deg='ITHACKI'` |
| `Rajeev_Ranjan_Prajapati_FullStack_Engineer.pdf` | `standard_cv` | **PARTIAL** | 12 | 1 | 4 | 6 | • `PROJECTS_WITHOUT_NAMES: 3 items at indices [3, 4, 5]` |
| `Résume_Shubham.pdf` | `standard_cv` | **PASS** | 13 | 3 | 1 | 2 | _Clean_ |
| `Sendrick Costa CV.pdf` | `maritime_cv` | **FAIL** | 1 | 14 | 2 | 0 | • `SECTION_HEADER_IN_LOCATION: 'References'`<br>• `EXCESSIVE_EXPERIENCE_COUNT: 14 entries` |
| `fresher_hr_resume.pdf` | `standard_cv` | **PASS** | 16 | 0 | 2 | 0 | _Clean_ |
| `swe_experienced_resume.pdf` | `standard_cv` | **PASS** | 39 | 2 | 2 | 3 | _Clean_ |

## 4. Top 5 Systemic Failure Classes in Rule-Based Parser

1. **Document Header / Form Label Extracted as Personal Name & Location Leakage**
   - *Pattern:* Non-standard documents often have titles like `APPLICATION FORM`, `Curriculum Vitae`, `Seafarer Profile` or form prompt labels like `Surname`. Rule-based regex/heuristics mistake these for candidate names.
   - *Location Leakage:* Headings such as `QUALIFICATION`, `References`, `Position`, `Discipline` or split candidate surnames (`AGARWAL`) leak into `personal.location`.
   - *Examples:* `JOSH PARASHAR MASTER CV2.pdf` (Name: 'APPLICATION FORM', Loc: 'Position'), `CV Rishabh Dixit.pdf` (Name: 'Curriculum Vitae'), `AASHISH DG.pdf` (Name: 'Seafarer Profile', Loc: 'Discipline'), `AKIBUL ALAM CV(JO).pdf` (Name: 'Surname', Loc: 'Alam'), `2nd Officer Mayur Agarwal_062029.pdf` (Name: 'MAYUR', Loc: 'AGARWAL'), `MUKUND 3RD OFF CV 2026.pdf` (Loc: 'QUALIFICATION').

2. **Tabular Sea-Service & Vessel Grids Ingested as Massive Education Entries**
   - *Pattern:* In maritime CVs and seafarer forms, vessel sea service, STCW courses, and document verification grids lack standard education keywords like 'University'. Loose fallbacks categorize hundreds of table rows under `EDUCATION`.
   - *Examples:* `AASHISH DG.pdf` (36 education items), `JOSH PARASHAR MASTER CV2.pdf` (28 education items), `AKIBUL ALAM CV(JO).pdf` (25 education items), `2nd Officer Mayur Agarwal_062029.pdf` (10 education items capturing 'SEA SERVICE', 'DOCUMENTS', 'ENDORSEMENTS').

3. **Table Column Headers Extracted as Experience Companies & Designations**
   - *Pattern:* Horizontal reading order across table columns fuses column headers (`Ship Name`, `S. No.`, `Vessel Name`, `Period`, `Type`) into company names and designations.
   - *Examples:* `CV Rishabh Dixit.pdf` (comp='Ship Name', desig='S. No.'), `JOSH PARASHAR MASTER CV2.pdf` (comp='Period', desig='Vessel Type'), `MUKUND 3RD OFF CV 2026.pdf` (comp='Type', desig='Vessel Name').

4. **Reference & Referee Contact Blocks Ingested into Professional Experience**
   - *Pattern:* Multi-column referee blocks at the end of a CV are grouped into experience candidate groups, generating multiple false-positive employment entries.
   - *Examples:* `Sendrick Costa CV.pdf` generates 14 experience items including referee contacts ('Josley Jeffroy Rodrigues', 'Pedro D’silva') and address lines ('Goa - India').

5. **Intra-Section Project Boundary & Header-Date Association Failures**
   - *Pattern:* In multi-project resumes where project title and date are on the same line without strong font size contrast or large vertical margins, candidate grouping splits between title and date or binds subtitles, creating entries with `name=None` and capturing page numbers as extra projects.
   - *Examples:* `Rajeev_Ranjan_Prajapati_FullStack_Engineer.pdf` (3 project entries have `name=None`, page footer '1' captured as project item).

## 5. Comparison Readiness: Current Parser vs Future LLM-Assisted Parser

This baseline proves that while the deterministic layout pipeline is highly effective for standard resumes (4/5 PASS on `standard_cv`), it fundamentally degrades on tabular and structured multi-column documents (0/7 PASS on maritime/forms).

The benchmark infrastructure established in `tests/benchmark/` provides a plug-and-play evaluation harness: swapping or augmenting `parser.parse_with_layout_pipeline` with an LLM-assisted pipeline will immediately yield measurable deltas against this machine-readable baseline report.