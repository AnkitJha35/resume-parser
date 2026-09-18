# Phase 8-2A: LLM-Assisted Semantic Extraction Contract

This document defines the architecture, data contracts, validation invariants, and migration strategy for integrating an LLM-assisted semantic extraction layer into the resume parser.

---

## 1. Current-State Findings & Information Loss Points

The parser currently uses a two-phase architecture:
1. **Physical & Layout Document IR:** [`app/domain/document.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/domain/document.py), [`app/pipeline/stages/reconstruction.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/reconstruction.py), [`app/pipeline/stages/layout.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/layout.py), [`app/pipeline/stages/structural_roles.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/structural_roles.py).
2. **Deterministic Semantic Pathing & Heuristic Extractors:** [`app/pipeline/stages/semantic_paths.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/semantic_paths.py), [`app/pipeline/stages/semantic_compat.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/semantic_compat.py), [`app/pipeline/stages/candidate_grouping.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/candidate_grouping.py), [`app/extractors/`](file:///home/ankit-jha/my-workspace/resume-parser/app/extractors/).

### Structured Information Available in Existing Document IR
At the output of `interpret_layout()` and `build_structural_blocks()`:
- **Line & Span Level:** Exact text, bounding boxes (`x0, y0, x1, y1`), page number, typography (`font_size`, `bold`, `italic`, `font_name`), line reading order, and source span IDs.
- **Region Level:** Region ID, kind (`header`, `column`, `physical_region`), column ID, and spatial bounding box.
- **Structural Role Level:** Synthesized structural roles (`SECTION_HEADING`, `ENTRY_TITLE`, `ORGANIZATION`, `LOCATION`, `DATE`, `BULLET`, `DESCRIPTION`, `TECHNOLOGY`, `CREDENTIAL`, `CONTACT`, `HEADER`, `FOOTER`, `SIDEBAR`, `TABLE_CELL`, `UNKNOWN`) in [`app/domain/structural.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/domain/structural.py).

### Where Layout & Region Information Is Currently Lost
1. **Loss Point 1: Compatibility Flattening ([`app/pipeline/stages/semantic_compat.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/semantic_compat.py#L21-L38))**
   `semantic_sections_to_text_blocks()` flattens `SemanticDocument` into `dict[str, list[TextBlock]]`. This strips region boundaries, column identities, and horizontal geometry down to flat lists of text lines.
2. **Loss Point 2: Global Unassigned Contact Pool ([`app/pipeline/stages/semantic_compat.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/semantic_compat.py#L41-L45))**
   `semantic_header_to_text_blocks()` treats all `unassigned_lines` across the entire document as candidates for `ContactExtractor`. When tables, footers, or referee lists are unassigned, they are scanned by contact heuristics, directly causing location leakage (e.g. `QUALIFICATION`, `References`, `Discipline`).
3. **Loss Point 3: 1D Candidate Grouping ([`app/pipeline/stages/candidate_grouping.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/candidate_grouping.py#L305-L415))**
   `group_candidates()` clusters blocks in a section based on vertical gaps and lookahead. For tabular documents (e.g. maritime sea-service logs), horizontal column headers (`Ship Name`, `S. No.`, `Vessel Type`, `Period`) are read in 1D sequence, creating false experience/education entries.
4. **Loss Point 4: Line-Level Fallback Misclassification ([`app/extractors/education.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/extractors/education.py), [`app/extractors/experience.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/extractors/experience.py))**
   Heuristics lack multi-modal domain context: government registry forms (`AASHISH DG.pdf`) or crew bio-data (`AKIBUL ALAM CV(JO).pdf`) explode into 25–36 education items because any line with dates or alphanumeric codes in an unclosed section is captured as a credential.

---

## 2. Proposed SemanticInput Contract

The `SemanticInput` model serializes the layout Document IR into a clean, compact, non-redundant JSON payload. It preserves spatial boxes, reading order, page boundaries, column identities, generic table structure, and suggested structural roles without duplicating block storage.

### Key Architectural Decisions
- **Non-Redundant Block Storage:** Canonical blocks reside only in the top-level `blocks` list. `pages` holds only page-level spatial dimensions (`page_number`, `width`, `height`), preventing divergent representations and minimizing payload size.
- **Generic Table Representation:** Generic table properties (`table_id`, `row_index`, `column_index`, `cell_role`) are supported on any block without domain-specific schemas.
- **Suggested Roles via Existing Structural Roles:** `suggested_role` is populated directly from existing [`build_structural_blocks()`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/stages/structural_roles.py#L101), avoiding redundant heuristic classifiers.

### Python Definition ([`app/domain/semantic_contract.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/domain/semantic_contract.py))
```python
class SemanticBlockInput(BaseModel):
    block_id: str
    text: str
    page: int
    bbox: list[float]  # [x0, y0, x1, y1]
    region_id: str
    region_kind: str   # "header", "column", "physical_region"
    reading_order: int
    column_id: int | None = None
    is_bold: bool | None = None
    font_size: float | None = None
    suggested_role: str | None = None  # from existing StructuralRole

    # Generic table representation (None when outside a table)
    table_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    cell_role: str | None = None  # e.g. "HEADER", "DATA"

class SemanticPageMeta(BaseModel):
    page_number: int
    width: float | None = None
    height: float | None = None

class SemanticInput(BaseModel):
    document_id: str
    page_count: int
    pages: list[SemanticPageMeta] = Field(default_factory=list)
    blocks: list[SemanticBlockInput] = Field(default_factory=list)
```

### Concrete SemanticInput JSON Example
```json
{
  "document_id": "doc_mayur_01",
  "page_count": 1,
  "pages": [
    {
      "page_number": 1,
      "width": 595.0,
      "height": 842.0
    }
  ],
  "blocks": [
    {
      "block_id": "b_p1_0",
      "text": "MAYUR AGARWAL",
      "page": 1,
      "bbox": [50.4, 48.0, 280.5, 68.0],
      "region_id": "p1-r0",
      "region_kind": "header",
      "reading_order": 0,
      "is_bold": true,
      "font_size": 16.0,
      "suggested_role": "HEADER",
      "table_id": null,
      "row_index": null,
      "column_index": null,
      "cell_role": null
    },
    {
      "block_id": "b_p1_1",
      "text": "mayur.ag01@gmail.com | +91 98295 19017",
      "page": 1,
      "bbox": [50.4, 72.0, 310.0, 84.0],
      "region_id": "p1-r0",
      "region_kind": "header",
      "reading_order": 1,
      "is_bold": false,
      "font_size": 10.0,
      "suggested_role": "CONTACT",
      "table_id": null,
      "row_index": null,
      "column_index": null,
      "cell_role": null
    },
    {
      "block_id": "b_p1_2",
      "text": "Vessel Name",
      "page": 1,
      "bbox": [50.4, 140.0, 120.0, 152.0],
      "region_id": "p1-r1",
      "region_kind": "physical_region",
      "reading_order": 2,
      "is_bold": true,
      "font_size": 10.0,
      "suggested_role": "TABLE_CELL",
      "table_id": "tbl_sea_service",
      "row_index": 0,
      "column_index": 0,
      "cell_role": "HEADER"
    },
    {
      "block_id": "b_p1_3",
      "text": "Darya Shaan",
      "page": 1,
      "bbox": [50.4, 156.0, 130.0, 168.0],
      "region_id": "p1-r1",
      "region_kind": "physical_region",
      "reading_order": 3,
      "is_bold": false,
      "font_size": 10.0,
      "suggested_role": "TABLE_CELL",
      "table_id": "tbl_sea_service",
      "row_index": 1,
      "column_index": 0,
      "cell_role": "DATA"
    },
    {
      "block_id": "b_p1_4",
      "text": "Jan 2023 – May 2024",
      "page": 1,
      "bbox": [150.0, 156.0, 260.0, 168.0],
      "region_id": "p1-r1",
      "region_kind": "physical_region",
      "reading_order": 4,
      "is_bold": false,
      "font_size": 10.0,
      "suggested_role": "DATE",
      "table_id": "tbl_sea_service",
      "row_index": 1,
      "column_index": 1,
      "cell_role": "DATA"
    }
  ]
}
```

---

## 3. Proposed SemanticOutput Contract

The `SemanticOutput` model is the constrained schema produced by the LLM. It guarantees:
1. **Provenance Evidence vs Conservative Normalization:** Every entity field references the exact `source_block_ids` providing evidence. `raw_value` retains verbatim text, and `value` is restricted to safe, deterministic canonical normalizations.
2. **Provenance on Boolean Derived Fields:** Derived booleans such as `current` use `GroundedBool` with mandatory `source_block_ids`.
3. **Generic Semantic Classification & Exclusion:** An explicit `block_classifications` list where blocks are mapped to `SemanticBlockCategory` (`SECTION_HEADING`, `PERSONAL`, `EXPERIENCE`, `EDUCATION`, `PROJECT`, `SKILL`, `CERTIFICATION`, `REFERENCE`, `TABLE_HEADER`, `TABLE_DATA`, `BOILERPLATE`, `UNKNOWN`).
4. **Constrained Document Archetype:** `document_archetype` uses a fixed `DocumentArchetype` enum (`STANDARD_CV`, `MARITIME_CV`, `MARITIME_TABULAR`, `STRUCTURED_FORM`, `ACADEMIC_CV`, `UNKNOWN`). Archetype routing remains deterministic outside the LLM; the LLM's reported archetype is observational.
5. **Complete Provenance on All Entities:** Every entity—including individual skills, technologies, languages, achievements, and certifications—is provenance-grounded via `GroundedString`.

### Python Definition ([`app/domain/semantic_contract.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/domain/semantic_contract.py))
```python
class GroundedString(BaseModel):
    value: str                     # Canonical/normalized representation
    raw_value: str | None = None   # Verbatim raw text from source
    source_block_ids: list[str] = Field(default_factory=list)

class GroundedBool(BaseModel):
    value: bool                    # e.g. current employment boolean
    source_block_ids: list[str] = Field(default_factory=list)

class BlockClassification(BaseModel):
    block_id: str
    category: SemanticBlockCategory
    exclusion_reason: str | None = None

class GroundedPersonal(BaseModel):
    name: GroundedString | None = None
    email: GroundedString | None = None
    phone: GroundedString | None = None
    location: GroundedString | None = None
    linkedin: GroundedString | None = None
    github: GroundedString | None = None
    portfolio: GroundedString | None = None

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
    technologies: list[GroundedString] = Field(default_factory=list)
    startDate: GroundedString | None = None
    endDate: GroundedString | None = None
    current: GroundedBool | None = None
    source_block_ids: list[str] = Field(default_factory=list)

class SemanticOutput(BaseModel):
    document_archetype: DocumentArchetype = DocumentArchetype.UNKNOWN
    block_classifications: list[BlockClassification] = Field(default_factory=list)
    personal: GroundedPersonal = Field(default_factory=GroundedPersonal)
    summary: GroundedString | None = None
    skills: list[GroundedString] = Field(default_factory=list)
    experience: list[GroundedExperienceItem] = Field(default_factory=list)
    education: list[GroundedEducationItem] = Field(default_factory=list)
    projects: list[GroundedProjectItem] = Field(default_factory=list)
    certifications: list[GroundedString] = Field(default_factory=list)
    languages: list[GroundedString] = Field(default_factory=list)
    achievements: list[GroundedString] = Field(default_factory=list)
```

### Concrete SemanticOutput JSON Example
```json
{
  "document_archetype": "maritime_cv",
  "block_classifications": [
    { "block_id": "b_p1_0", "category": "PERSONAL" },
    { "block_id": "b_p1_1", "category": "PERSONAL" },
    { "block_id": "b_p1_2", "category": "TABLE_HEADER", "exclusion_reason": "table_column_header" },
    { "block_id": "b_p1_3", "category": "EXPERIENCE" },
    { "block_id": "b_p1_4", "category": "EXPERIENCE" }
  ],
  "personal": {
    "name": {
      "value": "MAYUR AGARWAL",
      "raw_value": "MAYUR AGARWAL",
      "source_block_ids": ["b_p1_0"]
    },
    "email": {
      "value": "mayur.ag01@gmail.com",
      "raw_value": "mayur.ag01@gmail.com",
      "source_block_ids": ["b_p1_1"]
    },
    "phone": {
      "value": "+919829519017",
      "raw_value": "+91 98295 19017",
      "source_block_ids": ["b_p1_1"]
    },
    "location": null
  },
  "summary": null,
  "skills": [],
  "experience": [
    {
      "company": {
        "value": "Darya Shaan",
        "raw_value": "Darya Shaan",
        "source_block_ids": ["b_p1_3"]
      },
      "designation": {
        "value": "Second Officer",
        "raw_value": "Second Officer",
        "source_block_ids": ["b_p1_0"]
      },
      "startDate": {
        "value": "2023-01",
        "raw_value": "Jan 2023",
        "source_block_ids": ["b_p1_4"]
      },
      "endDate": {
        "value": "2024-05",
        "raw_value": "May 2024",
        "source_block_ids": ["b_p1_4"]
      },
      "current": {
        "value": false,
        "source_block_ids": ["b_p1_4"]
      },
      "source_block_ids": ["b_p1_3", "b_p1_4"]
    }
  ],
  "education": [],
  "projects": [],
  "certifications": [],
  "languages": [],
  "achievements": []
}
```

---

## 4. Deterministic Validation Invariants

To eliminate systemic failure modes without fuzzy heuristics, [`validate_semantic_output()`](file:///home/ankit-jha/my-workspace/resume-parser/app/domain/semantic_contract.py#L292) enforces deterministic boundaries:

1. **Strict Provenance Integrity:**
   Every ID in `source_block_ids` must exist in `SemanticInput.blocks`. Empty `source_block_ids` on any grounded field triggers `MISSING_PROVENANCE`.
2. **Safe Deterministic Canonical Normalization Classes:**
   Canonical values must be supported by referenced source blocks under strictly defined normalization classes:
   - *Whitespace / Case / Punctuation Normalization:* Exact alphanumeric substring match (`norm_val in norm_src`).
   - *ISO Date Normalization:* If canonical value is in ISO format (`YYYY` or `YYYY-MM`), the year digits and month name/number must be explicitly present in source text (e.g. source `"July 2018"` supports canonical `"2018-07"`).
   - *Phone Digit Normalization:* All digits of the canonical phone must appear in order in the source text (e.g. source `"+91 98295 19017"` supports canonical `"+919829519017"`).
   - *Boolean Current Status Normalization:* `current=True` must reference source blocks containing accepted markers (`Present`, `current`, `currently employed`, `till date`, `now`, `ongoing`).
   - **Strictly Forbidden:** Token subset combinations, synonym replacement, semantic enrichment, company renaming (e.g. `"Darya Shaan"` cannot support `"Darya Shipping"`), title expansion (e.g. `"Senior Software Engineer"` cannot support `"Principal Software Engineer"`), fuzzy matching, or probabilistic thresholds. Values failing deterministic boundaries trigger `UNSUPPORTED_CANONICAL_VALUE`.
3. **Document Title & Label Rejection Guard:**
   If `personal.name` matches generic document titles (`APPLICATION FORM`, `Curriculum Vitae`, `Seafarer Profile`, `Surname`, `Resume`, `Biodata`), validation rejects it.
4. **Structural Header Region Location Scope:**
   Blocks mapped to `personal.location` must belong to the structural header region (`page == 1` and `region_kind == "header"`). No arbitrary y-coordinate thresholds are used.
5. **Table Column Header Exclusion:**
   `company`, `designation`, `degree`, or `institution` cannot consist of table column header tokens (`Ship Name`, `S. No.`, `Vessel Name`, `Period`, `Type`, `Documents Details`, `DOI`, `POI`, `Sea Service`, `Endorsements`, `Vaccinations`).
6. **Referee / Reference Separation:**
   Any block explicitly categorized as `SemanticBlockCategory.REFERENCE` (or `BOILERPLATE` or `TABLE_HEADER`) cannot be mapped into `experience` items.

---

## 5. Pipeline Integration & Fallback Strategy

### Integration Seam
In [`app/pipeline/parser.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/parser.py):
```
PDFExtractor.extract(raw_bytes)
  → reconstruct_document()
  → interpret_layout()
      │
      ├── [Mode: Legacy / Deterministic Layout Pipeline]
      │     → detect_region_aware_sections()
      │     → candidate_grouping()
      │     → Deterministic Extractors
      │     → validate_resume()
      │
      └── [Mode: Semantic LLM with Deterministic Fallback]
            → build_semantic_input(layout_document)
            → LLM Semantic Extractor (JSON mode, constrained schema)
            → validate_semantic_output(output, input)
                  │
                  ├── If Valid: semantic_output_to_resume(output)
                  └── If Invalid / Timeout: Fall back to Deterministic Layout Pipeline
```

### Deterministic Routing vs Observational Archetype
- **Deterministic Routing Outside the LLM:** The pipeline evaluates document features (`has_tables`, page count, multi-column presence) to decide whether to invoke the LLM layer or use the deterministic pipeline.
- **LLM Archetype Observability:** The LLM reports `document_archetype` (`STANDARD_CV`, `MARITIME_CV`, etc.) inside `SemanticOutput` for logging, diagnostics, and benchmarking, but does not control routing decisions.
- **Fail-Safe Fallback:** If the LLM call times out, encounters an API error, or produces invariant violations, the pipeline automatically falls back to `parse_with_layout_pipeline()`, ensuring zero production downtime.

---

## 6. Explicit Non-Goals for Phase 8-2A

- **No LLM Provider Implementation:** Do not bind the parser to vendor SDKs (Gemini, OpenAI, Anthropic) in this contract design phase.
- **No Public Schema Mutations:** The public `Resume` model ([`app/domain/resume.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/domain/resume.py)) remains 100% unchanged.
- **No Parser Behavior Modification:** The existing deterministic pipeline behavior and test suites remain untouched.
- **No Fixture-Specific Logic:** All schemas, categories, and invariants are fully generic.

---

## 7. Phase 8-2B: Provider-Independent Extraction Interface

Phase 8-2B operationalizes the Phase 8-2A data contracts through a minimal, vendor-agnostic interface and service seam:

- **Scope Boundary:**
  - **Phase 8-2A** defined the data contract (`SemanticInput`, `SemanticOutput`, `validate_semantic_output()`).
  - **Phase 8-2B** defines the provider-independent extraction interface and mock execution seam.
  - **Real LLM Providers** (Gemini, Anthropic, OpenAI, Ollama) are **intentionally deferred**.
  - The **deterministic validator remains the trust boundary** between proposed structures and public domain objects.
  - The **existing parser remains 100% unchanged**.

### Architecture Rule: The LLM as Interpreter, Not Source of Truth
The LLM does not originate ground truth:
```
Document IR (Layout & Spatial Coordinates)
      │
      ▼
build_semantic_input()           --> Ground truth layout & text evidence
      │
      ▼
SemanticExtractor.extract()      --> Proposes semantic candidate structure
      │
      ▼
validate_semantic_output()       --> Trust boundary: validates support & invariants
      │
      ▼
semantic_output_to_resume()      --> Only executes upon 100% valid invariant check
```

### Interface & Service Definitions
- **Interface Protocol ([`app/extractors/semantic_extractor.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/extractors/semantic_extractor.py)):**
  ```python
  @runtime_checkable
  class SemanticExtractor(Protocol):
      def extract(self, input_data: SemanticInput) -> SemanticOutput:
          ...
  ```
- **Pipeline Service Seam ([`app/pipeline/semantic_pipeline.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/pipeline/semantic_pipeline.py)):**
  ```python
  def parse_document_semantically(
      document: Document,
      extractor: SemanticExtractor,
      document_id: str = "doc-1",
  ) -> Resume:
      semantic_input = build_semantic_input(document, document_id=document_id)
      output = extractor.extract(semantic_input)
      violations = validate_semantic_output(output, semantic_input)
      if violations:
          raise SemanticValidationError(violations)
      return semantic_output_to_resume(output)
  ```
- **Mock Implementation ([`MockSemanticExtractor`](file:///home/ankit-jha/my-workspace/resume-parser/app/extractors/semantic_extractor.py#L48)):**
  A zero-network, zero-LLM, deterministic implementation that extracts grounded personal information, experience, and block classifications solely from evident source blocks for seam testing.

---

## 8. Table-Grid Findings & Contract Boundary

### Current State
- `app/domain/document.py` includes `Page.tables: list[object] = field(default_factory=list)`, but this list is currently unpopulated by default extractors.
- `app/pipeline/stages/layout.py` (`interpret_layout`) splits pages into `header`, `column`, and `physical_region` kinds without detecting tabular cell grids or table boundaries.
- As a result, `table_id`, `row_index`, `column_index`, and `cell_role` on `SemanticBlockInput` remain `None` during default layout extraction.

### Precise Contract Boundary
1. **Fields Already Exist on Input Contract:** `SemanticBlockInput` already includes `table_id: str | None`, `row_index: int | None`, `column_index: int | None`, and `cell_role: str | None`.
2. **Optional Upstream Presence:** These fields are optional because upstream layout-aware table detection is not yet implemented.
3. **Enrichment Precedes Semantic Input:** Any future table enrichment pass must populate these fields upstream before `build_semantic_input()` runs.
4. **No Extractor Inference of Table Geometry:** The semantic extractor (mock or future LLM) must never attempt to infer or guess missing table coordinates.
5. **No Fabricated Coordinates:** Missing table metadata must remain `None`; it must never be represented as fabricated or guessed coordinates.
6. **Provenance Preservation:** The future upstream table detector must preserve original line and block provenance (`line_ids`, `source_span_ids`, `bbox`).

---

## 9. Phase 8-3: Real LLM Adapter, Serialization & Structured Output Parsing

Phase 8-3 connects the provider-independent interface from Phase 8-2B to a production-ready LLM adapter using the Gemini REST API, backed by deterministic serialization, structured output parsing, and invariant validation.

### Architecture Flow
```text
SemanticInput
    ↓
serialize_semantic_input()       --> Compact, deterministic JSON of canonical blocks
    ↓
build_extraction_prompt()        --> System rules + serialized document blocks
    ↓
Gemini REST API                  --> generateContent with responseMimeType="application/json"
    ↓
Envelope unwrapping              --> Candidate content parts extraction
    ↓
parse_semantic_output()          --> Strict Pydantic validation into SemanticOutput
    ↓
validate_semantic_output()       --> Deterministic trust boundary (provenance & rules)
    ↓
semantic_output_to_resume()      --> Resume domain model
```

### 1. SemanticInput Serialization (`app/extractors/semantic_prompt.py`)
- `serialize_semantic_input()` formats blocks deterministically sorted by `(page, reading_order, block_id)`.
- Preserves exact `block_id`, `text`, `page`, `bbox`, `region_id`, `region_kind`, `reading_order`, `column_id`, style hints (`is_bold`, `font_size`), `suggested_role`, and generic table metadata (`table_id`, `row_index`, `column_index`, `cell_role`).
- Contains zero duplicate block copies; blocks are referenced strictly by their canonical ID.

### 2. Provider-Neutral System Prompt (`SEMANTIC_EXTRACTION_SYSTEM_PROMPT`)
- Requires that every non-null value reference one or more supplied `source_block_ids`.
- Forbids semantic invention, company renaming, title expansion, date fabrication, and synonym replacement.
- Forbids treating document titles/form labels as personal names.
- Forbids treating table column headers as company/designation/degree values.
- Forbids classifying referee contacts as employment experience.
- Contains no fixture-specific rules or benchmark resume names.

### 3. Structured Response Parsing (`parse_semantic_output()`)
- Strips markdown code fences if present.
- Deserializes JSON and validates strictly against `SemanticOutput`.
- Rejects malformed JSON, array roots, or incompatible top-level structures with `SemanticExtractionError`.
- Preserves `null` fields; never coerces ungrounded model text into valid entities.

### 4. Provider Adapter Boundary (`app/extractors/providers/gemini.py`)
- `GeminiSemanticExtractor` implements `SemanticExtractor.extract(SemanticInput) -> SemanticOutput`.
- Makes HTTP POST requests to Google's Generative Language REST API (`models/{model}:generateContent`) with `temperature=0.0`.
- Isolates `httpx` dependencies inside `app/extractors/providers/gemini.py`. Domain and pipeline modules remain completely provider-agnostic.
- Maps HTTP status errors, timeouts, network failures, and empty candidates to `SemanticExtractionError`.

### 5. Deterministic Validation as the Trust Boundary
- The LLM is an interpreter, never the source of truth.
- `parse_document_semantically()` always executes `validate_semantic_output()` on the parsed `SemanticOutput`.
- If violations occur (e.g. hallucinated block IDs, document titles as names, ungrounded values), the pipeline halts with `SemanticValidationError`, preventing invalid output from reaching `Resume`.

### 6. Configuration Requirements
- Added to `Settings` (`app/core/config.py`):
  - `gemini_api_key: str | None = None`
  - `gemini_model: str = "gemini-2.5-flash"`
  - `gemini_timeout: float = 30.0`
- Resolves configuration hierarchically: explicit constructor arguments $\rightarrow$ environment variables (`GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_TIMEOUT`) $\rightarrow$ `Settings`.
- Importing the application or running tests never fails if the API key is missing. The provider raises `SemanticExtractionError` only at invocation time if credentials are absent.

### 7. Why the Normal Parser Does Not Automatically Call the LLM
- Standard resumes (`standard_cv`) achieve an 80% pass rate on the deterministic layout pipeline with zero token cost and ~0.15s latency.
- Automatically calling an LLM for all parses would introduce external network latency (1–3s), token costs, and rate limits into production message consumers.
- The semantic pipeline is exposed as an opt-in/routing seam via `parse_document_semantically()`.

### 8. Table Detection Deferred
- As established in Phase 8-2B, upstream table detection remains intentionally deferred. `SemanticBlockInput` supports table coordinates, but un-enriched blocks retain `table_id=None`. The extractor does not invent missing table coordinates.

---

## 10. Phase 8-4: Real LLM Semantic Benchmark Evaluation

Phase 8-4 introduces instrumentation and a repeatable benchmark harness to evaluate the Gemini semantic pipeline against the 12 real resume PDFs.

### 1. Opt-in Evaluation Architecture
- **No Live LLM Calls in Normal Pytest:** Standard test commands (`pytest -q`, `pytest -q tests/benchmark`) run purely offline with mocked HTTP transport. Live Gemini extraction requires explicit execution:
  ```bash
  python -m tests.benchmark.run_semantic [--model gemini-2.5-flash] [--timeout 30.0] [--compare]
  ```
- **Fail-Fast Credential Check:** If `GEMINI_API_KEY` is not present in the environment or configuration, the CLI fails cleanly without making network requests.
- **Safety & Secret Hygiene:** API keys and credentials are never printed to the console or serialized into benchmark JSON reports.

### 2. Deterministic Baseline as the Control
- The existing deterministic layout benchmark (`tests/benchmark/runner.py`) remains the unchanged baseline.
- The comparison helper (`tests/benchmark/compare.py`) compares field-level extractions between deterministic and semantic results across identical fixtures.
- Categorizes deltas using objective terms (`improved`, `regressed`, `unchanged`, `validation failure`, `extraction failure`) rather than synthetic "accuracy scores".

### 3. Separate Tracking of Validation Failures
- The benchmark runner (`SemanticBenchmarkRunner`) explicitly distinguishes between:
  - **Extraction Failures (`EXTRACTION_ERROR`):** Provider API errors, HTTP status codes, network timeouts, malformed JSON envelopes.
  - **Validation Failures (`VALIDATION_ERROR`):** Responses where the model proposed output that violated deterministic domain invariants (e.g. document titles as names, hallucinated block IDs, unsupported strings).
- An individual failure never crashes the benchmark run. Every fixture executes independently.

### 4. Token & Cost Instrumentation
- Token metrics (`prompt_tokens`, `output_tokens`, `total_tokens`) are captured solely when provided in the provider's response envelope (`usageMetadata`).
- If usage metadata is omitted by the provider or mock, it remains `None`. No extra API calls or synthetic estimates are introduced.

---

## 11. Local Development LLM Provider: Ollama

To enable local, zero-cloud development and evaluation without external API fees or token quotas, the semantic pipeline introduces a native Ollama adapter.

### 1. Architecture Alignment
- **Provider Protocol Equivalence:** `OllamaSemanticExtractor` ([`app/extractors/providers/ollama.py`](file:///home/ankit-jha/my-workspace/resume-parser/app/extractors/providers/ollama.py)) implements the same `SemanticExtractor` interface (`extract(SemanticInput) -> SemanticOutput`) as `GeminiSemanticExtractor` and `MockSemanticExtractor`.
- **Zero Domain Contamination:** All HTTP calls to Ollama's `/api/chat` and schema formatting remain inside the provider module. Domain models and pipeline validation remain 100% provider-agnostic.
- **Structured Schema Handling:** Uses the shared `resolve_schema_defs(SemanticOutput)` helper to pass an inlined JSON Schema to Ollama's `format` parameter.

### 2. Configuration & Defaults
- **Base URL:** `http://localhost:11434` (configurable via `OLLAMA_BASE_URL` or constructor).
- **Default Model:** `qwen2.5-coder:7b` (configurable via `OLLAMA_MODEL` or constructor).
- **Timeout:** `120.0s` (configurable via `OLLAMA_TIMEOUT` or constructor) to accommodate local inference times.
- **No Credentials Required:** Operates entirely offline without API keys or external authentication.

### 3. Pre-Flight Availability Checks
- The CLI command (`python -m tests.benchmark.run_semantic --provider ollama`) executes `OllamaSemanticExtractor.check_availability()` against `/api/tags` prior to benchmark execution.
- If Ollama is not running or the requested model is not downloaded, it prints an actionable message with the exact `ollama serve` or `ollama pull <model>` command required.

### 4. Provider-Neutral Benchmark Harness
- `SemanticBenchmarkRunner` accepts an injected `SemanticExtractor`, operating identically whether driven by Ollama, Gemini, or a Mock.

---

## 12. Phase 10C: Two-Pass Semantic Benchmark & Quality Gate

Phase 10C introduces benchmarking for the production concurrent two-pass Gemini semantic extractor against the 12-resume regression corpus and comparison with the Phase 8-5 single-pass Candidate-B baseline.

### 1. Benchmark Representations
- **`single_pass_candidate_b`:** Single-pass semantic extraction using compact Candidate-B serialization and full monolithic `SemanticOutput` schema.
- **`two_pass_candidate_b`:** Production concurrent two-pass semantic extraction using compact Candidate-B serialization with Pass 1 (`PersonalSemanticOutput`) and Pass 2 (`BodySemanticOutput`) executed concurrently and merged deterministically.

### 2. Exact Benchmark Commands

#### a) Candidate-B Single Pass Benchmark
```bash
python -m tests.benchmark.run_semantic --provider gemini --representation single_pass_candidate_b [--model gemini-3.5-flash-lite]
```

#### b) Candidate-B Two Pass Benchmark (Production Default)
```bash
python -m tests.benchmark.run_semantic --provider gemini --representation two_pass_candidate_b [--model gemini-3.5-flash-lite]
```

#### c) Semantic Comparison (Baseline vs Candidate)
```bash
python -m tests.benchmark.compare --baseline tests/benchmark/reports/candidate_b_single_pass_baseline.json --candidate benchmark_results/<candidate_run>.json [--markdown]
```
Or directly during benchmark run:
```bash
python -m tests.benchmark.run_semantic --provider gemini --representation two_pass_candidate_b --compare-baseline tests/benchmark/reports/candidate_b_single_pass_baseline.json
```

#### d) Production Quality Gate Evaluation
```bash
python -m tests.benchmark.quality_gate --results benchmark_results/<candidate_run>.json [--output-json <path.json>]
```
Or directly during benchmark run:
```bash
python -m tests.benchmark.run_semantic --provider gemini --representation two_pass_candidate_b --quality-gate
```

### 3. Metric Aggregation & Reporting
- **Aggregate Telemetry:** In two-pass mode, prompt tokens, output tokens, total tokens, latency, and retries are aggregated across both Gemini calls while preserving pass-specific telemetry under `usage.pass_metadata`.
- **Quality Gate Invariants:** Enforces `0` validation failures, `0` provider errors, `min_pass_rate_pct >= 80%`, `max_fail_rate_pct = 0%`, and 10 historical regression rules.
- **Sanitized Outputs:** Zero API keys, prompts, or PII appear in comparison tables or serialized reports.
