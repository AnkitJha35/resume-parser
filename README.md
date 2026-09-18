# Resume Parser Service

Production-ready resume parsing service providing layout-aware document reconstruction, spatial table binding, single-request semantic extraction, strict provenance grounding, and interactive diagnostic tooling.

---

## Architecture

```
Upload
  ↓
Validation (PDF header %PDF-, file extension, 10MB payload limit)
  ↓
PDF Ingestion (In-memory PyMuPDF stream — zero temporary files)
  ↓
Layout / Reading Order (Spatial reconstruction, multi-column topological ordering)
  ↓
Structural / Table Binding (Archetype-agnostic table & form cell binding)
  ↓
Semantic Serialization (Compact block IDs `b_p<page>_<idx>`)
  ↓
ONE LLM Request (Strict invariant: exactly 1.00 LLM call per resume)
  ↓
Provenance Repair & Validation (Token coverage ≥ 0.70, cross-entity boundary enforcement)
  ↓
Resume Envelope (ParseStatus, canonical schema, complete provenance map)
  ↓
Frontend (React SPA with interactive bounding box & provenance inspection)
```

---

## Key Guarantees & Runtime Characteristics

1. **Purely In-Memory Lifecycle**: Request bytes are streamed directly into memory. PyMuPDF processes documents in-memory via `fitz.open(stream=..., filetype="pdf")`. **No temporary files are created on disk**, eliminating file descriptor leaks and disk cleanup overhead.
2. **Deterministic Single-Call Extraction**: Exactly **one LLM request per resume**. No secondary repair passes, recovery loops, or sub-section prompt roundtrips.
3. **Strict Provenance & Grounding**: Extracted fields cite exact source block IDs with bounding boxes. Hallucinations or ungrounded attributes trigger validation failures (`VALIDATION_FAILED`) or deterministic repairs without LLM retries.
4. **Runtime Dependencies**: Self-contained PyMuPDF wheels provide bundled MuPDF binaries. **No system Poppler, Tesseract, or `libmupdf-dev` packages are required at runtime**. The production container only adds `curl` for container health monitoring.
5. **Upload Boundaries**: The `10 MB` (`MAX_UPLOAD_SIZE = 10485760` bytes) limit protects the extracted PDF payload bytes after multipart transmission.

---

## API Status Terminology (`ParseStatus`)

All responses return a typed uppercase `ParseStatus`:

- `SUCCESS`: Validated resume with candidate identity (name), contact information (email/phone), and professional/educational history.
- `PARTIAL`: Validated resume missing one or more foundational pillars (e.g. contact info or history), but passing all grounding checks.
- `VALIDATION_FAILED`: Extracted content violated grounding contracts (hallucinations, token coverage < 0.70, cross-entity boundary contamination).
- `EXTRACTION_FAILED`: Downstream LLM provider returned empty, malformed, or truncated semantic JSON.
- `ERROR`: System, configuration, or transport exception (e.g. invalid file format, payload too large, gateway timeout).

---

## Health Endpoints

### Container Liveness (`/health`)
- **Method**: `GET /health`
- **Status**: `200 OK`
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```
- **Semantics**: Lightweight container probe without calling external LLM providers or databases. Used by container orchestrators.

### API v1 Health & Versions (`/api/v1/health`)
- **Method**: `GET /api/v1/health`
- **Status**: `200 OK`
- **Response**:
  ```json
  {
    "status": "ok",
    "parser_version": "1.0.0",
    "application_version": "1.0.0"
  }
  ```
- **Semantics**: Confirms API routing and returns service & parser version metadata.

### Active Configuration (`/api/v1/config`)
- **Method**: `GET /api/v1/config`
- **Status**: `200 OK`
- **Semantics**: Reports active provider, model, representation format, and OCR engine without exposing credentials.

---

## Quickstart

### Local Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env and supply GEMINI_API_KEY

# 3. Run FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build production image
docker build -t resume-parser:latest .

# Run non-root container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY="your-gemini-api-key" \
  resume-parser:latest
```

---

## Frontend Setup & Production Build

The UI is a React 19 + TypeScript SPA located in `frontend/`.

### Development
```bash
cd frontend
npm install
npm run dev
```
The dev server runs at `http://localhost:5173` and connects to the backend at `http://localhost:8000` by default.

### Production Build & Serving
```bash
cd frontend
# Configure backend URL if not localhost:8000
export VITE_API_BASE_URL="http://localhost:8000"

# Build static assets
npm run build
```
Static artifacts are output to `frontend/dist/` and can be served using Nginx, Caddy, Cloudflare Pages, AWS S3/CloudFront, or any static file host.

