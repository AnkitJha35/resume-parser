# Production Deployment Guide

## Overview

The Resume Parser Service is containerized as an immutable, non-root Docker image designed to run standalone or orchestrated in Kubernetes, Docker Compose, AWS ECS, or Google Cloud Run.

---

## Canonical Architecture

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

## Runtime Dependencies & System Requirements

- **Base Image**: `python:3.12-slim`
- **System Packages**: Only `curl` is installed (used for container healthcheck execution).
- **No Native OCR/C System Dependencies**:
  - `PyMuPDF` (`pymupdf==1.28.2`) includes bundled native MuPDF binaries.
  - **No `libmupdf-dev`**, **no `tesseract-ocr`**, and **no `poppler-utils`** are needed or installed in production.
- **Runtime User**: Runs as unprivileged `appuser:appuser` (`uid=1000, gid=1000`).

---

## In-Memory Request & Temporary File Lifecycle

- **Upload Streaming**: Uploaded multipart PDF files are streamed directly into memory buffer (`bytes`).
- **Direct PyMuPDF Ingestion**: PyMuPDF consumes the bytes via `fitz.open(stream=pdf_bytes, filetype="pdf")`.
- **Zero Disk Writes**: **No temporary files are created on disk** throughout the entire detection, layout reconstruction, and extraction pipeline.
- **Upload Boundary**: The 10 MB limit (`MAX_UPLOAD_SIZE = 10485760` bytes) is enforced directly against the decoded PDF payload bytes, rejecting oversized files with `HTTP 413 Payload Too Large`.

---

## Versioning & Tagging Strategy

Images follow Semantic Versioning (`MAJOR.MINOR.PATCH`):
- `resume-parser:1.0.0`: Immutable release candidate tag.
- `resume-parser:1.0`: Minor tracking tag.
- `resume-parser:latest`: Latest production release.

---

## Docker Build & Run

### 1. Build the Production Image
```bash
docker build -t resume-parser:1.0.0 -t resume-parser:latest .
```

### 2. Run the Container
Inject runtime secrets via environment variables:
```bash
docker run -d \
  --name resume_parser \
  -p 8000:8000 \
  -e GEMINI_API_KEY="your-gemini-api-key" \
  -e WORKERS=2 \
  resume-parser:latest
```

---

## Healthcheck & Readiness

### 1. Container Liveness (`/health`)
Used by Kubernetes probes and Docker `HEALTHCHECK`. Does not invoke external APIs.
- **Method**: `GET /health`
- **HTTP Status**: `200 OK`
- **Exact Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. API v1 Status (`/api/v1/health`)
Validates application routing and version metadata.
- **Method**: `GET /api/v1/health`
- **HTTP Status**: `200 OK`
- **Exact Response**:
  ```json
  {
    "status": "ok",
    "parser_version": "1.0.0",
    "application_version": "1.0.0"
  }
  ```

### 3. Active Configuration (`/api/v1/config`)
Inspects active LLM provider, models, and parser settings without exposing credentials.
- **Method**: `GET /api/v1/config`
- **HTTP Status**: `200 OK`

---

## API Status Terminology (`ParseStatus`)

The service outputs standard uppercase statuses in the response envelope:
- `SUCCESS`: Complete parse with valid contact details and experience/education.
- `PARTIAL`: Grounded parse with missing contact or experience/education sections.
- `VALIDATION_FAILED`: Grounding or token-coverage failure (returned with HTTP 200 or 422 depending on conversion status).
- `EXTRACTION_FAILED`: Downstream provider returned unparseable or incomplete response.
- `ERROR`: File format error, payload excess, timeout, or configuration exception.

---

## LLM Request Invariant

The pipeline strictly guarantees **exactly one LLM request per resume**. There are no secondary prompt roundtrips, no dynamic re-prompt loops, and no hidden recovery LLM passes.

---

## Frontend Architecture & Production Serving

The frontend is a React 19 + TypeScript SPA in `frontend/`.

- **Build Output**: `npm run build` generates optimized static assets in `frontend/dist/`.
- **Backend Configuration**: Set `VITE_API_BASE_URL` at build time or runtime (defaults to `http://localhost:8000`).
- **Production Serving**: Serve `frontend/dist/` via Nginx, Caddy, Cloudflare Pages, or AWS S3/CloudFront reverse-proxying requests to `/api/v1/` on the FastAPI container.

---

## API Usage Examples

### Parse Resume
```bash
curl -X POST http://localhost:8000/api/v1/parse \
  -F "file=@/path/to/candidate_resume.pdf"
```

### Parse Diagnostics (Layout IR without LLM)
```bash
curl -X POST http://localhost:8000/api/v1/parse/diagnostic \
  -F "file=@/path/to/candidate_resume.pdf"
```
