# Production Deployment Guide

## Overview

The Resume Parser Service is containerized as an immutable, non-root Docker image designed to run standalone or orchestrated in Kubernetes, Docker Compose, AWS ECS, or Google Cloud Run.

## Versioning & Tagging Strategy

Images follow Semantic Versioning (`MAJOR.MINOR.PATCH`):
- `resume-parser:1.0.0`: Immutable release candidate tag.
- `resume-parser:1.0`: Minor tracking tag.
- `resume-parser:latest`: Latest production release.

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

## Healthcheck & Readiness

- **Container Healthcheck**: `GET /health` returns HTTP 200 `{"status": "ok"}` without calling LLMs or external services.
  ```bash
  curl http://localhost:8000/health
  ```
- **API Status**: `GET /api/v1/health` returns version and service status.
  ```bash
  curl http://localhost:8000/api/v1/health
  ```
- **Active Configuration**: `GET /api/v1/config` returns active provider, model, and parser settings without exposing credentials.
  ```bash
  curl http://localhost:8000/api/v1/config
  ```

## API Usage

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
