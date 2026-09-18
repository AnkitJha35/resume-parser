# Resume Parser Service

Production resume parsing foundation service providing layout-aware semantic parsing, provenance validation, and diagnostic APIs.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
# Build production image
docker build -t resume-parser:latest .

# Run container
docker run -p 8000:8000 resume-parser:latest
```

## Endpoints

- `GET /health`: Healthcheck endpoint
- `GET /api/v1/health`: API v1 health & version status
- `GET /api/v1/config`: Active provider configuration
- `POST /api/v1/parse`: Production resume parsing endpoint
- `POST /api/v1/parse/diagnostic`: Layout and structural IR diagnostic endpoint
