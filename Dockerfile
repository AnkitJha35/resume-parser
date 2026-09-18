# Production Dockerfile for Resume Parser Service
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    WORKERS=1

# Install runtime system utilities (curl for container healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root application user
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Install pinned production dependencies deterministically
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and project metadata
COPY pyproject.toml /app/
COPY app/ /app/app/

# Install application package
RUN pip install --no-cache-dir --no-deps -e .

# Switch ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
