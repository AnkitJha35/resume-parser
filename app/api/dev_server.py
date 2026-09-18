"""Local development FastAPI application for resume-parser test harness."""

from __future__ import annotations

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import router as api_v1_router
from app.core.config import Settings
from app.core.logging import configure_logging

logger = logging.getLogger("app.api.dev_server")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        configure_logging()
    except Exception:
        pass
    try:
        settings = Settings()
        logger.info(
            "Resume Parser Dev Server started. Active provider: %s, model: %s",
            settings.semantic_provider,
            settings.gemini_model if settings.semantic_provider == "gemini" else "configured",
        )
    except Exception as exc:
        logger.warning("Starting dev server with default configuration (%s)", exc)
    yield

app = FastAPI(
    title="Resume Parser Local Test Harness",
    description="Local developer API and test harness for resume parsing, structural diagnostics, and benchmark evaluation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Allow local frontend development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_v1_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred while processing the request.",
            "statusCode": 500,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> dict[str, str]:
    return {
        "service": "Resume Parser Local Test Harness",
        "docs": "/docs",
        "health": "/health",
        "api_v1_health": "/api/v1/health",
        "api_v1_config": "/api/v1/config",
    }
