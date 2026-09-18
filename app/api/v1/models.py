"""Pydantic schemas for the test harness API."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ParseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    ERROR = "ERROR"


class HealthResponse(BaseModel):
    status: str = "ok"
    parser_version: str = "1.0.0"
    application_version: str = "1.0.0"


class ConfigResponse(BaseModel):
    provider: str
    model: str
    representation: str
    parser_version: str
    ocr_available: bool
    ocr_engine: str | None = None
    two_pass: bool = False
    fallback_enabled: bool = False


class ParseMetadata(BaseModel):
    filename: str
    pageCount: int
    ocrUsed: bool
    provider: str
    model: str
    representation: str
    latencyMs: float
    requestCount: int = 1


class ProvenanceRecord(BaseModel):
    canonicalField: str
    extractedValue: Any = None
    rawValue: str | None = None
    sourceBlockIds: list[str] = Field(default_factory=list)
    sourceTexts: list[str] = Field(default_factory=list)
    pageNumbers: list[int] = Field(default_factory=list)
    confidence: float | None = None


class ParseResponse(BaseModel):
    success: bool
    status: ParseStatus = ParseStatus.SUCCESS
    resume: dict[str, Any] | None = None
    violations: list[str] = Field(default_factory=list)
    metadata: ParseMetadata
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class DiagnosticBlockItem(BaseModel):
    block_id: str
    page_number: int
    reading_order: int
    text: str
    bbox: list[float]
    suggested_role: str | None = None
    table_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    is_continuation: bool = False
    section_hint: str | None = None


class DiagnosticPageItem(BaseModel):
    page_number: int
    width: float
    height: float
    region_count: int


class DiagnosticRegionItem(BaseModel):
    page_number: int
    region_id: str
    bbox: list[float]
    reading_order: int
    region_type: str
    line_count: int


class DiagnosticTableItem(BaseModel):
    table_id: str
    page_number: int
    row_count: int
    column_count: int
    headers: list[str] = Field(default_factory=list)


class DiagnosticResponse(BaseModel):
    page_count: int
    block_count: int
    archetype: str
    pages: list[DiagnosticPageItem] = Field(default_factory=list)
    regions: list[DiagnosticRegionItem] = Field(default_factory=list)
    blocks: list[DiagnosticBlockItem] = Field(default_factory=list)
    tables: list[DiagnosticTableItem] = Field(default_factory=list)
    truncated: bool = False


class FixtureItem(BaseModel):
    id: str
    suite: str
    filename: str
    candidate_name: str | None = None
    target_domain: str | None = None
    archetype: str | None = None
    page_count_estimate: int | None = None
    has_tables: bool | None = None
    notes: str | None = None


class BenchmarkRequest(BaseModel):
    suite: str = "regression_12"
    provider: str | None = None
    model: str | None = None
    representation: str | None = None


class ApiErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int
    status: ParseStatus = ParseStatus.ERROR
    detail: dict[str, Any] | None = None
