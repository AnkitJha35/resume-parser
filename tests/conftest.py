"""Shared test helpers for fixture availability."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Permanently removed historical fixtures (Phase 6+ corpus).
DELETED_FIXTURES = frozenset(
    {
        "resume_1.pdf",
        "resume_2.pdf",
        "resume_6.pdf",
        "resume_7.pdf",
        "single-column.pdf",
        "two-column.pdf",
        "image-only.pdf",
    }
)


def require_fixture(name: str) -> Path:
    """Return fixture path, or skip the calling test if the PDF is unavailable."""
    path = FIXTURES_DIR / name
    if path.exists():
        return path
    reason = (
        f"deleted historical fixture: {name}"
        if name in DELETED_FIXTURES
        else f"missing fixture: {name}"
    )
    pytest.skip(reason)
