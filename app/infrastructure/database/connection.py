"""Database connection and engine lifecycle management."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Connection, Engine, create_engine, text

from app.core.config import Settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def get_engine(settings: Settings | None = None) -> Engine:
    """Return singleton SQLAlchemy engine initialized from application settings."""
    global _engine
    if _engine is not None:
        return _engine

    cfg = settings or Settings()
    if not cfg.database_url:
        raise ValueError(
            "DATABASE_URL is not configured. Set DATABASE_URL environment variable to connect to PostgreSQL."
        )

    _engine = create_engine(
        cfg.database_url,
        pool_size=cfg.database_pool_size,
        max_overflow=cfg.database_max_overflow,
        pool_timeout=cfg.database_pool_timeout,
        pool_pre_ping=True,
    )
    return _engine


def reset_engine() -> None:
    """Dispose and reset the cached engine (used in tests)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


@contextmanager
def get_connection(settings: Settings | None = None) -> Generator[Connection, None, None]:
    """Provide a transactional database connection."""
    engine = get_engine(settings)
    with engine.connect() as conn:
        yield conn


def try_acquire_document_advisory_lock(conn: Connection, content_hash: str) -> bool:
    """Attempt to acquire a PostgreSQL transaction-scoped advisory lock on a document's content hash.

    The lock is automatically released when the transaction ends (commit or rollback).
    """
    stmt = text("SELECT pg_try_advisory_xact_lock(hashtext(:hash))")
    result = conn.execute(stmt, {"hash": content_hash}).scalar()
    return bool(result)
