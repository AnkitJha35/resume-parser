"""Parse configuration hashing implementation."""

from __future__ import annotations

import hashlib


def compute_parse_config_hash(
    parser_version: str,
    schema_version: str,
    provider: str,
    model: str,
    representation: str,
) -> str:
    """Compute deterministic SHA-256 hash of normalized parser configuration.

    Normalizes every component by stripping surrounding whitespace and converting to lowercase.
    Components are joined with a pipe delimiter ('|').
    """
    normalized = (
        f"{str(parser_version).strip().lower()}|"
        f"{str(schema_version).strip().lower()}|"
        f"{str(provider).strip().lower()}|"
        f"{str(model).strip().lower()}|"
        f"{str(representation).strip().lower()}"
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
