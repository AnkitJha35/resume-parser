from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_RESOURCE_DIR = Path(__file__).resolve().parents[2] / "resources"


class SemanticResources:
    """Cached semantic vocabularies independent of extractor implementations."""

    _cache: dict[str, Any] = {}

    @classmethod
    def vocabulary(cls, name: str) -> dict[str, str]:
        if name not in cls._cache:
            path = _RESOURCE_DIR / f"{name}.json"
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            cls._cache[name] = value if isinstance(value, dict) else {}
        return dict(cls._cache[name])

    @classmethod
    def skills(cls) -> dict[str, str]:
        return cls.vocabulary("skills")
