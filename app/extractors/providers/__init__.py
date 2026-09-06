"""Provider-specific semantic extraction adapters."""

from __future__ import annotations

from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.providers.ollama import OllamaSemanticExtractor
from app.extractors.providers.nvidia import NvidiaSemanticExtractor
from app.extractors.providers.openrouter import OpenRouterSemanticExtractor

__all__ = [
    "FallbackSemanticExtractor",
    "GeminiSemanticExtractor",
    "NvidiaSemanticExtractor",
    "OllamaSemanticExtractor",
    "OpenRouterSemanticExtractor",
]
