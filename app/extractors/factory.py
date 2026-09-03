"""Factory for constructing configured semantic extractors."""

from __future__ import annotations

from app.core.config import Settings
from app.extractors.providers.fallback import FallbackSemanticExtractor
from app.extractors.providers.gemini import GeminiSemanticExtractor
from app.extractors.providers.ollama import OllamaSemanticExtractor
from app.extractors.semantic_extractor import SemanticExtractor


def get_semantic_extractor(
    settings: Settings | None = None,
) -> SemanticExtractor:
    """Construct a SemanticExtractor instance based on application settings.

    When semantic_fallback_enabled is False (default):
        Returns a configured GeminiSemanticExtractor.

    When semantic_fallback_enabled is True:
        Returns a FallbackSemanticExtractor with Gemini as primary and Ollama as fallback.
    """
    st = settings or Settings()

    primary = GeminiSemanticExtractor(
        api_key=st.gemini_api_key,
        model=st.gemini_model,
        base_url=st.gemini_base_url,
        timeout=st.gemini_timeout,
        max_retries=st.gemini_max_retries,
    )

    if not st.semantic_fallback_enabled:
        return primary

    fallback = OllamaSemanticExtractor(
        base_url=st.ollama_base_url,
        model=st.ollama_model,
        timeout=st.ollama_timeout,
        num_threads=st.ollama_num_threads,
        think=st.ollama_think,
    )

    return FallbackSemanticExtractor(primary=primary, fallback=fallback)
