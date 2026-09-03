"""Ollama REST API adapter for local semantic resume extraction."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.semantic_contract import SemanticInput, SemanticOutput
from app.extractors.semantic_extractor import SemanticExtractionError
from app.extractors.semantic_prompt import (
    build_compact_extraction_prompt,
    build_extraction_prompt,
    get_compact_schema,
    parse_semantic_output,
    resolve_schema_defs,
    serialize_compact_semantic_input,
)

# Provider-local aliases for backwards compatibility
serialize_ollama_compact_input = serialize_compact_semantic_input
get_ollama_compact_schema = get_compact_schema
build_ollama_extraction_prompt = build_compact_extraction_prompt


class OllamaSemanticExtractor:
    """Production provider adapter for local Ollama models via HTTP API.

    Implements SemanticExtractor protocol: extract(SemanticInput) -> SemanticOutput.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._explicit_base_url = base_url
        self._explicit_model = model
        self._explicit_timeout = timeout
        self._client = client
        self.last_usage_metadata: dict[str, int | None] | None = None

    def _resolve_config(self) -> tuple[str, str, float]:
        """Resolve base_url, model, and timeout from arguments, environment, or settings."""
        # 1. Base URL (defaults to http://localhost:11434)
        base_url = self._explicit_base_url or os.environ.get("OLLAMA_BASE_URL")
        # 2. Model (defaults to qwen2.5-coder:7b)
        model = self._explicit_model or os.environ.get("OLLAMA_MODEL")
        # 3. Timeout (defaults to 120.0s)
        timeout = self._explicit_timeout
        if timeout is None:
            env_timeout = os.environ.get("OLLAMA_TIMEOUT")
            if env_timeout:
                try:
                    timeout = float(env_timeout)
                except ValueError:
                    timeout = 120.0
            else:
                timeout = 120.0

        if not base_url or not model:
            try:
                settings = Settings()
                base_url = base_url or getattr(settings, "ollama_base_url", "http://localhost:11434")
                model = model or getattr(settings, "ollama_model", "qwen2.5-coder:7b")
                if timeout == 120.0:
                    timeout = getattr(settings, "ollama_timeout", 120.0)
            except (ValidationError, OSError):
                pass

        base_url = (base_url or "http://localhost:11434").rstrip("/")
        model = model or "qwen2.5-coder:7b"

        return base_url, model, timeout

    @staticmethod
    def get_chat_endpoint_url(base_url: str) -> str:
        """Return the chat completions endpoint URL."""
        return f"{base_url.rstrip('/')}/api/chat"

    @classmethod
    def check_availability(
        cls,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ) -> tuple[bool, str | None]:
        """Check if the Ollama service is reachable and the specified model is present."""
        resolved_base = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        resolved_model = model or os.environ.get("OLLAMA_MODEL") or "qwen2.5-coder:7b"

        tags_url = f"{resolved_base}/api/tags"
        try:
            if client is not None:
                resp = client.get(tags_url, timeout=5.0)
            else:
                with httpx.Client(timeout=5.0) as cl:
                    resp = cl.get(tags_url)
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError:
            return (
                False,
                f"Ollama service is not reachable at '{resolved_base}'. "
                f"Please ensure Ollama is running (e.g. 'ollama serve').",
            )
        except Exception as exc:
            return (
                False,
                f"Failed to query Ollama at '{resolved_base}': {exc}",
            )

        installed_models = [m.get("name", "") for m in data.get("models", [])]
        # Match exact name or base name without tag
        model_found = any(
            installed == resolved_model
            or installed.split(":")[0] == resolved_model.split(":")[0]
            or resolved_model == installed.split(":")[0]
            for installed in installed_models
        )

        if not model_found:
            return (
                False,
                f"Model '{resolved_model}' is not available in Ollama at '{resolved_base}'. "
                f"Installed models: {installed_models}. "
                f"Pull it using: 'ollama pull {resolved_model}'.",
            )

        return True, None

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        """Extract structured SemanticOutput from SemanticInput using Ollama API."""
        base_url, model, timeout = self._resolve_config()
        prompt = build_ollama_extraction_prompt(input_data)
        schema = get_ollama_compact_schema(SemanticOutput)

        endpoint_url = self.get_chat_endpoint_url(base_url)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0.0,
            },
        }

        # Send request
        try:
            if self._client is not None:
                response = self._client.post(endpoint_url, json=payload, timeout=timeout)
            else:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(endpoint_url, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as err:
            raise SemanticExtractionError(
                f"Ollama connection failed at '{base_url}'. Is Ollama running? Error: {err}"
            ) from err
        except httpx.HTTPStatusError as err:
            raise SemanticExtractionError(
                f"Ollama API HTTP error {err.response.status_code}: {err.response.text}"
            ) from err
        except httpx.TimeoutException as err:
            raise SemanticExtractionError(
                f"Ollama API request timed out after {timeout}s"
            ) from err
        except httpx.RequestError as err:
            raise SemanticExtractionError(
                f"Ollama API request failed: {err}"
            ) from err

        # Parse envelope
        try:
            res_json = response.json()
        except Exception as err:
            raise SemanticExtractionError(
                f"Failed to decode Ollama API response envelope: {err}"
            ) from err

        # Capture token metrics if reported
        p_tokens = res_json.get("prompt_eval_count")
        o_tokens = res_json.get("eval_count")
        if p_tokens is not None or o_tokens is not None:
            tot = (p_tokens or 0) + (o_tokens or 0)
            self.last_usage_metadata = {
                "prompt_tokens": p_tokens,
                "output_tokens": o_tokens,
                "total_tokens": tot,
            }
        else:
            self.last_usage_metadata = None

        message = res_json.get("message", {})
        content = message.get("content")
        if content is None:
            raise SemanticExtractionError(
                f"Ollama API response missing 'message.content': {res_json}"
            )

        return parse_semantic_output(content)
