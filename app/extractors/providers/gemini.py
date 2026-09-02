"""Gemini REST API adapter for semantic resume extraction."""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.domain.semantic_contract import SemanticInput, SemanticOutput
from app.extractors.semantic_extractor import SemanticExtractionError
from app.extractors.semantic_prompt import build_extraction_prompt, parse_semantic_output


def pydantic_to_gemini_schema(pydantic_model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model's JSON schema to a self-contained schema for Gemini REST responseSchema.

    Inlines all $defs references recursively so the schema is completely self-contained.
    """
    raw_schema = pydantic_model.model_json_schema()
    defs = raw_schema.pop("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                if ref_key in defs:
                    resolved = _resolve(defs[ref_key])
                    merged = dict(resolved)
                    for k, v in node.items():
                        if k != "$ref":
                            merged[k] = _resolve(v)
                    return merged
            res = {}
            for k, v in node.items():
                if k in ("title", "$defs"):
                    continue
                res[k] = _resolve(v)
            return res
        elif isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(raw_schema)


class GeminiSemanticExtractor:
    """Production provider adapter for Gemini models via REST API.

    Implements SemanticExtractor protocol: extract(SemanticInput) -> SemanticOutput.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._explicit_api_key = api_key
        self._explicit_model = model
        self._explicit_timeout = timeout
        self._client = client

    def _resolve_config(self) -> tuple[str, str, float]:
        """Resolve API key, model, and timeout from arguments, environment, or settings."""
        # 1. API Key
        api_key = self._explicit_api_key or os.environ.get("GEMINI_API_KEY")
        # 2. Model
        model = self._explicit_model or os.environ.get("GEMINI_MODEL")
        # 3. Timeout
        timeout = self._explicit_timeout
        if timeout is None:
            env_timeout = os.environ.get("GEMINI_TIMEOUT")
            if env_timeout:
                try:
                    timeout = float(env_timeout)
                except ValueError:
                    timeout = 30.0
            else:
                timeout = 30.0

        if not api_key:
            # Fall back to settings if available
            try:
                settings = Settings()
                api_key = getattr(settings, "gemini_api_key", None) or api_key
                model = model or getattr(settings, "gemini_model", "gemini-2.5-flash")
                if timeout == 30.0:
                    timeout = getattr(settings, "gemini_timeout", 30.0)
            except (ValidationError, OSError):
                # When Settings cannot be instantiated due to missing Kafka/MinIO test env,
                # fall back to environment variables or explicit parameters.
                pass

        if not model:
            model = "gemini-2.5-flash"

        if not api_key:
            raise SemanticExtractionError(
                "Gemini API key is required but not configured. "
                "Provide api_key to GeminiSemanticExtractor or set GEMINI_API_KEY environment variable."
            )

        return api_key, model, timeout

    @staticmethod
    def get_endpoint_url(model: str) -> str:
        """Return the exact Gemini REST API generateContent endpoint URL."""
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        """Extract structured SemanticOutput from SemanticInput using Gemini REST API."""
        api_key, model, timeout = self._resolve_config()
        prompt = build_extraction_prompt(input_data)
        response_schema = pydantic_to_gemini_schema(SemanticOutput)

        endpoint_url = self.get_endpoint_url(model)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
                "temperature": 0.0,
            },
        }

        # Send request
        try:
            if self._client is not None:
                response = self._client.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
            else:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise SemanticExtractionError(
                f"Gemini API HTTP error {err.response.status_code}: {err.response.text}"
            ) from err
        except httpx.TimeoutException as err:
            raise SemanticExtractionError(
                f"Gemini API request timed out after {timeout}s"
            ) from err
        except httpx.RequestError as err:
            raise SemanticExtractionError(
                f"Gemini API request failed: {err}"
            ) from err

        # Parse envelope
        try:
            res_json = response.json()
        except Exception as err:
            raise SemanticExtractionError(
                f"Failed to decode Gemini API response envelope: {err}"
            ) from err

        candidates = res_json.get("candidates")
        if not candidates or not isinstance(candidates, list):
            prompt_feedback = res_json.get("promptFeedback")
            raise SemanticExtractionError(
                f"Gemini API returned no candidates. Prompt feedback: {prompt_feedback}"
            )

        first_candidate = candidates[0]
        content = first_candidate.get("content", {})
        parts = content.get("parts", [])
        if not parts or not isinstance(parts, list) or "text" not in parts[0]:
            raise SemanticExtractionError(
                f"Gemini API candidate missing text part: {first_candidate}"
            )

        raw_text = parts[0]["text"]
        return parse_semantic_output(raw_text)
