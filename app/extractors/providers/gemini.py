"""Gemini REST API adapter for semantic resume extraction."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.domain.semantic_contract import SemanticInput, SemanticOutput
from app.extractors.semantic_extractor import (
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticRateLimitError,
    SemanticResponseError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
    SemanticValidationError,
)
from app.extractors.semantic_prompt import (
    build_compact_extraction_prompt,
    build_extraction_prompt,
    get_compact_schema,
    parse_semantic_output,
    resolve_schema_defs,
)

logger = logging.getLogger(__name__)

# Backwards-compatible aliases for existing tests and provider typing
pydantic_to_gemini_schema = resolve_schema_defs
GeminiConfigurationError = SemanticConfigurationError
GeminiRateLimitError = SemanticRateLimitError
GeminiServerError = SemanticServerError
GeminiTimeoutError = SemanticTimeoutError
GeminiTransportError = SemanticTransportError
GeminiResponseError = SemanticResponseError


def _sanitize_error_message(msg: str, api_key: str | None = None) -> str:
    """Ensure API key never leaks in exception strings or logs."""
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "[REDACTED]")
    return msg


class GeminiSemanticExtractor:
    """Production provider adapter for Gemini models via REST API.

    Implements SemanticExtractor protocol: extract(SemanticInput) -> SemanticOutput.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_multiplier: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.perf_counter,
        client: httpx.Client | None = None,
        compact: bool = True,
    ) -> None:
        self._explicit_api_key = api_key
        self._explicit_model = model
        self._explicit_base_url = base_url
        self._explicit_timeout = timeout
        self._explicit_max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff_multiplier = backoff_multiplier
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn
        self._client = client
        self._compact = compact
        self.last_usage_metadata: dict[str, Any] | None = None

    def _resolve_config(self) -> tuple[str, str, str, float, int]:
        """Resolve API key, model, base_url, timeout, and max_retries with strict precedence.

        Precedence order:
            explicit constructor argument (if not None)
            -> environment variable (if present and valid)
            -> application Settings
            -> safe default
        """
        # Tier 1 & 2: Explicit constructor arguments and Environment variables
        api_key = self._explicit_api_key
        if api_key is None and "GEMINI_API_KEY" in os.environ:
            api_key = os.environ["GEMINI_API_KEY"]

        model = self._explicit_model
        if model is None and "GEMINI_MODEL" in os.environ:
            model = os.environ["GEMINI_MODEL"]

        base_url = self._explicit_base_url
        if base_url is None and "GEMINI_BASE_URL" in os.environ:
            base_url = os.environ["GEMINI_BASE_URL"]

        timeout = self._explicit_timeout
        if timeout is not None and timeout <= 0:
            timeout = None
        if timeout is None and "GEMINI_TIMEOUT" in os.environ:
            try:
                parsed_timeout = float(os.environ["GEMINI_TIMEOUT"])
                if parsed_timeout > 0:
                    timeout = parsed_timeout
            except ValueError:
                pass

        max_retries = self._explicit_max_retries
        if max_retries is not None and max_retries < 0:
            max_retries = None
        if max_retries is None and "GEMINI_MAX_RETRIES" in os.environ:
            try:
                parsed_retries = int(os.environ["GEMINI_MAX_RETRIES"])
                if parsed_retries >= 0:
                    max_retries = parsed_retries
            except ValueError:
                pass

        # Tier 3: Settings for any unresolved fields
        if any(v is None for v in (api_key, model, base_url, timeout, max_retries)):
            try:
                settings = Settings()
                if api_key is None:
                    api_key = getattr(settings, "gemini_api_key", None)
                if model is None:
                    model = getattr(settings, "gemini_model", None)
                if base_url is None:
                    base_url = getattr(settings, "gemini_base_url", None)
                if timeout is None:
                    st_timeout = getattr(settings, "gemini_timeout", None)
                    if isinstance(st_timeout, (int, float)) and st_timeout > 0:
                        timeout = float(st_timeout)
                if max_retries is None:
                    st_retries = getattr(settings, "gemini_max_retries", None)
                    if isinstance(st_retries, int) and st_retries >= 0:
                        max_retries = st_retries
            except (ValidationError, OSError):
                pass

        # Tier 4: Safe defaults
        model = model or "gemini-2.5-flash"
        base_url = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        timeout = timeout if timeout is not None else 30.0
        max_retries = max_retries if max_retries is not None else 2

        if not api_key:
            raise SemanticConfigurationError(
                "Gemini API key is required but not configured. "
                "Provide api_key to GeminiSemanticExtractor or set GEMINI_API_KEY environment variable."
            )

        return api_key, model, base_url, timeout, max_retries

    @staticmethod
    def get_endpoint_url(model: str, base_url: str = "https://generativelanguage.googleapis.com") -> str:
        """Return the exact Gemini REST API generateContent endpoint URL."""
        return f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        """Extract structured SemanticOutput from SemanticInput using Gemini REST API."""
        start_time = self._time_fn()
        retry_count = 0
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None
        model_for_metadata = self._explicit_model or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

        try:
            api_key, model, base_url, timeout, max_retries = self._resolve_config()
            model_for_metadata = model
        except Exception as exc:
            total_latency_ms = max(0.0, (self._time_fn() - start_time) * 1000.0)
            self.last_usage_metadata = {
                "provider": "gemini",
                "model": model_for_metadata,
                "prompt_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "latency_ms": round(total_latency_ms, 2),
                "retry_count": 0,
                "status": "failure",
                "error_type": type(exc).__name__,
            }
            logger.error(
                "Gemini request configuration failed model=%s error_type=%s latency_ms=%.2f",
                model_for_metadata,
                type(exc).__name__,
                total_latency_ms,
            )
            raise

        if self._compact:
            prompt = build_compact_extraction_prompt(input_data)
            response_schema = get_compact_schema(SemanticOutput)
        else:
            prompt = build_extraction_prompt(input_data)
            response_schema = pydantic_to_gemini_schema(SemanticOutput)

        endpoint_url = self.get_endpoint_url(model, base_url=base_url)
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

        # Send request with transient retry loop, bounded backoff, and observability
        attempt = 0
        while True:
            logger.debug(
                "Gemini request started model=%s attempt=%d/%d",
                model,
                attempt + 1,
                max_retries + 1,
            )
            try:
                try:
                    if self._client is not None:
                        response = self._client.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
                    else:
                        with httpx.Client(timeout=timeout) as client:
                            response = client.post(endpoint_url, headers=headers, json=payload)
                    response.raise_for_status()

                except httpx.HTTPStatusError as err:
                    status_code = err.response.status_code
                    clean_body = _sanitize_error_message(err.response.text, api_key)

                    # Rate limit / Quota (429) -> SemanticRateLimitError
                    if status_code == 429:
                        retry_header = err.response.headers.get("retry-after")
                        retry_after = float(retry_header) if retry_header and retry_header.isdigit() else None
                        exc = SemanticRateLimitError(
                            f"Gemini API rate limit exceeded (HTTP 429): {clean_body}",
                            status_code=429,
                            retry_after=retry_after,
                        )
                    # 5xx Server Errors -> SemanticServerError
                    elif 500 <= status_code < 600:
                        exc = SemanticServerError(
                            f"Gemini API server error (HTTP {status_code}): {clean_body}",
                            status_code=status_code,
                        )
                    # Other 4xx Client Errors -> SemanticExtractionError (Non-transient, never retried)
                    else:
                        raise SemanticExtractionError(
                            f"Gemini API client HTTP error {status_code}: {clean_body}"
                        ) from err

                    # Retry transient errors if attempts remain with bounded exponential backoff
                    if attempt < max_retries:
                        if isinstance(exc, SemanticRateLimitError) and exc.retry_after is not None and exc.retry_after > 0:
                            delay = min(exc.retry_after, self._max_backoff)
                        else:
                            delay = min(self._initial_backoff * (self._backoff_multiplier ** attempt), self._max_backoff)
                        logger.warning(
                            "Gemini transient failure error_type=%s status_code=%d attempt=%d/%d retrying in %.2fs",
                            type(exc).__name__,
                            status_code,
                            attempt + 1,
                            max_retries + 1,
                            delay,
                        )
                        self._sleep_fn(delay)
                        attempt += 1
                        retry_count += 1
                        continue
                    raise exc from err

                except httpx.TimeoutException as err:
                    exc = SemanticTimeoutError(
                        f"Gemini API request timed out after {timeout}s"
                    )
                    if attempt < max_retries:
                        delay = min(self._initial_backoff * (self._backoff_multiplier ** attempt), self._max_backoff)
                        logger.warning(
                            "Gemini timeout error_type=%s attempt=%d/%d retrying in %.2fs",
                            type(exc).__name__,
                            attempt + 1,
                            max_retries + 1,
                            delay,
                        )
                        self._sleep_fn(delay)
                        attempt += 1
                        retry_count += 1
                        continue
                    raise exc from err

                except (httpx.ConnectError, httpx.NetworkError, httpx.RequestError) as err:
                    clean_err = _sanitize_error_message(str(err), api_key)
                    exc = SemanticTransportError(
                        f"Gemini API transport failure: {clean_err}"
                    )
                    if attempt < max_retries:
                        delay = min(self._initial_backoff * (self._backoff_multiplier ** attempt), self._max_backoff)
                        logger.warning(
                            "Gemini transport failure error_type=%s attempt=%d/%d retrying in %.2fs",
                            type(exc).__name__,
                            attempt + 1,
                            max_retries + 1,
                            delay,
                        )
                        self._sleep_fn(delay)
                        attempt += 1
                        retry_count += 1
                        continue
                    raise exc from err

                # Parse envelope
                try:
                    res_json = response.json()
                except Exception as err:
                    raise SemanticResponseError(
                        f"Failed to decode Gemini API response envelope as JSON: {err}"
                    ) from err

                usage = res_json.get("usageMetadata")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("promptTokenCount")
                    output_tokens = usage.get("candidatesTokenCount")
                    total_tokens = usage.get("totalTokenCount")

                candidates = res_json.get("candidates")
                if not candidates or not isinstance(candidates, list):
                    prompt_feedback = res_json.get("promptFeedback")
                    raise SemanticResponseError(
                        f"Gemini API returned no candidates. Prompt feedback: {prompt_feedback}"
                    )

                first_candidate = candidates[0]
                content = first_candidate.get("content", {})
                parts = content.get("parts", [])
                if not parts or not isinstance(parts, list) or "text" not in parts[0]:
                    raise SemanticResponseError(
                        f"Gemini API candidate missing text part: {first_candidate}"
                    )

                raw_text = parts[0]["text"]
                result = parse_semantic_output(raw_text)

                total_latency_ms = max(0.0, (self._time_fn() - start_time) * 1000.0)
                self.last_usage_metadata = {
                    "provider": "gemini",
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": round(total_latency_ms, 2),
                    "retry_count": retry_count,
                    "status": "success",
                }

                logger.info(
                    "Gemini request completed model=%s latency_ms=%.2f retries=%d prompt_tokens=%s output_tokens=%s",
                    model,
                    total_latency_ms,
                    retry_count,
                    prompt_tokens,
                    output_tokens,
                )
                return result

            except Exception as exc:
                total_latency_ms = max(0.0, (self._time_fn() - start_time) * 1000.0)
                self.last_usage_metadata = {
                    "provider": "gemini",
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": round(total_latency_ms, 2),
                    "retry_count": retry_count,
                    "status": "failure",
                    "error_type": type(exc).__name__,
                }
                logger.error(
                    "Gemini request failed model=%s error_type=%s latency_ms=%.2f retries=%d",
                    model,
                    type(exc).__name__,
                    total_latency_ms,
                    retry_count,
                )
                raise
