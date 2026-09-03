"""Gemini REST API adapter for semantic resume extraction."""

from __future__ import annotations

import json
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
    build_full_extraction_prompt,
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


def _extract_api_error_message(text: str, api_key: str | None = None) -> str:
    """Extract diagnostic message from Gemini error response and sanitize secrets."""
    clean_text = _sanitize_error_message(text, api_key)
    if not clean_text or not clean_text.strip():
        return "Empty response body"
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            if isinstance(err, dict):
                msg = err.get("message")
                status = err.get("status")
                code = err.get("code")
                parts = []
                if msg:
                    parts.append(str(msg))
                if status:
                    parts.append(f"status={status}")
                if code:
                    parts.append(f"code={code}")
                if parts:
                    return " - ".join(parts)
    except Exception:
        pass
    # For plain text or unparseable JSON, return sanitized excerpt
    excerpt = clean_text[:300].strip()
    return excerpt


def _parse_retry_after(header_value: str | None, max_backoff: float = 30.0) -> float | None:
    """Parse HTTP Retry-After header.

    Returns float delay (capped at max_backoff), 0.0 for zero,
    or None if missing, negative, decimal, non-numeric, or invalid.
    """
    if not header_value:
        return None
    header_str = header_value.strip()
    if not header_str.lstrip("-").isdigit():
        return None
    try:
        val = int(header_str)
        if val < 0:
            return None
        if val == 0:
            return 0.0
        return min(float(val), max_backoff)
    except (ValueError, OverflowError):
        return None


def _classify_http_error(err: httpx.HTTPStatusError, api_key: str | None = None) -> SemanticExtractionError:
    """Classify an HTTPStatusError into an appropriate typed Semantic error.

    Classification rules:
    - 400: SemanticResponseError (deterministic invalid request)
    - 401, 403: SemanticConfigurationError (authentication / authorization failure)
    - 404: SemanticConfigurationError (model or endpoint not found)
    - 429: SemanticRateLimitError (rate limit / quota exceeded)
    - 5xx: SemanticServerError (transient server error)
    - Other 4xx: SemanticExtractionError (deterministic client error)
    """
    status_code = err.response.status_code
    error_msg = _extract_api_error_message(err.response.text, api_key)

    if status_code == 400:
        return SemanticResponseError(
            f"Gemini API request invalid (HTTP 400): {error_msg}"
        )
    if status_code in (401, 403):
        return SemanticConfigurationError(
            f"Gemini API authentication failed (HTTP {status_code}): {error_msg}"
        )
    if status_code == 404:
        return SemanticConfigurationError(
            f"Gemini API model or endpoint not found (HTTP 404): {error_msg}"
        )
    if status_code == 429:
        retry_header = err.response.headers.get("retry-after")
        retry_after = _parse_retry_after(retry_header)
        return SemanticRateLimitError(
            f"Gemini API rate limit exceeded (HTTP 429): {error_msg}",
            status_code=429,
            retry_after=retry_after,
        )
    if 500 <= status_code < 600:
        return SemanticServerError(
            f"Gemini API server error (HTTP {status_code}): {error_msg}",
            status_code=status_code,
        )
    return SemanticExtractionError(
        f"Gemini API client HTTP error {status_code}: {error_msg}"
    )


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
        if api_key is not None and not api_key.strip():
            api_key = None
        if api_key is None and "GEMINI_API_KEY" in os.environ:
            env_key = os.environ["GEMINI_API_KEY"].strip()
            if env_key:
                api_key = env_key

        model = self._explicit_model
        if model is not None and not model.strip():
            model = None
        if model is None and "GEMINI_MODEL" in os.environ:
            env_model = os.environ["GEMINI_MODEL"].strip()
            if env_model:
                model = env_model

        base_url = self._explicit_base_url
        if base_url is not None and not base_url.strip():
            base_url = None
        if base_url is None and "GEMINI_BASE_URL" in os.environ:
            env_url = os.environ["GEMINI_BASE_URL"].strip()
            if env_url:
                base_url = env_url

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
                    st_key = getattr(settings, "gemini_api_key", None)
                    if st_key and st_key.strip():
                        api_key = st_key.strip()
                if model is None:
                    st_model = getattr(settings, "gemini_model", None)
                    if st_model and st_model.strip():
                        model = st_model.strip()
                if base_url is None:
                    st_url = getattr(settings, "gemini_base_url", None)
                    if st_url and st_url.strip():
                        base_url = st_url.strip()
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

        if not api_key or not api_key.strip():
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
                "representation": "candidate_b_compact" if self._compact else "full",
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
            response_schema = pydantic_to_gemini_schema(SemanticOutput)
        else:
            prompt = build_full_extraction_prompt(input_data)
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
                    exc = _classify_http_error(err, api_key)
                    status_code = err.response.status_code

                    # Only retry transient errors (429 and 5xx)
                    if isinstance(exc, (SemanticRateLimitError, SemanticServerError)) and attempt < max_retries:
                        if isinstance(exc, SemanticRateLimitError) and exc.retry_after is not None:
                            delay = exc.retry_after
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
                        if delay > 0.0:
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

                if not isinstance(res_json, dict):
                    raise SemanticResponseError(
                        f"Gemini API response envelope must be a JSON object, got {type(res_json).__name__}"
                    )

                # Extract usage metadata immediately so it is preserved even if subsequent parsing fails
                usage = res_json.get("usageMetadata")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("promptTokenCount")
                    output_tokens = usage.get("candidatesTokenCount")
                    total_tokens = usage.get("totalTokenCount")

                # Check promptFeedback for block reason
                prompt_feedback = res_json.get("promptFeedback")
                if isinstance(prompt_feedback, dict):
                    block_reason = prompt_feedback.get("blockReason")
                    if block_reason:
                        raise SemanticResponseError(
                            f"Gemini API prompt blocked with blockReason={block_reason}"
                        )

                # Validate candidates container
                if "candidates" not in res_json:
                    raise SemanticResponseError("Gemini API response missing 'candidates' field")

                candidates = res_json.get("candidates")
                if candidates is None or not isinstance(candidates, list):
                    raise SemanticResponseError(
                        f"Gemini API response 'candidates' must be a list, got {type(candidates).__name__}"
                    )

                if len(candidates) == 0:
                    raise SemanticResponseError(
                        f"Gemini API returned no candidates. Prompt feedback: {prompt_feedback}"
                    )

                first_candidate = candidates[0]
                if not isinstance(first_candidate, dict):
                    raise SemanticResponseError(
                        f"Gemini API candidate must be an object, got {type(first_candidate).__name__}"
                    )

                # Inspect finishReason for blocked or abnormal termination
                finish_reason = first_candidate.get("finishReason")
                if finish_reason and str(finish_reason).upper() in {
                    "SAFETY",
                    "RECITATION",
                    "BLOCKLIST",
                    "PROHIBITED_CONTENT",
                    "SPII",
                    "MALFORMED_FUNCTION_CALL",
                    "IMAGE_SAFETY",
                    "IMAGE_PROHIBITED_CONTENT",
                    "NO_IMAGE",
                    "UNEXPECTED_TOOL_CALL",
                    "TOO_MANY_TOOL_CALLS",
                    "MAX_TOKENS",
                    "OTHER",
                }:
                    raise SemanticResponseError(
                        f"Gemini API response candidate blocked with finishReason={finish_reason}"
                    )

                # Validate content
                if "content" not in first_candidate or first_candidate.get("content") is None:
                    raise SemanticResponseError("Gemini API candidate is missing 'content'")

                content = first_candidate.get("content")
                if not isinstance(content, dict):
                    raise SemanticResponseError(
                        f"Gemini API candidate 'content' must be an object, got {type(content).__name__}"
                    )

                # Validate parts
                if "parts" not in content or content.get("parts") is None:
                    raise SemanticResponseError("Gemini API candidate content is missing 'parts'")

                parts = content.get("parts")
                if not isinstance(parts, list):
                    raise SemanticResponseError(
                        f"Gemini API candidate content 'parts' must be a list, got {type(parts).__name__}"
                    )

                if len(parts) == 0:
                    raise SemanticResponseError("Gemini API candidate content contains no parts")

                first_part = parts[0]
                if not isinstance(first_part, dict):
                    raise SemanticResponseError(
                        f"Gemini API candidate content part must be an object, got {type(first_part).__name__}"
                    )

                if "text" not in first_part or first_part.get("text") is None:
                    raise SemanticResponseError("Gemini API candidate content part is missing 'text'")

                raw_text = first_part.get("text")
                if not isinstance(raw_text, str) or not raw_text.strip():
                    raise SemanticResponseError("Gemini API candidate content part contains empty text")

                result = parse_semantic_output(raw_text)

                total_latency_ms = max(0.0, (self._time_fn() - start_time) * 1000.0)
                self.last_usage_metadata = {
                    "provider": "gemini",
                    "model": model,
                    "representation": "candidate_b_compact" if self._compact else "full",
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
                    "representation": "candidate_b_compact" if self._compact else "full",
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
