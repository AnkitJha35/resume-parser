"""NVIDIA NIM REST API adapter for semantic resume extraction."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from typing import Any, Callable, Literal

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.domain.semantic_contract import (
    BodySemanticOutput,
    DocumentArchetype,
    PersonalSemanticOutput,
    SemanticInput,
    SemanticOutput,
    filter_semantic_input_to_blocks,
    get_body_evidence_category,
    group_sections_by_target,
    group_sections_for_recovery,
    is_body_output_suspiciously_empty,
    merge_body_outputs,
    merge_semantic_passes,
    partition_semantic_input_into_sections,
    constrain_appointment_experience_provenance,
    enforce_single_experience_entity,
    isolate_unit_target_collections,
    plan_section_aware_body_passes,
    sanitize_grounded_current_status,
    should_use_section_aware_body_extraction,
)
from app.extractors.semantic_extractor import (
    SemanticCompletenessError,
    SemanticConfigurationError,
    SemanticExtractionError,
    SemanticRateLimitError,
    SemanticResponseError,
    SemanticServerError,
    SemanticTimeoutError,
    SemanticTransportError,
)
from app.extractors.semantic_prompt import (
    build_body_extraction_prompt,
    build_body_recovery_prompt,
    build_compact_extraction_prompt,
    build_personal_extraction_prompt,
    get_body_schema,
    get_compact_schema,
    get_personal_schema,
    parse_body_output,
    parse_personal_output,
    parse_semantic_output,
)

logger = logging.getLogger(__name__)


def _sanitize_error_message(msg: str, api_key: str | None = None) -> str:
    """Ensure API key never leaks in exception strings or logs."""
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "[REDACTED]")
    return msg


def _extract_api_error_message(text: str, api_key: str | None = None) -> str:
    """Extract diagnostic message from NVIDIA error response and sanitize secrets."""
    clean_text = _sanitize_error_message(text, api_key)
    if not clean_text or not clean_text.strip():
        return "Empty response body"
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            if isinstance(err, dict):
                msg = err.get("message")
                code = err.get("code")
                parts = []
                if msg:
                    parts.append(str(msg))
                if code:
                    parts.append(f"code={code}")
                if parts:
                    return "; ".join(parts)
            return str(err)
    except Exception:
        pass
    return clean_text[:300].strip()


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
    - 402: SemanticConfigurationError (insufficient credits / payment required)
    - 404: SemanticConfigurationError (model or endpoint not found)
    - 429: SemanticRateLimitError (rate limit / quota exceeded)
    - 5xx: SemanticServerError (transient server error)
    - Other 4xx: SemanticExtractionError (deterministic client error)
    """
    status_code = err.response.status_code
    error_msg = _extract_api_error_message(err.response.text, api_key)

    if status_code == 400:
        return SemanticResponseError(
            f"NVIDIA API request invalid (HTTP 400): {error_msg}"
        )
    if status_code in (401, 403):
        return SemanticConfigurationError(
            f"NVIDIA API authentication failed (HTTP {status_code}): {error_msg}"
        )
    if status_code == 402:
        return SemanticConfigurationError(
            f"NVIDIA API payment required / insufficient credits (HTTP 402): {error_msg}"
        )
    if status_code == 404:
        return SemanticConfigurationError(
            f"NVIDIA API model or endpoint not found (HTTP 404): {error_msg}"
        )
    if status_code == 429:
        retry_header = err.response.headers.get("retry-after")
        retry_after = _parse_retry_after(retry_header)
        return SemanticRateLimitError(
            f"NVIDIA API rate limit exceeded (HTTP 429): {error_msg}",
            status_code=429,
            retry_after=retry_after,
        )
    if 500 <= status_code < 600:
        return SemanticServerError(
            f"NVIDIA API server error (HTTP {status_code}): {error_msg}",
            status_code=status_code,
        )
    return SemanticExtractionError(
        f"NVIDIA API client HTTP error {status_code}: {error_msg}"
    )


class NvidiaSemanticExtractor:
    """Production provider adapter for NVIDIA NIM REST API.

    Implements SemanticExtractor protocol: extract(SemanticInput) -> SemanticOutput.
    Preserves structured JSON schema/object modes, two-pass extraction, section grouping,
    transient retry backoff, and usage-token telemetry.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_tokens: int | None = None,
        response_format_type: Literal["json_object", "json_schema"] | None = None,
        enable_thinking: bool | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_multiplier: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.perf_counter,
        client: httpx.Client | None = None,
        compact: bool = True,
        two_pass: bool | None = None,
    ) -> None:
        self._explicit_api_key = api_key
        self._explicit_model = model
        self._explicit_base_url = base_url
        self._explicit_timeout = timeout
        self._explicit_max_retries = max_retries
        self._explicit_max_tokens = max_tokens
        self._explicit_response_format_type = response_format_type
        self._explicit_enable_thinking = enable_thinking
        self._explicit_two_pass = two_pass
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff_multiplier = backoff_multiplier
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn
        self._client = client
        self._compact = compact
        self.last_usage_metadata: dict[str, Any] | None = None
        self._last_call_meta: dict[str, Any] = {}

    def _resolve_two_pass(self) -> bool:
        """Resolve two-pass mode configuration."""
        if self._explicit_two_pass is not None:
            return self._explicit_two_pass
        if "NVIDIA_TWO_PASS" in os.environ:
            env_two_pass = os.environ["NVIDIA_TWO_PASS"].strip().lower()
            if env_two_pass in ("1", "true", "yes"):
                return True
            elif env_two_pass in ("0", "false", "no"):
                return False
        try:
            settings = Settings()
            st_two_pass = getattr(settings, "nvidia_two_pass", None)
            if isinstance(st_two_pass, bool):
                return st_two_pass
        except (ValidationError, OSError):
            pass
        return False

    def _resolve_config(self) -> tuple[str, str, str, float, int, int, str, bool]:
        """Resolve API key, model, base_url, timeout, max_retries, max_tokens, response_format_type, and enable_thinking with strict precedence."""
        api_key = self._explicit_api_key
        if api_key is not None and not api_key.strip():
            api_key = None
        if api_key is None and "NVIDIA_API_KEY" in os.environ:
            env_key = os.environ["NVIDIA_API_KEY"].strip()
            if env_key:
                api_key = env_key

        model = self._explicit_model
        if model is not None and not model.strip():
            model = None
        if model is None and "NVIDIA_MODEL" in os.environ:
            env_model = os.environ["NVIDIA_MODEL"].strip()
            if env_model:
                model = env_model

        base_url = self._explicit_base_url
        if base_url is not None and not base_url.strip():
            base_url = None
        if base_url is None and "NVIDIA_BASE_URL" in os.environ:
            env_url = os.environ["NVIDIA_BASE_URL"].strip()
            if env_url:
                base_url = env_url

        timeout = self._explicit_timeout
        if timeout is not None and timeout <= 0:
            timeout = None
        if timeout is None and "NVIDIA_TIMEOUT" in os.environ:
            try:
                parsed_timeout = float(os.environ["NVIDIA_TIMEOUT"])
                if parsed_timeout > 0:
                    timeout = parsed_timeout
            except ValueError:
                pass

        max_retries = self._explicit_max_retries
        if max_retries is not None and max_retries < 0:
            max_retries = None
        if max_retries is None and "NVIDIA_MAX_RETRIES" in os.environ:
            try:
                parsed_retries = int(os.environ["NVIDIA_MAX_RETRIES"])
                if parsed_retries >= 0:
                    max_retries = parsed_retries
            except ValueError:
                pass

        max_tokens = self._explicit_max_tokens
        if max_tokens is not None and max_tokens <= 0:
            max_tokens = None
        if max_tokens is None and "NVIDIA_MAX_TOKENS" in os.environ:
            try:
                parsed_max_tokens = int(os.environ["NVIDIA_MAX_TOKENS"])
                if parsed_max_tokens > 0:
                    max_tokens = parsed_max_tokens
            except ValueError:
                pass

        response_format_type = self._explicit_response_format_type
        if response_format_type is None and "NVIDIA_RESPONSE_FORMAT_TYPE" in os.environ:
            env_fmt = os.environ["NVIDIA_RESPONSE_FORMAT_TYPE"].strip().lower()
            if env_fmt in ("json_object", "json_schema"):
                response_format_type = env_fmt

        enable_thinking = self._explicit_enable_thinking
        if enable_thinking is None and "NVIDIA_ENABLE_THINKING" in os.environ:
            env_thinking = os.environ["NVIDIA_ENABLE_THINKING"].strip().lower()
            if env_thinking in ("1", "true", "yes"):
                enable_thinking = True
            elif env_thinking in ("0", "false", "no"):
                enable_thinking = False

        # Check Settings for unresolved fields
        if any(v is None for v in (api_key, model, base_url, timeout, max_retries, max_tokens, response_format_type, enable_thinking)):
            try:
                settings = Settings()
                if api_key is None:
                    st_key = getattr(settings, "nvidia_api_key", None)
                    if st_key and st_key.strip():
                        api_key = st_key.strip()
                if model is None:
                    st_model = getattr(settings, "nvidia_model", None)
                    if st_model and st_model.strip():
                        model = st_model.strip()
                if base_url is None:
                    st_url = getattr(settings, "nvidia_base_url", None)
                    if st_url and st_url.strip():
                        base_url = st_url.strip()
                if timeout is None:
                    st_timeout = getattr(settings, "nvidia_timeout", None)
                    if isinstance(st_timeout, (int, float)) and st_timeout > 0:
                        timeout = float(st_timeout)
                if max_retries is None:
                    st_retries = getattr(settings, "nvidia_max_retries", None)
                    if isinstance(st_retries, int) and st_retries >= 0:
                        max_retries = st_retries
                if max_tokens is None:
                    st_tokens = getattr(settings, "nvidia_max_tokens", None)
                    if isinstance(st_tokens, int) and st_tokens > 0:
                        max_tokens = st_tokens
                if response_format_type is None:
                    st_fmt = getattr(settings, "nvidia_response_format_type", None)
                    if st_fmt in ("json_object", "json_schema"):
                        response_format_type = st_fmt
                if enable_thinking is None:
                    st_thinking = getattr(settings, "nvidia_enable_thinking", None)
                    if isinstance(st_thinking, bool):
                        enable_thinking = st_thinking
            except (ValidationError, OSError):
                pass

        # Final defaults
        if model is None:
            model = "nvidia/nemotron-3.5-lightning-30b-a3b"
        if base_url is None:
            base_url = "https://integrate.api.nvidia.com/v1"
        if timeout is None:
            timeout = 60.0
        if max_retries is None:
            max_retries = 2
        if max_tokens is None:
            max_tokens = 16384
        if response_format_type is None:
            response_format_type = "json_schema"
        if enable_thinking is None:
            enable_thinking = False

        if not api_key:
            raise SemanticConfigurationError(
                "NVIDIA_API_KEY is not configured. Set NVIDIA_API_KEY environment variable "
                "or pass api_key explicitly."
            )

        return api_key, model, base_url, timeout, max_retries, max_tokens, response_format_type, enable_thinking

    @property
    def last_run_meta(self) -> dict[str, Any]:
        """Return execution metadata from the most recent run."""
        return dict(self._last_call_meta)

    def get_endpoint_url(self, base_url: str) -> str:
        """Construct full NVIDIA chat completions endpoint URL."""
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    def _execute_prompt_request(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        max_retries: int,
        max_tokens: int = 16384,
        response_format_type: str = "json_schema",
        enable_thinking: bool = False,
        pass_name: str = "single",
    ) -> tuple[str, dict[str, Any], int]:
        """Execute a single HTTP request to the NVIDIA chat/completions endpoint with retries and return (text, usage_dict, retry_count)."""
        endpoint_url = self.get_endpoint_url(base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            },
        }

        if response_format_type == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_output",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        attempt = 0
        retry_count = 0
        self._last_call_meta = {
            "prompt_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "retry_count": 0,
        }

        while True:
            logger.debug(
                "NVIDIA %s request started model=%s attempt=%d/%d max_tokens=%d format=%s",
                pass_name,
                model,
                attempt + 1,
                max_retries + 1,
                max_tokens,
                response_format_type,
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

                    # Only retry transient errors (429 and 5xx). Note: 401, 402, 403, 404, 400 are non-transient.
                    if isinstance(exc, (SemanticRateLimitError, SemanticServerError)) and attempt < max_retries:
                        if isinstance(exc, SemanticRateLimitError) and exc.retry_after is not None:
                            delay = exc.retry_after
                        else:
                            delay = min(self._initial_backoff * (self._backoff_multiplier ** attempt), self._max_backoff)
                        logger.warning(
                            "NVIDIA %s transient failure error_type=%s status_code=%d attempt=%d/%d retrying in %.2fs",
                            pass_name,
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
                        self._last_call_meta["retry_count"] = retry_count
                        continue
                    raise exc from err

                except httpx.TimeoutException as err:
                    exc = SemanticTimeoutError(
                        f"NVIDIA API request timed out after {timeout}s"
                    )
                    if attempt < max_retries:
                        delay = min(self._initial_backoff * (self._backoff_multiplier ** attempt), self._max_backoff)
                        logger.warning(
                            "NVIDIA %s timeout error_type=%s attempt=%d/%d retrying in %.2fs",
                            pass_name,
                            type(exc).__name__,
                            attempt + 1,
                            max_retries + 1,
                            delay,
                        )
                        self._sleep_fn(delay)
                        attempt += 1
                        retry_count += 1
                        self._last_call_meta["retry_count"] = retry_count
                        continue
                    raise exc from err

                except (httpx.ConnectError, httpx.NetworkError, httpx.RequestError) as err:
                    clean_err = _sanitize_error_message(str(err), api_key)
                    exc = SemanticTransportError(
                        f"NVIDIA API transport failure: {clean_err}"
                    )
                    if attempt < max_retries:
                        delay = min(self._initial_backoff * (self._backoff_multiplier ** attempt), self._max_backoff)
                        logger.warning(
                            "NVIDIA %s transport failure error_type=%s attempt=%d/%d retrying in %.2fs",
                            pass_name,
                            type(exc).__name__,
                            attempt + 1,
                            max_retries + 1,
                            delay,
                        )
                        self._sleep_fn(delay)
                        attempt += 1
                        retry_count += 1
                        self._last_call_meta["retry_count"] = retry_count
                        continue
                    raise exc from err

                # Parse envelope
                try:
                    res_json = response.json()
                except Exception as err:
                    raise SemanticResponseError(
                        f"Failed to decode NVIDIA API response envelope as JSON: {err}"
                    ) from err

                if not isinstance(res_json, dict):
                    raise SemanticResponseError(
                        f"NVIDIA API response envelope must be a JSON object, got {type(res_json).__name__}"
                    )

                # Check if envelope contains an error
                if "error" in res_json:
                    err_msg = _extract_api_error_message(json.dumps(res_json), api_key)
                    raise SemanticResponseError(f"NVIDIA API returned error envelope: {err_msg}")

                # Extract usage metadata
                prompt_tokens: int | None = None
                output_tokens: int | None = None
                total_tokens: int | None = None
                usage = res_json.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens")
                    output_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")
                    self._last_call_meta["prompt_tokens"] = prompt_tokens
                    self._last_call_meta["output_tokens"] = output_tokens
                    self._last_call_meta["total_tokens"] = total_tokens

                # Validate choices
                if "choices" not in res_json:
                    raise SemanticResponseError("NVIDIA API response missing 'choices' field")

                choices = res_json.get("choices")
                if choices is None or not isinstance(choices, list) or len(choices) == 0:
                    raise SemanticResponseError("NVIDIA API response 'choices' must be a non-empty list")

                first_choice = choices[0]
                if not isinstance(first_choice, dict):
                    raise SemanticResponseError(
                        f"NVIDIA API choice must be an object, got {type(first_choice).__name__}"
                    )

                message = first_choice.get("message")
                if not isinstance(message, dict):
                    raise SemanticResponseError(
                        f"NVIDIA API message must be an object, got {type(message).__name__}"
                    )

                content = message.get("content")
                if content is None:
                    finish_reason = first_choice.get("finish_reason")
                    raise SemanticResponseError(
                        f"NVIDIA API returned empty content (finish_reason={finish_reason})"
                    )

                usage_dict = {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                return content, usage_dict, retry_count

            except (SemanticExtractionError, httpx.HTTPError):
                raise
            except Exception as unhandled_err:
                clean_msg = _sanitize_error_message(str(unhandled_err), api_key)
                raise SemanticExtractionError(
                    f"Unexpected failure during NVIDIA request execution: {clean_msg}"
                ) from unhandled_err

    def extract_single_pass(self, input_data: SemanticInput) -> SemanticOutput:
        """Single-pass extraction mode."""
        api_key, model, base_url, timeout, max_retries, max_tokens, response_format_type, enable_thinking = self._resolve_config()
        start_time = self._time_fn()
        schema = get_compact_schema(SemanticOutput)
        schema["required"] = [
            "document_archetype",
            "personal",
            "skills",
            "experience",
            "education",
            "projects",
            "certifications",
            "languages",
            "achievements",
        ]
        if "properties" in schema and "personal" in schema["properties"]:
            p_prop = schema["properties"]["personal"]
            if isinstance(p_prop, dict) and "properties" in p_prop:
                p_prop["required"] = ["name", "email", "phone", "location"]
        prompt = build_compact_extraction_prompt(input_data)

        raw_text, usage_dict, retry_count = self._execute_prompt_request(
            prompt=prompt,
            response_schema=schema,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
            response_format_type=response_format_type,
            enable_thinking=enable_thinking,
            pass_name="single",
        )

        result = parse_semantic_output(raw_text)
        ev_cat = get_body_evidence_category(input_data)
        is_empty = not bool(
            result.skills
            or result.experience
            or result.education
            or result.projects
            or result.certifications
            or result.languages
            or result.achievements
        )
        body_recovery_invoked = False
        body_recovery_attempts = 0
        if is_empty and ev_cat:
            prompt_recovery = build_body_recovery_prompt(input_data)
            schema_body = get_body_schema()
            raw_rec_text, rec_usage, rec_retries = self._execute_prompt_request(
                prompt=prompt_recovery,
                response_schema=schema_body,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                max_tokens=max_tokens,
                response_format_type=response_format_type,
                enable_thinking=enable_thinking,
                pass_name="recovery",
            )
            rec_parsed = parse_body_output(raw_rec_text)
            result.skills = rec_parsed.skills
            result.experience = rec_parsed.experience
            result.education = rec_parsed.education
            result.projects = rec_parsed.projects
            result.certifications = rec_parsed.certifications
            result.languages = rec_parsed.languages
            result.achievements = rec_parsed.achievements
            if rec_parsed.summary and not result.summary:
                result.summary = rec_parsed.summary
            if rec_parsed.block_classifications:
                result.block_classifications.extend(rec_parsed.block_classifications)
            body_recovery_invoked = True
            body_recovery_attempts = 1
            retry_count += rec_retries
            p_tok = (usage_dict.get("prompt_tokens") or 0) + (rec_usage.get("prompt_tokens") or 0)
            o_tok = (usage_dict.get("output_tokens") or 0) + (rec_usage.get("output_tokens") or 0)
            tot_tok = (usage_dict.get("total_tokens") or 0) + (rec_usage.get("total_tokens") or 0)
            usage_dict["prompt_tokens"] = p_tok
            usage_dict["output_tokens"] = o_tok
            usage_dict["total_tokens"] = tot_tok
            is_empty = not bool(
                result.skills
                or result.experience
                or result.education
                or result.projects
                or result.certifications
                or result.languages
                or result.achievements
            )

        elapsed_ms = (self._time_fn() - start_time) * 1000
        usage_meta = {
            **usage_dict,
            "latency_ms": elapsed_ms,
            "retry_count": retry_count,
            "two_pass": False,
            "body_recovery_invoked": body_recovery_invoked,
            "body_recovery_reason": "suspicious_empty_body_recovery" if body_recovery_invoked else None,
            "body_recovery_attempts": body_recovery_attempts,
            "final_body_empty": is_empty,
            "body_completeness_failure": False,
            "evidence_category": ev_cat,
        }
        self.last_usage_metadata = usage_meta
        self._last_call_meta = usage_meta
        return result

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        """Extract structured resume entities from layout-aware SemanticInput."""
        if not self._resolve_two_pass():
            return self.extract_single_pass(input_data)

        api_key, model, base_url, timeout, max_retries, max_tokens, response_format_type, enable_thinking = self._resolve_config()
        start_time = self._time_fn()

        schema_personal = get_personal_schema()
        schema_body = get_body_schema()

        prompt_personal = build_personal_extraction_prompt(input_data)

        def _run_personal() -> tuple[PersonalSemanticOutput, dict[str, Any], int]:
            raw_text, usage, retries = self._execute_prompt_request(
                prompt=prompt_personal,
                response_schema=schema_personal,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                max_tokens=max_tokens,
                response_format_type=response_format_type,
                enable_thinking=enable_thinking,
                pass_name="personal",
            )
            parsed = parse_personal_output(raw_text)
            return parsed, usage, retries

        def _run_body_for_section_input(
            section_input: SemanticInput,
            pass_name: str,
            is_recovery: bool = False,
        ) -> tuple[BodySemanticOutput, dict[str, Any], int]:
            if is_recovery:
                prompt = build_body_recovery_prompt(section_input)
            else:
                prompt = build_body_extraction_prompt(section_input)
            raw_text, usage, retries = self._execute_prompt_request(
                prompt=prompt,
                response_schema=schema_body,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                max_tokens=max_tokens,
                response_format_type=response_format_type,
                enable_thinking=enable_thinking,
                pass_name=pass_name,
            )
            parsed = parse_body_output(raw_text)
            usage_enriched = {
                **usage,
                "body_recovery_invoked": is_recovery,
                "body_recovery_reason": "suspicious_empty_body_recovery" if is_recovery else None,
                "body_recovery_attempts": 1 if is_recovery else 0,
                "final_body_empty": not bool(
                    parsed.skills or parsed.experience or parsed.education
                    or parsed.projects or parsed.certifications
                    or parsed.languages or parsed.achievements
                ),
                "body_completeness_failure": False,
                "evidence_category": get_body_evidence_category(section_input),
            }
            return parsed, usage_enriched, retries

        def _run_body() -> tuple[BodySemanticOutput, dict[str, Any], int]:
            use_section_aware = should_use_section_aware_body_extraction(input_data)

            if use_section_aware:
                units = plan_section_aware_body_passes(input_data)
                arch_name = (
                    input_data.archetype.value
                    if hasattr(input_data.archetype, "value")
                    else str(input_data.archetype)
                )
                logger.info(
                    "NVIDIA body pass section-aware mode doc_id=%s archetype=%s units=%d",
                    input_data.document_id,
                    arch_name,
                    len(units),
                )

                if not units:
                    logger.warning(
                        "NVIDIA body pass section-aware mode found no units; "
                        "falling through to monolithic pass doc_id=%s",
                        input_data.document_id,
                    )
                    use_section_aware = False

                if use_section_aware:
                    section_outputs: list[BodySemanticOutput] = []
                    total_p_t = 0
                    total_o_t = 0
                    total_tot_t = 0
                    total_retries = 0

                    for unit in units:
                        if unit.is_appointment:
                            logger.info(
                                "NVIDIA body pass appointment extraction doc_id=%s section=%r "
                                "appt=%d/%d title_id=%s blocks=%d",
                                input_data.document_id,
                                unit.section_heading,
                                unit.appt_index,
                                unit.total_appts,
                                unit.title_block_id,
                                len(unit.section_input.blocks),
                            )
                        else:
                            logger.info(
                                "NVIDIA body pass section extraction doc_id=%s section=%r "
                                "target=%s blocks=%d",
                                input_data.document_id,
                                unit.section_heading,
                                unit.canonical_target,
                                len(unit.section_input.blocks),
                            )
                        try:
                            sec_res, sec_usage, sec_retries = _run_body_for_section_input(
                                unit.section_input, pass_name=unit.pass_name
                            )
                            sec_res = isolate_unit_target_collections(sec_res, unit.canonical_target)
                            if unit.is_appointment:
                                sec_res = enforce_single_experience_entity(sec_res, unit.title_block_id)
                                allowed_bids = unit.appointment_block_ids or [
                                    b.block_id for b in unit.section_input.blocks
                                ]
                                sec_res = constrain_appointment_experience_provenance(sec_res, allowed_bids)
                            section_outputs.append(sec_res)
                            total_p_t += (sec_usage.get("prompt_tokens") or 0)
                            total_o_t += (sec_usage.get("output_tokens") or 0)
                            total_tot_t += (sec_usage.get("total_tokens") or 0)
                            total_retries += sec_retries
                        except Exception as sec_exc:
                            if isinstance(sec_exc, SemanticConfigurationError):
                                raise
                            logger.warning(
                                "NVIDIA %s extraction failed doc_id=%s section=%r "
                                "target=%s error_type=%s: %s",
                                "appointment" if unit.is_appointment else "section",
                                input_data.document_id,
                                unit.section_heading,
                                unit.canonical_target,
                                type(sec_exc).__name__,
                                sec_exc,
                            )

                    merged_body = merge_body_outputs(section_outputs)
                    ev_category = get_body_evidence_category(input_data)

                    if is_body_output_suspiciously_empty(merged_body, input_data):
                        logger.warning(
                            "NVIDIA sectioned body passes returned suspiciously empty collections "
                            "(evidence_category=%s); triggering bounded recovery doc_id=%s model=%s",
                            ev_category,
                            input_data.document_id,
                            model,
                        )
                        supported_block_ids: list[str] = list(
                            dict.fromkeys(b.block_id for u in units for b in u.section_input.blocks)
                        )
                        recovery_input = filter_semantic_input_to_blocks(
                            input_data, supported_block_ids, include_headers=False
                        )
                        prompt_body_recovery = build_body_recovery_prompt(recovery_input)
                        raw_text_rec, usage_rec, retries_rec = self._execute_prompt_request(
                            prompt=prompt_body_recovery,
                            response_schema=schema_body,
                            api_key=api_key,
                            model=model,
                            base_url=base_url,
                            timeout=timeout,
                            max_retries=max_retries,
                            max_tokens=max_tokens,
                            response_format_type=response_format_type,
                            enable_thinking=enable_thinking,
                            pass_name="body_sectioned_recovery",
                        )
                        parsed_rec = parse_body_output(raw_text_rec)

                        p_t = total_p_t + (usage_rec.get("prompt_tokens") or 0)
                        o_t = total_o_t + (usage_rec.get("output_tokens") or 0)
                        tot_t = total_tot_t + (usage_rec.get("total_tokens") or 0)

                        still_empty = is_body_output_suspiciously_empty(parsed_rec, input_data)

                        combined_usage = {
                            "prompt_tokens": p_t,
                            "output_tokens": o_t,
                            "total_tokens": tot_t,
                            "body_recovery_invoked": True,
                            "body_recovery_reason": "suspicious_empty_body_sectioned",
                            "body_recovery_attempts": 1,
                            "final_body_empty": still_empty,
                            "body_completeness_failure": still_empty,
                            "evidence_category": ev_category,
                            "recovery_pass": usage_rec,
                        }
                        combined_retries = total_retries + retries_rec

                        if still_empty:
                            self._last_call_meta = combined_usage
                            logger.error(
                                "NVIDIA body sectioned recovery failed: body collections remain empty "
                                "doc_id=%s model=%s evidence_category=%s",
                                input_data.document_id,
                                model,
                                ev_category,
                            )
                            raise SemanticCompletenessError(
                                f"Empty body output on evidence-rich document {input_data.document_id!r} "
                                f"after sectioned bounded recovery (evidence: {ev_category})",
                                reason="empty_body_after_recovery",
                                evidence_category=ev_category,
                            )

                        return parsed_rec, combined_usage, combined_retries

                    is_empty = not bool(
                        merged_body.skills or merged_body.experience or merged_body.education
                        or merged_body.projects or merged_body.certifications
                        or merged_body.languages or merged_body.achievements
                    )
                    usage_final = {
                        "prompt_tokens": total_p_t,
                        "output_tokens": total_o_t,
                        "total_tokens": total_tot_t,
                        "body_recovery_invoked": False,
                        "body_recovery_reason": None,
                        "body_recovery_attempts": 0,
                        "final_body_empty": is_empty,
                        "body_completeness_failure": False,
                        "evidence_category": ev_category,
                    }
                    return merged_body, usage_final, total_retries

            # Normal monolithic body pass (STANDARD_CV and other archetypes)
            sections = partition_semantic_input_into_sections(input_data)
            has_unsupported = any(s.canonical_target == "unsupported" for s in sections)
            supported_blocks: list[str] = [
                bid for s in sections if s.canonical_target != "unsupported" for bid in s.block_ids
            ]
            if has_unsupported and supported_blocks:
                normal_body_input = filter_semantic_input_to_blocks(input_data, supported_blocks)
            else:
                normal_body_input = input_data

            prompt_body_req = build_body_extraction_prompt(normal_body_input)
            raw_text, usage, retries = self._execute_prompt_request(
                prompt=prompt_body_req,
                response_schema=schema_body,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                max_tokens=max_tokens,
                response_format_type=response_format_type,
                enable_thinking=enable_thinking,
                pass_name="body",
            )
            parsed = parse_body_output(raw_text)

            # Bounded semantic recovery for silent empty-body omission
            if is_body_output_suspiciously_empty(parsed, input_data):
                ev_category = get_body_evidence_category(input_data)
                logger.warning(
                    "NVIDIA body pass returned suspiciously empty collections on evidence-rich document "
                    "(evidence_category=%s); triggering bounded recovery attempt doc_id=%s model=%s",
                    ev_category,
                    input_data.document_id,
                    model,
                )
                recovery_sections = group_sections_for_recovery(sections)
                if recovery_sections:
                    rec_outputs: list[BodySemanticOutput] = []
                    p_t = usage.get("prompt_tokens") or 0
                    o_t = usage.get("output_tokens") or 0
                    tot_t = usage.get("total_tokens") or 0
                    total_rec_retries = retries
                    for rsec in recovery_sections:
                        rsec_input = filter_semantic_input_to_blocks(
                            input_data, rsec.block_ids, include_headers=False
                        )
                        rsec_res, rsec_usage, rsec_ret = _run_body_for_section_input(
                            rsec_input,
                            pass_name=f"body_recovery_{rsec.canonical_target}",
                            is_recovery=True,
                        )
                        rec_outputs.append(rsec_res)
                        p_t += (rsec_usage.get("prompt_tokens") or 0)
                        o_t += (rsec_usage.get("output_tokens") or 0)
                        tot_t += (rsec_usage.get("total_tokens") or 0)
                        total_rec_retries += rsec_ret
                    parsed_rec = merge_body_outputs(rec_outputs)
                else:
                    prompt_body_recovery = build_body_recovery_prompt(normal_body_input)
                    raw_text_rec, usage_rec, retries_rec = self._execute_prompt_request(
                        prompt=prompt_body_recovery,
                        response_schema=schema_body,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                        timeout=timeout,
                        max_retries=max_retries,
                        max_tokens=max_tokens,
                        response_format_type=response_format_type,
                        enable_thinking=enable_thinking,
                        pass_name="body_recovery",
                    )
                    parsed_rec = parse_body_output(raw_text_rec)
                    p_t = (usage.get("prompt_tokens") or 0) + (usage_rec.get("prompt_tokens") or 0)
                    o_t = (usage.get("output_tokens") or 0) + (usage_rec.get("output_tokens") or 0)
                    tot_t = (usage.get("total_tokens") or 0) + (usage_rec.get("total_tokens") or 0)
                    total_rec_retries = retries + retries_rec

                still_empty = is_body_output_suspiciously_empty(parsed_rec, input_data)

                combined_usage = {
                    "prompt_tokens": p_t,
                    "output_tokens": o_t,
                    "total_tokens": tot_t,
                    "body_recovery_invoked": True,
                    "body_recovery_reason": "suspicious_empty_body",
                    "body_recovery_attempts": len(recovery_sections) if recovery_sections else 1,
                    "final_body_empty": still_empty,
                    "body_completeness_failure": still_empty,
                    "evidence_category": ev_category,
                    "initial_pass": usage,
                }

                if still_empty:
                    self._last_call_meta = combined_usage
                    logger.error(
                        "NVIDIA body recovery failed: body collections remain empty "
                        "doc_id=%s model=%s evidence_category=%s",
                        input_data.document_id,
                        model,
                        ev_category,
                    )
                    raise SemanticCompletenessError(
                        f"Empty body output on evidence-rich document {input_data.document_id!r} "
                        f"after bounded recovery (evidence: {ev_category})",
                        reason="empty_body_after_recovery",
                        evidence_category=ev_category,
                    )

                return parsed_rec, combined_usage, total_rec_retries

            ev_cat = get_body_evidence_category(input_data)
            is_empty = not bool(
                parsed.skills
                or parsed.experience
                or parsed.education
                or parsed.projects
                or parsed.certifications
                or parsed.languages
                or parsed.achievements
            )
            usage_enriched = {
                **usage,
                "body_recovery_invoked": False,
                "body_recovery_reason": None,
                "body_recovery_attempts": 0,
                "final_body_empty": is_empty,
                "body_completeness_failure": False,
                "evidence_category": ev_cat,
            }
            return parsed, usage_enriched, retries

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_personal = executor.submit(_run_personal)
            fut_body = executor.submit(_run_body)

            personal_res, personal_usage, personal_retries = fut_personal.result()
            body_res, body_usage, body_retries = fut_body.result()

        merged_result = merge_semantic_passes(personal_res, body_res)
        merged_result = sanitize_grounded_current_status(merged_result, input_data)

        p_p = personal_usage.get("prompt_tokens")
        b_p = body_usage.get("prompt_tokens")
        prompt_tokens = (p_p or 0) + (b_p or 0) if (p_p is not None or b_p is not None) else None

        p_o = personal_usage.get("output_tokens")
        b_o = body_usage.get("output_tokens")
        output_tokens = (p_o or 0) + (b_o or 0) if (p_o is not None or b_o is not None) else None

        p_t = personal_usage.get("total_tokens")
        b_t = body_usage.get("total_tokens")
        total_tokens = (p_t or 0) + (b_t or 0) if (p_t is not None or b_t is not None) else None

        elapsed_ms = (self._time_fn() - start_time) * 1000
        total_retries = personal_retries + body_retries

        combined_usage_metadata = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": elapsed_ms,
            "retry_count": total_retries,
            "two_pass": True,
            "personal_pass": personal_usage,
            "body_pass": body_usage,
        }
        self.last_usage_metadata = combined_usage_metadata
        self._last_call_meta = combined_usage_metadata
        return merged_result
