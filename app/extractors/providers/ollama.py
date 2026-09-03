"""Ollama REST API adapter for local semantic resume extraction."""

from __future__ import annotations

import json
import logging
import os
import time
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

logger = logging.getLogger(__name__)

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
        num_threads: int | None = None,
        think: bool | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._explicit_base_url = base_url
        self._explicit_model = model
        self._explicit_timeout = timeout
        self._explicit_num_threads = num_threads
        self._explicit_think = think
        self._client = client
        self.last_usage_metadata: dict[str, Any] | None = None

    def _resolve_config(self) -> tuple[str, str, float, int, bool]:
        """Resolve base_url, model, timeout, num_threads, and think from arguments, environment, or settings."""
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
        # 4. CPU Threads (defaults to 8)
        num_threads = self._explicit_num_threads
        if num_threads is not None and num_threads <= 0:
            num_threads = 8

        if num_threads is None:
            env_threads = os.environ.get("OLLAMA_NUM_THREADS")
            if env_threads:
                try:
                    parsed_threads = int(env_threads)
                    num_threads = parsed_threads if parsed_threads > 0 else 8
                except ValueError:
                    num_threads = 8
            else:
                num_threads = 8

        # 5. Thinking / Reasoning control (defaults to False)
        think = self._explicit_think
        if think is None:
            env_think = os.environ.get("OLLAMA_THINK")
            if env_think is not None:
                cleaned = env_think.strip().lower()
                if cleaned in ("1", "true", "yes", "on"):
                    think = True
                elif cleaned in ("0", "false", "no", "off"):
                    think = False
                else:
                    think = False
            else:
                think = None

        if not base_url or not model or num_threads == 8 or think is None:
            try:
                settings = Settings()
                base_url = base_url or getattr(settings, "ollama_base_url", "http://localhost:11434")
                model = model or getattr(settings, "ollama_model", "qwen2.5-coder:7b")
                if timeout == 120.0:
                    timeout = getattr(settings, "ollama_timeout", 120.0)
                if num_threads == 8:
                    st_threads = getattr(settings, "ollama_num_threads", 8)
                    if isinstance(st_threads, int) and st_threads > 0:
                        num_threads = st_threads
                if think is None:
                    think = getattr(settings, "ollama_think", False)
            except (ValidationError, OSError):
                pass

        base_url = (base_url or "http://localhost:11434").rstrip("/")
        model = model or "qwen2.5-coder:7b"
        num_threads = num_threads if (num_threads and num_threads > 0) else 8
        think = bool(think) if think is not None else False

        return base_url, model, timeout, num_threads, think

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
        start_time = time.monotonic()
        base_url, model, timeout, num_threads, think = self._resolve_config()
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
            "think": think,
            "format": schema,
            "options": {
                "temperature": 0.0,
                "num_thread": num_threads,
            },
        }

        logger.debug("Ollama request started model=%s base_url=%s", model, base_url)

        # Send request
        try:
            if self._client is not None:
                response = self._client.post(endpoint_url, json=payload, timeout=timeout)
            else:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(endpoint_url, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as err:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._record_failure(model, "httpx.ConnectError", elapsed_ms)
            raise SemanticExtractionError(
                f"Ollama connection failed at '{base_url}'. Is Ollama running? Error: {err}"
            ) from err
        except httpx.HTTPStatusError as err:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._record_failure(model, "httpx.HTTPStatusError", elapsed_ms)
            raise SemanticExtractionError(
                f"Ollama API HTTP error {err.response.status_code}: {err.response.text}"
            ) from err
        except httpx.TimeoutException as err:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._record_failure(model, "httpx.TimeoutException", elapsed_ms)
            raise SemanticExtractionError(
                f"Ollama API request timed out after {timeout}s"
            ) from err
        except httpx.RequestError as err:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._record_failure(model, "httpx.RequestError", elapsed_ms)
            raise SemanticExtractionError(
                f"Ollama API request failed: {err}"
            ) from err

        # Parse envelope
        try:
            res_json = response.json()
        except Exception as err:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._record_failure(model, type(err).__name__, elapsed_ms)
            raise SemanticExtractionError(
                f"Failed to decode Ollama API response envelope: {err}"
            ) from err

        # Capture token metrics and metadata if reported
        p_tokens = res_json.get("prompt_eval_count")
        o_tokens = res_json.get("eval_count")
        if p_tokens is not None or o_tokens is not None:
            tot = (p_tokens or 0) + (o_tokens or 0)
            self.last_usage_metadata = {
                "prompt_tokens": p_tokens,
                "output_tokens": o_tokens,
                "total_tokens": tot,
                "num_threads": num_threads,
                "think": think,
            }
        else:
            self.last_usage_metadata = {
                "num_threads": num_threads,
                "think": think,
            }

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Ollama request completed model=%s latency_ms=%.2f prompt_tokens=%s output_tokens=%s",
            model,
            elapsed_ms,
            p_tokens,
            o_tokens,
        )

        message = res_json.get("message", {})
        content = message.get("content")
        if content is None:
            raise SemanticExtractionError(
                f"Ollama API response missing 'message.content': {res_json}"
            )

        return parse_semantic_output(content)

    def _record_failure(self, model: str, error_type: str, latency_ms: float) -> None:
        self.last_usage_metadata = {
            "provider": "ollama",
            "model": model,
            "representation": "candidate_b_compact",
            "prompt_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "latency_ms": round(latency_ms, 2),
            "status": "failure",
            "error_type": error_type,
        }
        logger.error(
            "Ollama request failed model=%s error_type=%s latency_ms=%.2f",
            model,
            error_type,
            latency_ms,
        )
