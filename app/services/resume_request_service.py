from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import StorageClientError
from app.extractors.factory import get_semantic_extractor
from app.extractors.semantic_extractor import SemanticExtractor
from app.infrastructure.kafka.producer import KafkaProducerClient
from app.infrastructure.storage.minio_client import MinioClient
from app.pipeline.parser import PipelineError, ResumeParser

logger = logging.getLogger(__name__)


class ResumeRequestService:
    def __init__(
        self,
        settings: Settings,
        storage_client_cls: type[MinioClient] = MinioClient,
        kafka_producer_cls: type[KafkaProducerClient] = KafkaProducerClient,
        parser_cls: type[ResumeParser] = ResumeParser,
        semantic_extractor: SemanticExtractor | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage_client_cls(settings)
        self._producer = kafka_producer_cls(settings)
        self._parser = parser_cls()
        self._semantic_extractor = semantic_extractor or get_semantic_extractor(settings)

    async def start(self) -> None:
        await self._producer.start()

    async def shutdown(self) -> None:
        await self._producer.shutdown()

    async def process(self, message: dict[str, Any]) -> None:
        job_id = message.get("jobId")
        resume_id = message.get("resumeId")
        storage_key = message.get("storageKey")

        if not job_id or not resume_id or not storage_key:
            await self._publish_failed_event(
                job_id or "",
                resume_id or "",
                "RESUME_PARSE_FAILED",
                "Missing required resume request fields.",
            )
            return

        try:
            pdf_bytes = await self._download_pdf(storage_key)
            start_time = time.monotonic()
            parser_method = (
                self._parser.parse
                if self._settings.parser_mode == "legacy"
                else self._parser.parse_with_layout_pipeline
            )
            resume = await asyncio.to_thread(parser_method, pdf_bytes)
            processing_time_ms = int((time.monotonic() - start_time) * 1000)
            await self._publish_completed_event(job_id, resume_id, resume, processing_time_ms)
        except StorageClientError as exc:
            logger.exception("Storage error while processing resume request")
            await self._publish_failed_event(job_id, resume_id, "STORAGE_ERROR", str(exc))
        except PipelineError as exc:
            logger.exception("Pipeline error while processing resume request")
            await self._publish_failed_event(job_id, resume_id, exc.code, exc.message)
        except ValidationError as exc:
            logger.exception("Validation error while processing resume request")
            await self._publish_failed_event(job_id, resume_id, "VALIDATION_ERROR", str(exc))
        except Exception as exc:
            logger.exception("Unexpected error while processing resume request")
            await self._publish_failed_event(job_id, resume_id, "RESUME_PARSE_FAILED", "An unexpected error occurred.")

    async def _download_pdf(self, storage_key: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return await asyncio.to_thread(self._storage.download_pdf, storage_key)
            except StorageClientError as exc:
                last_error = exc
                if attempt < 3:
                    await asyncio.sleep(0.5)
        raise last_error

    async def _publish_completed_event(
        self,
        job_id: str,
        resume_id: str,
        resume: Any,
        processing_time_ms: int,
    ) -> None:
        metadata = dict(resume.metadata)
        metadata["processingTimeMs"] = processing_time_ms
        message = {
            "jobId": job_id,
            "resumeId": resume_id,
            "status": "COMPLETED",
            "parserVersion": self._settings.parser_version,
            "result": resume.model_dump(),
            "metadata": metadata,
        }
        await self._producer.send_json(self._settings.kafka_topic_completed, message)

    async def _publish_failed_event(self, job_id: str, resume_id: str, code: str, message: str) -> None:
        event = {
            "jobId": job_id,
            "resumeId": resume_id,
            "status": "FAILED",
            "error": {
                "code": code,
                "message": message,
            },
        }
        await self._producer.send_json(self._settings.kafka_topic_failed, event)
