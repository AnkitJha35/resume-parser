from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiokafka import AIOKafkaProducer

from app.core.config import Settings

logger = logging.getLogger(__name__)


class KafkaProducerClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._settings.kafka_brokers.split(",")
        )
        await self._producer.start()

    async def send_json(self, topic: str, message: dict[str, Any]) -> None:
        if self._producer is None:
            raise RuntimeError("Kafka producer is not started")
        await self._producer.send_and_wait(topic, json.dumps(message).encode("utf-8"))

    async def shutdown(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
