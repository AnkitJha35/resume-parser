import asyncio
import json
import logging
from typing import Any, Callable

from aiokafka import AIOKafkaConsumer

from app.core.config import Settings
from app.core.exceptions import KafkaConsumerError

logger = logging.getLogger(__name__)


class KafkaConsumerClient:
    def __init__(self, settings: Settings, message_handler: Callable[[dict[str, Any]], Any]) -> None:
        self._settings = settings
        self._message_handler = message_handler
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        try:
            self._consumer = AIOKafkaConsumer(
                self._settings.kafka_topic_request,
                bootstrap_servers=self._settings.kafka_brokers.split(","),
                loop=asyncio.get_running_loop(),
                group_id="resume-parser-group",
                auto_offset_reset="earliest",
            )
            await self._consumer.start()
            asyncio.create_task(self._consume_loop())
        except Exception as exc:
            logger.exception("Failed to start Kafka consumer")
            raise KafkaConsumerError("Failed to start Kafka consumer") from exc

    async def _consume_loop(self) -> None:
        assert self._consumer is not None
        try:
            async for msg in self._consumer:
                try:
                    payload = json.loads(msg.value.decode("utf-8"))
                    await self._message_handler(payload)
                except Exception:
                    logger.exception("Failed to process Kafka message")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()
