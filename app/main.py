import logging
from fastapi import FastAPI
from app.core.config import Settings
from app.core.logging import configure_logging
from app.infrastructure.kafka.consumer import KafkaConsumerClient
from app.services.resume_request_service import ResumeRequestService

logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Parser Service")

@app.on_event("startup")
async def startup_event() -> None:
    configure_logging()
    settings = Settings()
    app.state.resume_request_service = ResumeRequestService(settings)
    await app.state.resume_request_service.start()
    app.state.kafka_consumer = KafkaConsumerClient(settings, app.state.resume_request_service.process)
    await app.state.kafka_consumer.start()

@app.on_event("shutdown")
async def shutdown_event() -> None:
    kafka_consumer = getattr(app.state, "kafka_consumer", None)
    if kafka_consumer is not None:
        await kafka_consumer.shutdown()
    resume_request_service = getattr(app.state, "resume_request_service", None)
    if resume_request_service is not None:
        await resume_request_service.shutdown()

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
