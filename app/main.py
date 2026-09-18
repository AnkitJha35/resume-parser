import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import router as api_v1_router
from app.core.config import Settings
from app.core.logging import configure_logging
from app.infrastructure.kafka.consumer import KafkaConsumerClient
from app.services.resume_request_service import ResumeRequestService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = Settings()
    app.state.settings = settings

    if getattr(settings, "kafka_enabled", False):
        try:
            logger.info("Starting background Kafka consumer on %s", settings.kafka_brokers)
            app.state.resume_request_service = ResumeRequestService(settings)
            await app.state.resume_request_service.start()
            app.state.kafka_consumer = KafkaConsumerClient(settings, app.state.resume_request_service.process)
            await app.state.kafka_consumer.start()
        except Exception as exc:
            logger.error("Failed to start Kafka worker: %s", exc)
    yield

    kafka_consumer = getattr(app.state, "kafka_consumer", None)
    if kafka_consumer is not None:
        await kafka_consumer.shutdown()
    resume_request_service = getattr(app.state, "resume_request_service", None)
    if resume_request_service is not None:
        await resume_request_service.shutdown()


app = FastAPI(
    title="Resume Parser Service",
    description="Production Resume Parser service providing layout-aware semantic parsing and diagnostic APIs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend / client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> dict[str, str]:
    return {
        "service": "Resume Parser Service",
        "docs": "/docs",
        "health": "/health",
        "api_v1_health": "/api/v1/health",
        "api_v1_config": "/api/v1/config",
    }
