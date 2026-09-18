"""Re-export API endpoints for top-level app.api package access."""

from app.api.v1.endpoints import (
    DiagnosticExtractorWrapper,
    MAX_UPLOAD_SIZE,
    router,
    set_extractor_override,
)

__all__ = [
    "DiagnosticExtractorWrapper",
    "MAX_UPLOAD_SIZE",
    "router",
    "set_extractor_override",
]
