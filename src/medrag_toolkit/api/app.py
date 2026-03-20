"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from medrag_toolkit.api.routes import router
from medrag_toolkit.config import Settings
from medrag_toolkit.core import MedRAG

log = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_file()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        medrag = MedRAG(config)
        app.state.medrag = medrag
        log.info("medrag_api_started", model=config.ollama.model)
        yield
        await medrag.close()
        log.info("medrag_api_stopped")

    app = FastAPI(
        title="MedRAG Toolkit API",
        description="Production Medical RAG with mandatory citation grounding",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
