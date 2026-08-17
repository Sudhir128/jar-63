"""FastAPI application factory and lifecycle handling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.agents import router as agents_router
from app.api.llm import router as llm_router
from app.api.memory import router as memory_router
from app.api.routes import router as health_router
from app.api.tasks import router as tasks_router
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.memory import close_redis
from app.runtime import RuntimeManager
from app.runtime.bootstrap import register_demo_agents
from app.runtime.manager import register_default_tools

logger = get_logger("app")

__all__ = ["create_app", "app"]


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.logging)

    runtime = RuntimeManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.bind(event="app.startup", version=__version__).info(
            "Starting {} ({})", settings.app.name, settings.app.env.value
        )
        # Register default tools before start() so the Phase 7 router/dispatcher
        # see all tools (the agent orchestration is wired inside start()).
        await register_default_tools(runtime.tool_registry)
        await runtime.start()
        await register_demo_agents(runtime.agent_registry, settings)
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.shutdown()
            await close_redis()
            logger.bind(event="app.shutdown").info("Application stopped")

    app = FastAPI(
        title=settings.app.name,
        version=__version__,
        description="JAR-63 — a modular personal AI operating system.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=settings.app.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(tasks_router)
    app.include_router(llm_router)
    app.include_router(memory_router)
    app.include_router(agents_router)
    return app


# Module-level app instance for ``uvicorn app.main:app``.
app = create_app()
