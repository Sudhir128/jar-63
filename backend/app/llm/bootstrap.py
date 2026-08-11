"""LLM runtime bootstrap.

Registers default model definitions and providers into the LLM registries at
startup, driven by configuration. Ollama is always registered (local-first).
The OpenAI-compatible provider is registered only when configured.

No model is downloaded or modified. If the configured Ollama model is not
installed, the provider reports ``MODEL_NOT_FOUND`` at request time.
"""

from __future__ import annotations

from app.config import Settings
from app.core.logging import get_logger
from app.events import EventBus
from app.llm.factory import ProviderFactory
from app.llm.models import ModelCapability, ModelDefinition
from app.llm.providers.ollama import OLLAMA_PROVIDER_ID
from app.llm.providers.openai_compatible import OPENAI_COMPATIBLE_PROVIDER_ID
from app.llm.registry import ModelRegistry, ProviderRegistry

logger = get_logger("llm.bootstrap")

__all__ = ["register_default_models", "register_default_providers", "default_model_definitions"]


def default_model_definitions(settings: Settings) -> list[ModelDefinition]:
    """Build the default model definitions from settings."""
    models: list[ModelDefinition] = []
    ollama_model = settings.llm.ollama_default_model or settings.llm.default_model
    models.append(
        ModelDefinition(
            model_id=ollama_model,
            provider=OLLAMA_PROVIDER_ID,
            display_name=ollama_model,
            capabilities={
                ModelCapability.CHAT,
                ModelCapability.CODING,
                ModelCapability.REASONING,
            },
            context_window=32768 if "coder" in ollama_model else 8192,
            supports_structured_output=True,
            supports_tools=False,
            supports_reasoning=True,
            coding_score=0.8 if "coder" in ollama_model else 0.5,
            local=True,
            enabled=True,
        )
    )
    if settings.llm.openai_compatible_configured:
        cloud_model = settings.llm.openai_compatible_model
        models.append(
            ModelDefinition(
                model_id=cloud_model,
                provider=OPENAI_COMPATIBLE_PROVIDER_ID,
                display_name=cloud_model,
                capabilities={
                    ModelCapability.CHAT,
                    ModelCapability.CODING,
                    ModelCapability.REASONING,
                    ModelCapability.TOOL_CALLING,
                    ModelCapability.STRUCTURED_OUTPUT,
                },
                context_window=128000,
                supports_structured_output=True,
                supports_tools=True,
                supports_reasoning=True,
                coding_score=0.9,
                local=False,
                enabled=True,
            )
        )
    return models


def register_default_models(model_registry: ModelRegistry, settings: Settings) -> None:
    """Register default model definitions (idempotent, skips existing)."""
    for model in default_model_definitions(settings):
        if model_registry.exists(model.provider, model.model_id):
            continue
        model_registry.register(model)
        logger.bind(
            event="llm.model.registered", provider=model.provider, model=model.model_id
        ).info("Registered default model '{}'", model.display)


def register_default_providers(
    provider_registry: ProviderRegistry,
    settings: Settings,
    event_bus: EventBus | None = None,
) -> ProviderFactory:
    """Register default LLM providers and return the factory.

    Ollama is always registered. The OpenAI-compatible provider is registered
    only when configured.
    """
    factory = ProviderFactory(settings.llm, event_bus=event_bus)
    if not provider_registry.exists(OLLAMA_PROVIDER_ID):
        provider_registry.register(OLLAMA_PROVIDER_ID, factory.create(OLLAMA_PROVIDER_ID))
    if settings.llm.openai_compatible_configured and not provider_registry.exists(
        OPENAI_COMPATIBLE_PROVIDER_ID
    ):
        provider_registry.register(
            OPENAI_COMPATIBLE_PROVIDER_ID, factory.create(OPENAI_COMPATIBLE_PROVIDER_ID)
        )
    return factory
