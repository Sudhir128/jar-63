"""LLM provider configuration (provider/model independent).

JAR-63 is **local-first**: Ollama is the primary provider. Cloud inference is
optional via a generic OpenAI-compatible endpoint. No specific cloud provider
is required, and no API key is needed to run the application.

All settings are loaded from environment variables (optionally a ``.env``
file). Secrets are stored as :class:`SecretStr` and never logged.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["LLMSettings"]


class LLMSettings(BaseSettings):
    """LLM configuration: providers, routing, privacy, timeouts.

    Environment variables (see ``.env.example``):

    * ``LLM_ENABLED`` — master switch (default ``true``).
    * ``LLM_DEFAULT_PROVIDER`` — default provider id (default ``ollama``).
    * ``LLM_DEFAULT_MODEL`` — default model id (default ``qwen2.5-coder:7b``).
    * ``LLM_ROUTING_POLICY`` — routing policy (default ``local_first``).
    * ``LLM_REQUEST_TIMEOUT`` — per-request timeout seconds (default ``60``).
    * ``LLM_VERBOSE_LOGGING`` — dev-only full prompt/completion logging
      (default ``false``). Never enable in production.
    * ``LLM_ALLOW_CLOUD_FOR_PRIVATE`` — allow PRIVATE/SENSITIVE requests to
      use cloud providers (default ``false``).
    * ``OLLAMA_BASE_URL`` — Ollama HTTP base URL (default
      ``http://localhost:11434``).
    * ``OLLAMA_DEFAULT_MODEL`` — default Ollama model.
    * ``OPENAI_COMPATIBLE_ENABLED`` — enable the generic cloud provider
      (default ``false``).
    * ``OPENAI_COMPATIBLE_BASE_URL`` — OpenAI-compatible base URL.
    * ``OPENAI_COMPATIBLE_API_KEY`` — optional API key (SecretStr).
    * ``OPENAI_COMPATIBLE_MODEL`` — default cloud model id.

    Legacy provider keys (``OPENAI_API_KEY`` etc.) are retained as optional
    SecretStr fields for forward compatibility but are not required.
    """

    # --- Master / routing ---
    enabled: bool = Field(default=True, alias="LLM_ENABLED")
    default_provider: str = Field(default="ollama", alias="LLM_DEFAULT_PROVIDER")
    default_model: str = Field(default="qwen2.5-coder:7b", alias="LLM_DEFAULT_MODEL")
    routing_policy: str = Field(default="local_first", alias="LLM_ROUTING_POLICY")
    request_timeout: int = Field(default=60, alias="LLM_REQUEST_TIMEOUT")
    verbose_logging: bool = Field(default=False, alias="LLM_VERBOSE_LOGGING")
    allow_cloud_for_private: bool = Field(default=False, alias="LLM_ALLOW_CLOUD_FOR_PRIVATE")

    # --- Ollama (primary, local) ---
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_default_model: str = Field(default="qwen2.5-coder:7b", alias="OLLAMA_DEFAULT_MODEL")

    # --- Generic OpenAI-compatible provider (optional cloud) ---
    openai_compatible_enabled: bool = Field(default=False, alias="OPENAI_COMPATIBLE_ENABLED")
    openai_compatible_base_url: str = Field(default="", alias="OPENAI_COMPATIBLE_BASE_URL")
    openai_compatible_api_key: SecretStr | None = Field(
        default=None, alias="OPENAI_COMPATIBLE_API_KEY"
    )
    openai_compatible_model: str = Field(default="", alias="OPENAI_COMPATIBLE_MODEL")

    # --- Legacy optional provider keys (never required) ---
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, alias="OPENROUTER_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LLM_", extra="ignore", frozen=True, populate_by_name=True
    )

    @property
    def openai_compatible_configured(self) -> bool:
        """Whether the OpenAI-compatible provider is usable."""
        return (
            self.openai_compatible_enabled
            and bool(self.openai_compatible_base_url)
            and bool(self.openai_compatible_model)
        )
