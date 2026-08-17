"""Memory subsystem configuration (Phase 6).

All memory settings are loaded from environment variables. The memory
subsystem is **local-first**: no cloud embedding API or hosted vector
database is required. PostgreSQL is the durable source of truth; Redis
handles working memory and cache; vector storage is optional and pluggable.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["MemorySettings"]


class MemorySettings(BaseSettings):
    """Persistent memory subsystem configuration."""

    # --- Master switch ---
    enabled: bool = Field(default=True, alias="MEMORY_ENABLED")

    # --- PostgreSQL durable store ---
    pg_persistence: bool = Field(default=True, alias="MEMORY_PG_PERSISTENCE")

    # --- Redis working memory / cache ---
    redis_working_memory: bool = Field(default=True, alias="MEMORY_REDIS_WORKING")
    working_memory_ttl_seconds: int = Field(default=3600, alias="MEMORY_WORKING_TTL")
    cache_ttl_seconds: int = Field(default=600, alias="MEMORY_CACHE_TTL")

    # --- Vector memory ---
    vector_enabled: bool = Field(default=False, alias="MEMORY_VECTOR_ENABLED")
    vector_store_backend: str = Field(default="memory", alias="MEMORY_VECTOR_BACKEND")
    # pgvector is optional; when "pgvector" is selected the backend requires
    # the pgvector extension. Tests always run without it.
    vector_dimensions: int = Field(default=256, alias="MEMORY_VECTOR_DIMENSIONS")

    # --- Embedding provider (local-first) ---
    embedding_provider: str = Field(default="hashing", alias="MEMORY_EMBEDDING_PROVIDER")
    # Future: a local Ollama embedding model name (e.g. "nomic-embed-text").
    embedding_model: str = Field(default="", alias="MEMORY_EMBEDDING_MODEL")

    # --- Retrieval limits (context window control) ---
    retrieval_max_items: int = Field(default=10, alias="MEMORY_MAX_ITEMS")
    retrieval_max_chars: int = Field(default=4000, alias="MEMORY_MAX_CHARS")
    retrieval_max_tokens: int = Field(default=2000, alias="MEMORY_MAX_TOKENS")
    # Character approximation: ~4 chars per token (documented, not exact).
    chars_per_token_approx: int = Field(default=4, alias="MEMORY_CHARS_PER_TOKEN")

    # --- Retention ---
    conversation_retention_hours: int = Field(
        default=48, alias="MEMORY_CONVERSATION_RETENTION_HOURS"
    )
    task_retention_hours: int = Field(default=168, alias="MEMORY_TASK_RETENTION_HOURS")
    # PREFERENCE and SEMANTIC are long-lived (no expiry by default).

    # --- Privacy ---
    # Reject content matching these substrings (case-insensitive) unless
    # explicitly allowed. Basic guardrail, not comprehensive detection.
    sensitive_patterns: list[str] = Field(
        default_factory=lambda: [
            "api_key",
            "apikey",
            "secret_key",
            "password",
            "token",
            "bearer ",
            "authorization",
        ],
        alias="MEMORY_SENSITIVE_PATTERNS",
    )
    allow_sensitive_storage: bool = Field(default=False, alias="MEMORY_ALLOW_SENSITIVE")

    # --- Duplicate detection ---
    duplicate_similarity_threshold: float = Field(default=0.85, alias="MEMORY_DUPLICATE_THRESHOLD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MEMORY_",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )
