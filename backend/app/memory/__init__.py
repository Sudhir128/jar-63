"""Memory subsystem package (Phase 0 foundation + Phase 6 persistence).

Phase 0 provided only the Redis connection abstraction. Phase 6 adds the
persistent memory layer: stores, vector index, embeddings, write policy,
retriever, conversation store, consolidator, and the coordinator
(:class:`MemoryManager`).
"""

from app.memory.consolidation import (
    BasicMemoryConsolidator,
    ConsolidationReport,
    MemoryConsolidator,
)
from app.memory.conversation import (
    ConversationStore,
    ConversationSummarizer,
    summarize_turns,
)
from app.memory.db_init import init_memory_tables
from app.memory.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from app.memory.health import MemoryHealthChecker, MemoryHealthSnapshot
from app.memory.manager import (
    MemoryManager,
    MemoryManagerState,
    build_memory_manager,
)
from app.memory.models import (
    ConversationMessage,
    MemoryContext,
    MemoryRecord,
    MemorySearchQuery,
    MemorySearchResult,
)
from app.memory.policy import MemoryWritePolicy, WriteDecision
from app.memory.redis import (
    RedisClient,
    check_redis_connection,
    close_redis,
    get_redis,
    get_shared_redis,
)
from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore, PostgreSQLStore
from app.memory.types import (
    MEMORY_TYPE_DEFAULT_RETENTION,
    MemorySource,
    MemoryType,
    MemoryWriteDecision,
    RetentionPolicy,
)
from app.memory.vector import InMemoryVectorStore, VectorMemoryStore, VectorRecord
from app.memory.working_memory import RedisWorkingMemoryStore

__all__ = [
    # Redis foundation (Phase 0)
    "RedisClient",
    "check_redis_connection",
    "close_redis",
    "get_redis",
    "get_shared_redis",
    # Phase 6 domain models
    "ConversationMessage",
    "MemoryContext",
    "MemoryRecord",
    "MemorySearchQuery",
    "MemorySearchResult",
    # Phase 6 types
    "MEMORY_TYPE_DEFAULT_RETENTION",
    "MemorySource",
    "MemoryType",
    "MemoryWriteDecision",
    "RetentionPolicy",
    # Phase 6 stores
    "MemoryStore",
    "PostgreSQLStore",
    "RedisWorkingMemoryStore",
    "VectorMemoryStore",
    "InMemoryVectorStore",
    "VectorRecord",
    # Phase 6 embeddings
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    # Phase 6 policy + retriever
    "MemoryWritePolicy",
    "WriteDecision",
    "MemoryRetriever",
    # Phase 6 conversation
    "ConversationStore",
    "ConversationSummarizer",
    "summarize_turns",
    # Phase 6 consolidation
    "BasicMemoryConsolidator",
    "ConsolidationReport",
    "MemoryConsolidator",
    # Phase 6 manager + health
    "MemoryManager",
    "MemoryManagerState",
    "build_memory_manager",
    "MemoryHealthChecker",
    "MemoryHealthSnapshot",
    "init_memory_tables",
]
