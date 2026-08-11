# JAR-63 — Architecture

This document describes the intended architecture of JAR-63 and how the
Phase 0 foundation maps onto it. Phase 0 establishes the skeleton only;
later phases fill in the behavior without restructuring the project.

## High-level request flow

```
User
  │
  ▼
Interaction Layer
  │
  ▼
Runtime Manager
  │
  ▼
Central Orchestrator
  │
  ├──► Planner        (decompose intent into steps)
  ├──► Router         (select agents/tools for each step)
  └──► Policy         (guardrails, permissions, cost limits)
          │
          ▼
    Agent Dispatcher
          │
          ▼
    Specialized Agents
          │
          ▼
         Tools
          │
          ▼
     Verification
          │
          ▼
      Reflection
          │
          ▼
    Response Builder
          │
          ▼
        User
```

### Voice as the primary interface

The **Interaction Layer** is intentionally abstract. In later phases the
**voice / speech** subsystem (speech-to-text, TTS, wake-word, streaming
status events) becomes the primary interface, with text/HTTP as a secondary
channel. The Phase 0 event bus already defines `VoiceStatus` events so that
future voice components can report state through the same event pipeline as
every other subsystem.

---

## Layer responsibilities

| Layer               | Responsibility                                        | Phase 0 status |
|---------------------|--------------------------------------------------------|----------------|
| Interaction Layer   | Receive user input (text now, voice later)             | HTTP `/health`, `/version` only |
| Runtime Manager     | Owns lifecycle of all subsystems                       | `RuntimeManager` with start/stop lifecycle |
| Central Orchestrator| Drives the plan→route→dispatch→verify→reflect loop     | Contract only (no behavior) |
| Planner             | Decomposes intent into ordered steps                   | Reserved |
| Router              | Maps steps to agents/tools                             | Reserved |
| Policy              | Enforces permissions / limits                          | Reserved |
| Agent Dispatcher    | Executes agents with lifecycle hooks                   | `Dispatcher` contract + `AgentRegistry` |
| Specialized Agents  | Domain agents (research, math, email, …)               | `AgentInterface` only — no concrete agents |
| Tools               | Capabilities used by agents                            | `ToolRegistry` + `ToolInterface` — no concrete tools |
| Verification        | Validates outputs before delivery                      | Reserved |
| Reflection          | Self-critique / retry decisions                        | Reserved |
| Response Builder    | Formats final response for the interaction layer       | Reserved |

---

## Foundation components (Phase 0)

### Configuration (`app/config`)

Centralized, environment-driven configuration via `pydantic-settings`. Each
category (`AppSettings`, `DatabaseSettings`, `RedisSettings`, `LLMSettings`,
`SecuritySettings`, `LoggingSettings`) is an independent `BaseSettings` that
reads its own env vars and can be used standalone. `Settings` aggregates
them and `get_settings()` caches the frozen result. Secrets use `SecretStr`.

### Logging (`app/core/logging.py`)

Structured logging via Loguru with:

- timestamp, component, event, task_id, agent_id, session_id, error context
- secret masking (passwords, api keys, tokens, URLs with credentials,
  `Bearer` headers) — secrets are never written to logs
- optional JSON serialization and file sinks

### Events (`app/events`)

Asynchronous event bus (`EventBus` ABC + `InMemoryEventBus`) supporting:

- `publish`, `subscribe`, `unsubscribe`
- typed events via `EventType`
- multiple subscribers and wildcard (`None`) subscriptions
- event metadata, unique event IDs, timestamps, task/agent/session ids
- isolated error handling (one failing handler does not stop others)
- both async and sync handlers

Initial event types: `TaskStarted/Completed/Failed`,
`AgentStarted/Completed/Failed`, `ToolStarted/Completed`,
`MemoryUpdated`, `WorkflowStarted/Completed`, `VoiceStatus`.

### Agents (`app/agents`)

`AgentInterface` defines the contract every future agent implements:

- `agent_id`, `name`, `description`, `capabilities`
- async `execute(context) -> AgentResult`
- lifecycle hooks: `on_register`, `on_unregister`, `on_start`,
  `on_complete`, `on_error`

`AgentRegistry` supports register/unregister/get/list/exists and
capability-based lookup, with no dependency on concrete agent types.

### Tools (`app/tools`)

`ToolInterface` + `ToolRegistry` mirror the agent pattern. The registry
supports the future tool catalog (python, shell, browser, git, github,
docker, filesystem, database, search, email, calendar, weather, http,
vector_db). Only registration/discovery is implemented.

### Runtime (`app/runtime`)

Minimal but structured managers that later phases implement without
restructuring:

- `RuntimeManager` — owns lifecycle and composes all registries/managers
- `TaskManager` / `SessionManager` — task and session state
- `Dispatcher` — agent dispatch contract (`DefaultDispatcher` placeholder)
- `WorkflowManager` — workflow contract; LangGraph integration deferred

### Database (`app/database`)

SQLAlchemy foundation: sync + async engines, `SessionLocal`, async session
factory, declarative `Base`, FastAPI dependency (`get_db`, `get_async_db`),
and a `check_db_connection` / `check_async_db_connection` health mechanism.
SQLite is supported transparently (used by the test suite); PostgreSQL is
the production target. No application tables are defined yet.

### Memory / Redis (`app/memory`)

`RedisClient` connection abstraction with health-check. No caching or queue
implementations yet.

### API (`app/api`, `app/main.py`)

FastAPI app factory with startup/shutdown lifecycle. Exposes:

- `GET /health` — application status + component health
- `GET /version` — version info
- `/docs`, `/redoc`, `/openapi.json`

---

## Provider/model independence

The LLM configuration exposes optional keys for multiple providers but the
system never depends on a specific one. A future `LLMClient` abstraction
(Phase 2) will sit behind a provider interface so that OpenAI, Anthropic,
Google, Ollama, OpenRouter, and other OpenAI-compatible providers can be
swapped via configuration alone.

---

## Phase roadmap

1. **Phase 0 — Foundation** (complete): skeleton, config, runtime
   interfaces, registries, event bus, DB/Redis, FastAPI, Docker, tests.
2. **Phase 1 — Universal Loop Engine & Runtime Orchestration** (complete):
   goal-directed verifiable loop (DISCOVER → PLAN → EXECUTE → VERIFY →
   DECIDE → ITERATE), stop conditions, evidence-based verification,
   typed loop events, LangGraph adapter, loop service, task/loop REST API,
   deterministic demo workflows. See [loop-engine.md](./loop-engine.md).
3. **Phase 2 — LLM provider abstraction:** provider-independent client.
4. **Phase 3 — Agents & Tools:** concrete agents and tool implementations.
5. **Phase 4 — Memory:** persistent + vector memory, Redis caching/queues.
6. **Phase 5 — Voice / Interaction:** voice as the primary interface.
7. **Phase 6 — Android client.**
8. **Phase 7 — Builder & research agents.**
