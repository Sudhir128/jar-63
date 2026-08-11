# JAR-63

A modular, provider-independent personal AI operating system — a Jarvis-style
assistant built on Clean Architecture, an event-driven runtime, and a
plugin-based agent/tool system.

> **Status: Phase 5 — Real Local LLM Runtime Integration.**
> Phase 0 established the project skeleton, configuration, runtime interfaces,
> registries, event bus, database/Redis foundations, FastAPI app, Docker
> environment, and tests. Phase 1 added the goal-directed, verifiable Universal
> Loop Engine. Phase 2 added the provider-independent LLM abstraction layer
> (Ollama + OpenAI-compatible). Phase 3 added the tool catalog and tool-calling.
> Phase 4 added the MathAgent and confirmation resume flow. Phase 5 wires the
> LLM subsystem into the running runtime: health checker, model discovery,
> capability sync, LLM status API, and the full plan → tool → verify chain.
> No voice, memory, or Android components are implemented yet — only the
> infrastructure to build them later. See [`docs/llm.md`](docs/llm.md).

---

## Project purpose

JAR-63 is designed to be a personal AI operating system that can:

- Route user intent through a central orchestrator to specialized agents.
- Use a pluggable set of tools (Python, shell, browser, git, docker, …).
- Remain independent of any single LLM provider (OpenAI, Anthropic, Google,
  Ollama, OpenRouter, and other OpenAI-compatible providers).
- Eventually expose voice as the primary interaction layer.

The foundation phase deliberately avoids building features; it builds the
extensible surface that future phases implement.

---

## Architecture overview

```
User
  │
  ▼
Interaction Layer  ──► Runtime Manager
                          │
                          ▼
                  Central Orchestrator
                          │
            ┌─────────────┼──────────────┐
            ▼             ▼              ▼
         Planner       Router         Policy
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

The **Voice / Interaction layer** will later become the primary interface.
See [`docs/architecture.md`](docs/architecture.md) for the full intended
architecture and phase roadmap.

---

## Technology stack

**Backend**

- Python 3.13+
- FastAPI
- Pydantic / pydantic-settings
- SQLAlchemy (async-capable) + PostgreSQL
- Redis
- LangGraph / LangChain (reserved for later phases)
- pytest, ruff
- Docker / Docker Compose

The LLM layer is provider-independent by design; provider-specific packages
are not installed in this phase.

---

## Repository structure

```
jar-63/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes (/health, /version)
│   │   ├── core/         # logging, exceptions, identifiers
│   │   ├── config/       # pydantic-settings categories + aggregator
│   │   ├── runtime/      # RuntimeManager, Task/Session/Dispatcher/Workflow
│   │   ├── agents/       # AgentInterface + AgentRegistry
│   │   ├── memory/       # Redis connection abstraction
│   │   ├── tools/        # ToolInterface + ToolRegistry
│   │   ├── events/       # typed events + async EventBus
│   │   ├── workflows/    # workflow manager contract (LangGraph deferred)
│   │   ├── database/     # SQLAlchemy engine/session/base + health-check
│   │   ├── models/       # (reserved for ORM models)
│   │   ├── schemas/      # Pydantic API schemas
│   │   ├── services/     # (reserved for domain services)
│   │   ├── plugins/      # (reserved for plugin loading)
│   │   ├── utils/        # (reserved for helpers)
│   │   └── main.py       # FastAPI app factory + lifecycle
│   └── tests/            # pytest suite
├── android/              # (reserved for later phases)
├── docs/                 # architecture documentation
├── docker/               # reserved for environment-specific images
├── scripts/              # dev convenience scripts
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Local setup

Requirements: Python 3.13+ and (optionally) Docker.

```bash
# 1. Clone and enter the repo
cd jar-63

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install the project + dev dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# edit .env as needed

# 5. Run the API (no DB/Redis required for /health)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

---

## Docker setup

The compose environment brings up `backend`, `postgres`, and `redis` on a
shared network. The backend waits for healthy Postgres and Redis before
starting.

```bash
# Build and start all services
docker compose up --build

# Check health
curl -s http://localhost:8000/health | python -m json.tool

# Stop
docker compose down
```

PostgreSQL is exposed on `5432` and Redis on `6379`. The backend connects
to them via the `postgres` / `redis` service names on the `jar63-net`
network.

---

## Environment variables

All configuration is environment-driven. See [`.env.example`](.env.example)
for the full list. Categories:

| Category  | Prefix / vars                                                       |
|-----------|---------------------------------------------------------------------|
| App       | `APP_ENV`, `APP_NAME`, `APP_HOST`, `APP_PORT`, `APP_DEBUG`, `APP_LOG_LEVEL` |
| Database  | `DATABASE_URL` or `POSTGRES_USER/PASSWORD/HOST/PORT/DB`, `DATABASE_ECHO`, pool sizes |
| Redis     | `REDIS_URL` or `REDIS_HOST/PORT/DB`                                 |
| LLM       | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`, `LLM_DEFAULT_PROVIDER/MODEL`, `LLM_REQUEST_TIMEOUT` |
| Security  | `SECRET_KEY`, `SECURITY_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`  |
| Logging   | `LOG_LEVEL`, `LOG_JSON`, `LOG_TO_FILE`, `LOG_FILE`, `LOG_SERIALIZE_BACKTRACE` |

`APP_ENV` accepts `development`, `testing`, or `production`. **Never commit
real secrets.** API keys are stored as `SecretStr` and masked in logs.

---

## Running tests

```bash
cd backend
APP_ENV=testing python -m pytest            # full suite
APP_ENV=testing python -m pytest -v         # verbose
APP_ENV=testing python -m pytest -k phase5  # Phase 5 tests only
```

The unit/integration tests use an in-memory SQLite database and do not
require a running Postgres or Redis. All LLM tests run fully offline using
`httpx.MockTransport`. An integration test boots the FastAPI app via
`TestClient` (exercising the startup/shutdown lifecycle) and verifies
`/health`.

Optional real-Ollama integration tests (requires a local Ollama with
`qwen2.5-coder:7b` installed — see [`docs/local-ollama.md`](docs/local-ollama.md)):

```bash
OLLAMA_INTEGRATION=1 python -m pytest -m ollama -v
```

Phase 5 demos (auto-detect Ollama, fall back to mock):

```bash
cd backend
python -m app.llm.phase5_demos
```

---

## Development workflow

```bash
# Format + lint
ruff format backend scripts
ruff check --fix backend scripts

# Run tests
pytest

# Convenience script (wraps the above)
./scripts/dev.sh {install|test|lint|format|run|up|down|health}
```

Recommended branch flow: create a feature branch, run `ruff` + `pytest`,
then open a pull request.

---

## Future roadmap

- **Phase 0 — Foundation** (done): Project skeleton, configuration, runtime
  interfaces, registries, event bus, database/Redis foundations, FastAPI app,
  Docker environment, tests.
- **Phase 1 — Universal Loop Engine & Runtime Orchestration** (done):
  Goal-directed verifiable loop (DISCOVER → PLAN → EXECUTE → VERIFY → DECIDE
  → ITERATE), planner, router/policy, dispatcher behavior, LangGraph workflow
  execution, loop service, task/loop REST API. See
  [`docs/loop-engine.md`](docs/loop-engine.md).
- **Phase 2 — LLM provider abstraction** (done): Provider/model-independent
  LLM client (Ollama, OpenAI-compatible). Local-first routing. See
  [`docs/llm.md`](docs/llm.md).
- **Phase 3 — Agents & Tools** (done): Specialized agents and the tool
  catalog (calculator, time, echo, health, filesystem, …). Native tool
  calling. See [`docs/tools.md`](docs/tools.md).
- **Phase 4 — Math Agent & Confirmation Resume** (done): First specialized
  agent (MathAgent with Maker/Checker verification), confirmation pause/resume
  flow for high-risk tools.
- **Phase 5 — Real Local LLM Runtime Integration** (done): LLM health
  checker, model discovery, capability sync, LLM status API, full
  plan → tool → verify chain against real Ollama. See
  [`docs/llm.md`](docs/llm.md#phase-5-real-local-llm-runtime-integration).
- **Phase 6 — Memory:** Persistent + vector memory, Redis caching/queues.
- **Phase 7 — Voice / Interaction:** Voice as the primary interface.
- **Phase 8 — Android client.**
- **Phase 9 — Builder & research agents.**

See [`docs/architecture.md`](docs/architecture.md) for details.

---

## License

Proprietary — internal project.
