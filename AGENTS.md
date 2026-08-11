# JAR-63 — Agent Memory

Repository-specific knowledge for OpenHands sessions working on JAR-63.

## Project

JAR-63 is a modular, provider-independent personal AI operating system
(Jarvis-style). Phase 0 (foundation) is complete. Later phases implement
agents, tools, voice, memory, Android — do not implement those prematurely.

## Stack

- Python 3.13+, FastAPI, Pydantic, pydantic-settings, SQLAlchemy (sync+async),
  PostgreSQL, Redis, Loguru, anyio, **LangGraph + langchain-core** (installed,
  core runtime dep).
- Dev: pytest, pytest-asyncio, ruff, httpx, httpx2, aiosqlite.

## Layout

- `backend/app/` — application package (import as `app`).
- `backend/tests/` — pytest suite; `conftest.py` forces APP_ENV=testing + sqlite.
- Config: `backend/app/config/` — each category is an independent `BaseSettings`;
  `Settings` (plain BaseModel) aggregates them; `get_settings()` is lru_cached.
- Runtime: `backend/app/runtime/manager.py` (`RuntimeManager`) owns lifecycle.
- DB: `backend/app/database/session.py` — sync `engine`/`SessionLocal` + async
  `async_engine`; SQLite-aware (skips pool args, maps async sqlite→aiosqlite).
- Redis: `backend/app/memory/redis.py` — use `get_redis()`/`check_redis_connection()`
  (NOT `RedisClient()` which is just a `Redis` type alias defaulting to localhost).

## Commands

- Install: `pip install -e ".[dev]"`
- Tests: `cd backend && APP_ENV=testing pytest` (94 tests; uses in-memory sqlite)
- Lint/format: `ruff format backend scripts && ruff check --fix backend scripts`
- Run app: `uvicorn app.main:app --reload` (from `backend/`)
- Docker: `sudo docker compose up --build -d` (backend/postgres/redis on jar63-net)
- Health: `curl -s http://localhost:8000/health`
- Loop API: `POST /api/v1/tasks` (create+run), `GET /api/v1/tasks/{id}` (state),
  `GET /api/v1/tasks/{id}/result`, `GET /api/v1/tasks/{id}/iterations`,
  `POST /api/v1/tasks/{id}/cancel`, `GET /api/v1/tasks` (list).

## Conventions

- Enums use `enum.StrEnum` (not `str, Enum`) — ruff UP042.
- Optional no-op lifecycle hooks use `# noqa: B027`.
- Tests set env via `os.environ.setdefault` BEFORE importing `app` modules
  (conftest marks those imports `# noqa: E402`).
- pyproject `filterwarnings = ["error", "ignore::DeprecationWarning"]` is strict;
  starlette TestClient needs `httpx2` installed (added to dev deps).
- Secrets as `SecretStr`, masked in logs; `.env` is gitignored, `.env.example` kept.
- Docker backend WORKDIR is `/app/backend`; compose mounts `./backend` for dev.

## Loop engine (Phase 1)

- Location: `backend/app/runtime/loop/`.
- Flow: DISCOVER → PLAN → EXECUTE → VERIFY → DECIDE (conditional: success→END,
  retry→PLAN, max_iter→END, cancelled→END, failed→END) → ITERATE.
- `LoopController` orchestrates stages, stop conditions, events, policy.
- `LoopState` evolves functionally (immutable `evolve()`); history append-only.
- Verification is evidence-based: success REQUIRES passed verification +
  ≥1 evidence. `ExactMatchVerifier`, `CallableVerifier`, `CompositeVerifier`.
- `LoopGraphAdapter` wraps controller as LangGraph `StateGraph` (replaceable).
- `LoopService` bridges TaskManager↔LoopController; owned by `RuntimeManager`.
- `TaskManager.register(task)` / `find(task_id)` added for pre-built tasks.
- Deterministic demos in `loop/demos.py` (A=success, B=retry, C=max-iter, D=cancel).
- **Bootstrap**: `app/runtime/bootstrap.py` `register_demo_agents()` registers the
  demo agents into the running app's `AgentRegistry` at lifespan startup (skipped in
  production, collision-safe). This is the supported mechanism for wiring demo agents
  into the REST API runtime. `build_demo_agents()` returns: `demo.echo`,
  `demo.math`, `demo.failing`, `demo.cancel`. `ScriptedAgent` tracks script position
  **per task_id** so a shared instance behaves correctly for every task.
- **Iteration history**: `IterationRecord` is created by the controller after every
  completed VERIFY (via `LoopController._record_iteration`), NOT by the ITERATE stage.
  This records every attempt including the final stopping one. ITERATE only computes
  the next action. The record is skipped if one already exists for the current
  iteration_number (idempotent).
- Gotcha: `LOOP_VERIFICATION_PASSED/FAILED` published right after VERIFY (before
  DECIDE), so they always fire. `LOOP_ITERATION_COMPLETED` only fires when the
  loop actually iterates (after ITERATE) — NOT on a first-iteration success.
- Gotcha: terminal events — `LOOP_COMPLETED` (success), `LOOP_STOPPED` (cancel AND
  max-iterations), `LOOP_FAILED` (failure/no-plan). There is no separate
  `LOOP_CANCELLED`/`LOOP_MAX_ITERATIONS` event type.
- Gotcha: controller-published events put correlation fields (loop_id, iteration) in
  `metadata`; execute-stage agent events (`AGENT_STARTED/COMPLETED/FAILED`) put them
  in `payload`. Both carry task_id/session_id at top level. All required fields are
  present; a future cleanup could normalize this.
- Gotcha: `anyio.fail_after` is a SYNC context manager — use `with`, not
  `async with`, or it raises TypeError at runtime.
- Gotcha: `TaskManager.update_status` does `model_copy` (returns a NEW task).
  To let callers observe updates, mutate the tracked task in place in
  `LoopService` rather than relying on `update_status`.

## Docker notes

- Start daemon in this env: `sudo dockerd > /tmp/docker.log 2>&1 &` then `sudo docker ...`
- compose `env_file: .env` requires a `.env` (copy from `.env.example`).
- Running pytest inside the dev-configured container will FAIL (APP_ENV=development,
  real DB) — tests are meant for APP_ENV=testing locally/CI. Don't "fix" by changing
  the container env; that's by design.

## Phase status

- Phase 0: DONE.
- Phase 1: DONE — Universal Loop Engine + Runtime Orchestration. 94 tests pass,
  ruff clean, Docker stack healthy (backend/postgres/redis), /health ok, loop API
  verified end-to-end in-container. No git commits made yet (uncommitted tree).
- Acceptance verification DONE — all 5 workflows (success, retry, max-iter, cancel,
  unknown-agent) verified through the Dockerized REST API; iteration history captures
  every attempt; event sequences verified; DB+Redis connectivity confirmed.
- Phase 2: DONE — Local-First LLM & Model Abstraction. 207 tests pass (94 Phase 0/1
  + 113 Phase 2), 4 optional Ollama integration tests skipped by default, ruff clean.
  - LLM abstraction: `LLMClient` interface, typed `LLMRequest`/`LLMResponse` models,
    `ModelRegistry`/`ProviderRegistry`, `ProviderFactory`, deterministic `ModelRouter`.
  - Local-first: Ollama is the primary provider; no API key required. OpenAI-compatible
    provider is optional (disabled by default).
  - Routing: `LOCAL_FIRST` policy (default) — local model selected when available,
    regardless of network. Privacy gating: PRIVATE/SENSITIVE disallow cloud unless
    `LLM_ALLOW_CLOUD_FOR_PRIVATE=true`. Network is a signal, not a rule.
  - Planner: `LLMPlanner` produces a validated structured `LLMPlan`, translates to
    loop `PlanResult`/`NextAction`, falls back to `DefaultPlanStage` on any failure.
    LLM only plans; it never executes.
  - Verifier extension point: `LLMVerifier` (not the default; objective verifiers
    remain first choice). Still produces `VerificationEvidence`.
  - Security: API keys are `SecretStr`, never logged. LLM events carry only
    provider/model/latency metadata — never prompt contents. `LLM_VERBOSE_LOGGING`
    (default false) is the only prompt-logging path.
  - Demos: 6 deterministic routing/planning demos in `app/llm/demos.py` (all pass).
  - Tests fully offline via `httpx.MockTransport`; optional real-Ollama tests behind
    `OLLAMA_INTEGRATION=1` + `-m ollama` marker.
  - Config: `LLMSettings` extended (`populate_by_name=True`); `.env.example` updated.
  - Docs: `docs/llm.md` created.
- Recommended next: Phase 3 — Wire the LLM planner into the default loop (opt-in via
  config), add tool-calling support, and begin the specialized agents (math, research).
- Phase 3: DONE — LLM-Powered Loop + Tool Execution Foundation. 288 tests pass
  (207 Phase 0–2 + 81 Phase 3), 4 optional Ollama integration tests skipped, ruff clean.
  - Tool interface extended: `RiskLevel` (NONE/LOW/MEDIUM/HIGH/CRITICAL), `ToolContext`,
    `ToolCallRequest`. Tools declare risk + input_schema (jsonschema).
  - `ToolPolicy` (interface) + `DefaultToolPolicy` (risk-based: LOW→allow,
    MEDIUM/HIGH→confirm, CRITICAL→deny) + `AllowAllToolPolicy` (tests).
    `PolicyDecision` has `allow()/deny()/confirm()` factories.
  - `ConfirmationStore` — in-memory pending/approved/rejected store.
  - `ToolExecutor` — single entry point: resolve→validate(jsonschema)→policy→
    (confirm?)→execute→normalize→Observation. Publishes events at every step.
    Tracks call history for repeated-call detection.
  - `Observation` model — what was learned (TOOL_RESULT/TOOL_FAILURE/TOOL_DENIED/
    TOOL_CONFIRMATION_REQUIRED). Feeds back into LoopState for next plan.
  - Concrete tools: `CalculatorTool` (safe AST eval, NO eval/exec), `TimeTool`,
    `HealthTool`, `EchoTool` — all LOW risk. `DEFAULT_TOOLS()` factory.
  - Loop integration: `LoopState` extended (tool_call_count, observations,
    confirmation fields); `LoopPolicy` extended (max_tool_calls_per_iteration=8,
    per_task=32, max_repeated=3); `ExecuteStage` delegates to ToolExecutor;
    `LLMPlanStage` translates tool plans; `LLMPlanner` prompt includes tools.
  - `LoopService` wires ToolExecutor + policy into every controller; exposes
    confirmation management API (`/confirmations/pending`, `/approve`, `/reject`).
  - Events: 10 new types (tool.call.*, tool.policy.denied, tool.confirmation.*,
    observation.created, llm.fallback).
  - Demos: 7 end-to-end workflows in `app/runtime/loop/phase3_demos.py` (all pass):
    calculator, time, tool failure, policy denial, confirmation pause/resume,
    iteration-after-replan, full LLM planner.
  - Safety: LLM never executes tools directly; schema validation; risk-based
    confirmation; no eval/exec; tool call limits.
  - Docs: `docs/tools.md` created.
- Recommended next: Phase 4 — specialized agents (math, research) that use the
  tool execution foundation, and wire a real LLM provider for planning.


## Phase 4 — Real Local LLM Tool Calling + Confirmation Resume + Math Agent

**Status:** Implementation complete; 99 Phase 4 tests written and passing.

### What was implemented

- **Tool definition conversion** (`app/llm/tool_conversion.py`):
  `tool_to_definition` and `tool_info_to_definition` convert ToolRegistry
  ToolInfo into LLMToolDefinition. Exported from `app.llm.__init__`.
  Also: `build_assistant_tool_call_message` and `build_tool_result_message`.

- **Ollama native tool-calling** (`app/llm/providers/ollama.py`):
  Tool defs mapped to Ollama envelope; tool_calls parsed (dict or JSON string;
  malformed skipped). Finish reason TOOL_CALLS when present.

- **OpenAI-compatible tool-calling** (`app/llm/providers/openai_compatible.py`):
  Same coverage for OpenAI format (arguments as JSON string).

- **AgentStatus.WAITING_FOR_CONFIRMATION** (`app/agents/interface.py`).

- **MathAgent** (`app/agents/math/`): First specialized agent. Uses
  CalculatorTool via ToolExecutor. Extracts arithmetic from natural language.
  MathVerifier independently re-computes (Maker/Checker).

- **Confirmation resume flow**: LoopController pause/resume; LoopService
  approve/reject/resume; API endpoints for confirmations.

### Bugs found and fixed during testing

- `service.py`: EventType used but not imported -> NameError on approve/reject.
- `_extract_expression`: "17 % 5" matched percentage regex. Fixed with
  negative lookahead.

### Test results

- Phase 4: 99 passed (7 test files)
- Full suite: 387 passed, 4 skipped (real Ollama integration)


## Phase 5 — Real Local LLM Runtime Integration

**Status:** Implementation complete; 54 Phase 5 tests written and passing.

### What was implemented

- **LLMHealthChecker** (`app/llm/health.py`): Probes Ollama at startup,
  discovers installed models via `/api/tags`, queries per-model capabilities
  via `/api/show`, and syncs real capabilities (tool calling, context length)
  into the `ModelRegistry`. Produces an immutable `LLMHealthSnapshot`. Never
  raises — wraps all probes in try/except. Accepts `LLMSettings` (not
  top-level `Settings`). `_discover_capabilities` falls back to the model
  registry when `get_model_info` returns empty/None.

- **HttpxNetworkChecker** (`app/llm/network.py`): Real network connectivity
  probe for cloud-fallback routing decisions. Uses httpx to probe configured
  endpoints.

- **RuntimeManager LLM wiring** (`app/runtime/manager.py`): Builds
  model/provider registries, router, planner, and health checker at startup.
  Passes `self.settings.llm` to `LLMHealthChecker`. `LLMPlanner` imported
  lazily (TYPE_CHECKING + string annotation) to avoid a circular import.
  Passes `LLMPlanner` to `LoopService`. Runs a non-blocking startup health
  check.

- **LLM status API** (`app/api/llm.py`): `GET /api/v1/llm/status` and
  `POST /api/v1/llm/health-check`. Registered in `main.py` alongside
  health_router and tasks_router.

- **Health surface**: LLM availability surfaced in `/health` (degraded, not
  unhealthy, when Ollama is unreachable).

- **Phase 5 demos** (`app/llm/phase5_demos.py`): Runnable demos showing
  health check → discovery → routing → LLM plan → tool execution →
  verification. Auto-detects real Ollama; falls back to a deterministic
  `httpx.MockTransport` when unreachable. 3/3 demos pass.

### Bugs found and fixed during testing

- `LoopFinalStatus.SUCCESS` value is `"success"`, not `"completed"` — e2e
  test assertion corrected.
- Calculator tool returns `{"result": 42, "expression": "7 * 6"}`, not just
  `{"result": 42}` — e2e test assertion corrected to check
  `final_response["result"]`.
- Events test: `ev.model_dump()` returns enum/datetime objects that
  `json.dumps` can't serialize by default — added `default=str`.
- E2e events test: the `OllamaClient` needs the shared `event_bus` passed
  via `make_client(event_bus=bus)` for LLM_REQUEST_STARTED/COMPLETED events
  to reach subscribers.
- `LLMHealthChecker._sync_model`: replaced try/except/pass with
  `contextlib.suppress(Exception)` for collision-safe registration.

### Test results

- Phase 5 offline (mocked): 54 passed (6 test files)
  - test_phase5_llm_health.py — 15
  - test_phase5_network_checker.py — 9
  - test_phase5_runtime_llm.py — 11
  - test_phase5_llm_api.py — 6
  - test_phase5_e2e_ollama.py — 8
  - test_phase5_demos.py — 5
- Real Ollama integration: 8 tests (4 original + 4 new) behind `-m ollama`
  marker, properly skipped when OLLAMA_INTEGRATION!=1.
- Full suite: 441 passed, 8 skipped.

### Key design decisions

- **Local-first routing is a policy, not a network check.** The router
  selects a local model when available regardless of network status.
- **LLM unavailable = degraded, not unhealthy.** The loop falls back to the
  deterministic planner; `/health` reports the LLM component as degraded.
- **The LLM only plans; it never executes.** Tool execution remains the
  Loop Engine's responsibility.
- **No secret leakage.** LLM events carry provider/model/latency metadata
  only — never prompt contents or completions (verified by e2e test).
