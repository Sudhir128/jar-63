# LLM Abstraction Layer (Phase 2)

JAR-63 is **local-first**: Ollama is the primary LLM provider and no cloud API
key is required to run the application. This document describes the Phase 2 LLM
abstraction layer that introduces provider/model independence without coupling
to a specific vendor.

## Design goals

- **Provider independence.** Agents and the planner depend on `LLMClient`, never
  on a provider SDK. Provider-specific raw responses are normalized at the
  provider boundary into typed `LLMResponse` objects.
- **Local-first.** The default routing policy selects a local model when one is
  available, regardless of network status. "Internet available → cloud" is
  explicitly NOT the routing rule.
- **Privacy-aware.** `PRIVATE` and `SENSITIVE` requests are not sent to cloud
  providers unless `LLM_ALLOW_CLOUD_FOR_PRIVATE=true`.
- **Deterministic fallback.** If the LLM is unavailable, misconfigured, or
  returns invalid output, the planner falls back to the deterministic Phase 1
  planner. The loop never blocks on LLM availability.
- **No secret leakage.** API keys are `SecretStr`, sent only as `Bearer`
  headers, and never logged. LLM events carry provider/model/latency metadata
  only — never prompt contents or completions.
- **Extensible.** New providers, models, and capabilities can be added without
  restructuring the layer.

## Architecture

```
Agent / Planner
    ↓
LLMClient (abstract interface)        app/llm/client.py
    ↓
ModelRouter → ProviderFactory         app/llm/router.py, factory.py
    ↓
ProviderRegistry / ModelRegistry      app/llm/registry.py
    ↓
OllamaClient | OpenAICompatibleClient app/llm/providers/
```

### Layering rule

The application depends on the interface (`LLMClient`), never on concrete
providers. Provider SDKs (if any) are confined to `app/llm/providers/` and
never imported by agents, the planner, or the loop engine.

## Key components

| Component | File | Responsibility |
|-----------|------|----------------|
| `LLMClient` | `app/llm/client.py` | Abstract async client contract |
| `LLMRequest` / `LLMResponse` | `app/llm/models.py` | Provider-independent typed models |
| `ModelRegistry` | `app/llm/registry.py` | Model definitions keyed by (provider, model_id) |
| `ProviderRegistry` | `app/llm/registry.py` | LLM client instances keyed by provider id |
| `ProviderFactory` | `app/llm/factory.py` | Builds clients from settings |
| `ModelRouter` | `app/llm/router.py` | Deterministic local-first model selection |
| `OllamaClient` | `app/llm/providers/ollama.py` | Local Ollama HTTP client (primary) |
| `OpenAICompatibleClient` | `app/llm/providers/openai_compatible.py` | Generic cloud client (optional) |
| `LLMPlanner` | `app/llm/planner.py` | LLM-backed planner with deterministic fallback |
| `LLMPlan` / `LLMPlanStep` | `app/llm/plan_schema.py` | Strict structured plan schema |
| `LLMVerifier` | `app/llm/verifier.py` | LLM judge extension point (not the default) |
| `bootstrap` | `app/llm/bootstrap.py` | Registers default models/providers at startup |

## Routing policy

The default policy is `LOCAL_FIRST`:

1. Find local models matching the required capabilities and context window.
2. If a local model is available, select it (regardless of network status).
3. If no local model is available and the privacy level allows cloud, consider
   cloud models whose provider is reachable (network is a *signal*, not a rule).
4. If nothing satisfies the request, raise `ModelUnavailableError` with a
   structured reason.

`LOCAL_ONLY` restricts selection to local models only.

## Configuration

See `.env.example` for all LLM environment variables. Key settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_ENABLED` | `true` | Master switch |
| `LLM_DEFAULT_PROVIDER` | `ollama` | Default provider id |
| `LLM_DEFAULT_MODEL` | `qwen2.5-coder:7b` | Default model id |
| `LLM_ROUTING_POLICY` | `local_first` | Routing policy |
| `LLM_REQUEST_TIMEOUT` | `60` | Per-request timeout (seconds) |
| `LLM_VERBOSE_LOGGING` | `false` | Dev-only full prompt logging (never in prod) |
| `LLM_ALLOW_CLOUD_FOR_PRIVATE` | `false` | Allow PRIVATE/SENSITIVE on cloud |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP base URL |
| `OLLAMA_DEFAULT_MODEL` | `qwen2.5-coder:7b` | Default Ollama model |
| `OPENAI_COMPATIBLE_ENABLED` | `false` | Enable generic cloud provider |
| `OPENAI_COMPATIBLE_BASE_URL` | (empty) | OpenAI-compatible base URL |
| `OPENAI_COMPATIBLE_API_KEY` | (none) | Optional API key (SecretStr) |
| `OPENAI_COMPATIBLE_MODEL` | (empty) | Default cloud model id |

## Installing a local model

JAR-63 never downloads or modifies models. To use a local model, install it
manually:

```bash
ollama pull qwen2.5-coder:7b
```

If the configured model is not installed, the Ollama provider reports
`MODEL_NOT_FOUND` at request time with install guidance.

## Events

LLM events are published to the EventBus (never containing prompt contents):

| Event | When |
|-------|------|
| `LLM_REQUEST_STARTED` | An LLM request begins |
| `LLM_REQUEST_COMPLETED` | An LLM request succeeds |
| `LLM_REQUEST_FAILED` | An LLM request fails |
| `MODEL_SELECTED` | The router selects a model |
| `MODEL_FALLBACK` | The planner falls back to deterministic |
| `MODEL_UNAVAILABLE` | No model can satisfy the request |

## Loop integration

The LLM planner integrates with the loop engine via `LLMPlanStage`, which
delegates to `LLMPlanner`. The planner:

1. Selects a model via the router (local-first).
2. Builds a prompt and requests a structured `LLMPlan`.
3. Validates the plan against the strict schema.
4. Translates the plan into the loop's `PlanResult` / `NextAction`.
5. On any failure, falls back to `DefaultPlanStage` and records the reason.

The LLM **only** produces a plan. It never executes shell commands, Python,
browser actions, filesystem changes, or database writes. Execution remains the
Loop Engine's responsibility.

## Testing

All unit tests run fully offline using `httpx.MockTransport` to emulate the
Ollama and OpenAI-compatible HTTP APIs. No real server is required.

```bash
cd backend && APP_ENV=testing python -m pytest tests/test_llm_*.py -q
```

Optional integration tests against a real Ollama instance:

```bash
OLLAMA_INTEGRATION=1 pytest -m ollama
```

## Phase 5: Real local LLM runtime integration

Phase 5 wires the Phase 2 LLM abstraction layer into the **running runtime**.
Previously the LLM subsystem existed as a library; Phase 5 makes it a live part
of the application's startup, health surface, and task loop.

### What Phase 5 adds

| Component | File | Responsibility |
|-----------|------|----------------|
| `LLMHealthChecker` | `app/llm/health.py` | Probes Ollama, discovers installed models and their real capabilities, syncs them into the `ModelRegistry`, and publishes an `LLMHealthSnapshot` |
| `LLMHealthSnapshot` | `app/llm/health.py` | Immutable snapshot: availability, model, installed models, capabilities, latency, provider health per provider |
| `HttpxNetworkChecker` | `app/llm/network.py` | Real network connectivity probe for cloud-fallback routing decisions |
| `RuntimeManager` LLM wiring | `app/runtime/manager.py` | Builds model/provider registries, router, planner, and health checker at startup; passes `LLMPlanner` to `LoopService`; runs a non-blocking startup health check |
| LLM status API | `app/api/llm.py` | `GET /api/v1/llm/status` and `POST /api/v1/llm/health-check` endpoints |
| `/health` LLM component | `app/api/routes.py` | LLM availability surfaced in the aggregate health response (degraded, not unhealthy) |
| Phase 5 demos | `app/llm/phase5_demos.py` | Runnable demos: health check → discovery → routing → LLM plan → tool execution → verification |

### Startup flow

```
RuntimeManager.startup()
    ├── builds ModelRegistry + ProviderRegistry
    ├── registers OllamaClient (and optional cloud client)
    ├── builds ModelRouter + LLMHealthChecker + LLMPlanner
    ├── runs LLMHealthChecker.check() (non-blocking, never raises)
    │     ├── GET /api/tags   → installed models
    │     ├── POST /api/show  → per-model capabilities (tool calling, context length)
    │     └── syncs real capabilities into ModelRegistry
    └── constructs LoopService with llm_planner wired in
```

If Ollama is unreachable at startup, the system starts in **degraded** mode: the
loop falls back to the deterministic planner, and `/health` reports the LLM
component as degraded (the rest of the system remains healthy).

### Runtime flow (per task)

```
LoopService.run_task_loop()
    └── LLMPlanStage → LLMPlanner.plan()
          ├── ModelRouter.select() (local-first)
          ├── OllamaClient.generate_structured() → LLMPlan
          ├── validate against strict schema
          └── on any failure → DefaultPlanStage (deterministic fallback)
```

### Health surface

`GET /api/v1/llm/status` returns:

```json
{
  "enabled": true,
  "available": true,
  "status": "available",
  "model": "qwen2.5-coder:7b",
  "provider": "ollama",
  "installed_models": ["qwen2.5-coder:7b"],
  "capabilities": ["chat", "coding", "tool_calling"],
  "providers": [{"provider": "ollama", "status": "available"}]
}
```

### Discovery and capability sync

`LLMHealthChecker` queries Ollama's `/api/show` for each installed model and
syncs the **real** capabilities (tool calling, context length, families) into
the `ModelRegistry`. This means the router's selection is based on what the
model actually supports, not a static declaration. Models pre-registered without
tool-calling are upgraded in place when discovery confirms support.

### Events

Phase 5 reuses the existing LLM event types (`LLM_REQUEST_STARTED`,
`LLM_REQUEST_COMPLETED`, `LLM_REQUEST_FAILED`, `MODEL_SELECTED`,
`MODEL_FALLBACK`, `MODEL_UNAVAILABLE`). The health checker publishes
`MODEL_UNAVAILABLE` when a configured model is not found on the provider.

### Demos

```bash
cd backend
python -m app.llm.phase5_demos            # auto-detect Ollama (falls back to mock)
python -m app.llm.phase5_demos --mock     # force mocked transport
python -m app.llm.phase5_demos --real     # force real Ollama (fail if unreachable)
```

### Phase 5 tests

```bash
cd backend
# Offline (mocked) — always run
APP_ENV=testing python -m pytest tests/test_phase5_*.py -q

# Real Ollama — opt-in
OLLAMA_INTEGRATION=1 pytest -m ollama
```

## Security

- API keys are stored as `SecretStr` and never appear in logs or events.
- The `Authorization` header is never logged.
- LLM events carry only provider/model/latency/success metadata.
- `LLM_VERBOSE_LOGGING` (default `false`) is the only way prompt/completion
  content reaches logs, and it must never be enabled in production.
- The provider never downloads, deletes, or modifies user models.
