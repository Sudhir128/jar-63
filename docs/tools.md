# Tool Execution Foundation (Phase 3)

Phase 3 adds **safe, policy-governed tool execution** to the Universal Loop
Engine. The LLM planner can now request tool calls; a `ToolExecutor` resolves,
validates, policy-checks, and (when required) asks for human confirmation
before any tool runs. The LLM never executes tools directly.

## Design

```
LLM Planner
  │  produces LLMPlanStep(tool="calculator", tool_arguments={...})
  ▼
LLMPlanStage._translate
  │  validates tool exists in ToolRegistry → ActionType.EXECUTE_TOOL
  ▼
ExecuteStage
  │  builds ToolCallRecord, delegates to ToolExecutor
  ▼
ToolExecutor
  │  1. resolve tool (ToolRegistry.get)
  │  2. validate arguments (jsonschema)
  │  3. evaluate policy (ToolPolicy.evaluate)
  │     ├── ALLOW  → execute
  │     ├── DENY   → skip, emit Observation(TOOL_DENIED)
  │     └── CONFIRM → create ConfirmationRequest, pause
  │  4. execute tool (tool.execute)
  │  5. normalize result → Observation
  │  6. publish events
  ▼
Observation → LoopState → Verify → Iterate
```

## Key Components

### `app/tools/interface.py`

Extended with:

- **`RiskLevel`** — `NONE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`. Tools declare
  their risk; policy uses it to decide whether confirmation is needed.
- **`ToolContext`** — the execution context passed to `tool.execute()`: tool
  call ID, arguments, invocation ID, task/session IDs.
- **`ToolCallRequest`** — a request to call a tool (resolved by the executor).

### `app/tools/policy.py`

- **`ToolPolicy`** (interface) — `evaluate(info) → PolicyDecision`.
- **`PolicyDecision`** — `allow()`, `deny()`, `confirm()` factories. Exactly
  one of `allowed`/`denied`/`requires_confirmation` is true.
- **`DefaultToolPolicy`** — risk-based:
  - `LOW`/`NONE` → ALLOW
  - `MEDIUM` → CONFIRM
  - `HIGH` → CONFIRM
  - `CRITICAL` → DENY (unless explicitly in `allow`)
  - Explicit `deny` set always denies.
  - Explicit `allow` set bypasses confirmation (auto-approve).
  - `network_allowed=False` denies tools that `requires_network`.
- **`AllowAllToolPolicy`** — for tests/demos; allows everything.

### `app/tools/confirmation.py`

- **`ConfirmationRequest`** — pending, approved, or rejected. Carries tool
  name, arguments, risk level, reason, and who decided.
- **`ConfirmationStore`** — in-memory store: `create`, `get`, `approve`,
  `reject`, `list_pending`. Only pending requests can be decided.

### `app/tools/executor.py`

- **`ToolExecutor`** — the single entry point for tool execution. Enforces
  the full pipeline above. Tracks call history per task for repeated-call
  detection. Publishes events at every step.
- **`ToolCallRecord`** — the normalized call (tool name + arguments).
- **`ToolExecutionOutcome`** — the result: `result` (ToolResult or None),
  `observation`, `decision`, `confirmation`, `skipped`/`skipped_reason`.

### `app/tools/impl.py`

Concrete foundation tools (all `LOW` risk, safe):

| Tool | Category | Description |
|------|----------|-------------|
| `CalculatorTool` | `CALCULATOR` | Safe arithmetic via AST (no `eval`/`exec`). Supports `+ - * / // % **`, parentheses, unary minus. Rejects function calls, assignments, names, expressions > 100 chars. |
| `TimeTool` | `TIME` | Returns current UTC time (ISO, date, time). |
| `HealthTool` | `HEALTH` | Returns runtime health status + component checks. |
| `EchoTool` | `UTILITY` | Echoes arguments back (for testing). |

`DEFAULT_TOOLS()` returns a fresh list of all four.

### `app/runtime/loop/observation.py`

- **`Observation`** — what the system learned after an action. Distinct from
  execution result and verification:
  - `TOOL_RESULT` — tool succeeded
  - `TOOL_FAILURE` — tool ran but failed
  - `TOOL_DENIED` — policy denied execution
  - `TOOL_CONFIRMATION_REQUIRED` — tool needs human confirmation

Observations are immutable and feed back into `LoopState` for the next plan.

## Loop Integration

- **`LoopState`** — extended with `tool_call_count`,
  `tool_call_count_per_iteration`, `observations`, `tool_results`,
  `pending_tool_calls`, `confirmation_required`, `confirmation_request`,
  `current_plan`, `current_step`.
- **`LoopPolicy`** — extended with `max_tool_calls_per_iteration` (8),
  `max_tool_calls_per_task` (32), `max_repeated_identical_tool_calls` (3).
  Validates `per_task >= per_iteration`.
- **`ExecuteStage`** — delegates `EXECUTE_TOOL` actions to `ToolExecutor`
  (when provided); handles denial/confirmation outcomes gracefully.
- **`LLMPlanStage`** — translates `LLMPlanStep(tool=...)` into
  `ActionType.EXECUTE_TOOL`; unknown tools become no-ops (never executed).
- **`LLMPlanner`** — prompt now includes available tools; plan schema
  supports `tool` and `tool_arguments` fields.
- **`LoopService`** — wires `ToolExecutor` + policy into every controller;
  exposes `list_pending_confirmations`, `approve_confirmation`,
  `reject_confirmation`.

## Events

Phase 3 adds:

| Event Type | When |
|------------|------|
| `tool.call.requested` | A tool call is submitted to the executor |
| `tool.call.started` | Tool execution begins (after policy ALLOW) |
| `tool.call.completed` | Tool execution finished successfully |
| `tool.call.failed` | Tool execution raised an exception |
| `tool.policy.denied` | Policy denied the tool call |
| `tool.confirmation.required` | Policy requires human confirmation |
| `tool.confirmation.approved` | Confirmation was approved |
| `tool.confirmation.rejected` | Confirmation was rejected |
| `observation.created` | An Observation was created |
| `llm.fallback` | LLM planner fell back to deterministic |

## API Endpoints

Confirmation management (under `/api/v1/tasks`):

- `GET /confirmations/pending` — list pending confirmations
- `POST /confirmations/{id}/approve` — approve a confirmation
- `POST /confirmations/{id}/reject` — reject a confirmation

## Safety Guarantees

1. **No direct LLM execution** — the LLM only *plans* tool calls; the
   `ToolExecutor` resolves and runs them.
2. **Schema validation** — arguments are validated against the tool's
   `input_schema` (jsonschema) before execution.
3. **Policy enforcement** — every call passes through `ToolPolicy.evaluate`.
4. **Risk-based confirmation** — `MEDIUM`/`HIGH` tools require explicit
   human approval before execution.
5. **No `eval`/`exec`** — `CalculatorTool` uses a restricted AST evaluator.
6. **Tool limits** — `LoopPolicy` caps tool calls per iteration and per task.
7. **Repeated-call tracking** — the executor records call history per task.

## Demos

`app/runtime/loop/phase3_demos.py` contains 7 end-to-end workflows that run
the real `LoopController`:

1. **Calculator** — LLM plans `238 * 47` → tool → verify → success (11186)
2. **Time** — LLM plans time query → tool → verify → success
3. **Tool failure** — `1 / 0` → tool fails → no false success
4. **Policy denial** — denied tool → not executed → no false success
5. **Confirmation** — HIGH-risk tool → pause → approve → execute
6. **Iteration** — wrong plan → verify fails → replan → correct → success
7. **LLM planner** — full end-to-end with stub LLM

Run them:

```python
import asyncio
from app.runtime.loop.phase3_demos import run_calculator_workflow
result = asyncio.run(run_calculator_workflow())
print(result.final_status, result.final_response)
```

## Future Tools (Not Yet Implemented)

The `ToolRegistry` is designed to support: Python, Shell, Browser, Git,
GitHub, Docker, Filesystem, Database, Search, Email, Calendar, Weather, HTTP,
Vector Database. These will be added in later phases, each with appropriate
`RiskLevel` and `input_schema`.
