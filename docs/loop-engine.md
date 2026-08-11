# Universal Loop Engine

The Universal Loop Engine is the cognitive core of JAR-63. It implements a
goal-directed, verifiable, self-correcting loop that any task flows through.
This document describes the Phase 1 implementation.

## Design goals

- **Verifiable** — every result must pass objective verification before the loop
  accepts it as a success.
- **Self-correcting** — when verification fails, the loop iterates (re-plans and
  re-executes) up to a configurable limit.
- **Observable** — every stage transition publishes typed events so the (future)
  voice/status layer can narrate progress.
- **Provider-independent** — the loop knows nothing about LLM providers; it talks
  to agents and tools through registries.
- **Replaceable** — the controller is wrapped by a LangGraph adapter, so the
  orchestration graph can be swapped without touching stage logic.

## The loop

```
DISCOVER → PLAN → EXECUTE → VERIFY → DECIDE → ITERATE
                                   │
                                   └─ success?  → END (success)
                                      retry?     → PLAN
                                      max iter?  → END (max_iterations)
                                      cancelled? → END (cancelled)
                                      failed?    → END (failed)
```

### Stages

| Stage    | Responsibility                                              |
| -------- | ----------------------------------------------------------- |
| DISCOVER | Inspect the task/goal and available agents/tools.           |
| PLAN     | Produce the next action (which agent/tool, with what args). |
| EXECUTE  | Run the planned action through the agent registry.          |
| VERIFY   | Evaluate the output against success criteria / expected.    |
| DECIDE   | Evaluate stop conditions (cancel, success, max, failure).   |
| ITERATE  | Record the iteration, update state, prepare for re-plan.    |

### Stop conditions (evaluated in order)

1. **Cancellation** — external cancel requested.
2. **Success** — verification passed with objective evidence.
3. **Max iterations** — iteration budget exhausted.
4. **Timeout** — execution time budget exceeded.
5. **Failure** — execution failed and retries are not permitted.

## Verification

Verification is evidence-based. A `Verifier` produces a `VerificationResult`
with one or more `VerificationEvidence` entries. Built-in verifiers:

- `ExactMatchVerifier` — output equals expected value.
- `CallableVerifier` — sync or async predicate over the output.
- `CompositeVerifier` — combines multiple verifiers (any fail ⇒ fail).
- `always_pass_verifier` / `always_fail_verifier` — testing helpers.

A success requires **passed verification with at least one piece of evidence**.
Without evidence, the loop cannot conclude success — it must keep iterating.

## State

`LoopState` is a functional, immutable-evolving Pydantic model. Each stage
produces a new state via `evolve(...)`, and iteration history is append-only.
This makes state snapshots safe to serialize for the (future) persistence and
voice/status layers.

Key fields: `loop_id`, `goal`, `status`, `current_stage`, `iteration_count`,
`max_iterations`, `next_action`, `completed_steps`, `failed_steps`, `changes`,
`remaining_work`, `blockers`, `last_error`, `iteration_history`,
`last_verification`, `cancel_requested`.

`iteration_history` is append-only and records **every** completed attempt,
including the final stopping one. Records are created by `LoopController` right
after VERIFY (so the final iteration that stops the loop is still recorded),
not by the ITERATE stage.

## Events

The loop publishes typed events (see `app/events/types.py`, `LOOP_*`):

- `LOOP_STARTED` / `LOOP_COMPLETED`
- `LOOP_ITERATION_STARTED` / `LOOP_ITERATION_COMPLETED`
- `LOOP_STAGE_STARTED` / `LOOP_STAGE_COMPLETED`
- `LOOP_VERIFICATION_PASSED` / `LOOP_VERIFICATION_FAILED`
- `LOOP_STOPPED` (cancellation AND max-iterations) / `LOOP_FAILED` (failure/no-plan)

Every event carries `loop_id`, `task_id`, `session_id`, `event_id`, `timestamp`,
and iteration metadata. Controller-published events put correlation fields in
`metadata`; execute-stage agent events put them in `payload`.

## Demo agents (development)

`app/runtime/bootstrap.py` registers deterministic demo agents into the running
app's `AgentRegistry` at startup (skipped in production, collision-safe):
`demo.echo` (success), `demo.math` (retry: 99→100), `demo.failing` (max-iter),
`demo.cancel` (slow, for cancellation). `ScriptedAgent` tracks its script position
per task so a shared instance behaves correctly across tasks.

## LangGraph adapter

`LoopGraphAdapter` wraps the `LoopController` as a LangGraph `StateGraph`. Each
stage becomes a node; `DECIDE` is a conditional edge that routes to `END` on
stop or back to `PLAN` on retry. The adapter is a replaceable boundary — the
controller and stages are unaware of LangGraph.

## REST surface

| Method | Path                          | Purpose                          |
| ------ | ----------------------------- | -------------------------------- |
| POST   | `/api/v1/tasks`               | Create + run a task (inline/bg). |
| GET    | `/api/v1/tasks`               | List all loops.                  |
| GET    | `/api/v1/tasks/{task_id}`     | Loop state + iteration history.  |
| GET    | `/api/v1/tasks/{task_id}/result`        | Final loop result.      |
| GET    | `/api/v1/tasks/{task_id}/iterations`    | Iteration history.      |
| POST   | `/api/v1/tasks/{task_id}/cancel`        | Request cancellation.   |

## What is NOT implemented yet

- LLM-backed planning (current planner is deterministic).
- Persistent loop state (in-memory only).
- Real tool execution (tool registry is wired but tools are stubs).
- Reflection / policy-based routing.
- Voice/status narration (events are published but not consumed).
