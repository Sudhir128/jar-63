"""Phase 1 acceptance verification harness.

Runs every documented workflow through the LIVE Dockerized API and prints
structured evidence for the verification report. This is a verification tool,
not a test-suite member (it hits a running server).
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE = "http://localhost:8000"


def _post(client: httpx.Client, body: dict) -> tuple[int, dict]:
    r = client.post("/api/v1/tasks", json=body, timeout=30)
    return r.status_code, r.json()


def _get(client: httpx.Client, task_id: str, suffix: str = "") -> dict:
    r = client.get(f"/api/v1/tasks/{task_id}{suffix}", timeout=30)
    return r.json()


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def workflow_success(client: httpx.Client) -> dict:
    section("WORKFLOW 1 — SUCCESS (demo.echo)")
    code, body = _post(
        client,
        {
            "goal": "Return the word hello.",
            "agent_id": "demo.echo",
            "input": "hello",
            "success_criteria": ["expected:hello"],
            "expected_output": "hello",
            "max_iterations": 5,
            "background": False,
        },
    )
    print(f"POST -> {code}")
    print(json.dumps(body, indent=2))
    tid = body["task_id"]
    state = _get(client, tid)
    result = _get(client, tid, "/result")
    iters = _get(client, tid, "/iterations")
    print("\nSTATE:")
    print(
        json.dumps(
            {
                "status": state["status"],
                "current_stage": state["current_stage"],
                "iteration_count": state["iteration_count"],
                "final_result_status": (state["final_result"] or {}).get("final_status"),
            },
            indent=2,
        )
    )
    print("\nRESULT:")
    print(
        json.dumps(
            {
                "final_status": result["final_status"],
                "success": result["success"],
                "final_response": result["final_response"],
                "iterations_used": result["iterations_used"],
                "verification_evidence": result["verification_evidence"],
            },
            indent=2,
        )
    )
    print(f"\nITERATION COUNT: {len(iters)}")
    return {"code": code, "body": body, "state": state, "result": result, "iters": iters}


def workflow_retry(client: httpx.Client) -> dict:
    section("WORKFLOW 2 — RETRY (demo.math: 99 then 100)")
    code, body = _post(
        client,
        {
            "goal": "Return 100.",
            "agent_id": "demo.math",
            "success_criteria": ["expected:100"],
            "expected_output": 100,
            "max_iterations": 5,
            "background": False,
        },
    )
    print(f"POST -> {code}")
    print(json.dumps(body, indent=2))
    tid = body["task_id"]
    result = _get(client, tid, "/result")
    iters = _get(client, tid, "/iterations")
    print("\nRESULT:")
    print(
        json.dumps(
            {
                "final_status": result["final_status"],
                "success": result["success"],
                "final_response": result["final_response"],
                "iterations_used": result["iterations_used"],
            },
            indent=2,
        )
    )
    print(f"\nITERATION HISTORY ({len(iters)} records):")
    for i in iters:
        exec_out = (i["result"] or {}).get("output")
        verif_passed = (i["verification"] or {}).get("passed")
        print(f"  iter {i['iteration_number']}: output={exec_out!r} verify_passed={verif_passed}")
    return {"code": code, "body": body, "result": result, "iters": iters}


def workflow_max_iter(client: httpx.Client) -> dict:
    section("WORKFLOW 3 — MAX ITERATIONS (demo.failing, max=3)")
    code, body = _post(
        client,
        {
            "goal": "Return 100.",
            "agent_id": "demo.failing",
            "success_criteria": ["expected:100"],
            "expected_output": 100,
            "max_iterations": 3,
            "background": False,
        },
    )
    print(f"POST -> {code}")
    print(json.dumps(body, indent=2))
    tid = body["task_id"]
    result = _get(client, tid, "/result")
    state = _get(client, tid)
    iters = _get(client, tid, "/iterations")
    print("\nRESULT:")
    print(
        json.dumps(
            {
                "final_status": result["final_status"],
                "success": result["success"],
                "iterations_used": result["iterations_used"],
                "failure_reason": result["failure_reason"],
                "stopped_reason": result["stopped_reason"],
                "remaining_work": result["remaining_work"],
            },
            indent=2,
        )
    )
    print(f"\nITERATION COUNT: {len(iters)} (expected exactly 3)")
    print(f"STATE status: {state['status']}")
    return {"code": code, "body": body, "result": result, "state": state, "iters": iters}


def workflow_cancel(client: httpx.Client) -> dict:
    section("WORKFLOW 4 — CANCELLATION (demo.cancel, background)")
    code, body = _post(
        client,
        {
            "goal": "Return 100.",
            "agent_id": "demo.cancel",
            "success_criteria": ["expected:100"],
            "expected_output": 100,
            "max_iterations": 50,
            "background": True,
        },
    )
    print(f"POST (background) -> {code}")
    print(json.dumps(body, indent=2))
    tid = body["task_id"]
    time.sleep(0.4)  # let one iteration start
    cancel = client.post(f"/api/v1/tasks/{tid}/cancel", timeout=10).json()
    print("\nCANCEL:")
    print(json.dumps(cancel, indent=2))
    # Poll until background loop settles
    for _ in range(20):
        time.sleep(0.3)
        result = _get(client, tid, "/result")
        if result.get("final_status"):
            break
    state = _get(client, tid)
    print("\nFINAL STATE:")
    print(
        json.dumps(
            {
                "status": state["status"],
                "current_stage": state["current_stage"],
                "iteration_count": state["iteration_count"],
                "final_result_status": (state["final_result"] or {}).get("final_status"),
            },
            indent=2,
        )
    )
    print("\nFINAL RESULT:")
    print(
        json.dumps(
            {
                "final_status": result.get("final_status"),
                "stopped_reason": result.get("stopped_reason"),
                "iterations_used": result.get("iterations_used"),
            },
            indent=2,
        )
    )
    return {"code": code, "body": body, "cancel": cancel, "state": state, "result": result}


def workflow_unknown_agent(client: httpx.Client) -> dict:
    section("WORKFLOW 5 — UNKNOWN AGENT FAILURE")
    code, body = _post(
        client,
        {
            "goal": "Do something.",
            "agent_id": "nonexistent-agent",
            "expected_output": "anything",
            "max_iterations": 3,
            "background": False,
        },
    )
    print(f"POST -> {code}")
    print(json.dumps(body, indent=2))
    tid = body["task_id"]
    result = _get(client, tid, "/result")
    print("\nRESULT:")
    print(
        json.dumps(
            {
                "final_status": result["final_status"],
                "success": result["success"],
                "failure_reason": result["failure_reason"],
            },
            indent=2,
        )
    )
    return {"code": code, "body": body, "result": result}


def main() -> int:
    with httpx.Client(base_url=BASE) as client:
        # Environment
        section("ENVIRONMENT")
        health = client.get("/health").json()
        version = client.get("/version").json()
        print("HEALTH:", json.dumps(health, indent=2))
        print("VERSION:", json.dumps(version, indent=2))

        w1 = workflow_success(client)
        w2 = workflow_retry(client)
        w3 = workflow_max_iter(client)
        w4 = workflow_cancel(client)
        w5 = workflow_unknown_agent(client)
        _ = (w1, w2, w3, w4, w5)

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
