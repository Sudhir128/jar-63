"""Tests for the Phase 5 demo module (mocked transport)."""

from __future__ import annotations

from app.llm.phase5_demos import (
    run_calculator_e2e_demo,
    run_health_check_demo,
    run_phase5_demo,
    run_routing_demo,
)


async def test_health_check_demo_succeeds_mocked() -> None:
    result = await run_health_check_demo(use_mock=True)
    assert result.success
    assert result.ollama_real is False
    assert result.model == "qwen2.5-coder:7b"
    assert result.health_status == "available"


async def test_routing_demo_succeeds_mocked() -> None:
    result = await run_routing_demo(use_mock=True)
    assert result.success
    assert "ollama" in result.final_status
    assert "local=True" in result.final_status


async def test_calculator_e2e_demo_succeeds_mocked() -> None:
    result = await run_calculator_e2e_demo(use_mock=True)
    assert result.success
    assert result.final_status == "success"
    assert result.final_response["result"] == 60


async def test_run_phase5_demo_all_succeed_mocked() -> None:
    results = await run_phase5_demo(use_mock=True)
    assert len(results) == 3
    assert all(r.success for r in results), [r.demo_name for r in results if not r.success]


async def test_health_check_demo_real_unreachable_reports_error() -> None:
    """When --real is forced and Ollama is unreachable, the demo reports the error."""
    result = await run_health_check_demo(use_mock=False)
    assert not result.success
    assert result.ollama_real is False
    assert result.error is not None
