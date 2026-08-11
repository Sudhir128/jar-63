"""Tests for the ToolRegistry."""

from __future__ import annotations

import pytest

from app.core.exceptions import ToolAlreadyRegisteredError, ToolNotFoundError
from app.tools.interface import ToolCategory
from app.tools.registry import ToolRegistry
from tests._fakes import EchoTool


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


async def test_register_and_get(registry: ToolRegistry) -> None:
    tool = EchoTool()
    await registry.register(tool)
    assert registry.exists("echo")
    assert registry.get("echo") is tool
    assert len(registry) == 1
    assert "echo" in registry


async def test_register_duplicate_raises(registry: ToolRegistry) -> None:
    await registry.register(EchoTool())
    with pytest.raises(ToolAlreadyRegisteredError):
        await registry.register(EchoTool())


async def test_unregister(registry: ToolRegistry) -> None:
    await registry.register(EchoTool())
    removed = await registry.unregister("echo")
    assert removed.name == "echo"
    assert not registry.exists("echo")


async def test_unregister_missing_raises(registry: ToolRegistry) -> None:
    with pytest.raises(ToolNotFoundError):
        await registry.unregister("nope")


async def test_get_missing_raises(registry: ToolRegistry) -> None:
    with pytest.raises(ToolNotFoundError):
        registry.get("nope")


async def test_list_and_find_by_category(registry: ToolRegistry) -> None:
    await registry.register(EchoTool("a"))
    await registry.register(EchoTool("b"))
    assert {t.name for t in registry.list()} == {"a", "b"}
    assert {t.name for t in registry.find_by_category(ToolCategory.PYTHON)} == {"a", "b"}
    assert registry.find_by_category(ToolCategory.SHELL) == []


def test_tool_categories_documented() -> None:
    from app.tools.interface import TOOL_CATEGORIES

    expected = {
        "python",
        "shell",
        "browser",
        "git",
        "github",
        "docker",
        "filesystem",
        "database",
        "search",
        "email",
        "calendar",
        "weather",
        "http",
        "vector_db",
    }
    names = {c.value for c in TOOL_CATEGORIES}
    assert expected.issubset(names)
