"""Tool interface and registry foundation.

Tools perform concrete actions (calculator, time, filesystem, etc.). Agents
and the LLM may *request* tool execution, but actual execution always goes
through the :class:`ToolRegistry` and the :class:`ToolPolicy` — the LLM never
executes tools directly.
"""

from __future__ import annotations

import abc
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.identifiers import generate_id

__all__ = [
    "ToolCategory",
    "RiskLevel",
    "ToolEnvironment",
    "ToolInfo",
    "ToolDecisionContext",
    "ToolExecutionMetadata",
    "ToolContext",
    "ToolResult",
    "ToolInterface",
    "TOOL_CATEGORIES",
]


class ToolCategory(StrEnum):
    """Categories of tools the registry will eventually support."""

    PYTHON = "python"
    SHELL = "shell"
    BROWSER = "browser"
    GIT = "git"
    GITHUB = "github"
    DOCKER = "docker"
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    SEARCH = "search"
    EMAIL = "email"
    CALENDAR = "calendar"
    WEATHER = "weather"
    HTTP = "http"
    VECTOR_DB = "vector_db"
    CALCULATOR = "calculator"
    TIME = "time"
    HEALTH = "health"
    CUSTOM = "custom"


# Convenience list of intended tool categories (documented in docs).
TOOL_CATEGORIES: list[ToolCategory] = list(ToolCategory)


class RiskLevel(StrEnum):
    """Risk classification of a tool.

    Initial classifications only — not a complete security model. The
    :class:`~app.tools.policy.ToolPolicy` uses these to decide whether a tool
    may run automatically, requires confirmation, or is denied.

    * ``LOW``      — read-only / side-effect-free (calculator, time, echo).
    * ``MEDIUM``   — read-only with broader access (read-only filesystem).
    * ``HIGH``     — mutating side effects (write filesystem, shell, DB mutation).
    * ``CRITICAL`` — irreversible / external impact (deployment, git push).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def requires_confirmation_by_default(self) -> bool:
        """Whether this risk level requires confirmation unless policy overrides."""
        return self in (RiskLevel.HIGH, RiskLevel.CRITICAL)


class ToolEnvironment(StrEnum):
    """Execution environments a tool may target.

    Used for discovery/filtering and as a *future* dimension of capability
    checks. ``LOCAL`` and ``CONTAINER`` are the primary JAR-63 targets; the
    host is ``LOCAL``.
    """

    LOCAL = "local"
    CONTAINER = "container"
    SANDBOX = "sandbox"
    REMOTE = "remote"
    ANY = "any"


class ToolInfo(BaseModel):
    """Static metadata describing a tool.

    ``input_schema`` and ``output_schema`` are JSON Schema objects used to
    validate tool arguments and results before/after execution.

    ``required_capabilities`` lets a tool declare what it needs in order to
    run. The Phase 8.1 foundation only *declares* them; no capability policy is
    enforced yet.
    """

    model_config = ConfigDict(frozen=True)

    tool_id: str = ""
    name: str
    description: str = ""
    category: ToolCategory = ToolCategory.CUSTOM
    version: str = "0.1.0"
    risk_level: RiskLevel = RiskLevel.LOW
    requires_network: bool = False
    requires_confirmation: bool = False
    read_only: bool | None = None
    modifies_external_state: bool | None = None
    supported_environments: list[ToolEnvironment] = Field(
        default_factory=lambda: [ToolEnvironment.LOCAL]
    )
    default_timeout_seconds: float | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)


class ToolContext(BaseModel):
    """Input passed to :meth:`ToolInterface.execute`."""

    model_config = ConfigDict(extra="allow")

    invocation_id: str = Field(default_factory=lambda: generate_id("tool"))
    tool_call_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolDecisionContext(BaseModel):
    """Capability-relevant context supplied to discovery and policy evaluation.

    Only safe identity/metadata fields — never arguments, prompts, or secrets.
    Used by the Phase 8.1 *foundation*; capability policy is not yet enforced.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    agent_id: str | None = None
    agent_version: str | None = None
    agent_dynamic: bool = False
    task_id: str | None = None
    session_id: str | None = None
    operation: str = "execute"
    environment: ToolEnvironment = ToolEnvironment.ANY


class ToolExecutionMetadata(BaseModel):
    """Safe execution metadata attached to a :class:`ToolResult`.

    Contains only identity, timing, and verdict facts — never API keys,
    passwords, tokens, credentials, raw prompts, secrets, unrestricted
    arguments, or unrestricted tool output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str = Field(default_factory=lambda: generate_id("texec"))
    tool: str = ""
    tool_call_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    loop_id: str | None = None
    iteration: int | None = None
    agent_id: str | None = None
    agent_version: str | None = None
    risk: RiskLevel = RiskLevel.LOW
    policy_verdict: str = ""
    policy_reason: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    success: bool = True


class ToolResult(BaseModel):
    """Output returned by :meth:`ToolInterface.execute`.

    Errors are normalized: a tool never returns an arbitrary unstructured
    exception to the caller. ``error`` carries a safe, typed message.
    ``partial`` marks a partial result, ``evidence`` carries structured
    evidence (kept out of ``output`` when it must not be reported verbatim),
    and ``execution_metadata`` safely records how the tool ran.
    """

    model_config = ConfigDict(extra="allow")

    invocation_id: str
    tool_call_id: str | None = None
    name: str
    success: bool = True
    partial: bool = False
    output: Any = None
    error: str | None = None
    error_type: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_metadata: ToolExecutionMetadata | None = None

    @property
    def failed(self) -> bool:
        """Whether the tool did not complete successfully."""
        return not self.success


class ToolInterface(abc.ABC):
    """Common interface/base class for all tools.

    Subclasses must implement :meth:`execute` and provide :attr:`info`.
    Lifecycle hooks have default no-op implementations.
    """

    @property
    @abc.abstractmethod
    def info(self) -> ToolInfo:
        """Return the tool's static metadata."""

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def description(self) -> str:
        return self.info.description

    @property
    def category(self) -> ToolCategory:
        return self.info.category

    @property
    def risk_level(self) -> RiskLevel:
        return self.info.risk_level

    @abc.abstractmethod
    async def execute(self, context: ToolContext) -> ToolResult:
        """Perform the tool's work asynchronously."""

    async def on_register(self) -> None:  # noqa: B027
        """Called once when the tool is registered."""

    async def on_unregister(self) -> None:  # noqa: B027
        """Called once when the tool is unregistered."""
