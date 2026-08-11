"""Tool policy: safety/authorization layer for tool execution.

The policy is evaluated *before* any tool runs. It decides whether a tool
call is ALLOWED, DENIED, or requires confirmation. The LLM can never bypass
the policy — tool calls that are denied or pending confirmation are not
executed.

This is a basic extension point, not a sophisticated security engine. Later
phases may replace the implementation without changing the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.tools.interface import RiskLevel, ToolInfo

__all__ = [
    "PolicyDecision",
    "PolicyVerdict",
    "ToolPolicy",
    "AllowAllToolPolicy",
    "DefaultToolPolicy",
    "ToolPolicyError",
]


class ToolPolicyError(Exception):
    """Raised when a policy configuration is invalid."""


class PolicyVerdict(StrEnum):
    """The verdict a :class:`ToolPolicy` returns for a tool call."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"


@dataclass(frozen=True)
class PolicyDecision:
    """The result of evaluating a tool call against the policy.

    ``reason`` is a human-readable explanation suitable for events and the
    observation record. It never contains secrets.
    """

    verdict: PolicyVerdict
    reason: str = ""
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.verdict is PolicyVerdict.ALLOW

    @property
    def denied(self) -> bool:
        return self.verdict is PolicyVerdict.DENY

    @property
    def requires_confirmation(self) -> bool:
        return self.verdict is PolicyVerdict.REQUIRE_CONFIRMATION

    @classmethod
    def allow(cls, reason: str = "allowed by policy") -> PolicyDecision:
        return cls(verdict=PolicyVerdict.ALLOW, reason=reason)

    @classmethod
    def deny(cls, reason: str, *, detail: str | None = None) -> PolicyDecision:
        return cls(verdict=PolicyVerdict.DENY, reason=reason, detail=detail)

    @classmethod
    def confirm(cls, reason: str, *, detail: str | None = None) -> PolicyDecision:
        return cls(verdict=PolicyVerdict.REQUIRE_CONFIRMATION, reason=reason, detail=detail)


class ToolPolicy:
    """Abstract tool policy contract.

    A policy evaluates a tool call (identified by its :class:`ToolInfo` and
    arguments) in the context of the current task/session and returns a
    :class:`PolicyDecision`.
    """

    def evaluate(
        self,
        info: ToolInfo,
        *,
        arguments: dict[str, Any] | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        privacy: str = "public",
    ) -> PolicyDecision:
        raise NotImplementedError


class AllowAllToolPolicy(ToolPolicy):
    """Permissive policy that allows everything (backward compatibility).

    Used when no explicit policy is configured so existing Phase 0/1 behavior
    (direct tool execution without policy checks) is preserved.
    """

    def evaluate(self, info: ToolInfo, **_: Any) -> PolicyDecision:
        return PolicyDecision.allow(reason="allow-all policy")


class DefaultToolPolicy(ToolPolicy):
    """Basic risk-based policy.

    Rules (evaluated in order):

    1. Tools in the explicit deny list are DENIED.
    2. CRITICAL tools are DENIED unless explicitly allowed.
    3. HIGH tools require confirmation (unless auto-approved).
    4. MEDIUM tools require confirmation if ``require_confirmation_for_medium``
       is set.
    5. LOW tools are allowed.
    6. Tools requiring network are DENIED if ``network_allowed`` is False.

    This is intentionally simple. The interface allows a sophisticated policy
    engine to replace it later.
    """

    def __init__(
        self,
        *,
        deny: set[str] | None = None,
        allow: set[str] | None = None,
        auto_approve: set[str] | None = None,
        network_allowed: bool = True,
        require_confirmation_for_medium: bool = False,
    ) -> None:
        self._deny = deny or set()
        self._allow = allow or set()
        self._auto_approve = auto_approve or set()
        self._network_allowed = network_allowed
        self._confirm_medium = require_confirmation_for_medium

    def evaluate(
        self,
        info: ToolInfo,
        *,
        arguments: dict[str, Any] | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        privacy: str = "public",
    ) -> PolicyDecision:
        name = info.name

        if name in self._deny:
            return PolicyDecision.deny(
                f"Tool '{name}' is denied by policy.", detail="explicit deny list"
            )

        if info.requires_network and not self._network_allowed:
            return PolicyDecision.deny(
                f"Tool '{name}' requires network access, which is not allowed.",
                detail="network disabled",
            )

        if name in self._auto_approve:
            return PolicyDecision.allow(reason=f"Tool '{name}' auto-approved by policy.")

        if info.risk_level is RiskLevel.CRITICAL and name not in self._allow:
            return PolicyDecision.deny(
                f"Tool '{name}' is CRITICAL risk and not explicitly allowed.",
                detail="critical risk denied",
            )

        if info.risk_level is RiskLevel.HIGH:
            return PolicyDecision.confirm(
                f"Tool '{name}' is HIGH risk; confirmation required.",
                detail="high risk requires confirmation",
            )

        if info.risk_level is RiskLevel.MEDIUM and self._confirm_medium:
            return PolicyDecision.confirm(
                f"Tool '{name}' is MEDIUM risk; confirmation required.",
                detail="medium risk requires confirmation",
            )

        return PolicyDecision.allow(reason=f"Tool '{name}' allowed (risk={info.risk_level.value}).")
