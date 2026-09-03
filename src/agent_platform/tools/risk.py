from dataclasses import dataclass
from enum import StrEnum

from agent_platform.tools.definition import ToolDefinition


class ToolRiskLevel(StrEnum):
    """Platform risk classification for governed tools."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ToolRiskRule:
    """Risk and approval requirements for a specific tool identity."""

    tool_name: str
    tool_version: str
    risk_level: ToolRiskLevel
    approval_required: bool


@dataclass(frozen=True)
class ToolRiskDecision:
    """Resolved risk decision for a governed tool execution."""

    risk_level: ToolRiskLevel
    approval_required: bool


@dataclass(frozen=True)
class ToolApprovalContext:
    """Approval information supplied to governed tool execution."""

    approved: bool
    approver_id: str | None = None


class ToolRiskError(Exception):
    """Base error raised by tool risk policy enforcement."""


class ToolRiskPolicyNotConfiguredError(ToolRiskError):
    """Raised when no risk policy exists for a governed tool."""


class ToolApprovalRequiredError(ToolRiskError):
    """Raised when a tool requires approval but valid approval is absent."""


class ToolRiskPolicy:
    """Resolves and enforces tool risk and approval requirements."""

    def __init__(
        self,
        *,
        rules: tuple[ToolRiskRule, ...] = (),
    ) -> None:
        self._rules = {(rule.tool_name, rule.tool_version): rule for rule in rules}

    def evaluate(
        self,
        *,
        definition: ToolDefinition,
    ) -> ToolRiskDecision:
        key = (
            definition.metadata.name,
            definition.metadata.version,
        )

        rule = self._rules.get(key)

        if rule is None:
            raise ToolRiskPolicyNotConfiguredError(
                f"no risk policy configured for tool "
                f"{definition.metadata.name}:{definition.metadata.version}"
            )

        return ToolRiskDecision(
            risk_level=rule.risk_level,
            approval_required=rule.approval_required,
        )

    def enforce_approval(
        self,
        *,
        definition: ToolDefinition,
        decision: ToolRiskDecision,
        approval_context: ToolApprovalContext | None,
    ) -> None:
        if not decision.approval_required:
            return

        if approval_context is None or not approval_context.approved:
            raise ToolApprovalRequiredError(
                f"approval required for tool "
                f"{definition.metadata.name}:{definition.metadata.version}"
            )

        if not approval_context.approver_id:
            raise ToolApprovalRequiredError(
                f"approved execution requires approver identity for tool "
                f"{definition.metadata.name}:{definition.metadata.version}"
            )
