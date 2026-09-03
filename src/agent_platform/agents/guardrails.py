from dataclasses import dataclass


class AgentGuardrailError(Exception):
    """Base error raised by agent execution guardrails."""


class AgentStepLimitExceededError(AgentGuardrailError):
    """Raised when the agent exceeds the configured step limit."""


class AgentToolLimitExceededError(AgentGuardrailError):
    """Raised when the agent exceeds the configured tool execution limit."""


@dataclass(frozen=True)
class AgentExecutionLimits:
    """Deterministic limits applied to an agent execution."""

    max_steps: int = 8
    max_tool_executions: int = 5

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        if self.max_tool_executions < 0:
            raise ValueError("max_tool_executions cannot be negative")


class AgentGuardrailPolicy:
    """Enforces deterministic execution limits for an agent."""

    def __init__(
        self,
        *,
        limits: AgentExecutionLimits | None = None,
    ) -> None:
        self._limits = limits or AgentExecutionLimits()

    def check_step_limit(
        self,
        *,
        step_count: int,
    ) -> None:
        if step_count >= self._limits.max_steps:
            raise AgentStepLimitExceededError(
                f"agent exceeded maximum step limit of {self._limits.max_steps}"
            )

    def check_tool_limit(
        self,
        *,
        tool_execution_count: int,
    ) -> None:
        if tool_execution_count >= self._limits.max_tool_executions:
            raise AgentToolLimitExceededError(
                f"agent exceeded maximum tool execution limit "
                f"of {self._limits.max_tool_executions}"
            )
