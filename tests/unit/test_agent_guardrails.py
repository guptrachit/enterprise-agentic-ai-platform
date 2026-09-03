import pytest

from agent_platform.agents import (
    AgentExecutionLimits,
    AgentGuardrailPolicy,
    AgentStepLimitExceededError,
    AgentToolLimitExceededError,
)


def test_execution_limits_reject_invalid_step_limit() -> None:
    with pytest.raises(ValueError):
        AgentExecutionLimits(
            max_steps=0,
        )


def test_execution_limits_reject_negative_tool_limit() -> None:
    with pytest.raises(ValueError):
        AgentExecutionLimits(
            max_tool_executions=-1,
        )


def test_step_limit_allows_execution_below_limit() -> None:
    policy = AgentGuardrailPolicy(
        limits=AgentExecutionLimits(
            max_steps=3,
        )
    )

    policy.check_step_limit(
        step_count=2,
    )


def test_step_limit_rejects_execution_at_limit() -> None:
    policy = AgentGuardrailPolicy(
        limits=AgentExecutionLimits(
            max_steps=3,
        )
    )

    with pytest.raises(AgentStepLimitExceededError):
        policy.check_step_limit(
            step_count=3,
        )


def test_tool_limit_allows_execution_below_limit() -> None:
    policy = AgentGuardrailPolicy(
        limits=AgentExecutionLimits(
            max_tool_executions=2,
        )
    )

    policy.check_tool_limit(
        tool_execution_count=1,
    )


def test_tool_limit_rejects_execution_at_limit() -> None:
    policy = AgentGuardrailPolicy(
        limits=AgentExecutionLimits(
            max_tool_executions=2,
        )
    )

    with pytest.raises(AgentToolLimitExceededError):
        policy.check_tool_limit(
            tool_execution_count=2,
        )
