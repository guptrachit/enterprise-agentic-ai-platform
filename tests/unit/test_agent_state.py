from agent_platform.agents import (
    AgentExecutionState,
    AgentObservation,
)
from agent_platform.tools import (
    ToolAuthorizationContext,
    ToolExecutionContext,
)


def create_state() -> AgentExecutionState:
    return AgentExecutionState(
        user_message="Find customer 123",
        execution_context=ToolExecutionContext(
            correlation_id="corr-123",
        ),
        authorization_context=ToolAuthorizationContext(
            subject_id="user-123",
            granted_scopes=frozenset(
                {
                    "tool:search:execute",
                }
            ),
        ),
    )


def test_execution_state_starts_empty() -> None:
    state = create_state()

    assert state.observations == ()
    assert state.step_count == 0
    assert state.tool_execution_count == 0


def test_execution_state_records_tool_observation_immutably() -> None:
    state = create_state()

    observation = AgentObservation(
        tool_name="search",
        tool_version="1.0.0",
        output={
            "result": "customer 123",
        },
    )

    updated = state.record_tool_observation(
        observation,
    )

    assert state.observations == ()
    assert state.tool_execution_count == 0

    assert updated.observations == (observation,)
    assert updated.tool_execution_count == 1


def test_execution_state_increments_step_immutably() -> None:
    state = create_state()

    updated = state.increment_step()

    assert state.step_count == 0
    assert updated.step_count == 1
