import pytest
from pydantic import ValidationError

from agent_platform.agents import (
    AgentDecision,
    AgentFinalResponse,
)
from agent_platform.tools import ToolSelection


def test_agent_decision_accepts_tool_selection() -> None:
    decision = AgentDecision(
        tool_selection=ToolSelection(
            tool_name="search",
            tool_version="1.0.0",
            arguments={
                "query": "customer 123",
            },
        )
    )

    assert decision.tool_selection is not None
    assert decision.final_response is None


def test_agent_decision_accepts_final_response() -> None:
    decision = AgentDecision(
        final_response=AgentFinalResponse(
            content="Here is the result.",
        )
    )

    assert decision.final_response is not None
    assert decision.tool_selection is None


def test_agent_decision_rejects_both_decisions() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one",
    ):
        AgentDecision(
            tool_selection=ToolSelection(
                tool_name="search",
                tool_version="1.0.0",
                arguments={},
            ),
            final_response=AgentFinalResponse(
                content="Done",
            ),
        )


def test_agent_decision_rejects_empty_decision() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one",
    ):
        AgentDecision()
