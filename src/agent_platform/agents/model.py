from typing import Protocol

from agent_platform.agents.contracts import AgentDecision
from agent_platform.agents.state import AgentExecutionState


class AgentModel(Protocol):
    """Provider-neutral model interface used by the agent runtime."""

    async def decide(
        self,
        *,
        state: AgentExecutionState,
    ) -> AgentDecision:
        """Produce the next governed agent decision."""
