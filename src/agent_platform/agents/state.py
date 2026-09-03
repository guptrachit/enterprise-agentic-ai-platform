from pydantic import BaseModel, ConfigDict, Field

from agent_platform.agents.contracts import AgentObservation
from agent_platform.tools.authorization import ToolAuthorizationContext
from agent_platform.tools.contracts import ToolExecutionContext
from agent_platform.tools.risk import ToolApprovalContext


class AgentExecutionState(BaseModel):
    """Typed state carried across a governed agent execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    user_message: str = Field(
        min_length=1,
        description="Original user request.",
    )

    execution_context: ToolExecutionContext

    authorization_context: ToolAuthorizationContext

    approval_context: ToolApprovalContext | None = None

    observations: tuple[AgentObservation, ...] = ()

    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of agent decisions processed.",
    )

    tool_execution_count: int = Field(
        default=0,
        ge=0,
        description="Number of governed tools executed.",
    )

    def record_tool_observation(
        self,
        observation: AgentObservation,
    ) -> "AgentExecutionState":
        return self.model_copy(
            update={
                "observations": (
                    *self.observations,
                    observation,
                ),
                "tool_execution_count": (self.tool_execution_count + 1),
            }
        )

    def increment_step(self) -> "AgentExecutionState":
        return self.model_copy(
            update={
                "step_count": self.step_count + 1,
            }
        )
