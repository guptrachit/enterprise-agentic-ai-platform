from pydantic import BaseModel, ConfigDict

from agent_platform.agents.retrieval import (
    AgentRetrievedContext,
)


class AgentRetrievalExecutionContext(BaseModel):
    """Retrieval state attached to one agent execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    retrieved_context: AgentRetrievedContext | None = None

    @property
    def has_retrieval_context(
        self,
    ) -> bool:
        return self.retrieved_context is not None

    @property
    def citation_ids(
        self,
    ) -> tuple[str, ...]:
        if self.retrieved_context is None:
            return ()

        return self.retrieved_context.citation_registry.citation_ids
