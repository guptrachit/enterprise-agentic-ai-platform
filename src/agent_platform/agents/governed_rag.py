from pydantic import BaseModel, ConfigDict, Field

from agent_platform.agents.citation_validation import (
    CitationValidationResult,
)
from agent_platform.agents.governed_execution import (
    GovernedAgentExecutionResult,
)
from agent_platform.agents.retrieval import (
    AgentRetrievalRequest,
    AgentRetrievedContext,
)


class GovernedRAGAgentRequest(BaseModel):
    """End-to-end governed RAG request."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    user_input: str = Field(
        min_length=1,
    )

    retrieval_request: AgentRetrievalRequest


class GovernedRAGAgentResult(BaseModel):
    """Final governed RAG response."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    response_text: str

    retrieved_context: AgentRetrievedContext

    agent_execution: GovernedAgentExecutionResult

    citation_validation: CitationValidationResult

    trusted_citation_ids: tuple[str, ...]

    @property
    def citations_are_valid(
        self,
    ) -> bool:
        return self.citation_validation.is_valid
