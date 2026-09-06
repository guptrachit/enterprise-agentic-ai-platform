from agent_platform.agents.citation_validation import (
    CitationValidationService,
)
from agent_platform.agents.governed_execution import (
    GovernedAgentExecutionPort,
    GovernedAgentExecutionRequest,
)
from agent_platform.agents.governed_rag import (
    GovernedRAGAgentRequest,
    GovernedRAGAgentResult,
)
from agent_platform.agents.retrieval_service import (
    AgentRetrievalService,
)


class GovernedRAGAgentService:
    """Coordinates governed retrieval and governed agent execution."""

    def __init__(
        self,
        *,
        retrieval_service: AgentRetrievalService,
        execution_port: GovernedAgentExecutionPort,
        citation_validation_service: (CitationValidationService),
    ) -> None:
        self._retrieval_service = retrieval_service
        self._execution_port = execution_port
        self._citation_validation_service = citation_validation_service

    async def execute(
        self,
        *,
        request: GovernedRAGAgentRequest,
    ) -> GovernedRAGAgentResult:
        retrieved_context = await self._retrieval_service.retrieve(
            request=request.retrieval_request,
        )

        allowed_citation_ids = frozenset(
            retrieved_context.citation_registry.citation_ids
        )

        agent_execution = await self._execution_port.execute(
            request=(
                GovernedAgentExecutionRequest(
                    user_input=request.user_input,
                    context_fragments=(retrieved_context.assembled_context.fragments),
                    allowed_citation_ids=(allowed_citation_ids),
                )
            )
        )

        citation_validation = self._citation_validation_service.validate(
            cited_ids=(agent_execution.cited_ids),
            allowed_citation_ids=(allowed_citation_ids),
        )

        return GovernedRAGAgentResult(
            response_text=(agent_execution.response_text),
            retrieved_context=retrieved_context,
            agent_execution=agent_execution,
            citation_validation=(citation_validation),
            trusted_citation_ids=(citation_validation.valid_citation_ids),
        )
