from agent_platform.agents.retrieval import (
    AgentKnowledgeRetriever,
    AgentRetrievalRequest,
    AgentRetrievedContext,
)
from agent_platform.memory.citation_mapping import (
    build_citation_source_map,
)
from agent_platform.memory.context_assembly import (
    ContextAssemblyInput,
    ContextAssemblyService,
)
from agent_platform.memory.provenance_service import (
    EvidenceProvenanceService,
)


class AgentRetrievalService:
    """Build governed retrieval context for an agent execution."""

    def __init__(
        self,
        *,
        retriever: AgentKnowledgeRetriever,
        provenance_service: EvidenceProvenanceService,
        context_assembly_service: ContextAssemblyService,
    ) -> None:
        self._retriever = retriever
        self._provenance_service = provenance_service
        self._context_assembly_service = context_assembly_service

    async def retrieve(
        self,
        *,
        request: AgentRetrievalRequest,
    ) -> AgentRetrievedContext:
        retrieval_result = await self._retriever.retrieve(
            query=request.query,
            context=request.authorization_context,
        )

        citation_registry = self._provenance_service.from_enterprise_knowledge(
            result=retrieval_result,
        )

        citation_ids_by_source = build_citation_source_map(
            registry=citation_registry,
        )

        assembled_context = self._context_assembly_service.assemble(
            input_data=ContextAssemblyInput(
                enterprise_knowledge=retrieval_result,
                citation_ids_by_source=(citation_ids_by_source),
            ),
            budget=request.context_budget,
        )

        return AgentRetrievedContext(
            retrieval_result=retrieval_result,
            citation_registry=citation_registry,
            assembled_context=assembled_context,
        )
