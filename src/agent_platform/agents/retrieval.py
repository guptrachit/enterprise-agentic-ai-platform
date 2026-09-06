from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from agent_platform.memory.citation_registry import (
    CitationRegistry,
)
from agent_platform.memory.context_budget import (
    ContextAssemblyResult,
    ContextBudget,
)
from agent_platform.memory.knowledge import (
    KnowledgeQuery,
    KnowledgeRetrievalResult,
)
from agent_platform.memory.knowledge_authorization import (
    KnowledgeAuthorizationContext,
)


class AgentRetrievalRequest(BaseModel):
    """Knowledge request made on behalf of one governed agent execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: KnowledgeQuery

    authorization_context: KnowledgeAuthorizationContext

    context_budget: ContextBudget


class AgentRetrievedContext(BaseModel):
    """Governed retrieval context made available to the agent."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    retrieval_result: KnowledgeRetrievalResult

    citation_registry: CitationRegistry

    assembled_context: ContextAssemblyResult


@runtime_checkable
class AgentKnowledgeRetriever(Protocol):
    """Provider-neutral knowledge retrieval boundary for agents."""

    async def retrieve(
        self,
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeRetrievalResult:
        """Return authorized and ranked enterprise knowledge."""
        ...
