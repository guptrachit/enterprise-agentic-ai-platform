from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.knowledge import (
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeQuery,
)


class KnowledgeAccessDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class KnowledgeAuthorizationContext(BaseModel):
    """Identity and security boundary for enterprise knowledge retrieval."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    subject_id: str | None = None
    tenant_id: str = Field(min_length=1)

    allowed_classifications: frozenset[KnowledgeClassification] = frozenset()

    granted_scopes: frozenset[str] = frozenset()


class KnowledgeAccessEvaluation(BaseModel):
    """Authorization outcome for knowledge retrieval."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    decision: KnowledgeAccessDecision
    reason: str = Field(min_length=1)


@runtime_checkable
class KnowledgeAuthorizationPolicy(Protocol):
    """Provider-neutral security policy for enterprise knowledge."""

    def authorize_query(
        self,
        *,
        query: KnowledgeQuery,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeAccessEvaluation:
        """Authorize a knowledge retrieval request."""
        ...

    def authorize_chunk(
        self,
        *,
        chunk: KnowledgeChunk,
        context: KnowledgeAuthorizationContext,
    ) -> KnowledgeAccessEvaluation:
        """Authorize one retrieved knowledge chunk."""
        ...


class KnowledgeAccessDeniedError(Exception):
    """Raised when enterprise knowledge access is denied."""
