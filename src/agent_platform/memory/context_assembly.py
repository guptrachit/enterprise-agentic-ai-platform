from pydantic import BaseModel, ConfigDict

from agent_platform.memory.context_budget import (
    ContextAssemblyResult,
    ContextBudget,
    ContextFragment,
    ContextSourceType,
    TokenEstimator,
)
from agent_platform.memory.contracts import RetrievedItem
from agent_platform.memory.conversation import ConversationContext
from agent_platform.memory.knowledge import KnowledgeRetrievalResult
from agent_platform.memory.working import WorkingMemory


class ContextAssemblyInput(BaseModel):
    """All governed context sources available for one model call."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    conversation: ConversationContext | None = None

    working_memory: WorkingMemory | None = None

    long_term_memory: tuple[RetrievedItem, ...] = ()

    enterprise_knowledge: KnowledgeRetrievalResult | None = None

    citation_ids_by_source: dict[str, str] = {}


class ContextAssemblyService:
    """Build bounded model context from governed context sources."""

    def __init__(
        self,
        *,
        token_estimator: TokenEstimator,
    ) -> None:
        self._token_estimator = token_estimator

    def assemble(
        self,
        *,
        input_data: ContextAssemblyInput,
        budget: ContextBudget,
    ) -> ContextAssemblyResult:
        candidates = self._build_candidates(
            input_data=input_data,
        )

        ordered_candidates = tuple(
            sorted(
                candidates,
                key=lambda fragment: fragment.priority,
            )
        )

        selected: list[ContextFragment] = []

        used_tokens = 0
        truncated = False

        for fragment in ordered_candidates:
            next_total = used_tokens + fragment.estimated_tokens

            if next_total > budget.max_tokens:
                truncated = True
                continue

            selected.append(fragment)

            used_tokens = next_total

        return ContextAssemblyResult(
            fragments=tuple(selected),
            used_tokens=used_tokens,
            max_tokens=budget.max_tokens,
            truncated=truncated,
        )

    def _build_candidates(
        self,
        *,
        input_data: ContextAssemblyInput,
    ) -> tuple[ContextFragment, ...]:
        fragments: list[ContextFragment] = []

        if input_data.conversation is not None:
            fragments.extend(
                self._conversation_fragments(
                    context=input_data.conversation,
                )
            )

        if input_data.working_memory is not None:
            fragments.extend(
                self._working_memory_fragments(
                    memory=input_data.working_memory,
                )
            )

        fragments.extend(
            self._long_term_memory_fragments(
                items=input_data.long_term_memory,
            )
        )

        if input_data.enterprise_knowledge is not None:
            fragments.extend(
                self._knowledge_fragments(
                    result=input_data.enterprise_knowledge,
                    citation_ids_by_source=(input_data.citation_ids_by_source),
                )
            )

        return tuple(fragments)

    def _conversation_fragments(
        self,
        *,
        context: ConversationContext,
    ) -> tuple[ContextFragment, ...]:
        fragments: list[ContextFragment] = []

        for message in context.messages:
            content = f"{message.role.value}: {message.content}"

            fragments.append(
                self._fragment(
                    content=content,
                    source_type=(ContextSourceType.CONVERSATION),
                    source_id=message.message_id,
                    priority=10,
                )
            )

        return tuple(fragments)

    def _working_memory_fragments(
        self,
        *,
        memory: WorkingMemory,
    ) -> tuple[ContextFragment, ...]:
        fragments: list[ContextFragment] = []

        for item in memory.items:
            content = f"{item.key}: {item.value}"

            fragments.append(
                self._fragment(
                    content=content,
                    source_type=(ContextSourceType.WORKING_MEMORY),
                    source_id=item.key,
                    priority=20,
                )
            )

        return tuple(fragments)

    def _long_term_memory_fragments(
        self,
        *,
        items: tuple[RetrievedItem, ...],
    ) -> tuple[ContextFragment, ...]:
        fragments: list[ContextFragment] = []

        for item in items:
            fragments.append(
                self._fragment(
                    content=item.record.content,
                    source_type=(ContextSourceType.LONG_TERM_MEMORY),
                    source_id=item.record.memory_id,
                    priority=30,
                )
            )

        return tuple(fragments)

    def _knowledge_fragments(
        self,
        *,
        result: KnowledgeRetrievalResult,
        citation_ids_by_source: dict[str, str],
    ) -> tuple[ContextFragment, ...]:
        fragments: list[ContextFragment] = []

        for item in result.items:
            chunk_id = item.chunk.chunk_id

            fragments.append(
                self._fragment(
                    content=item.chunk.content,
                    source_type=(ContextSourceType.ENTERPRISE_KNOWLEDGE),
                    source_id=chunk_id,
                    citation_id=(citation_ids_by_source.get(chunk_id)),
                    priority=40,
                )
            )

        return tuple(fragments)

    def _fragment(
        self,
        *,
        content: str,
        source_type: ContextSourceType,
        source_id: str | None,
        priority: int,
        citation_id: str | None = None,
    ) -> ContextFragment:
        return ContextFragment(
            content=content,
            source_type=source_type,
            source_id=source_id,
            citation_id=citation_id,
            estimated_tokens=(
                self._token_estimator.estimate(
                    text=content,
                )
            ),
            priority=priority,
        )
