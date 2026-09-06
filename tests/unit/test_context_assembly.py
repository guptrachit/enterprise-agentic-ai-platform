from agent_platform.memory import (
    ContextAssemblyInput,
    ContextAssemblyService,
    ContextBudget,
    ContextSourceType,
    ConversationContext,
    ConversationMessage,
    ConversationRole,
    DeterministicTokenEstimator,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeQuery,
    KnowledgeRetrievalResult,
    KnowledgeSourceType,
    MemoryMetadata,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RetrievedItem,
    RetrievedKnowledgeChunk,
    WorkingMemory,
)


def create_service() -> ContextAssemblyService:
    return ContextAssemblyService(token_estimator=(DeterministicTokenEstimator()))


def test_token_estimator_is_deterministic() -> None:
    estimator = DeterministicTokenEstimator()

    first = estimator.estimate(
        text="incident escalation policy",
    )

    second = estimator.estimate(
        text="incident escalation policy",
    )

    assert first == second
    assert first == 3


def test_empty_input_returns_empty_context() -> None:
    service = create_service()

    result = service.assemble(
        input_data=ContextAssemblyInput(),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert result.fragments == ()
    assert result.used_tokens == 0
    assert result.truncated is False


def test_conversation_context_is_included() -> None:
    service = create_service()

    conversation = ConversationContext(
        conversation_id="conversation-001",
        messages=(
            ConversationMessage(
                message_id="message-001",
                role=ConversationRole.USER,
                content="Show escalation policy.",
            ),
        ),
    )

    result = service.assemble(
        input_data=ContextAssemblyInput(
            conversation=conversation,
        ),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert len(result.fragments) == 1

    assert result.fragments[0].source_type is ContextSourceType.CONVERSATION


def test_working_memory_is_included() -> None:
    service = create_service()

    working_memory = WorkingMemory(
        execution_id="execution-001",
    ).set(
        key="customer_id",
        value="C-100",
    )

    result = service.assemble(
        input_data=ContextAssemblyInput(
            working_memory=working_memory,
        ),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert len(result.fragments) == 1

    assert result.fragments[0].source_type is ContextSourceType.WORKING_MEMORY


def test_long_term_memory_is_included() -> None:
    service = create_service()

    record = MemoryRecord(
        memory_id="memory-001",
        content="User prefers PDF reports.",
        metadata=MemoryMetadata(
            memory_type=MemoryType.LONG_TERM,
            scope=MemoryScope.USER,
            subject_id="user-123",
            tenant_id="tenant-001",
        ),
        created_at="2026-09-04T00:00:00Z",
    )

    item = RetrievedItem(
        record=record,
        score=1.0,
        rank=1,
    )

    result = service.assemble(
        input_data=ContextAssemblyInput(
            long_term_memory=(item,),
        ),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert len(result.fragments) == 1

    assert result.fragments[0].source_type is ContextSourceType.LONG_TERM_MEMORY


def test_enterprise_knowledge_is_included() -> None:
    service = create_service()

    query = KnowledgeQuery(
        query="incident escalation",
        tenant_id="tenant-001",
    )

    chunk = KnowledgeChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        tenant_id="tenant-001",
        content="Escalate incidents after 24 hours.",
        chunk_index=0,
        source_type=KnowledgeSourceType.DOCUMENT,
        classification=KnowledgeClassification.INTERNAL,
    )

    retrieval = KnowledgeRetrievalResult(
        query=query,
        items=(
            RetrievedKnowledgeChunk(
                chunk=chunk,
                rank=1,
                score=0.9,
            ),
        ),
    )

    result = service.assemble(
        input_data=ContextAssemblyInput(
            enterprise_knowledge=retrieval,
        ),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert len(result.fragments) == 1

    assert result.fragments[0].source_type is ContextSourceType.ENTERPRISE_KNOWLEDGE


def test_priority_order_is_deterministic() -> None:
    service = create_service()

    conversation = ConversationContext(
        conversation_id="conversation-001",
        messages=(
            ConversationMessage(
                message_id="message-001",
                role=ConversationRole.USER,
                content="Need escalation policy.",
            ),
        ),
    )

    working_memory = WorkingMemory(
        execution_id="execution-001",
    ).set(
        key="priority",
        value="critical",
    )

    result = service.assemble(
        input_data=ContextAssemblyInput(
            conversation=conversation,
            working_memory=working_memory,
        ),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert tuple(fragment.source_type for fragment in result.fragments) == (
        ContextSourceType.CONVERSATION,
        ContextSourceType.WORKING_MEMORY,
    )


def test_context_respects_token_budget() -> None:
    service = create_service()

    conversation = ConversationContext(
        conversation_id="conversation-001",
        messages=(
            ConversationMessage(
                message_id="message-001",
                role=ConversationRole.USER,
                content="one two three four",
            ),
            ConversationMessage(
                message_id="message-002",
                role=ConversationRole.ASSISTANT,
                content="five six seven eight",
            ),
        ),
    )

    result = service.assemble(
        input_data=ContextAssemblyInput(
            conversation=conversation,
        ),
        budget=ContextBudget(
            max_tokens=5,
        ),
    )

    assert result.used_tokens <= 5
    assert result.truncated is True


def test_context_never_exceeds_budget() -> None:
    service = create_service()

    working_memory = (
        WorkingMemory(
            execution_id="execution-001",
        )
        .set(
            key="one",
            value="alpha beta gamma",
        )
        .set(
            key="two",
            value="delta epsilon zeta",
        )
    )

    result = service.assemble(
        input_data=ContextAssemblyInput(
            working_memory=working_memory,
        ),
        budget=ContextBudget(
            max_tokens=4,
        ),
    )

    assert result.used_tokens <= result.max_tokens


def test_source_ids_are_preserved() -> None:
    service = create_service()

    conversation = ConversationContext(
        conversation_id="conversation-001",
        messages=(
            ConversationMessage(
                message_id="message-123",
                role=ConversationRole.USER,
                content="Hello",
            ),
        ),
    )

    result = service.assemble(
        input_data=ContextAssemblyInput(
            conversation=conversation,
        ),
        budget=ContextBudget(
            max_tokens=100,
        ),
    )

    assert result.fragments[0].source_id == "message-123"
