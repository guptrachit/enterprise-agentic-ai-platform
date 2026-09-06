from datetime import UTC, datetime

import pytest

from agent_platform.memory import (
    DefaultMemoryAuthorizationPolicy,
    InMemoryLongTermMemoryStore,
    MemoryAuthorizationContext,
    MemoryAuthorizationDecision,
    MemoryAuthorizationDeniedError,
    MemoryAuthorizationPolicy,
    MemoryAuthorizationService,
    MemoryMetadata,
    MemoryOperation,
    MemoryRecord,
    MemoryRepository,
    MemoryScope,
    MemoryType,
    SecuredMemoryRepository,
)


def create_record(
    *,
    memory_id: str = "memory-001",
    tenant_id: str | None = "tenant-001",
    subject_id: str | None = "user-123",
    scope: MemoryScope = MemoryScope.USER,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        content="Preferred reporting format is PDF.",
        metadata=MemoryMetadata(
            memory_type=MemoryType.LONG_TERM,
            scope=scope,
            subject_id=subject_id,
            tenant_id=tenant_id,
        ),
        created_at=datetime(
            2026,
            9,
            4,
            tzinfo=UTC,
        ),
    )


def create_context(
    *,
    tenant_id: str = "tenant-001",
    subject_id: str | None = "user-123",
    scopes: frozenset[str] = frozenset(
        {
            "memory:read",
            "memory:write",
            "memory:delete",
        }
    ),
) -> MemoryAuthorizationContext:
    return MemoryAuthorizationContext(
        tenant_id=tenant_id,
        subject_id=subject_id,
        granted_scopes=scopes,
    )


def create_secured_repository() -> SecuredMemoryRepository:
    repository = MemoryRepository(
        store=InMemoryLongTermMemoryStore(),
    )

    authorization_service = MemoryAuthorizationService(
        policy=DefaultMemoryAuthorizationPolicy(),
    )

    return SecuredMemoryRepository(
        repository=repository,
        authorization_service=authorization_service,
    )


def test_default_authorization_policy_satisfies_protocol() -> None:
    policy = DefaultMemoryAuthorizationPolicy()

    assert isinstance(
        policy,
        MemoryAuthorizationPolicy,
    )


def test_matching_tenant_subject_and_scope_is_allowed() -> None:
    policy = DefaultMemoryAuthorizationPolicy()

    evaluation = policy.authorize_record(
        operation=MemoryOperation.READ,
        record=create_record(),
        context=create_context(),
    )

    assert evaluation.decision is MemoryAuthorizationDecision.ALLOW


def test_missing_operation_scope_is_denied() -> None:
    policy = DefaultMemoryAuthorizationPolicy()

    evaluation = policy.authorize_record(
        operation=MemoryOperation.DELETE,
        record=create_record(),
        context=create_context(
            scopes=frozenset(
                {
                    "memory:read",
                    "memory:write",
                }
            ),
        ),
    )

    assert evaluation.decision is MemoryAuthorizationDecision.DENY


def test_cross_tenant_access_is_denied() -> None:
    policy = DefaultMemoryAuthorizationPolicy()

    evaluation = policy.authorize_record(
        operation=MemoryOperation.READ,
        record=create_record(
            tenant_id="tenant-999",
        ),
        context=create_context(
            tenant_id="tenant-001",
        ),
    )

    assert evaluation.decision is MemoryAuthorizationDecision.DENY


def test_cross_subject_user_memory_is_denied() -> None:
    policy = DefaultMemoryAuthorizationPolicy()

    evaluation = policy.authorize_record(
        operation=MemoryOperation.READ,
        record=create_record(
            subject_id="user-999",
        ),
        context=create_context(
            subject_id="user-123",
        ),
    )

    assert evaluation.decision is MemoryAuthorizationDecision.DENY


@pytest.mark.asyncio
async def test_secured_repository_allows_authorized_save_and_get() -> None:
    repository = create_secured_repository()

    record = create_record()
    context = create_context()

    saved = await repository.save(
        record=record,
        context=context,
    )

    loaded = await repository.get(
        memory_id=record.memory_id,
        context=context,
    )

    assert saved == record
    assert loaded == record


@pytest.mark.asyncio
async def test_secured_repository_blocks_cross_tenant_read() -> None:
    raw_repository = MemoryRepository(
        store=InMemoryLongTermMemoryStore(),
    )

    record = create_record(
        tenant_id="tenant-999",
    )

    await raw_repository.save(
        record=record,
    )

    secured_repository = SecuredMemoryRepository(
        repository=raw_repository,
        authorization_service=MemoryAuthorizationService(
            policy=DefaultMemoryAuthorizationPolicy(),
        ),
    )

    with pytest.raises(
        MemoryAuthorizationDeniedError,
        match="another tenant",
    ):
        await secured_repository.get(
            memory_id=record.memory_id,
            context=create_context(
                tenant_id="tenant-001",
            ),
        )


@pytest.mark.asyncio
async def test_secured_repository_requires_delete_scope() -> None:
    repository = create_secured_repository()

    record = create_record()

    await repository.save(
        record=record,
        context=create_context(),
    )

    with pytest.raises(
        MemoryAuthorizationDeniedError,
        match="memory:delete",
    ):
        await repository.delete(
            memory_id=record.memory_id,
            context=create_context(
                scopes=frozenset(
                    {
                        "memory:read",
                        "memory:write",
                    }
                ),
            ),
        )


@pytest.mark.asyncio
async def test_authorized_delete_removes_record() -> None:
    repository = create_secured_repository()

    record = create_record()
    context = create_context()

    await repository.save(
        record=record,
        context=context,
    )

    deleted = await repository.delete(
        memory_id=record.memory_id,
        context=context,
    )

    assert deleted is True

    assert (
        await repository.get(
            memory_id=record.memory_id,
            context=context,
        )
        is None
    )
