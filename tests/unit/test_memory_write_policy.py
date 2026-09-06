from datetime import UTC, datetime

import pytest

from agent_platform.memory import (
    DefaultMemoryWritePolicy,
    MemoryMetadata,
    MemoryScope,
    MemoryType,
    MemoryWriteDecision,
    MemoryWriteDeniedError,
    MemoryWritePolicy,
    MemoryWriteRequest,
    MemoryWriteService,
)


def create_request(
    *,
    memory_type: MemoryType = MemoryType.LONG_TERM,
    scope: MemoryScope = MemoryScope.USER,
    subject_id: str | None = "user-123",
    tenant_id: str | None = "tenant-001",
) -> MemoryWriteRequest:
    return MemoryWriteRequest(
        content="Preferred reporting format is PDF.",
        metadata=MemoryMetadata(
            memory_type=memory_type,
            scope=scope,
            subject_id=subject_id,
            tenant_id=tenant_id,
            tags=frozenset(
                {
                    "preference",
                    "reporting",
                }
            ),
        ),
        source="agent-memory-candidate",
    )


def test_default_policy_satisfies_protocol() -> None:
    policy = DefaultMemoryWritePolicy()

    assert isinstance(
        policy,
        MemoryWritePolicy,
    )


def test_policy_allows_valid_long_term_memory() -> None:
    policy = DefaultMemoryWritePolicy()

    evaluation = policy.evaluate(
        request=create_request(),
    )

    assert evaluation.decision is MemoryWriteDecision.ALLOW


def test_policy_denies_working_memory_persistence() -> None:
    policy = DefaultMemoryWritePolicy()

    evaluation = policy.evaluate(
        request=create_request(
            memory_type=MemoryType.WORKING,
        ),
    )

    assert evaluation.decision is MemoryWriteDecision.DENY


def test_policy_denies_missing_tenant() -> None:
    policy = DefaultMemoryWritePolicy()

    evaluation = policy.evaluate(
        request=create_request(
            tenant_id=None,
        ),
    )

    assert evaluation.decision is MemoryWriteDecision.DENY


def test_policy_denies_user_scope_without_subject() -> None:
    policy = DefaultMemoryWritePolicy()

    evaluation = policy.evaluate(
        request=create_request(
            subject_id=None,
        ),
    )

    assert evaluation.decision is MemoryWriteDecision.DENY


def test_write_service_fails_closed_when_denied() -> None:
    service = MemoryWriteService(
        policy=DefaultMemoryWritePolicy(),
    )

    with pytest.raises(
        MemoryWriteDeniedError,
        match="LONG_TERM",
    ):
        service.authorize(
            request=create_request(
                memory_type=MemoryType.WORKING,
            ),
        )


def test_write_service_builds_record_after_authorization() -> None:
    service = MemoryWriteService(
        policy=DefaultMemoryWritePolicy(),
    )

    request = create_request()

    created_at = datetime(
        2026,
        9,
        4,
        tzinfo=UTC,
    )

    record = service.build_record(
        memory_id="memory-001",
        created_at=created_at,
        request=request,
    )

    assert record.memory_id == "memory-001"
    assert record.content == request.content
    assert record.metadata == request.metadata
    assert record.created_at == created_at
    assert record.attributes["write_source"] == "agent-memory-candidate"
