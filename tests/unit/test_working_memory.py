import pytest
from pydantic import ValidationError

from agent_platform.memory import (
    WorkingMemory,
    WorkingMemoryItem,
)


def test_working_memory_item_is_immutable() -> None:
    item = WorkingMemoryItem(
        key="order_id",
        value="ORD-123",
        source="user-request",
    )

    with pytest.raises(ValidationError):
        item.value = "ORD-999"


def test_set_adds_new_item_without_mutating_original() -> None:
    original = WorkingMemory(
        execution_id="exec-001",
    )

    updated = original.set(
        key="order_id",
        value="ORD-123",
        source="user-request",
    )

    assert original.items == ()
    assert updated.get("order_id") == "ORD-123"
    assert updated is not original


def test_set_replaces_existing_key() -> None:
    memory = WorkingMemory(
        execution_id="exec-001",
    )

    memory = memory.set(
        key="order_status",
        value="PROCESSING",
        source="order-service",
    )

    memory = memory.set(
        key="order_status",
        value="DELAYED",
        source="order-service",
    )

    assert memory.get("order_status") == "DELAYED"
    assert len(memory.items) == 1


def test_get_item_preserves_provenance() -> None:
    memory = WorkingMemory(
        execution_id="exec-001",
    ).set(
        key="customer_tier",
        value="gold",
        source="crm",
    )

    item = memory.get_item("customer_tier")

    assert item is not None
    assert item.value == "gold"
    assert item.source == "crm"


def test_missing_key_returns_none() -> None:
    memory = WorkingMemory(
        execution_id="exec-001",
    )

    assert memory.get("missing") is None
    assert memory.get_item("missing") is None


def test_contains_returns_key_presence() -> None:
    memory = WorkingMemory(
        execution_id="exec-001",
    ).set(
        key="needs_escalation",
        value=True,
        source="sla-rule",
    )

    assert memory.contains("needs_escalation") is True
    assert memory.contains("missing") is False


def test_remove_returns_new_memory_without_key() -> None:
    original = WorkingMemory(
        execution_id="exec-001",
    ).set(
        key="order_id",
        value="ORD-123",
    )

    updated = original.remove("order_id")

    assert original.contains("order_id") is True
    assert updated.contains("order_id") is False
    assert updated is not original


def test_empty_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkingMemoryItem(
            key="",
            value="invalid",
        )
