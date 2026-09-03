import pytest

from agent_platform.agents import (
    AgentApprovalAlreadyResolvedError,
    AgentApprovalRejectedError,
    AgentApprovalStatus,
    AgentApprovalStore,
    AgentExecutionState,
)
from agent_platform.tools import (
    ToolAuthorizationContext,
    ToolExecutionContext,
    ToolSelection,
)


def create_state() -> AgentExecutionState:
    return AgentExecutionState(
        user_message="Delete customer 123",
        execution_context=ToolExecutionContext(
            correlation_id="corr-123",
        ),
        authorization_context=ToolAuthorizationContext(
            subject_id="user-123",
            granted_scopes=frozenset(
                {
                    "tool:delete_customer:execute",
                }
            ),
        ),
    )


def create_store() -> tuple[AgentApprovalStore, str]:
    store = AgentApprovalStore()

    request = store.create(
        tool_selection=ToolSelection(
            tool_name="delete_customer",
            tool_version="1.0.0",
            arguments={
                "customer_id": "123",
            },
        ),
        execution_state=create_state(),
    )

    return store, request.approval_id


def test_approval_request_starts_pending() -> None:
    store, approval_id = create_store()

    request = store.get(
        approval_id=approval_id,
    )

    assert request.status is AgentApprovalStatus.PENDING


def test_approval_request_freezes_tool_selection() -> None:
    store, approval_id = create_store()

    request = store.get(
        approval_id=approval_id,
    )

    assert request.tool_selection.tool_name == "delete_customer"
    assert request.tool_selection.arguments == {
        "customer_id": "123",
    }


def test_approval_request_can_be_approved() -> None:
    store, approval_id = create_store()

    result = store.approve(
        approval_id=approval_id,
        approver_id="manager-123",
    )

    assert result.status is AgentApprovalStatus.APPROVED
    assert result.approver_id == "manager-123"


def test_approved_request_creates_tool_approval_context() -> None:
    store, approval_id = create_store()

    store.approve(
        approval_id=approval_id,
        approver_id="manager-123",
    )

    context = store.to_tool_approval_context(
        approval_id=approval_id,
    )

    assert context.approved is True
    assert context.approver_id == "manager-123"


def test_rejected_request_cannot_create_execution_context() -> None:
    store, approval_id = create_store()

    store.reject(
        approval_id=approval_id,
        approver_id="manager-123",
    )

    with pytest.raises(
        AgentApprovalRejectedError,
    ):
        store.to_tool_approval_context(
            approval_id=approval_id,
        )


def test_resolved_request_cannot_be_approved_twice() -> None:
    store, approval_id = create_store()

    store.approve(
        approval_id=approval_id,
        approver_id="manager-123",
    )

    with pytest.raises(
        AgentApprovalAlreadyResolvedError,
    ):
        store.approve(
            approval_id=approval_id,
            approver_id="manager-456",
        )
