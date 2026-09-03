from agent_platform.tools.audit import (
    InMemoryToolAuditSink,
    ToolAuditDecision,
    ToolAuditEvent,
)


def test_audit_event_records_governance_decision() -> None:
    event = ToolAuditEvent(
        tool_name="issue_payment",
        tool_version="1.0.0",
        correlation_id="corr-123",
        subject_id="user-123",
        decision=ToolAuditDecision.AUTHORIZATION_DENIED,
    )

    assert event.decision is ToolAuditDecision.AUTHORIZATION_DENIED


def test_in_memory_audit_sink_records_event() -> None:
    sink = InMemoryToolAuditSink()

    event = ToolAuditEvent(
        tool_name="delete_customer",
        tool_version="1.0.0",
        correlation_id="corr-456",
        subject_id="user-456",
        decision=ToolAuditDecision.APPROVAL_REQUIRED,
    )

    sink.emit(event)

    assert sink.events == (event,)


def test_audit_event_records_approver_identity() -> None:
    event = ToolAuditEvent(
        tool_name="issue_payment",
        tool_version="1.0.0",
        correlation_id="corr-789",
        subject_id="user-789",
        decision=ToolAuditDecision.APPROVED,
        approver_id="manager-123",
    )

    assert event.approver_id == "manager-123"
