from agent_platform.tools import (
    InMemoryToolTelemetrySink,
    ToolExecutionEvent,
    ToolExecutionStatus,
    ToolRiskLevel,
)


def test_in_memory_sink_records_event() -> None:
    sink = InMemoryToolTelemetrySink()

    event = ToolExecutionEvent(
        tool_name="echo",
        tool_version="1.0.0",
        subject_id="user-123",
        correlation_id="corr-123",
        risk_level=ToolRiskLevel.LOW,
        approval_required=False,
        approver_id=None,
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.5,
    )

    sink.emit(event)

    assert sink.events == (event,)


def test_execution_event_records_failure_type() -> None:
    event = ToolExecutionEvent(
        tool_name="echo",
        tool_version="1.0.0",
        subject_id="user-123",
        correlation_id="corr-123",
        risk_level=ToolRiskLevel.HIGH,
        approval_required=True,
        approver_id="manager-123",
        status=ToolExecutionStatus.FAILURE,
        duration_ms=25.0,
        error_type="ToolRetryExhaustedError",
    )

    assert event.status == ToolExecutionStatus.FAILURE
    assert event.error_type == "ToolRetryExhaustedError"
    assert event.approver_id == "manager-123"
