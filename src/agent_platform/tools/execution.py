from typing import Any

from agent_platform.tools.audit import (
    ToolAuditDecision,
    ToolAuditEvent,
    ToolAuditSink,
)
from agent_platform.tools.authorization import (
    ToolAuthorizationContext,
    ToolAuthorizationDeniedError,
    ToolAuthorizationPolicy,
)
from agent_platform.tools.binding import ToolBinding
from agent_platform.tools.contracts import (
    ToolExecutionContext,
    ToolInput,
    ToolOutput,
)
from agent_platform.tools.retry import (
    ToolRetryExecutor,
    ToolRetryPolicy,
)
from agent_platform.tools.risk import (
    ToolApprovalContext,
    ToolApprovalRequiredError,
    ToolRiskPolicy,
)
from agent_platform.tools.telemetry import (
    ToolExecutionEvent,
    ToolExecutionStatus,
    ToolExecutionTimer,
    ToolTelemetrySink,
)
from agent_platform.tools.validation import ToolValidator


class ToolExecutionError(Exception):
    """Base error raised during governed tool execution."""


class ToolExecutionService:
    """Executes tools through governed validation and policy boundaries."""

    def __init__(
        self,
        *,
        authorization_policy: ToolAuthorizationPolicy,
        risk_policy: ToolRiskPolicy,
        retry_policy: ToolRetryPolicy | None = None,
        validator: ToolValidator | None = None,
        telemetry_sink: ToolTelemetrySink | None = None,
        audit_sink: ToolAuditSink | None = None,
    ) -> None:
        self._authorization_policy = authorization_policy
        self._risk_policy = risk_policy
        self._validator = validator or ToolValidator()
        self._retry_executor = ToolRetryExecutor(
            policy=retry_policy or ToolRetryPolicy(),
        )
        self._telemetry_sink = telemetry_sink
        self._audit_sink = audit_sink

    @property
    def risk_policy(self) -> ToolRiskPolicy:
        """Return the governed risk policy used by this execution service."""
        return self._risk_policy

    async def execute[
        InputT: ToolInput,
        OutputT: ToolOutput,
    ](
        self,
        *,
        binding: ToolBinding[InputT, OutputT],
        payload: Any,
        context: ToolExecutionContext,
        authorization_context: ToolAuthorizationContext,
        approval_context: ToolApprovalContext | None = None,
    ) -> OutputT:
        timer = ToolExecutionTimer()

        validated_input = self._validator.validate_input(
            definition=binding.definition,
            payload=payload,
        )

        try:
            self._authorization_policy.authorize(
                definition=binding.definition,
                context=authorization_context,
            )
        except ToolAuthorizationDeniedError:
            self._emit_audit_event(
                binding=binding,
                context=context,
                authorization_context=authorization_context,
                decision=ToolAuditDecision.AUTHORIZATION_DENIED,
            )
            raise

        self._emit_audit_event(
            binding=binding,
            context=context,
            authorization_context=authorization_context,
            decision=ToolAuditDecision.AUTHORIZED,
        )

        risk_decision = self._risk_policy.evaluate(
            definition=binding.definition,
        )

        try:
            self._risk_policy.enforce_approval(
                definition=binding.definition,
                decision=risk_decision,
                approval_context=approval_context,
            )
        except ToolApprovalRequiredError:
            self._emit_audit_event(
                binding=binding,
                context=context,
                authorization_context=authorization_context,
                decision=ToolAuditDecision.APPROVAL_REQUIRED,
            )
            raise

        if risk_decision.approval_required:
            self._emit_audit_event(
                binding=binding,
                context=context,
                authorization_context=authorization_context,
                decision=ToolAuditDecision.APPROVED,
                approver_id=(
                    approval_context.approver_id
                    if approval_context is not None
                    else None
                ),
            )

        attempt_count = 0

        async def invoke_tool() -> OutputT:
            nonlocal attempt_count
            attempt_count += 1

            return await binding.tool.execute(
                tool_input=validated_input,
                context=context,
            )

        try:
            raw_output = await self._retry_executor.execute(
                invoke_tool,
            )

            validated_output = self._validator.validate_output(
                definition=binding.definition,
                payload=raw_output,
            )

        except Exception as exc:
            self._emit_event(
                binding=binding,
                context=context,
                authorization_context=authorization_context,
                approval_context=approval_context,
                risk_level=risk_decision.risk_level,
                approval_required=risk_decision.approval_required,
                status=ToolExecutionStatus.FAILURE,
                duration_ms=timer.elapsed_ms(),
                attempt_count=attempt_count,
                error_type=type(exc).__name__,
            )
            raise

        self._emit_event(
            binding=binding,
            context=context,
            authorization_context=authorization_context,
            approval_context=approval_context,
            risk_level=risk_decision.risk_level,
            approval_required=risk_decision.approval_required,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=timer.elapsed_ms(),
            attempt_count=attempt_count,
        )

        return validated_output

    def _emit_audit_event[
        InputT: ToolInput,
        OutputT: ToolOutput,
    ](
        self,
        *,
        binding: ToolBinding[InputT, OutputT],
        context: ToolExecutionContext,
        authorization_context: ToolAuthorizationContext,
        decision: ToolAuditDecision,
        approver_id: str | None = None,
    ) -> None:
        if self._audit_sink is None:
            return

        self._audit_sink.emit(
            ToolAuditEvent(
                tool_name=binding.definition.metadata.name,
                tool_version=binding.definition.metadata.version,
                correlation_id=context.correlation_id,
                subject_id=authorization_context.subject_id,
                decision=decision,
                approver_id=approver_id,
            )
        )

    def _emit_event[
        InputT: ToolInput,
        OutputT: ToolOutput,
    ](
        self,
        *,
        binding: ToolBinding[InputT, OutputT],
        context: ToolExecutionContext,
        authorization_context: ToolAuthorizationContext,
        approval_context: ToolApprovalContext | None,
        risk_level,
        approval_required: bool,
        status: ToolExecutionStatus,
        duration_ms: float,
        attempt_count: int,
        error_type: str | None = None,
    ) -> None:
        if self._telemetry_sink is None:
            return

        approver_id = (
            approval_context.approver_id if approval_context is not None else None
        )

        self._telemetry_sink.emit(
            ToolExecutionEvent(
                tool_name=binding.definition.metadata.name,
                tool_version=binding.definition.metadata.version,
                subject_id=authorization_context.subject_id,
                correlation_id=context.correlation_id,
                risk_level=risk_level,
                approval_required=approval_required,
                approver_id=approver_id,
                status=status,
                duration_ms=duration_ms,
                attempt_count=attempt_count,
                error_type=error_type,
            )
        )
