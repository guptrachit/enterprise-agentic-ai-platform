import pytest

from agent_platform.tools import (
    Tool,
    ToolApprovalContext,
    ToolAuthorizationContext,
    ToolAuthorizationDeniedError,
    ToolAuthorizationPolicy,
    ToolAuthorizationRule,
    ToolBinding,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionService,
    ToolInput,
    ToolMetadata,
    ToolOutput,
    ToolRiskLevel,
    ToolRiskPolicy,
    ToolRiskRule,
)
from agent_platform.tools.audit import (
    InMemoryToolAuditSink,
    ToolAuditDecision,
)
from agent_platform.tools.risk import ToolApprovalRequiredError


class AuditInput(ToolInput):
    value: str


class AuditOutput(ToolOutput):
    value: str


class AuditTool(Tool[AuditInput, AuditOutput]):
    @property
    def name(self) -> str:
        return "audit_tool"

    async def execute(
        self,
        tool_input: AuditInput,
        context: ToolExecutionContext,
    ) -> AuditOutput:
        return AuditOutput(
            value=tool_input.value,
        )


def create_binding() -> ToolBinding[AuditInput, AuditOutput]:
    definition = ToolDefinition(
        metadata=ToolMetadata(
            name="audit_tool",
            version="1.0.0",
            description="Tool used to test governance audit.",
        ),
        input_type=AuditInput,
        output_type=AuditOutput,
    )

    return ToolBinding(
        definition=definition,
        tool=AuditTool(),
    )


def create_service(
    *,
    audit_sink: InMemoryToolAuditSink,
    approval_required: bool,
) -> ToolExecutionService:
    return ToolExecutionService(
        authorization_policy=ToolAuthorizationPolicy(
            rules=(
                ToolAuthorizationRule(
                    tool_name="audit_tool",
                    tool_version="1.0.0",
                    required_scopes=frozenset(
                        {
                            "tool:audit_tool:execute",
                        }
                    ),
                ),
            )
        ),
        risk_policy=ToolRiskPolicy(
            rules=(
                ToolRiskRule(
                    tool_name="audit_tool",
                    tool_version="1.0.0",
                    risk_level=(
                        ToolRiskLevel.HIGH if approval_required else ToolRiskLevel.LOW
                    ),
                    approval_required=approval_required,
                ),
            )
        ),
        audit_sink=audit_sink,
    )


@pytest.mark.asyncio
async def test_authorization_denial_is_audited() -> None:
    audit_sink = InMemoryToolAuditSink()

    service = create_service(
        audit_sink=audit_sink,
        approval_required=False,
    )

    with pytest.raises(ToolAuthorizationDeniedError):
        await service.execute(
            binding=create_binding(),
            payload={
                "value": "hello",
            },
            context=ToolExecutionContext(
                correlation_id="corr-denied",
            ),
            authorization_context=ToolAuthorizationContext(
                subject_id="user-denied",
                granted_scopes=frozenset(),
            ),
        )

    assert len(audit_sink.events) == 1

    event = audit_sink.events[0]

    assert event.decision is ToolAuditDecision.AUTHORIZATION_DENIED
    assert event.subject_id == "user-denied"
    assert event.correlation_id == "corr-denied"


@pytest.mark.asyncio
async def test_authorized_execution_is_audited() -> None:
    audit_sink = InMemoryToolAuditSink()

    service = create_service(
        audit_sink=audit_sink,
        approval_required=False,
    )

    await service.execute(
        binding=create_binding(),
        payload={
            "value": "hello",
        },
        context=ToolExecutionContext(
            correlation_id="corr-authorized",
        ),
        authorization_context=ToolAuthorizationContext(
            subject_id="user-authorized",
            granted_scopes=frozenset(
                {
                    "tool:audit_tool:execute",
                }
            ),
        ),
    )

    assert [event.decision for event in audit_sink.events] == [
        ToolAuditDecision.AUTHORIZED,
    ]


@pytest.mark.asyncio
async def test_missing_approval_is_audited() -> None:
    audit_sink = InMemoryToolAuditSink()

    service = create_service(
        audit_sink=audit_sink,
        approval_required=True,
    )

    with pytest.raises(ToolApprovalRequiredError):
        await service.execute(
            binding=create_binding(),
            payload={
                "value": "hello",
            },
            context=ToolExecutionContext(
                correlation_id="corr-approval",
            ),
            authorization_context=ToolAuthorizationContext(
                subject_id="user-approval",
                granted_scopes=frozenset(
                    {
                        "tool:audit_tool:execute",
                    }
                ),
            ),
        )

    assert [event.decision for event in audit_sink.events] == [
        ToolAuditDecision.AUTHORIZED,
        ToolAuditDecision.APPROVAL_REQUIRED,
    ]


@pytest.mark.asyncio
async def test_human_approval_is_audited_with_approver() -> None:
    audit_sink = InMemoryToolAuditSink()

    service = create_service(
        audit_sink=audit_sink,
        approval_required=True,
    )

    await service.execute(
        binding=create_binding(),
        payload={
            "value": "hello",
        },
        context=ToolExecutionContext(
            correlation_id="corr-approved",
        ),
        authorization_context=ToolAuthorizationContext(
            subject_id="user-approved",
            granted_scopes=frozenset(
                {
                    "tool:audit_tool:execute",
                }
            ),
        ),
        approval_context=ToolApprovalContext(
            approved=True,
            approver_id="manager-123",
        ),
    )

    assert [event.decision for event in audit_sink.events] == [
        ToolAuditDecision.AUTHORIZED,
        ToolAuditDecision.APPROVED,
    ]

    assert audit_sink.events[-1].approver_id == "manager-123"
