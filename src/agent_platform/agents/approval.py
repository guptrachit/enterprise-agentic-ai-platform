from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from agent_platform.agents.state import AgentExecutionState
from agent_platform.tools.risk import ToolApprovalContext
from agent_platform.tools.selection import ToolSelection


class AgentApprovalStatus(StrEnum):
    """Lifecycle state of an agent approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class AgentApprovalRequest:
    """Frozen high-risk action awaiting human approval."""

    approval_id: str
    tool_selection: ToolSelection
    execution_state: AgentExecutionState
    status: AgentApprovalStatus = AgentApprovalStatus.PENDING
    approver_id: str | None = None


class AgentApprovalError(Exception):
    """Base error raised by the agent approval workflow."""


class AgentApprovalNotFoundError(AgentApprovalError):
    """Raised when an approval request cannot be found."""


class AgentApprovalAlreadyResolvedError(AgentApprovalError):
    """Raised when an approval request was already resolved."""


class AgentApprovalRejectedError(AgentApprovalError):
    """Raised when execution is resumed after approval rejection."""


class AgentApprovalStore:
    """In-memory store for frozen human approval requests."""

    def __init__(self) -> None:
        self._requests: dict[str, AgentApprovalRequest] = {}

    def create(
        self,
        *,
        tool_selection: ToolSelection,
        execution_state: AgentExecutionState,
    ) -> AgentApprovalRequest:
        request = AgentApprovalRequest(
            approval_id=str(uuid4()),
            tool_selection=tool_selection,
            execution_state=execution_state,
        )

        self._requests[request.approval_id] = request
        return request

    def get(
        self,
        *,
        approval_id: str,
    ) -> AgentApprovalRequest:
        try:
            return self._requests[approval_id]
        except KeyError as exc:
            raise AgentApprovalNotFoundError(
                f"approval request not found: {approval_id}"
            ) from exc

    def approve(
        self,
        *,
        approval_id: str,
        approver_id: str,
    ) -> AgentApprovalRequest:
        request = self.get(
            approval_id=approval_id,
        )

        if request.status is not AgentApprovalStatus.PENDING:
            raise AgentApprovalAlreadyResolvedError(
                f"approval request already resolved: {approval_id}"
            )

        approved = AgentApprovalRequest(
            approval_id=request.approval_id,
            tool_selection=request.tool_selection,
            execution_state=request.execution_state,
            status=AgentApprovalStatus.APPROVED,
            approver_id=approver_id,
        )

        self._requests[approval_id] = approved
        return approved

    def reject(
        self,
        *,
        approval_id: str,
        approver_id: str,
    ) -> AgentApprovalRequest:
        request = self.get(
            approval_id=approval_id,
        )

        if request.status is not AgentApprovalStatus.PENDING:
            raise AgentApprovalAlreadyResolvedError(
                f"approval request already resolved: {approval_id}"
            )

        rejected = AgentApprovalRequest(
            approval_id=request.approval_id,
            tool_selection=request.tool_selection,
            execution_state=request.execution_state,
            status=AgentApprovalStatus.REJECTED,
            approver_id=approver_id,
        )

        self._requests[approval_id] = rejected
        return rejected

    def to_tool_approval_context(
        self,
        *,
        approval_id: str,
    ) -> ToolApprovalContext:
        request = self.get(
            approval_id=approval_id,
        )

        if request.status is AgentApprovalStatus.REJECTED:
            raise AgentApprovalRejectedError(
                f"approval request rejected: {approval_id}"
            )

        if request.status is not AgentApprovalStatus.APPROVED:
            return ToolApprovalContext(
                approved=False,
            )

        return ToolApprovalContext(
            approved=True,
            approver_id=request.approver_id,
        )
