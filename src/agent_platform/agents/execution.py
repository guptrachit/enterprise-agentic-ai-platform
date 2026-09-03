from agent_platform.agents.approval import (
    AgentApprovalStatus,
    AgentApprovalStore,
)
from agent_platform.agents.contracts import (
    AgentApprovalRequired,
    AgentFinalResponse,
    AgentObservation,
)
from agent_platform.agents.guardrails import AgentGuardrailPolicy
from agent_platform.agents.model import AgentModel
from agent_platform.agents.state import AgentExecutionState
from agent_platform.tools.binding_registry import ToolBindingRegistry
from agent_platform.tools.execution import ToolExecutionService


class AgentExecutionError(Exception):
    """Base error raised during governed agent execution."""


AgentExecutionResult = AgentFinalResponse | AgentApprovalRequired


class AgentExecutionService:
    """Coordinates bounded governed agent execution."""

    def __init__(
        self,
        *,
        model: AgentModel,
        binding_registry: ToolBindingRegistry,
        tool_execution_service: ToolExecutionService,
        guardrail_policy: AgentGuardrailPolicy | None = None,
        approval_store: AgentApprovalStore | None = None,
    ) -> None:
        self._model = model
        self._binding_registry = binding_registry
        self._tool_execution_service = tool_execution_service
        self._guardrail_policy = guardrail_policy or AgentGuardrailPolicy()
        self._approval_store = approval_store or AgentApprovalStore()

    async def execute(
        self,
        *,
        state: AgentExecutionState,
    ) -> AgentExecutionResult:
        current_state = state

        while True:
            self._guardrail_policy.check_step_limit(
                step_count=current_state.step_count,
            )

            decision = await self._model.decide(
                state=current_state,
            )

            current_state = current_state.increment_step()

            if decision.final_response is not None:
                return decision.final_response

            if decision.tool_selection is None:
                raise AgentExecutionError("agent model produced no executable decision")

            result = await self._execute_selection(
                state=current_state,
                selection=decision.tool_selection,
            )

            if isinstance(result, AgentApprovalRequired):
                return result

            current_state = result

    async def _execute_selection(
        self,
        *,
        state: AgentExecutionState,
        selection,
    ) -> AgentExecutionState | AgentApprovalRequired:
        self._guardrail_policy.check_tool_limit(
            tool_execution_count=state.tool_execution_count,
        )

        binding = self._binding_registry.get(
            name=selection.tool_name,
            version=selection.tool_version,
        )

        risk_decision = self._tool_execution_service.risk_policy.evaluate(
            definition=binding.definition,
        )

        approval_context = state.approval_context

        if risk_decision.approval_required and (
            approval_context is None or not approval_context.approved
        ):
            request = self._approval_store.create(
                tool_selection=selection,
                execution_state=state,
            )

            return AgentApprovalRequired(
                approval_id=request.approval_id,
                tool_selection=selection,
            )

        output = await self._tool_execution_service.execute(
            binding=binding,
            payload=selection.arguments,
            context=state.execution_context,
            authorization_context=state.authorization_context,
            approval_context=approval_context,
        )

        return state.record_tool_observation(
            AgentObservation(
                tool_name=selection.tool_name,
                tool_version=selection.tool_version,
                output=output.model_dump(),
            )
        )

    async def resume(
        self,
        *,
        approval_id: str,
    ) -> AgentExecutionResult:
        approval_request = self._approval_store.get(
            approval_id=approval_id,
        )

        if approval_request.status is AgentApprovalStatus.PENDING:
            return AgentApprovalRequired(
                approval_id=approval_request.approval_id,
                tool_selection=approval_request.tool_selection,
            )

        approval_context = self._approval_store.to_tool_approval_context(
            approval_id=approval_id,
        )

        resumed_state = approval_request.execution_state.model_copy(
            update={
                "approval_context": approval_context,
            }
        )

        result = await self._execute_selection(
            state=resumed_state,
            selection=approval_request.tool_selection,
        )

        if isinstance(result, AgentApprovalRequired):
            return result

        return await self.execute(
            state=result,
        )

    @property
    def approval_store(self) -> AgentApprovalStore:
        return self._approval_store
