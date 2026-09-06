from time import perf_counter

from agent_platform.agents.retrieval import (
    AgentRetrievalRequest,
    AgentRetrievedContext,
)
from agent_platform.agents.retrieval_service import (
    AgentRetrievalService,
)
from agent_platform.memory.observability import (
    MemoryRetrievalTelemetrySink,
    ObservabilityOutcome,
    RetrievalTelemetryEvent,
)


class ObservableAgentRetrievalService:
    """Adds safe operational telemetry around governed retrieval."""

    def __init__(
        self,
        *,
        service: AgentRetrievalService,
        telemetry_sink: MemoryRetrievalTelemetrySink,
    ) -> None:
        self._service = service
        self._telemetry_sink = telemetry_sink

    async def retrieve(
        self,
        *,
        request: AgentRetrievalRequest,
    ) -> AgentRetrievedContext:
        started_at = perf_counter()

        try:
            result = await self._service.retrieve(
                request=request,
            )

        except Exception as error:
            duration_ms = (perf_counter() - started_at) * 1000

            await self._telemetry_sink.record_retrieval(
                event=RetrievalTelemetryEvent(
                    tenant_id=request.query.tenant_id,
                    outcome=ObservabilityOutcome.FAILURE,
                    duration_ms=duration_ms,
                    retrieved_count=0,
                    context_fragment_count=0,
                    budget_dropped_count=0,
                    used_tokens=0,
                    max_tokens=(request.context_budget.max_tokens),
                    truncated=False,
                    error_type=type(error).__name__,
                )
            )

            raise

        duration_ms = (perf_counter() - started_at) * 1000

        retrieved_count = len(result.retrieval_result.items)

        context_fragment_count = len(result.assembled_context.fragments)

        budget_dropped_count = max(
            retrieved_count - context_fragment_count,
            0,
        )

        await self._telemetry_sink.record_retrieval(
            event=RetrievalTelemetryEvent(
                tenant_id=request.query.tenant_id,
                outcome=ObservabilityOutcome.SUCCESS,
                duration_ms=duration_ms,
                retrieved_count=retrieved_count,
                context_fragment_count=(context_fragment_count),
                budget_dropped_count=(budget_dropped_count),
                used_tokens=(result.assembled_context.used_tokens),
                max_tokens=(result.assembled_context.max_tokens),
                truncated=(result.assembled_context.truncated),
            )
        )

        return result
