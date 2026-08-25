import time
from collections.abc import Callable

from agent_platform.llm.base import LLMResponse
from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.execution_service import (
    LLMExecutionResult,
    LLMExecutionService,
)
from agent_platform.llm.runtime_llm_execution_service_factory import (
    RuntimeLLMExecutionServiceFactory,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshConfig,
    RuntimePolicyRefreshMode,
)
from agent_platform.llm.runtime_policy_refresh_health import (
    RuntimePolicyRefreshHealthSnapshot,
)
from agent_platform.llm.runtime_policy_refresh_metrics import (
    RuntimePolicyRefreshMetrics,
)
from agent_platform.llm.runtime_policy_refresh_telemetry import (
    create_runtime_policy_refresh_event,
    log_runtime_policy_refresh_event,
)


class RefreshAwareLLMExecutionService:
    """Execute requests using configurable routing policy refresh behavior."""

    def __init__(
        self,
        *,
        policy_name: str,
        factory: RuntimeLLMExecutionServiceFactory,
        refresh_config: RuntimePolicyRefreshConfig | None = None,
        refresh_metrics: RuntimePolicyRefreshMetrics | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy_name = policy_name
        self.factory = factory
        self.refresh_config = (
            refresh_config
            if refresh_config is not None
            else RuntimePolicyRefreshConfig()
        )
        self.refresh_metrics = refresh_metrics
        self.clock = clock

        self._service: LLMExecutionService | None = None
        self._resolved_at: float | None = None

        if self.refresh_config.mode is RuntimePolicyRefreshMode.PROCESS_STATIC:
            self._refresh()

    @property
    def policy_identifier(self) -> str | None:
        """Return the currently resolved policy identifier."""

        if self._service is None:
            return None

        return self._service.policy_identifier

    def health_snapshot(
        self,
    ) -> RuntimePolicyRefreshHealthSnapshot:
        """Return operational health for the current refresh state."""

        cache_age_seconds = None

        if self._resolved_at is not None:
            cache_age_seconds = self.clock() - self._resolved_at

        metrics_snapshot = (
            self.refresh_metrics.snapshot()
            if self.refresh_metrics is not None
            else None
        )

        return RuntimePolicyRefreshHealthSnapshot(
            policy_name=self.policy_name,
            policy_identifier=self.policy_identifier,
            refresh_mode=self.refresh_config.mode,
            ttl_seconds=self.refresh_config.ttl_seconds,
            resolved_at=self._resolved_at,
            cache_age_seconds=cache_age_seconds,
            metrics=metrics_snapshot,
        )

    async def execute(
        self,
        request: LLMExecutionRequest,
    ) -> LLMResponse:
        """Execute a request using the configured refresh strategy."""

        service = self._service_for_request()

        return await service.execute(request)

    async def execute_with_decision(
        self,
        request: LLMExecutionRequest,
    ) -> LLMExecutionResult:
        """Execute and expose the routing decision."""

        service = self._service_for_request()

        return await service.execute_with_decision(request)

    def _service_for_request(
        self,
    ) -> LLMExecutionService:
        """Resolve, refresh, or reuse the runtime execution service."""

        mode = self.refresh_config.mode

        if mode is RuntimePolicyRefreshMode.PER_REQUEST:
            return self._refresh()

        if mode is RuntimePolicyRefreshMode.TTL and self._ttl_refresh_required():
            return self._refresh()

        if self._service is None:
            return self._refresh()

        if self.refresh_metrics is not None:
            self.refresh_metrics.record_cache_hit()

        self._log_refresh_event(
            refreshed=False,
            previous_policy_identifier=(self._service.policy_identifier),
        )

        return self._service

    def _ttl_refresh_required(
        self,
    ) -> bool:
        """Return whether the TTL-backed service has expired."""

        if self._service is None:
            return True

        if self._resolved_at is None:
            return True

        ttl_seconds = self.refresh_config.ttl_seconds

        if ttl_seconds is None:
            raise RuntimeError("TTL refresh mode requires ttl_seconds.")

        elapsed = self.clock() - self._resolved_at

        return elapsed >= ttl_seconds

    def _refresh(
        self,
    ) -> LLMExecutionService:
        """Resolve the currently active policy."""

        previous_policy_identifier = (
            self._service.policy_identifier if self._service is not None else None
        )

        service = self.factory.create(self.policy_name)

        self._service = service
        self._resolved_at = self.clock()

        policy_identifier = service.policy_identifier

        if self.refresh_metrics is not None and policy_identifier is not None:
            self.refresh_metrics.record_refresh(
                previous_policy_identifier=(previous_policy_identifier),
                policy_identifier=policy_identifier,
            )

        self._log_refresh_event(
            refreshed=True,
            previous_policy_identifier=previous_policy_identifier,
        )

        return service

    def _log_refresh_event(
        self,
        *,
        refreshed: bool,
        previous_policy_identifier: str | None,
    ) -> None:
        """Emit runtime policy refresh telemetry."""

        if self._service is None:
            return

        policy_identifier = self._service.policy_identifier

        if policy_identifier is None:
            return

        log_runtime_policy_refresh_event(
            create_runtime_policy_refresh_event(
                policy_name=self.policy_name,
                refresh_mode=self.refresh_config.mode,
                policy_identifier=policy_identifier,
                refreshed=refreshed,
                previous_policy_identifier=(previous_policy_identifier),
            )
        )
