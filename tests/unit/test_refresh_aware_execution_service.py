from unittest.mock import AsyncMock, Mock

import pytest

from agent_platform.llm.execution import LLMExecutionRequest
from agent_platform.llm.refresh_aware_execution_service import (
    RefreshAwareLLMExecutionService,
)
from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshConfig,
    RuntimePolicyRefreshMode,
)


def create_runtime_service(
    *,
    policy_identifier: str,
):
    service = Mock()
    service.policy_identifier = policy_identifier
    service.execute = AsyncMock()
    service.execute_with_decision = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_process_static_resolves_once() -> None:
    service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = service

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    factory.create.assert_called_once_with("production-policy")

    assert wrapper.policy_identifier == ("production-policy@1.0.0")


@pytest.mark.asyncio
async def test_per_request_refreshes_every_request() -> None:
    first = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    second = create_runtime_service(
        policy_identifier="production-policy@1.1.0",
    )

    factory = Mock()

    factory.create.side_effect = (
        first,
        second,
    )

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    assert wrapper.policy_identifier == ("production-policy@1.0.0")

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    assert wrapper.policy_identifier == ("production-policy@1.1.0")

    assert factory.create.call_count == 2


@pytest.mark.asyncio
async def test_ttl_reuses_service_before_expiration() -> None:
    now = [100.0]

    def clock() -> float:
        return now[0]

    service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = service

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=60.0,
        ),
        clock=clock,
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    now[0] = 130.0

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    factory.create.assert_called_once_with("production-policy")


@pytest.mark.asyncio
async def test_ttl_refreshes_after_expiration() -> None:
    now = [100.0]

    def clock() -> float:
        return now[0]

    first = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    second = create_runtime_service(
        policy_identifier="production-policy@1.1.0",
    )

    factory = Mock()

    factory.create.side_effect = (
        first,
        second,
    )

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=60.0,
        ),
        clock=clock,
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    assert wrapper.policy_identifier == ("production-policy@1.0.0")

    now[0] = 160.0

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    assert wrapper.policy_identifier == ("production-policy@1.1.0")

    assert factory.create.call_count == 2


@pytest.mark.asyncio
async def test_ttl_boundary_refreshes_at_exact_expiration() -> None:
    now = [10.0]

    def clock() -> float:
        return now[0]

    first = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    second = create_runtime_service(
        policy_identifier="production-policy@1.1.0",
    )

    factory = Mock()

    factory.create.side_effect = (
        first,
        second,
    )

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=30.0,
        ),
        clock=clock,
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    now[0] = 40.0

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    assert factory.create.call_count == 2

    assert wrapper.policy_identifier == ("production-policy@1.1.0")


@pytest.mark.asyncio
async def test_execute_with_decision_uses_same_refresh_strategy() -> None:
    service = create_runtime_service(
        policy_identifier="production-policy@2.0.0",
    )

    expected = object()

    service.execute_with_decision.return_value = expected

    factory = Mock()
    factory.create.return_value = service

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
    )

    request = LLMExecutionRequest(
        prompt="Answer.",
    )

    result = await wrapper.execute_with_decision(request)

    assert result is expected

    service.execute_with_decision.assert_awaited_once_with(request)


def test_process_static_resolves_during_construction() -> None:
    service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = service

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
    )

    factory.create.assert_called_once_with("production-policy")

    assert wrapper.policy_identifier == ("production-policy@1.0.0")


def test_per_request_does_not_resolve_during_construction() -> None:
    factory = Mock()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
    )

    factory.create.assert_not_called()
    assert wrapper.policy_identifier is None


def test_ttl_does_not_resolve_until_first_request() -> None:
    factory = Mock()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=60.0,
        ),
    )

    factory.create.assert_not_called()
    assert wrapper.policy_identifier is None


@pytest.mark.asyncio
async def test_per_request_logs_refresh_event(
    caplog,
) -> None:
    import json
    import logging

    service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = service

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
    )

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        await wrapper.execute(
            LLMExecutionRequest(
                prompt="Answer.",
            )
        )

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_runtime_policy_refresh ")
    ]

    assert len(records) == 1

    payload = json.loads(
        records[0].getMessage().removeprefix("llm_runtime_policy_refresh ")
    )

    assert payload["refreshed"] is True

    assert payload["policy_identifier"] == ("production-policy@1.0.0")


@pytest.mark.asyncio
async def test_ttl_logs_cache_reuse_before_expiration(
    caplog,
) -> None:
    import json
    import logging

    now = [100.0]

    def clock() -> float:
        return now[0]

    service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = service

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=60.0,
        ),
        clock=clock,
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    caplog.clear()

    now[0] = 130.0

    with caplog.at_level(
        logging.INFO,
        logger="agent_platform.llm",
    ):
        await wrapper.execute(
            LLMExecutionRequest(
                prompt="Second.",
            )
        )

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("llm_runtime_policy_refresh ")
    ]

    assert len(records) == 1

    payload = json.loads(
        records[0].getMessage().removeprefix("llm_runtime_policy_refresh ")
    )

    assert payload["refresh_mode"] == "ttl"
    assert payload["refreshed"] is False


@pytest.mark.asyncio
async def test_refresh_metrics_record_per_request_refresh() -> None:
    from agent_platform.llm.runtime_policy_refresh_metrics import (
        RuntimePolicyRefreshMetrics,
    )

    service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = service

    metrics = RuntimePolicyRefreshMetrics()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
        refresh_metrics=metrics,
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_resolutions == 2
    assert snapshot.refreshes == 2
    assert snapshot.cache_hits == 0


@pytest.mark.asyncio
async def test_refresh_metrics_record_ttl_cache_hit() -> None:
    from agent_platform.llm.runtime_policy_refresh_metrics import (
        RuntimePolicyRefreshMetrics,
    )

    now = [100.0]

    def clock() -> float:
        return now[0]

    service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = service

    metrics = RuntimePolicyRefreshMetrics()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=60.0,
        ),
        refresh_metrics=metrics,
        clock=clock,
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    now[0] = 130.0

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_resolutions == 2
    assert snapshot.refreshes == 1
    assert snapshot.cache_hits == 1


@pytest.mark.asyncio
async def test_refresh_metrics_detect_policy_change() -> None:
    from agent_platform.llm.runtime_policy_refresh_metrics import (
        RuntimePolicyRefreshMetrics,
    )

    first = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    second = create_runtime_service(
        policy_identifier="production-policy@1.1.0",
    )

    factory = Mock()
    factory.create.side_effect = (
        first,
        second,
    )

    metrics = RuntimePolicyRefreshMetrics()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
        refresh_metrics=metrics,
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="First.",
        )
    )

    await wrapper.execute(
        LLMExecutionRequest(
            prompt="Second.",
        )
    )

    snapshot = metrics.snapshot()

    assert snapshot.refreshes == 2
    assert snapshot.policy_changes == 1


def test_health_snapshot_for_process_static_service() -> None:
    from agent_platform.llm.runtime_policy_refresh_metrics import (
        RuntimePolicyRefreshMetrics,
    )

    now = [100.0]

    def clock() -> float:
        return now[0]

    service = create_runtime_service(
        policy_identifier="production-policy@1.0.0",
    )

    factory = Mock()
    factory.create.return_value = service

    metrics = RuntimePolicyRefreshMetrics()

    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=factory,
        refresh_metrics=metrics,
        clock=clock,
    )

    now[0] = 125.0

    snapshot = wrapper.health_snapshot()

    assert snapshot.policy_identifier == ("production-policy@1.0.0")

    assert snapshot.resolved_at == 100.0
    assert snapshot.cache_age_seconds == 25.0

    assert snapshot.metrics is not None
    assert snapshot.metrics.refreshes == 1


def test_health_snapshot_before_lazy_resolution() -> None:
    wrapper = RefreshAwareLLMExecutionService(
        policy_name="production-policy",
        factory=Mock(),
        refresh_config=RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.PER_REQUEST,
        ),
    )

    snapshot = wrapper.health_snapshot()

    assert snapshot.policy_identifier is None
    assert snapshot.resolved is False
    assert snapshot.resolved_at is None
    assert snapshot.cache_age_seconds is None
