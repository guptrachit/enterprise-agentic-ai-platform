from agent_platform.memory.production_readiness import (
    ReadinessStatus,
)
from agent_platform.memory.production_readiness_service import (
    Layer3ProductionReadinessService,
)


def test_layer3_readiness_passes_when_all_controls_are_present() -> None:
    service = Layer3ProductionReadinessService()

    report = service.evaluate(
        tenant_isolation_enforced=True,
        retrieval_authorization_enforced=True,
        memory_write_policy_enforced=True,
        memory_read_policy_enforced=True,
        citations_validated=True,
        context_budget_enforced=True,
        sensitive_payload_logging_disabled=True,
        retrieval_evaluation_available=True,
        retrieval_observability_available=True,
        provider_boundaries_abstracted=True,
    )

    assert report.passed is True
    assert report.failed_check_ids == ()

    assert all(check.status == ReadinessStatus.PASS for check in report.checks)


def test_layer3_readiness_fails_closed() -> None:
    service = Layer3ProductionReadinessService()

    report = service.evaluate(
        tenant_isolation_enforced=True,
        retrieval_authorization_enforced=False,
        memory_write_policy_enforced=True,
        memory_read_policy_enforced=True,
        citations_validated=True,
        context_budget_enforced=True,
        sensitive_payload_logging_disabled=True,
        retrieval_evaluation_available=True,
        retrieval_observability_available=True,
        provider_boundaries_abstracted=True,
    )

    assert report.passed is False

    assert report.failed_check_ids == ("retrieval_authorization",)
