from agent_platform.llm.api_health_status import (
    LLMAPIHealthStatus,
    assess_llm_api_health,
    classify_llm_api_health,
)


def test_health_assessment_is_healthy_without_reasons() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=True,
        utilization_rate=0.25,
        capacity_rejections=0,
        failure_counts={},
    )

    assert assessment.status is LLMAPIHealthStatus.HEALTHY

    assert assessment.reasons == ()


def test_health_assessment_reports_unresolved_policy() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=False,
        utilization_rate=0.0,
        capacity_rejections=0,
        failure_counts={},
    )

    assert assessment.status is LLMAPIHealthStatus.DEGRADED

    assert assessment.reasons == ("routing_policy_unresolved",)


def test_health_assessment_reports_timeout() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=True,
        utilization_rate=0.25,
        capacity_rejections=0,
        failure_counts={
            "llm_request_timeout": 1,
        },
    )

    assert assessment.status is LLMAPIHealthStatus.DEGRADED

    assert assessment.reasons == ("llm_request_timeout",)


def test_health_assessment_reports_multiple_degraded_reasons() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=False,
        utilization_rate=0.25,
        capacity_rejections=0,
        failure_counts={
            "llm_request_timeout": 2,
            "llm_rate_limit_exceeded": 3,
        },
    )

    assert assessment.status is LLMAPIHealthStatus.DEGRADED

    assert assessment.reasons == (
        "routing_policy_unresolved",
        "llm_request_timeout",
        "llm_rate_limit_exceeded",
    )


def test_health_assessment_reports_full_capacity() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=True,
        utilization_rate=1.0,
        capacity_rejections=0,
        failure_counts={},
    )

    assert assessment.status is LLMAPIHealthStatus.SATURATED

    assert assessment.reasons == ("llm_capacity_saturated",)


def test_health_assessment_reports_capacity_rejection() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=True,
        utilization_rate=0.5,
        capacity_rejections=2,
        failure_counts={},
    )

    assert assessment.status is LLMAPIHealthStatus.SATURATED

    assert assessment.reasons == ("llm_capacity_exceeded",)


def test_saturated_reasons_take_precedence() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=False,
        utilization_rate=1.0,
        capacity_rejections=1,
        failure_counts={
            "llm_request_timeout": 1,
        },
    )

    assert assessment.status is LLMAPIHealthStatus.SATURATED

    assert assessment.reasons == (
        "llm_capacity_saturated",
        "llm_capacity_exceeded",
    )


def test_prompt_too_large_does_not_degrade_health() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=True,
        utilization_rate=0.25,
        capacity_rejections=0,
        failure_counts={
            "prompt_too_large": 10,
        },
    )

    assert assessment.status is LLMAPIHealthStatus.HEALTHY

    assert assessment.reasons == ()


def test_classifier_remains_backward_compatible() -> None:
    status = classify_llm_api_health(
        runtime_resolved=True,
        utilization_rate=0.25,
        capacity_rejections=0,
        failure_counts={
            "llm_request_timeout": 1,
        },
    )

    assert status is LLMAPIHealthStatus.DEGRADED


def test_health_assessment_is_degraded_after_export_failure() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=True,
        utilization_rate=0.25,
        capacity_rejections=0,
        failure_counts={},
        export_failed_count=1,
    )

    assert assessment.status is LLMAPIHealthStatus.DEGRADED

    assert assessment.reasons == ("telemetry_export_degraded",)


def test_export_failure_does_not_override_saturated_status() -> None:
    assessment = assess_llm_api_health(
        runtime_resolved=True,
        utilization_rate=1.0,
        capacity_rejections=0,
        failure_counts={},
        export_failed_count=2,
    )

    assert assessment.status is LLMAPIHealthStatus.SATURATED

    assert assessment.reasons == ("llm_capacity_saturated",)


def test_classifier_accepts_export_failure_count() -> None:
    status = classify_llm_api_health(
        runtime_resolved=True,
        utilization_rate=0.25,
        capacity_rejections=0,
        failure_counts={},
        export_failed_count=1,
    )

    assert status is LLMAPIHealthStatus.DEGRADED
