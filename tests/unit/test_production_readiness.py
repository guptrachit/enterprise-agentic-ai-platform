from unittest.mock import Mock

import pytest

from agent_platform.config import Settings
from agent_platform.production_readiness import (
    ProductionReadinessError,
    validate_production_readiness,
    validate_runtime_readiness,
)


def create_settings(
    **overrides,
) -> Settings:
    values = {
        "openai_api_key": "test-key",
        "app_env": "production",
        "llm_api_authentication_required": True,
        "llm_api_cors_allowed_origins": ("https://app.example.com",),
    }

    values.update(overrides)

    return Settings(**values)


def create_runtime(
    *,
    resolved: bool,
):
    runtime = Mock()

    snapshot = Mock()

    snapshot.to_dict.return_value = {
        "resolved": resolved,
        "policy_name": "production-routing-policy",
        "policy_identifier": ("production-routing-policy@1.0.0" if resolved else None),
    }

    runtime.health_snapshot.return_value = snapshot

    return runtime


def test_development_configuration_is_not_subject_to_production_rules() -> None:
    settings = Settings(
        openai_api_key="test-key",
        app_env="development",
        llm_api_authentication_required=False,
    )

    validate_production_readiness(settings)


def test_valid_production_configuration_passes() -> None:
    validate_production_readiness(create_settings())


def test_production_requires_authentication() -> None:
    settings = create_settings(llm_api_authentication_required=False)

    with pytest.raises(
        ProductionReadinessError,
        match="requires LLM API authentication",
    ):
        validate_production_readiness(settings)


def test_production_rejects_wildcard_cors() -> None:
    settings = create_settings(llm_api_cors_allowed_origins=("*",))

    with pytest.raises(
        ProductionReadinessError,
        match="Wildcard CORS origin",
    ):
        validate_production_readiness(settings)


def test_production_requires_explicit_cors_origin() -> None:
    settings = create_settings(llm_api_cors_allowed_origins=())

    with pytest.raises(
        ProductionReadinessError,
        match="explicit CORS origin",
    ):
        validate_production_readiness(settings)


def test_runtime_readiness_passes_when_resolved() -> None:
    validate_runtime_readiness(create_runtime(resolved=True))


def test_runtime_readiness_fails_when_unresolved() -> None:
    with pytest.raises(
        ProductionReadinessError,
        match="Governed LLM runtime is not ready",
    ):
        validate_runtime_readiness(create_runtime(resolved=False))


def test_runtime_readiness_defaults_missing_resolved_to_false() -> None:
    runtime = Mock()

    snapshot = Mock()

    snapshot.to_dict.return_value = {}

    runtime.health_snapshot.return_value = snapshot

    with pytest.raises(
        ProductionReadinessError,
        match="Governed LLM runtime is not ready",
    ):
        validate_runtime_readiness(runtime)
