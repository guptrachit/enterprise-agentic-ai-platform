from agent_platform.config import get_settings
from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.workload import LLMWorkload


def test_settings_load() -> None:
    settings = get_settings()

    assert settings.app_env == "development"
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-5-mini"
    assert settings.openai_api_key
    assert settings.llm_timeout_seconds == 30.0
    assert settings.llm_max_retries == 2
    assert settings.llm_initial_backoff_seconds == 0.5
    assert settings.llm_max_backoff_seconds == 5.0


def test_model_can_be_overridden_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LLM_MODEL",
        "environment-model",
    )

    get_settings.cache_clear()

    try:
        settings = get_settings()

        assert settings.llm_model == "environment-model"
    finally:
        get_settings.cache_clear()


def test_models_can_be_loaded_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LLM_MODELS",
        """
        [
            {
                "name": "fast_general",
                "provider": "openai",
                "provider_model": "gpt-5-mini",
                "workloads": ["general", "classification"],
                "cost_tier": 1,
                "latency_tier": 1,
                "capabilities": [
                    "structured_output",
                    "tool_calling"
                ]
            }
        ]
        """,
    )

    get_settings.cache_clear()

    try:
        settings = get_settings()

        assert len(settings.llm_models) == 1

        model = settings.llm_models[0]

        assert model.name == "fast_general"
        assert model.provider == "openai"
        assert model.provider_model == "gpt-5-mini"

        assert model.capabilities == (
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.TOOL_CALLING,
        )
    finally:
        get_settings.cache_clear()


def test_model_policy_can_be_loaded_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LLM_MODEL_POLICY",
        """
        {
            "assignments": {
                "classification": [
                    "classification_primary",
                    "classification_backup"
                ],
                "reasoning": "reasoning_model"
            }
        }
        """,
    )

    get_settings.cache_clear()

    try:
        settings = get_settings()

        assignments = settings.llm_model_policy.assignments

        assert assignments[LLMWorkload.CLASSIFICATION] == (
            "classification_primary",
            "classification_backup",
        )

        assert assignments[LLMWorkload.REASONING] == "reasoning_model"
    finally:
        get_settings.cache_clear()
