from agent_platform.config import get_settings


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
