from agent_platform.config import Settings, get_settings
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


def test_settings_default_llm_api_prompt_limit() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    assert settings.llm_api_max_prompt_chars == 20_000


def test_settings_reject_invalid_llm_api_prompt_limit() -> None:
    import pytest

    with pytest.raises(
        ValueError,
        match="llm_api_max_prompt_chars must be greater than 0",
    ):
        Settings(
            openai_api_key="test-key",
            llm_api_max_prompt_chars=0,
        )


def test_settings_default_llm_api_request_timeout() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    assert settings.llm_api_request_timeout_seconds == 60.0


def test_settings_reject_invalid_llm_api_request_timeout() -> None:
    import pytest

    with pytest.raises(
        ValueError,
        match=("llm_api_request_timeout_seconds must be greater than 0"),
    ):
        Settings(
            openai_api_key="test-key",
            llm_api_request_timeout_seconds=0.0,
        )


def test_settings_default_llm_api_rate_limit() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    assert settings.llm_api_rate_limit_requests == 60
    assert settings.llm_api_rate_limit_window_seconds == 60.0


def test_settings_reject_invalid_llm_api_rate_limit_requests() -> None:
    import pytest

    with pytest.raises(
        ValueError,
        match="llm_api_rate_limit_requests must be greater than 0",
    ):
        Settings(
            openai_api_key="test-key",
            llm_api_rate_limit_requests=0,
        )


def test_settings_reject_invalid_llm_api_rate_limit_window() -> None:
    import pytest

    with pytest.raises(
        ValueError,
        match=("llm_api_rate_limit_window_seconds must be greater than 0"),
    ):
        Settings(
            openai_api_key="test-key",
            llm_api_rate_limit_window_seconds=0.0,
        )


def test_settings_default_trusted_proxy_hosts() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    assert settings.llm_api_trusted_proxy_hosts == ()


def test_settings_preserves_trusted_proxy_hosts() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_api_trusted_proxy_hosts=(
            "10.0.0.10",
            "10.0.0.11",
        ),
    )

    assert settings.llm_api_trusted_proxy_hosts == (
        "10.0.0.10",
        "10.0.0.11",
    )


def test_settings_rejects_empty_trusted_proxy_host() -> None:
    import pytest

    with pytest.raises(
        ValueError,
        match=("llm_api_trusted_proxy_hosts must not contain empty values"),
    ):
        Settings(
            openai_api_key="test-key",
            llm_api_trusted_proxy_hosts=(
                "10.0.0.10",
                " ",
            ),
        )


def test_settings_default_llm_api_authentication_required() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    assert settings.llm_api_authentication_required is False


def test_settings_can_require_llm_api_authentication() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_api_authentication_required=True,
    )

    assert settings.llm_api_authentication_required is True


def test_settings_default_cors_allowed_origins() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    assert settings.llm_api_cors_allowed_origins == ("http://localhost:3000",)


def test_settings_accepts_cors_allowed_origins() -> None:
    settings = Settings(
        openai_api_key="test-key",
        llm_api_cors_allowed_origins=(
            "https://app.example.com",
            "https://admin.example.com",
        ),
    )

    assert settings.llm_api_cors_allowed_origins == (
        "https://app.example.com",
        "https://admin.example.com",
    )


def test_settings_rejects_empty_cors_origin() -> None:
    import pytest

    with pytest.raises(
        ValueError,
        match=("llm_api_cors_allowed_origins must not contain empty values"),
    ):
        Settings(
            openai_api_key="test-key",
            llm_api_cors_allowed_origins=(
                "https://app.example.com",
                " ",
            ),
        )


def test_settings_default_llm_api_max_request_body_bytes() -> None:
    settings = Settings(
        openai_api_key="test-key",
    )

    assert settings.llm_api_max_request_body_bytes == 65_536


def test_settings_reject_invalid_llm_api_max_request_body_bytes() -> None:
    import pytest

    with pytest.raises(
        ValueError,
        match=("llm_api_max_request_body_bytes must be greater than 0"),
    ):
        Settings(
            openai_api_key="test-key",
            llm_api_max_request_body_bytes=0,
        )
