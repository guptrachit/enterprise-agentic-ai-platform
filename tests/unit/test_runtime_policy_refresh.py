import pytest

from agent_platform.llm.runtime_policy_refresh import (
    RuntimePolicyRefreshConfig,
    RuntimePolicyRefreshMode,
)


def test_default_refresh_mode_is_process_static() -> None:
    config = RuntimePolicyRefreshConfig()

    assert config.mode is RuntimePolicyRefreshMode.PROCESS_STATIC

    assert config.ttl_seconds is None


@pytest.mark.parametrize(
    (
        "mode",
        "value",
    ),
    (
        (
            RuntimePolicyRefreshMode.PROCESS_STATIC,
            "process_static",
        ),
        (
            RuntimePolicyRefreshMode.PER_REQUEST,
            "per_request",
        ),
        (
            RuntimePolicyRefreshMode.TTL,
            "ttl",
        ),
    ),
)
def test_refresh_mode_values(
    mode: RuntimePolicyRefreshMode,
    value: str,
) -> None:
    assert mode.value == value


def test_per_request_refresh_configuration() -> None:
    config = RuntimePolicyRefreshConfig(
        mode=RuntimePolicyRefreshMode.PER_REQUEST,
    )

    assert config.mode is RuntimePolicyRefreshMode.PER_REQUEST

    assert config.ttl_seconds is None


def test_ttl_refresh_configuration() -> None:
    config = RuntimePolicyRefreshConfig(
        mode=RuntimePolicyRefreshMode.TTL,
        ttl_seconds=60.0,
    )

    assert config.mode is RuntimePolicyRefreshMode.TTL
    assert config.ttl_seconds == 60.0


def test_ttl_mode_requires_ttl_seconds() -> None:
    with pytest.raises(
        ValueError,
        match="ttl_seconds is required",
    ):
        RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
        )


@pytest.mark.parametrize(
    "ttl_seconds",
    (
        0.0,
        -1.0,
        -60.0,
    ),
)
def test_ttl_must_be_positive(
    ttl_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be greater than 0",
    ):
        RuntimePolicyRefreshConfig(
            mode=RuntimePolicyRefreshMode.TTL,
            ttl_seconds=ttl_seconds,
        )


@pytest.mark.parametrize(
    "mode",
    (
        RuntimePolicyRefreshMode.PROCESS_STATIC,
        RuntimePolicyRefreshMode.PER_REQUEST,
    ),
)
def test_non_ttl_modes_reject_ttl_seconds(
    mode: RuntimePolicyRefreshMode,
) -> None:
    with pytest.raises(
        ValueError,
        match="may only be configured",
    ):
        RuntimePolicyRefreshConfig(
            mode=mode,
            ttl_seconds=60.0,
        )
