from agent_platform.llm.retry import RetryPolicy


def test_exponential_backoff() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=0.5,
        max_backoff_seconds=5.0,
        jitter=False,
    )

    assert policy.backoff_seconds(0) == 0.5
    assert policy.backoff_seconds(1) == 1.0
    assert policy.backoff_seconds(2) == 2.0
    assert policy.backoff_seconds(3) == 4.0


def test_backoff_respects_maximum() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=0.5,
        max_backoff_seconds=2.0,
        jitter=False,
    )

    assert policy.backoff_seconds(0) == 0.5
    assert policy.backoff_seconds(1) == 1.0
    assert policy.backoff_seconds(2) == 2.0
    assert policy.backoff_seconds(3) == 2.0


def test_jitter_stays_within_backoff_range() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=1.0,
        max_backoff_seconds=5.0,
        jitter=True,
    )

    for _ in range(100):
        delay = policy.backoff_seconds(2)
        assert 0.0 <= delay <= 4.0


def test_provider_retry_after_takes_precedence() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=0.5,
        max_backoff_seconds=5.0,
        jitter=False,
    )

    assert (
        policy.retry_delay(
            retry_number=2,
            retry_after_seconds=20.0,
        )
        == 20.0
    )


def test_backoff_used_when_retry_after_missing() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=0.5,
        max_backoff_seconds=5.0,
        jitter=False,
    )

    assert (
        policy.retry_delay(
            retry_number=2,
            retry_after_seconds=None,
        )
        == 2.0
    )


def test_negative_retry_after_is_not_allowed() -> None:
    policy = RetryPolicy()

    assert (
        policy.retry_delay(
            retry_number=1,
            retry_after_seconds=-10.0,
        )
        == 0.0
    )
