import pytest

from agent_platform.llm.api_rate_limit import (
    LLMAPIRateLimiter,
    LLMRateLimitExceededError,
)


def test_rate_limiter_rejects_invalid_request_limit() -> None:
    with pytest.raises(
        ValueError,
        match="max_requests must be greater than 0",
    ):
        LLMAPIRateLimiter(
            max_requests=0,
            window_seconds=60.0,
        )


def test_rate_limiter_rejects_invalid_window() -> None:
    with pytest.raises(
        ValueError,
        match="window_seconds must be greater than 0",
    ):
        LLMAPIRateLimiter(
            max_requests=10,
            window_seconds=0.0,
        )


@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_within_limit() -> None:
    limiter = LLMAPIRateLimiter(
        max_requests=2,
        window_seconds=60.0,
    )

    await limiter.check("client-1")

    await limiter.check("client-1")


@pytest.mark.asyncio
async def test_rate_limiter_rejects_request_over_limit() -> None:
    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    await limiter.check("client-1")

    with pytest.raises(LLMRateLimitExceededError):
        await limiter.check("client-1")

    assert limiter.snapshot().rejected_requests == 1


@pytest.mark.asyncio
async def test_rate_limiter_tracks_callers_independently() -> None:
    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
    )

    await limiter.check("client-1")

    await limiter.check("client-2")

    assert limiter.snapshot().tracked_callers == 2


@pytest.mark.asyncio
async def test_rate_limiter_resets_after_window() -> None:
    now = [100.0]

    def clock() -> float:
        return now[0]

    limiter = LLMAPIRateLimiter(
        max_requests=1,
        window_seconds=60.0,
        clock=clock,
    )

    await limiter.check("client-1")

    now[0] = 160.0

    await limiter.check("client-1")


def test_rate_limiter_snapshot_to_dict() -> None:
    limiter = LLMAPIRateLimiter(
        max_requests=10,
        window_seconds=60.0,
    )

    assert limiter.snapshot().to_dict() == {
        "max_requests": 10,
        "window_seconds": 60.0,
        "tracked_callers": 0,
        "rejected_requests": 0,
    }
