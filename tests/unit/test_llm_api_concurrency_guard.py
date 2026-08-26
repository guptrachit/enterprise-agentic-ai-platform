import asyncio

import pytest

from agent_platform.llm.api_guardrails import (
    InFlightRequestGuard,
    LLMCapacityExceededError,
)


def test_concurrency_guard_rejects_invalid_capacity() -> None:
    with pytest.raises(
        ValueError,
        match="max_in_flight must be greater than 0",
    ):
        InFlightRequestGuard(max_in_flight=0)


@pytest.mark.asyncio
async def test_concurrency_guard_tracks_active_request() -> None:
    guard = InFlightRequestGuard(max_in_flight=2)

    assert guard.active_requests == 0
    assert guard.available_capacity == 2

    async with guard.slot():
        assert guard.active_requests == 1
        assert guard.available_capacity == 1

    assert guard.active_requests == 0
    assert guard.available_capacity == 2


@pytest.mark.asyncio
async def test_concurrency_guard_rejects_when_full() -> None:
    guard = InFlightRequestGuard(max_in_flight=1)

    async with guard.slot():
        with pytest.raises(
            LLMCapacityExceededError,
        ):
            async with guard.slot():
                pass

        assert guard.active_requests == 1

    assert guard.active_requests == 0


@pytest.mark.asyncio
async def test_concurrency_guard_releases_after_exception() -> None:
    guard = InFlightRequestGuard(max_in_flight=1)

    with pytest.raises(
        RuntimeError,
        match="boom",
    ):
        async with guard.slot():
            raise RuntimeError("boom")

    assert guard.active_requests == 0

    async with guard.slot():
        assert guard.active_requests == 1


@pytest.mark.asyncio
async def test_concurrency_guard_allows_configured_parallelism() -> None:
    guard = InFlightRequestGuard(max_in_flight=2)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def worker() -> None:
        async with guard.slot():
            if guard.active_requests == 2:
                entered.set()

            await release.wait()

    first = asyncio.create_task(worker())

    second = asyncio.create_task(worker())

    await asyncio.wait_for(
        entered.wait(),
        timeout=1.0,
    )

    assert guard.active_requests == 2
    assert guard.available_capacity == 0

    release.set()

    await asyncio.gather(
        first,
        second,
    )

    assert guard.active_requests == 0


def test_concurrency_guard_snapshot_starts_empty() -> None:
    guard = InFlightRequestGuard(max_in_flight=4)

    snapshot = guard.snapshot()

    assert snapshot.max_in_flight == 4
    assert snapshot.active_requests == 0
    assert snapshot.available_capacity == 4
    assert snapshot.capacity_rejections == 0
    assert snapshot.utilization_rate == 0.0


@pytest.mark.asyncio
async def test_concurrency_guard_snapshot_tracks_utilization() -> None:
    guard = InFlightRequestGuard(max_in_flight=4)

    async with guard.slot():
        snapshot = guard.snapshot()

        assert snapshot.active_requests == 1
        assert snapshot.available_capacity == 3

        assert snapshot.utilization_rate == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_concurrency_guard_counts_rejections() -> None:
    guard = InFlightRequestGuard(max_in_flight=1)

    async with guard.slot():
        with pytest.raises(LLMCapacityExceededError):
            async with guard.slot():
                pass

        snapshot = guard.snapshot()

        assert snapshot.capacity_rejections == 1

    final = guard.snapshot()

    assert final.active_requests == 0
    assert final.capacity_rejections == 1


def test_concurrency_snapshot_to_dict() -> None:
    guard = InFlightRequestGuard(max_in_flight=5)

    assert guard.snapshot().to_dict() == {
        "max_in_flight": 5,
        "active_requests": 0,
        "available_capacity": 5,
        "capacity_rejections": 0,
        "utilization_rate": 0.0,
    }
