from unittest.mock import AsyncMock, Mock

import pytest

from agent_platform.runtime_lifecycle import (
    close_runtime_resource,
)


@pytest.mark.asyncio
async def test_close_runtime_resource_ignores_resource_without_close() -> None:
    resource = object()

    await close_runtime_resource(resource)


@pytest.mark.asyncio
async def test_close_runtime_resource_calls_sync_close() -> None:
    resource = Mock()

    resource.close.return_value = None

    await close_runtime_resource(resource)

    resource.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_close_runtime_resource_awaits_async_close() -> None:
    resource = Mock()

    resource.close = AsyncMock()

    await close_runtime_resource(resource)

    resource.close.assert_awaited_once_with()
