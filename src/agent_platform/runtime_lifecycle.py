import inspect
from typing import Any


async def close_runtime_resource(
    resource: Any,
) -> None:
    """Close one runtime resource when it exposes a close method."""

    close = getattr(
        resource,
        "close",
        None,
    )

    if close is None:
        return

    result = close()

    if inspect.isawaitable(result):
        await result
