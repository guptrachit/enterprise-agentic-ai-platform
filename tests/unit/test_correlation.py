from uuid import UUID

from agent_platform.correlation import (
    CORRELATION_ID_HEADER,
    create_correlation_id,
)


def test_correlation_header_name() -> None:
    assert CORRELATION_ID_HEADER == ("X-Correlation-ID")


def test_create_correlation_id_returns_uuid() -> None:
    correlation_id = create_correlation_id()

    parsed = UUID(correlation_id)

    assert str(parsed) == correlation_id


def test_create_correlation_id_is_unique() -> None:
    first = create_correlation_id()
    second = create_correlation_id()

    assert first != second
