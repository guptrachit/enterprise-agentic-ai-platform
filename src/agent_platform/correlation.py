from uuid import uuid4

CORRELATION_ID_HEADER = "X-Correlation-ID"


def create_correlation_id() -> str:
    """Create a new request correlation identifier."""

    return str(uuid4())
