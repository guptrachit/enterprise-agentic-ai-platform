from unittest.mock import Mock

from agent_platform.llm.api_client_identity import (
    ANONYMOUS_CLIENT_ID,
    resolve_client_identity,
)


def create_request(
    *,
    host: str | None,
    forwarded_for: str | None = None,
):
    request = Mock()

    if host is None:
        request.client = None
    else:
        request.client = Mock()
        request.client.host = host

    headers = {}

    if forwarded_for is not None:
        headers["X-Forwarded-For"] = forwarded_for

    request.headers = headers

    return request


def test_client_identity_uses_direct_host() -> None:
    request = create_request(host="127.0.0.1")

    assert resolve_client_identity(request) == "127.0.0.1"


def test_client_identity_falls_back_when_client_missing() -> None:
    request = create_request(host=None)

    assert resolve_client_identity(request) == ANONYMOUS_CLIENT_ID


def test_client_identity_ignores_forwarded_for_from_untrusted_host() -> None:
    request = create_request(
        host="203.0.113.10",
        forwarded_for="198.51.100.25",
    )

    assert (
        resolve_client_identity(
            request,
            trusted_proxy_hosts=("10.0.0.10",),
        )
        == "203.0.113.10"
    )


def test_client_identity_uses_forwarded_for_from_trusted_proxy() -> None:
    request = create_request(
        host="10.0.0.10",
        forwarded_for="198.51.100.25",
    )

    assert (
        resolve_client_identity(
            request,
            trusted_proxy_hosts=("10.0.0.10",),
        )
        == "198.51.100.25"
    )


def test_client_identity_uses_first_forwarded_address() -> None:
    request = create_request(
        host="10.0.0.10",
        forwarded_for=("198.51.100.25, 10.0.0.20"),
    )

    assert (
        resolve_client_identity(
            request,
            trusted_proxy_hosts=("10.0.0.10",),
        )
        == "198.51.100.25"
    )


def test_client_identity_uses_direct_host_when_forwarded_header_empty() -> None:
    request = create_request(
        host="10.0.0.10",
        forwarded_for="   ",
    )

    assert (
        resolve_client_identity(
            request,
            trusted_proxy_hosts=("10.0.0.10",),
        )
        == "10.0.0.10"
    )
