from fastapi import Request

ANONYMOUS_CLIENT_ID = "anonymous"
FORWARDED_FOR_HEADER = "X-Forwarded-For"


def resolve_client_identity(
    request: Request,
    *,
    trusted_proxy_hosts: tuple[str, ...] = (),
) -> str:
    """Resolve a server-controlled client identity for rate limiting."""

    if request.client is None:
        return ANONYMOUS_CLIENT_ID

    direct_host = request.client.host.strip()

    if not direct_host:
        return ANONYMOUS_CLIENT_ID

    if direct_host not in trusted_proxy_hosts:
        return direct_host

    forwarded_for = request.headers.get(FORWARDED_FOR_HEADER)

    if forwarded_for is None:
        return direct_host

    first_hop = forwarded_for.split(
        ",",
        maxsplit=1,
    )[0].strip()

    if not first_hop:
        return direct_host

    return first_hop
