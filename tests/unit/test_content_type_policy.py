from unittest.mock import Mock

import pytest

from agent_platform.security.content_type_policy import (
    JSONContentTypePolicy,
    UnsupportedMediaTypeError,
)


def create_request(
    content_type: str | None,
):
    request = Mock()

    headers = {}

    if content_type is not None:
        headers["Content-Type"] = content_type

    request.headers = headers

    return request


def test_accepts_application_json() -> None:
    policy = JSONContentTypePolicy()

    policy.enforce(create_request("application/json"))


def test_accepts_application_json_with_charset() -> None:
    policy = JSONContentTypePolicy()

    policy.enforce(create_request("application/json; charset=utf-8"))


def test_rejects_missing_content_type() -> None:
    policy = JSONContentTypePolicy()

    with pytest.raises(
        UnsupportedMediaTypeError,
        match="application/json",
    ):
        policy.enforce(create_request(None))


def test_rejects_text_plain() -> None:
    policy = JSONContentTypePolicy()

    with pytest.raises(
        UnsupportedMediaTypeError,
    ):
        policy.enforce(create_request("text/plain"))


def test_rejects_form_encoded_content() -> None:
    policy = JSONContentTypePolicy()

    with pytest.raises(
        UnsupportedMediaTypeError,
    ):
        policy.enforce(create_request("application/x-www-form-urlencoded"))
