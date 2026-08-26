from dataclasses import dataclass

from fastapi import Request


class UnsupportedMediaTypeError(ValueError):
    """Raised when an API request uses an unsupported media type."""


@dataclass(frozen=True)
class JSONContentTypePolicy:
    """Require JSON content for protected API requests."""

    allowed_media_type: str = "application/json"

    def enforce(
        self,
        request: Request,
    ) -> None:
        """Require a JSON Content-Type header."""

        content_type = request.headers.get("Content-Type")

        if content_type is None:
            raise UnsupportedMediaTypeError("Content-Type must be application/json.")

        media_type = (
            content_type.split(
                ";",
                maxsplit=1,
            )[0]
            .strip()
            .lower()
        )

        if media_type != self.allowed_media_type:
            raise UnsupportedMediaTypeError("Content-Type must be application/json.")
