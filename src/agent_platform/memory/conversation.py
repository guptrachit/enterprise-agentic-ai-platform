from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ConversationRole(StrEnum):
    """Supported roles for governed conversation messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationMessage(BaseModel):
    """Immutable message stored in governed conversation context."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    message_id: str = Field(min_length=1)
    role: ConversationRole
    content: str = Field(min_length=1)
    source: str | None = None


class ConversationContext(BaseModel):
    """Immutable ordered conversation state for a single conversation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    conversation_id: str = Field(min_length=1)
    messages: tuple[ConversationMessage, ...] = ()

    def append(
        self,
        message: ConversationMessage,
    ) -> "ConversationContext":
        """Return a new context containing the appended message."""
        return self.model_copy(
            update={
                "messages": (
                    *self.messages,
                    message,
                )
            }
        )

    def latest(
        self,
        *,
        count: int,
    ) -> tuple[ConversationMessage, ...]:
        """Return the most recent messages in chronological order."""
        if count < 1:
            raise ValueError("count must be at least 1")

        return self.messages[-count:]

    def messages_by_role(
        self,
        role: ConversationRole,
    ) -> tuple[ConversationMessage, ...]:
        """Return all messages matching the requested role."""
        return tuple(message for message in self.messages if message.role is role)
