import pytest
from pydantic import ValidationError

from agent_platform.memory import (
    ConversationContext,
    ConversationMessage,
    ConversationRole,
)


def create_message(
    *,
    message_id: str,
    role: ConversationRole,
    content: str,
) -> ConversationMessage:
    return ConversationMessage(
        message_id=message_id,
        role=role,
        content=content,
    )


def test_conversation_message_is_immutable() -> None:
    message = create_message(
        message_id="msg-001",
        role=ConversationRole.USER,
        content="Show me the latest order status.",
    )

    with pytest.raises(ValidationError):
        message.content = "Changed"


def test_context_append_returns_new_context() -> None:
    original = ConversationContext(
        conversation_id="conversation-001",
    )

    message = create_message(
        message_id="msg-001",
        role=ConversationRole.USER,
        content="Hello",
    )

    updated = original.append(message)

    assert original.messages == ()
    assert updated.messages == (message,)
    assert updated is not original


def test_context_preserves_message_order() -> None:
    context = ConversationContext(
        conversation_id="conversation-001",
    )

    first = create_message(
        message_id="msg-001",
        role=ConversationRole.USER,
        content="What is the status?",
    )

    second = create_message(
        message_id="msg-002",
        role=ConversationRole.ASSISTANT,
        content="The order is processing.",
    )

    context = context.append(first)
    context = context.append(second)

    assert context.messages == (
        first,
        second,
    )


def test_latest_returns_recent_messages_in_order() -> None:
    context = ConversationContext(
        conversation_id="conversation-001",
        messages=(
            create_message(
                message_id="msg-001",
                role=ConversationRole.SYSTEM,
                content="System instructions",
            ),
            create_message(
                message_id="msg-002",
                role=ConversationRole.USER,
                content="Question one",
            ),
            create_message(
                message_id="msg-003",
                role=ConversationRole.ASSISTANT,
                content="Answer one",
            ),
        ),
    )

    latest = context.latest(
        count=2,
    )

    assert tuple(message.message_id for message in latest) == (
        "msg-002",
        "msg-003",
    )


def test_latest_requires_positive_count() -> None:
    context = ConversationContext(
        conversation_id="conversation-001",
    )

    with pytest.raises(
        ValueError,
        match="count must be at least 1",
    ):
        context.latest(
            count=0,
        )


def test_messages_can_be_filtered_by_role() -> None:
    context = ConversationContext(
        conversation_id="conversation-001",
        messages=(
            create_message(
                message_id="msg-001",
                role=ConversationRole.USER,
                content="First question",
            ),
            create_message(
                message_id="msg-002",
                role=ConversationRole.ASSISTANT,
                content="First answer",
            ),
            create_message(
                message_id="msg-003",
                role=ConversationRole.USER,
                content="Second question",
            ),
        ),
    )

    user_messages = context.messages_by_role(
        ConversationRole.USER,
    )

    assert tuple(message.message_id for message in user_messages) == (
        "msg-001",
        "msg-003",
    )


def test_empty_content_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ConversationMessage(
            message_id="msg-001",
            role=ConversationRole.USER,
            content="",
        )
