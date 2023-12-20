# ruff: noqa: S101
import pytest
from jinja2 import TemplateError
from transformers import AutoTokenizer

from aana.models.pydantic.chat_message import ChatDialog, ChatMessage
from aana.utils.chat_template import apply_chat_template

dialog = ChatDialog(
    messages=[
        ChatMessage(
            role="system",
            content="You are a friendly chatbot who always responds in the style of a pirate",
        ),
        ChatMessage(
            role="user",
            content="How many helicopters can a human eat in one sitting?",
        ),
        ChatMessage(role="assistant", content="I don't know, how many?"),
        ChatMessage(role="user", content="One, but only if they're really hungry!"),
    ]
)


def test_chat_template_blank():
    """Test that applying a blank chat template to a dialog fails."""
    checkpoint = "NousResearch/Llama-2-7b-hf"  # Doesn't have a chat template
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    with pytest.raises(ValueError):
        apply_chat_template(tokenizer, dialog)


def test_chat_template_custom():
    """Test that applying a custom chat template to a dialog succeeds."""
    checkpoint = "NousResearch/Llama-2-7b-hf"  # Doesn't have a chat template
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    prompt = apply_chat_template(
        tokenizer, dialog, "llama2"
    )  # Apply custom chat template "llama2"
    assert prompt == (
        "<s>[INST] <<SYS>>\\nYou are a friendly chatbot who always responds in the style "
        "of a pirate\\n<</SYS>>\\n\\nHow many helicopters can a human eat in one sitting? "
        "[/INST] I don't know, how many? </s><s>[INST] One, but only if they're really hungry! [/INST]"
    )


def test_chat_messages_without_system():
    """Test that applying a custom chat template to a dialog without a system message succeeds."""
    dialog = ChatDialog(
        messages=[
            ChatMessage(
                role="user",
                content="How many helicopters can a human eat in one sitting?",
            ),
            ChatMessage(role="assistant", content="I don't know, how many?"),
            ChatMessage(role="user", content="One, but only if they're really hungry!"),
        ]
    )

    checkpoint = "NousResearch/Llama-2-7b-hf"  # Doesn't have a chat template
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    prompt = apply_chat_template(
        tokenizer, dialog, "llama2"
    )  # Apply custom chat template "llama2"

    assert prompt == (
        "<s>[INST] How many helicopters can a human eat in one sitting? [/INST] I don't know, how many? </s>"
        "<s>[INST] One, but only if they're really hungry! [/INST]"
    )


def test_chat_messages_wrong_role():
    """Test that applying a custom chat template to a dialog with assistant message first fails."""
    dialog = ChatDialog(
        messages=[
            ChatMessage(
                role="assistant",
                content="How many helicopters can a human eat in one sitting?",
            ),
            ChatMessage(role="user", content="I don't know, how many?"),
            ChatMessage(
                role="assistant", content="One, but only if they're really hungry!"
            ),
        ]
    )

    checkpoint = "NousResearch/Llama-2-7b-hf"  # Doesn't have a chat template
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    with pytest.raises(TemplateError):
        apply_chat_template(
            tokenizer, dialog, "llama2"
        )  # Apply custom chat template "llama2"
