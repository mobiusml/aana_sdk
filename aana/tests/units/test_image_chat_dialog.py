# ruff: noqa: S101

from pathlib import Path

import pytest

from aana.core.models.image import Image
from aana.core.models.image_chat import (
    ImageChatDialog,
    ImageChatMessage,
    ImageContent,
    TextContent,
)

IMAGE_PATH = Path("aana/tests/files/images/Starry_Night.jpeg")
PROMPT = "describe images with more info!"

def test_text_content_creation():
    """Test that a text content can be created."""
    content = TextContent(text=PROMPT)
    assert content.type == "text"
    assert content.text == PROMPT

    with pytest.raises(ValueError):
        content = TextContent(text=PROMPT, type="image")

    with pytest.raises(ValueError):
        content = TextContent(text=Image(path=IMAGE_PATH), type="text")


def test_image_content_creation():
    """Test that a image content can be created."""
    content = ImageContent(image=Image(path=IMAGE_PATH, save_on_disk=False))
    assert content.type == "image"
    assert content.image.path == IMAGE_PATH

    with pytest.raises(ValueError):
        content = ImageContent(image=Image(path=IMAGE_PATH), type="text")
    
    with pytest.raises(ValueError):
        content = ImageContent(image="Image")

def test_image_dialog_creation():
    """Test that a image chat dialog can be created."""
    system_messages = ImageChatMessage(content=[
        ImageContent(image=Image(path=IMAGE_PATH, save_on_disk=False)),
        TextContent(text=PROMPT)
    ], role="system")

    user_messages = ImageChatMessage(content=[
        ImageContent(image=Image(path=IMAGE_PATH, save_on_disk=False)),
        TextContent(text=PROMPT)
    ], role="user")

    dialog = ImageChatDialog(messages=[system_messages, user_messages])

    assert len(dialog.messages) == 2

    for message, role in zip(dialog.messages, ["system", "user"], strict=False):
        assert type(message) == ImageChatMessage
        assert message.role == role
        assert len(message.content) == 2

        assert type(message.content[0]) == ImageContent
        assert message.content[0].image.path == IMAGE_PATH

        assert type(message.content[1]) == TextContent
        assert message.content[1].text == PROMPT
    
    messages, images = dialog.to_objects()
    assert len(messages) == 2
    assert len(images) == 2

    for message, role in zip(messages, ["system", "user"], strict=False):
        assert message["role"] == role
        assert len(message["content"]) == 2
        assert message["content"][0] == {"type": "image"}

        assert message["content"][1] == {"type": "text", "text": PROMPT}


def test_image_dialog_creation_from_prompt():
    """Test that a image chat dialog can be created from prompt."""
    image_list = [Image(path=IMAGE_PATH, save_on_disk=False) for _ in range(5)]

    dialog = ImageChatDialog.from_prompt(PROMPT, images=image_list)

    assert len(dialog.messages) == 1
    message = dialog.messages[0]

    assert type(message) == ImageChatMessage
    assert message.role == "user"
    assert len(message.content) == 6

    for i in range(5):
        assert type(message.content[i]) == ImageContent
        assert message.content[i].image.path == IMAGE_PATH

    assert type(message.content[5]) == TextContent
    assert message.content[5].text == PROMPT

    messages, images = dialog.to_objects()
    assert len(messages) == 1
    assert len(images) == 5

    message = messages[0]
    assert len(message["content"]) == 6
    for i in range(5):
        assert message["content"][i] == {"type": "image"}

    assert message["content"][5] == {"type": "text", "text": PROMPT}


def test_image_dialog_creation_from_list():
    """Test that a image chat dialog can be created from prompt."""
    messages = [
        {
            "content": [
                {
                    "type": "image",
                    "image": Image(path=IMAGE_PATH, save_on_disk=False)
                },
                {
                    "type": "text",
                    "text": PROMPT
                }
            ],
            "role": "system"
        },
        {
            "content": [
                {
                    "type": "image",
                    "image": Image(path=IMAGE_PATH, save_on_disk=False)
                },
                {
                    "type": "text",
                    "text": PROMPT
                }
            ],
            "role": "user"
        }
    ]

    dialog = ImageChatDialog.from_list(messages)

    assert len(dialog.messages) == 2

    for message, role in zip(dialog.messages, ["system", "user"], strict=False):
        assert type(message) == ImageChatMessage
        assert message.role == role
        assert len(message.content) == 2

        assert type(message.content[0]) == ImageContent
        assert message.content[0].image.path == IMAGE_PATH

        assert type(message.content[1]) == TextContent
        assert message.content[1].text == PROMPT
    
    messages, images = dialog.to_objects()
    assert len(messages) == 2
    assert len(images) == 2

    for message, role in zip(messages, ["system", "user"], strict=False):
        assert message["role"] == role
        assert len(message["content"]) == 2
        assert message["content"][0] == {"type": "image"}

        assert message["content"][1] == {"type": "text", "text": PROMPT}