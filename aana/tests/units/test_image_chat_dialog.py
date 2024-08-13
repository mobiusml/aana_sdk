# ruff: noqa: S101

from importlib import resources

import pytest

from aana.core.models.image import Image
from aana.core.models.image_chat import (
    ImageChatDialog,
    ImageChatMessage,
    ImageContent,
    TextContent,
)


@pytest.fixture
def input_image():
    """Gets an Image for test."""
    image_path = resources.files("aana.tests.files.images") / "Starry_Night.jpeg"
    return Image(path=image_path)


@pytest.fixture
def input_prompt():
    """Gets a text for text."""
    return "describe images with more info!"


def test_text_content_creation(input_image, input_prompt):
    """Test that a text content can be created."""
    content = TextContent(text=input_prompt)
    assert content.type == "text"
    assert content.text == input_prompt

    # Value error if we tried to init TextContent with different type attr
    with pytest.raises(ValueError):
        content = TextContent(text=input_prompt, type="image")

    # Value error if we tried to set text field with non-str value
    with pytest.raises(ValueError):
        content = TextContent(text=input_image, type="text")


def test_image_content_creation(input_image):
    """Test that a image content can be created."""
    content = ImageContent(image=input_image)
    assert content.type == "image"
    assert content.image.path == input_image.path

    # Value error if we tried to init ImageContent with different type attr
    with pytest.raises(ValueError):
        content = ImageContent(image=input_image, type="text")

    # Value error if we tried to set image field with non-image value
    with pytest.raises(ValueError):
        content = ImageContent(image="Image")


def test_image_dialog_creation(input_image, input_prompt):
    """Test that a image chat dialog can be created."""
    system_messages = ImageChatMessage(
        content=[ImageContent(image=input_image), TextContent(text=input_prompt)],
        role="system",
    )

    user_messages = ImageChatMessage(
        content=[ImageContent(image=input_image), TextContent(text=input_prompt)],
        role="user",
    )

    dialog = ImageChatDialog(messages=[system_messages, user_messages])

    assert len(dialog.messages) == 2

    for message, role in zip(dialog.messages, ["system", "user"], strict=False):
        assert isinstance(message, ImageChatMessage)
        assert message.role == role
        assert len(message.content) == 2

        assert isinstance(message.content[0], ImageContent)
        assert message.content[0].image.path == input_image.path

        assert isinstance(message.content[1], TextContent)
        assert message.content[1].text == input_prompt

    messages, images = dialog.to_objects()
    assert len(messages) == 2
    assert len(images) == 2

    for message, role in zip(messages, ["system", "user"], strict=False):
        assert message["role"] == role
        assert len(message["content"]) == 2
        assert message["content"][0] == {"type": "image"}

        assert message["content"][1] == {"type": "text", "text": input_prompt}


def test_image_dialog_creation_from_prompt(input_image, input_prompt):
    """Test that a image chat dialog can be created from prompt."""
    image_list = [input_image for _ in range(5)]

    dialog = ImageChatDialog.from_prompt(input_prompt, images=image_list)

    assert len(dialog.messages) == 1
    message = dialog.messages[0]

    assert isinstance(message, ImageChatMessage)
    assert message.role == "user"
    assert len(message.content) == 6

    for i in range(5):
        assert isinstance(message.content[i], ImageContent)
        assert message.content[i].image.path == input_image.path

    assert isinstance(message.content[5], TextContent)
    assert message.content[5].text == input_prompt

    messages, images = dialog.to_objects()
    assert len(messages) == 1
    assert len(images) == 5

    message = messages[0]
    assert len(message["content"]) == 6
    for i in range(5):
        assert message["content"][i] == {"type": "image"}

    assert message["content"][5] == {"type": "text", "text": input_prompt}


def test_image_dialog_creation_from_list(input_image, input_prompt):
    """Test that a image chat dialog can be created from prompt."""
    messages = [
        {
            "content": [
                {"type": "image", "image": input_image},
                {"type": "text", "text": input_prompt},
            ],
            "role": "system",
        },
        {
            "content": [
                {"type": "image", "image": input_image},
                {"type": "text", "text": input_prompt},
            ],
            "role": "user",
        },
    ]

    dialog = ImageChatDialog.from_list(messages)

    assert len(dialog.messages) == 2

    for message, role in zip(dialog.messages, ["system", "user"], strict=False):
        assert isinstance(message, ImageChatMessage)
        assert message.role == role
        assert len(message.content) == 2

        assert isinstance(message.content[0], ImageContent)
        assert message.content[0].image.path == input_image.path

        assert isinstance(message.content[1], TextContent)
        assert message.content[1].text == input_prompt

    messages, images = dialog.to_objects()
    assert len(messages) == 2
    assert len(images) == 2

    for message, role in zip(messages, ["system", "user"], strict=False):
        assert message["role"] == role
        assert len(message["content"]) == 2
        assert message["content"][0] == {"type": "image"}

        assert message["content"][1] == {"type": "text", "text": input_prompt}
