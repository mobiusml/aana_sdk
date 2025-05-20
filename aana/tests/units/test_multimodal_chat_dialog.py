# ruff: noqa: S101

import base64
from importlib import resources

import pytest

from aana.core.models.image import Image
from aana.core.models.multimodal_chat import (
    FrameVideo,
    ImageContent,
    MultimodalChatDialog,
    MultimodalChatMessage,
    TextContent,
    VideoContent,
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


@pytest.fixture
def input_video(input_image):
    """Gets a FrameVideo for test."""
    # Create a FrameVideo with two frames
    return FrameVideo(frames=[input_image, input_image])


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


def test_video_content_creation(input_image):
    """Test that a video content can be created."""
    content = VideoContent(video=FrameVideo(frames=[input_image, input_image]))
    assert content.type == "video"

    # Value error if we tried to init VideoContent with different type attr
    with pytest.raises(ValueError):
        content = VideoContent(video=FrameVideo(image=input_image), type="text")

    # Value error if we tried to set video field with non-video value
    with pytest.raises(ValueError):
        content = VideoContent(video="Video")


def test_multimodal_dialog_creation(input_image, input_video, input_prompt):
    """Test that a image chat dialog can be created."""
    system_messages = MultimodalChatMessage(
        content=[
            ImageContent(image=input_image),
            TextContent(text=input_prompt),
        ],
        role="system",
    )

    user_messages = MultimodalChatMessage(
        content=[
            ImageContent(image=input_image),
            VideoContent(video=input_video),
            TextContent(text=input_prompt),
        ],
        role="user",
    )

    dialog = MultimodalChatDialog(messages=[system_messages, user_messages])

    assert len(dialog.messages) == 2

    for message, role in zip(dialog.messages, ["system", "user"], strict=False):
        assert isinstance(message, MultimodalChatMessage)
        assert message.role == role

        if message.role == "system":
            assert len(message.content) == 2
        else:
            assert len(message.content) == 3

        assert isinstance(message.content[0], ImageContent)
        assert message.content[0].image.path == input_image.path

        if message.role == "user":
            assert isinstance(message.content[1], VideoContent)
            assert message.content[1].video.frames[0].path == input_image.path

            assert isinstance(message.content[2], TextContent)
            assert message.content[2].text == input_prompt
        else:
            assert isinstance(message.content[1], TextContent)
            assert message.content[1].text == input_prompt

    messages, images, videos = dialog.to_objects()
    assert len(messages) == 2
    assert len(images) == 2
    assert len(videos) == 1

    for message, role in zip(messages, ["system", "user"], strict=False):
        assert message["role"] == role

        if message["role"] == "system":
            assert len(message["content"]) == 2
        else:
            assert len(message["content"]) == 3

        assert message["content"][0] == {"type": "image"}

        if message["role"] == "user":
            assert message["content"][1] == {"type": "video"}
            assert message["content"][2] == {"type": "text", "text": input_prompt}
        else:
            assert message["content"][1] == {"type": "text", "text": input_prompt}


def test_image_dialog_creation_from_prompt(input_image, input_video, input_prompt):
    """Test that a image chat dialog can be created from prompt."""
    image_list = [input_image for _ in range(5)]
    video_list = [input_video for _ in range(3)]

    dialog = MultimodalChatDialog.from_prompt(
        input_prompt, images=image_list, videos=video_list
    )

    assert len(dialog.messages) == 1
    message = dialog.messages[0]

    assert isinstance(message, MultimodalChatMessage)
    assert message.role == "user"
    assert len(message.content) == 9

    for i in range(5):
        assert isinstance(message.content[i], ImageContent)
        assert message.content[i].image.path == input_image.path

    for i in range(5, 8):
        assert isinstance(message.content[i], VideoContent)

    assert isinstance(message.content[8], TextContent)
    assert message.content[8].text == input_prompt

    messages, images, videos = dialog.to_objects()
    assert len(messages) == 1
    assert len(images) == 5
    assert len(videos) == 3

    message = messages[0]
    assert len(message["content"]) == 9
    for i in range(5):
        assert message["content"][i] == {"type": "image"}

    for i in range(5, 8):
        assert message["content"][i] == {"type": "video"}

    assert message["content"][8] == {"type": "text", "text": input_prompt}


def test_image_dialog_creation_from_list(input_image, input_video, input_prompt):
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
                {"type": "video", "video": input_video},
                {"type": "text", "text": input_prompt},
            ],
            "role": "user",
        },
    ]

    dialog = MultimodalChatDialog.from_list(messages)

    assert len(dialog.messages) == 2

    for message, role in zip(dialog.messages, ["system", "user"], strict=False):
        assert isinstance(message, MultimodalChatMessage)
        assert message.role == role

        if message.role == "system":
            assert len(message.content) == 2
        else:
            assert len(message.content) == 3

        assert isinstance(message.content[0], ImageContent)
        assert message.content[0].image.path == input_image.path

        if message.role == "user":
            assert isinstance(message.content[1], VideoContent)
            assert message.content[1].video.frames[0].path == input_image.path

            assert isinstance(message.content[2], TextContent)
            assert message.content[2].text == input_prompt
        else:
            assert isinstance(message.content[1], TextContent)
            assert message.content[1].text == input_prompt

    messages, images, videos = dialog.to_objects()
    assert len(messages) == 2
    assert len(images) == 2
    assert len(videos) == 1

    for message, role in zip(messages, ["system", "user"], strict=False):
        assert message["role"] == role
        if message["role"] == "system":
            assert len(message["content"]) == 2
        else:
            assert len(message["content"]) == 3

        assert message["content"][0] == {"type": "image"}

        if message["role"] == "user":
            assert message["content"][1] == {"type": "video"}
            assert message["content"][2] == {"type": "text", "text": input_prompt}
        else:
            assert message["content"][1] == {"type": "text", "text": input_prompt}


def test_frame_video(input_image):
    """Test that a FrameVideo can be created and get_base64_url method works."""
    video = FrameVideo(frames=[input_image, input_image])
    assert len(video.frames) == 2
    assert video.frames[0].path == input_image.path
    assert video.frames[1].path == input_image.path

    # Test get_base64_url method
    base64_url = video.get_base64_url()
    assert base64_url.startswith("data:video/jpeg;base64,")

    images = [
        base64.b64decode(img_base64, validate=True)
        for img_base64 in base64_url.split(",")[1:]
    ]

    assert len(images) == 2
    for img_bytes in images:
        # Check if the image bytes start with the JPEG magic number
        # (0xFF, 0xD8) for JPEG files
        assert img_bytes.startswith(b"\xff\xd8")  # JPEG magic number
