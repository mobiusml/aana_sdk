# ruff: noqa: S101

from pathlib import Path

import pytest
from ray import serve

from aana.core.models.chat import ChatMessage, Prompt
from aana.core.models.image import Image
from aana.core.models.image_chat import ImageChatDialog
from aana.tests.utils import (
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)


@pytest.fixture(
    scope="function", params=get_deployments_by_type("Idefics2Deployment")
)
def setup_deployment(app_setup, request):
    """Setup Idefics 2 deployment."""
    name, deployment = request.param
    deployments = [
        {
            "name": "idefics_2_deployment",
            "instance": deployment,
        }
    ]
    endpoints = []

    return name, deployment, app_setup(deployments, endpoints)

@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, image_path, expected_output",
    [("Who is the painter of the image?", "aana/tests/files/images/Starry_Night.jpeg", "Van gogh.")],
)
async def test_idefics2_deployment_chat(setup_deployment, prompt, image_path, expected_output):
    """Test Idefics 2 deployments."""
    handle = serve.get_app_handle("idefics_2_deployment")
    image = Image(path=Path(image_path), save_on_disk=False, media_id="test_image")
    dialog = ImageChatDialog.from_prompt(prompt=prompt, images=[image])
    output = await handle.chat.remote(dialog=dialog)
    output_message = output["message"]

    assert type(output_message) == ChatMessage
    assert output_message.content == expected_output
    assert output_message.role == "assistant"


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, image_path, expected_output",
    [("Who is the painter of the image?", "aana/tests/files/images/Starry_Night.jpeg", "Van gogh.")],
)
async def test_idefics2_deployment_chat_stream(setup_deployment, prompt, image_path, expected_output):
    """Test Idefics 2 deployments chat stream."""
    handle = serve.get_app_handle("idefics_2_deployment")
    image = Image(path=Path(image_path), save_on_disk=False, media_id="test_image")
    dialog = ImageChatDialog.from_prompt(prompt=prompt, images=[image])
    output_message = ""
    stream = handle.options(stream=True).chat_stream.remote(dialog=dialog)
    async for chunk in stream:
        output_message += chunk["text"]

    assert output_message == expected_output


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, image_path, expected_output",
    [("Who is the painter of the image?", "aana/tests/files/images/Starry_Night.jpeg", "Van gogh.")],
)
async def test_idefics2_deployment_chat_batch(setup_deployment, prompt, image_path, expected_output):
    """Test Idefics 2 deployments in batch."""
    handle = serve.get_app_handle("idefics_2_deployment")
    image = Image(path=Path(image_path), save_on_disk=False, media_id="test_image")
    dialogs = [ImageChatDialog.from_prompt(prompt=prompt, images=[image]) for _ in range(10)]
    outputs = await handle.chat_batch.remote(dialogs=dialogs)
    for output in outputs:
        output_message = output["message"]

        assert type(output_message) == ChatMessage
        assert output_message.content == expected_output
        assert output_message.role == "assistant"
