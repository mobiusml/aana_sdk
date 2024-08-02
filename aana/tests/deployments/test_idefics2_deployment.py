# ruff: noqa: S101

from pathlib import Path

import pytest
from ray import serve

from aana.core.models.chat import Prompt
from aana.core.models.image import Image
from aana.core.models.image_chat import ImageChatDialog
from aana.tests.utils import (
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)


@pytest.fixture(
    scope="function", params=get_deployments_by_type("idefics_2_deployment")
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
    "prompt, image_path",
    [("Mona Lisa but from Picasso's Blue Period", "aana/tests/files/images/Starry_Night.jpeg")],
)
async def test_idefics2_deployment(setup_deployment, prompt, image_path):
    """Test Idefics 2 deployments."""
    handle = serve.get_app_handle("idefics_2_deployment")

    dialog = ImageChatDialog.from_prompt(prompt=prompt, images=[Image(path=Path(image_path), save_on_disk=False)])
    output = await handle.chat_batch.remote()
