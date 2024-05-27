# ruff: noqa: S101
from importlib import resources

import pytest
from ray import serve

from aana.core.models.image import Image
from aana.tests.utils import (
    compare_texts,
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)


@pytest.fixture(scope="function", params=get_deployments_by_type("HFBlip2Deployment"))
def setup_hf_blip2_deployment(app_setup, request):
    """Setup HF BLIP2 deployment."""
    name, deployment = request.param
    deployments = [
        {
            "name": "blip2_deployment",
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
    "image_name, expected_text",
    [
        (
            "Starry_Night.jpeg",
            "the starry night by vincent van gogh, 1884-1890, oil on canvas, 48 x 48 in, gilded frame, signed and dated",
        )
    ],
)
async def test_hf_blip2_deployments(
    setup_hf_blip2_deployment, image_name, expected_text
):
    """Test HuggingFace BLIP2 deployments."""
    handle = serve.get_app_handle("blip2_deployment")

    path = resources.path("aana.tests.files.images", image_name)
    image = Image(path=path, save_on_disk=False, media_id=image_name)

    output = await handle.generate.remote(image=image)
    caption = output["caption"]
    compare_texts(expected_text, caption)

    images = [image] * 8

    output = await handle.generate_batch.remote(images=images)
    captions = output["captions"]

    assert len(captions) == 8
    for caption in captions:
        compare_texts(expected_text, caption)
