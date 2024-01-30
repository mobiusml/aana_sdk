# ruff: noqa: S101
from importlib import resources

import pytest

from aana.models.core.image import Image
from aana.tests.utils import (
    compare_texts,
    get_deployments_by_type,
    is_gpu_available,
)


@pytest.fixture(scope="function", params=get_deployments_by_type("HFBlip2Deployment"))
def setup_hf_blip2_deployment(setup_deployment, request):
    """Setup HF BLIP2 deployment."""
    name, deployment = request.param
    binded_deployment = deployment.bind()
    return name, deployment, *setup_deployment(binded_deployment)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_name, expected_text",
    [("Starry_Night.jpeg", "the starry night by vincent van gogh")],
)
async def test_hf_blip2_deployments(
    setup_hf_blip2_deployment, image_name, expected_text
):
    """Test HuggingFace BLIP2 deployments."""
    name, deployment, handle, port, route_prefix = setup_hf_blip2_deployment

    path = resources.path("aana.tests.files.images", image_name)
    image = Image(path=path, save_on_disk=False)

    output = await handle.generate.remote(image=image)
    caption = output["caption"]
    compare_texts(expected_text, caption)

    images = [image] * 8

    output = await handle.generate_batch.remote(images=images)
    captions = output["captions"]

    assert len(captions) == 8
    for caption in captions:
        compare_texts(expected_text, caption)
