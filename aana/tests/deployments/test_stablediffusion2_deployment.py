# ruff: noqa: S101

import pytest

from aana.models.pydantic.prompt import Prompt
from aana.tests.utils import (
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)


@pytest.fixture(
    scope="function", params=get_deployments_by_type("StableDiffusion2Deployment")
)
def setup_deployment(setup_deployment, request):
    """Setup Stable Diffusion 2 deployment."""
    name, deployment = request.param
    return name, deployment, *setup_deployment(deployment, bind=True)


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    ["Mona Lisa but from Picasso's Blue Period"],
)
async def test_stablediffusion2_deployment(setup_deployment, prompt):
    """Test HuggingFace BLIP2 deployments."""
    name, deployment, handle, port, route_prefix = setup_deployment

    output = await handle.generate.remote(prompt=Prompt(prompt))

    image = output["image"]

    assert image is not None
    assert image.size == (768, 768)
    assert image.mode == "RGB"
