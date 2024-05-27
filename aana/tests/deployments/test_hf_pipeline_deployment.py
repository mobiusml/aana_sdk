# ruff: noqa: S101
from importlib import resources

import pytest
from ray import serve

from aana.core.models.image import Image
from aana.tests.utils import (
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)


def get_expected_output(name):
    """Gets expected output for a given deployment name."""
    if name == "hf_blip2_opt_2_7b_pipeline_deployment":
        return [[{"generated_text": "the starry night by van gogh\n"}]]
    else:
        raise ValueError(f"Unknown deployment name: {name}")  # noqa: TRY003


@pytest.fixture(
    scope="function", params=get_deployments_by_type("HfPipelineDeployment")
)
def setup_hf_pipeline_deployment(app_setup, request):
    """Setup HF Pipeline deployment."""
    name, deployment = request.param
    deployments = [
        {
            "name": "hf_pipeline_deployment",
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
    "image_name",
    ["Starry_Night.jpeg"],
)
async def test_hf_pipeline_deployments(setup_hf_pipeline_deployment, image_name):
    """Test HuggingFace Pipeline deployments."""
    name, deployment, app = setup_hf_pipeline_deployment

    handle = serve.get_app_handle("hf_pipeline_deployment")

    expected_output = get_expected_output(name)

    path = resources.path("aana.tests.files.images", image_name)
    output = await handle.call.remote(images=[str(path)])
    assert output == expected_output

    image = Image(path=path, save_on_disk=False, media_id=image_name)

    output = await handle.call.remote(images=[image])
    assert output == expected_output

    output = await handle.call.remote([image])
    assert output == expected_output

    output = await handle.call.remote(images=image)
    assert output == expected_output[0]

    output = await handle.call.remote(image)
    assert output == expected_output[0]
