# ruff: noqa: S101
from importlib import resources

import pytest

from aana.models.core.image import Image
from aana.tests.utils import (
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)


@pytest.fixture(
    scope="function", params=get_deployments_by_type("StandardConceptsV2Deployment")
)
def setup_std_concepts_v2_deployment(setup_deployment, request):
    """Setup StandardConceptsV2 deployment."""
    name, deployment = request.param
    return name, deployment, *setup_deployment(deployment, bind=True)


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_name",
    ["Starry_Night2.jpg"],
)
async def test_std_concepts_v2_deployment_smoke(
    image_name, setup_std_concepts_v2_deployment
):
    """Tests standard concepts v2 deployment."""
    name, deployment, handle, port, route_prefix = setup_std_concepts_v2_deployment
    path = resources.path("aana.tests.files.images", image_name)
    image = Image(path=path, save_on_disk=False, media_id=image_name)

    output = await handle.generate.remote(image=image)
    print(output)
    assert output is not None
