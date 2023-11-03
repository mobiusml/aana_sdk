from importlib import resources
import random
import pytest
import ray
from ray import serve

from aana.configs.deployments import deployments
from aana.models.core.image import Image
from aana.tests.utils import compare_texts, is_gpu_available


def ray_setup(deployment):
    # Setup ray environment and serve
    ray.init(ignore_reinit_error=True)
    app = deployment.bind()
    # random port from 30000 to 40000
    port = random.randint(30000, 40000)
    handle = serve.run(app, port=port)
    return handle


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.asyncio
async def test_hf_blip2_deployments():
    for name, deployment in deployments.items():
        # skip if not a VLLM deployment
        if deployment.name != "HFBlip2Deployment":
            continue

        handle = ray_setup(deployment)

        path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
        image = Image(path=path, save_on_disc=False)

        images = [image] * 8

        output = await handle.generate_captions.remote(images=images)
        captions = output["captions"]

        assert len(captions) == 8
        for caption in captions:
            compare_texts("the starry night by vincent van gogh", caption)
