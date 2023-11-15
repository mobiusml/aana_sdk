from importlib import resources
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
    port = 34422
    test_name = deployment.name
    route_prefix = f"/{test_name}"
    handle = serve.run(app, port=port, name=test_name, route_prefix=route_prefix)
    return handle


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_name, expected_text",
    [("Starry_Night.jpeg", "the starry night by vincent van gogh")],
)
async def test_hf_blip2_deployments(image_name, expected_text):
    for name, deployment in deployments.items():
        # skip if not a VLLM deployment
        if deployment.name != "HFBlip2Deployment":
            continue

        handle = ray_setup(deployment)

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
