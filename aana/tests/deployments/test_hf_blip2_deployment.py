# ruff: noqa: S101
from importlib import resources

import pytest

from aana.core.models.image import Image
from aana.core.models.types import Dtype
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.hf_blip2_deployment import HFBlip2Config, HFBlip2Deployment
from aana.tests.utils import verify_deployment_results

deployments = [
    (
        "blip2_deployment",
        HFBlip2Deployment.options(
            num_replicas=1,
            max_ongoing_requests=1000,
            ray_actor_options={"num_gpus": 0.25},
            user_config=HFBlip2Config(
                model="Salesforce/blip2-opt-2.7b",
                dtype=Dtype.FLOAT16,
                batch_size=2,
                num_processing_threads=2,
            ).model_dump(mode="json"),
        ),
    )
]


@pytest.mark.parametrize("setup_deployment", deployments, indirect=True)
class TestHFBlip2Deployment:
    """Test HuggingFace BLIP2 deployment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("image_name", ["Starry_Night.jpeg"])
    async def test_methods(self, setup_deployment, image_name):
        """Test HuggingFace BLIP2 methods."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "hf_blip2"
            / f"{deployment_name}_{image_name}.json"
        )

        path = resources.files("aana.tests.files.images") / image_name
        image = Image(path=path, save_on_disk=False, media_id=image_name)

        output = await handle.generate(image=image)
        caption = output["caption"]
        verify_deployment_results(expected_output_path, caption)

        images = [image] * 8

        output = await handle.generate_batch(images=images)
        captions = output["captions"]

        assert len(captions) == 8
        for caption in captions:
            verify_deployment_results(expected_output_path, caption)
