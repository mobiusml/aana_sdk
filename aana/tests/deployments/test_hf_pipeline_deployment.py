# ruff: noqa: S101
from importlib import resources

import pytest
from transformers import BitsAndBytesConfig

from aana.core.models.image import Image
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.hf_pipeline_deployment import (
    HfPipelineConfig,
    HfPipelineDeployment,
)
from aana.tests.utils import (
    is_gpu_available,
    verify_deployment_results,
)

deployments = [
    (
        "hf_pipeline_blip2_deployment",
        HfPipelineDeployment.options(
            num_replicas=1,
            ray_actor_options={"num_gpus": 1},
            user_config=HfPipelineConfig(
                model_id="Salesforce/blip2-opt-2.7b",
                model_kwargs={
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=False, load_in_4bit=True
                    ),
                },
            ).model_dump(mode="json"),
        ),
    )
]


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_name, deployment", deployments)
@pytest.mark.parametrize("image_name", ["Starry_Night.jpeg"])
async def test_hf_pipeline_deployments(
    setup_deployment, deployment_name, deployment, image_name
):
    """Test HuggingFace Pipeline deployments."""
    setup_deployment(deployment_name, deployment)

    handle = await AanaDeploymentHandle.create(deployment_name)

    expected_output_path = (
        resources.path("aana.tests.files.expected", "")
        / "hf_pipeline"
        / f"{deployment_name}_{image_name}.json"
    )
    path = resources.path("aana.tests.files.images", image_name)
    image = Image(path=path, save_on_disk=False, media_id=image_name)

    output = await handle.call(images=image)
    verify_deployment_results(expected_output_path, output)

    output = await handle.call(image)
    verify_deployment_results(expected_output_path, output)

    output = await handle.call(images=[str(path)])
    verify_deployment_results(expected_output_path, [output])

    output = await handle.call(images=[image])
    verify_deployment_results(expected_output_path, [output])

    output = await handle.call([image])
    verify_deployment_results(expected_output_path, [output])
