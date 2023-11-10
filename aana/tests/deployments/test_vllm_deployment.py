import random
import pytest
import ray
from ray import serve

from aana.configs.deployments import deployments
from aana.models.pydantic.sampling_params import SamplingParams
from aana.tests.utils import compare_texts, is_gpu_available


def expected_output(name):
    if name == "vllm_deployment_llama2_7b_chat":
        return (
            "  Elon Musk is a South African-born entrepreneur, inventor, and business magnate. "
            "He is best known for his revolutionary ideas"
        )
    if name == "vllm_deployment_zephyr_7b_beta":
        return (
            "\n\nElon Musk is an entrepreneur, business magnate, and investor. "
            "He is the founder, CEO, and Chief Designer of SpaceX"
        )
    else:
        raise ValueError(f"Unknown deployment name: {name}")


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
async def test_vllm_deployments():
    for name, deployment in deployments.items():
        # skip if not a VLLM deployment
        if deployment.name != "VLLMDeployment":
            continue

        handle = ray_setup(deployment)

        # test generate method
        output = await handle.generate.remote(
            prompt="[INST] Who is Elon Musk? [/INST]",
            sampling_params=SamplingParams(temperature=1.0, max_tokens=32),
        )
        text = output["text"]
        expected_text = expected_output(name)
        compare_texts(expected_text, text)

        # test generate_stream method
        stream = handle.options(stream=True).generate_stream.remote(
            prompt="[INST] Who is Elon Musk? [/INST]",
            sampling_params=SamplingParams(temperature=1.0, max_tokens=32),
        )
        text = ""
        async for chunk in stream:
            chunk = await chunk
            text += chunk["text"]
        expected_text = expected_output(name)
        compare_texts(expected_text, text)

        # test generate_batch method
        output = await handle.generate_batch.remote(
            prompts=[
                "[INST] Who is Elon Musk? [/INST]",
                "[INST] Who is Elon Musk? [/INST]",
            ],
            sampling_params=SamplingParams(temperature=1.0, max_tokens=32),
        )
        texts = output["texts"]
        expected_text = expected_output(name)

        for text in texts:
            compare_texts(expected_text, text)
