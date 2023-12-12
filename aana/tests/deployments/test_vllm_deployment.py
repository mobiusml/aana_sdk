# ruff: noqa: S101
import pytest
import ray
from ray import serve

from aana.configs.deployments import deployments
from aana.models.pydantic.chat_message import ChatDialog, ChatMessage
from aana.models.pydantic.sampling_params import SamplingParams
from aana.tests.utils import compare_texts, is_gpu_available


def expected_output(name):
    """Gets expected output for a given vLLM version."""
    if name == "vllm_deployment_llama2_7b_chat":
        return (
            "  Elon Musk is a South African-born entrepreneur, inventor, "
            "and business magnate who is best known for his innovative companies in"
        )
    else:
        raise ValueError(f"Unknown deployment name: {name}")  # noqa: TRY003


def ray_setup(deployment):
    """Setup Ray instance for the test."""
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
async def test_vllm_deployments():
    """Test VLLM deployments."""
    for name, deployment in deployments.items():
        # skip if not a VLLM deployment
        if deployment.name != "VLLMDeployment":
            continue

        expected_text = expected_output(name)

        handle = ray_setup(deployment)

        # test generate method
        output = await handle.generate.remote(
            prompt="[INST] Who is Elon Musk? [/INST]",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )
        text = output["text"]

        compare_texts(expected_text, text)

        # test generate_stream method
        stream = handle.options(stream=True).generate_stream.remote(
            prompt="[INST] Who is Elon Musk? [/INST]",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )
        text = ""
        async for chunk in stream:
            chunk = await chunk
            text += chunk["text"]

        compare_texts(expected_text, text)

        # test generate_batch method
        output = await handle.generate_batch.remote(
            prompts=[
                "[INST] Who is Elon Musk? [/INST]",
                "[INST] Who is Elon Musk? [/INST]",
            ],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )
        texts = output["texts"]

        for text in texts:
            compare_texts(expected_text, text)

        # test chat method
        dialog = ChatDialog(
            messages=[
                ChatMessage(
                    role="user",
                    content="Who is Elon Musk?",
                )
            ]
        )
        output = await handle.chat.remote(
            dialog=dialog,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )

        response_message = output["message"]
        assert response_message.role == "assistant"
        text = response_message.content

        compare_texts(expected_text, text)

        # test chat_stream method
        stream = handle.options(stream=True).chat_stream.remote(
            dialog=dialog,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )

        text = ""
        async for chunk in stream:
            chunk = await chunk
            text += chunk["text"]

        compare_texts(expected_text, text)
