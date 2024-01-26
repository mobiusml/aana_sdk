# ruff: noqa: S101
import pytest

from aana.configs.deployments import deployments
from aana.exceptions.general import PromptTooLongException
from aana.models.pydantic.chat_message import ChatDialog, ChatMessage
from aana.models.pydantic.sampling_params import SamplingParams
from aana.tests.deployments.utils import start_ray
from aana.tests.utils import compare_texts, is_gpu_available, ray_lock, ray_shutdown


def expected_output(name):
    """Gets expected output for a given vLLM version."""
    if name == "vllm_deployment_llama2_7b_chat":
        return (
            "  Elon Musk is a South African-born entrepreneur, inventor, "
            "and business magnate who is best known for his innovative companies in"
        )
    else:
        raise ValueError(f"Unknown deployment name: {name}")  # noqa: TRY003


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.asyncio
async def test_vllm_deployments():
    """Test VLLM deployments."""
    for name, deployment in deployments.items():
        # skip if not a VLLM deployment
        if deployment.name != "VLLMDeployment":
            continue

        expected_text = expected_output(name)

        with ray_lock():
            handle = start_ray(deployment)

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

            # test generate method with too long prompt
            with pytest.raises(PromptTooLongException):
                output = await handle.generate.remote(
                    prompt="[INST] Who is Elon Musk? [/INST]" * 1000,
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
                )
            ray_shutdown()
