# ruff: noqa: S101
from importlib import resources

import pytest

from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.hf_text_generation_deployment import (
    HfTextGenerationConfig,
    HfTextGenerationDeployment,
)
from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment
from aana.exceptions.runtime import PromptTooLongException
from aana.tests.utils import verify_deployment_results
from aana.utils.core import get_object_hash

deployments = [
    (
        (
            "phi3_mini_4k_instruct_hf_text_generation_deployment",
            HfTextGenerationDeployment.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 0.25},
                user_config=HfTextGenerationConfig(
                    model_id="microsoft/Phi-3-mini-4k-instruct",
                    model_kwargs={
                        "trust_remote_code": True,
                    },
                    default_sampling_params=SamplingParams(
                        temperature=0.0, kwargs={"diversity_penalty": 0.0}
                    ),
                ).model_dump(mode="json"),
            ),
        ),
        "<s><|user|>\n{query}<|end|>\n<|assistant|>\n",
    ),
    (
        (
            "phi3_mini_4k_instruct_vllm_deployment",
            VLLMDeployment.options(
                num_replicas=1,
                max_ongoing_requests=1000,
                ray_actor_options={"num_gpus": 0.25},
                user_config=VLLMConfig(
                    model="microsoft/Phi-3-mini-4k-instruct",
                    dtype=Dtype.FLOAT16,
                    gpu_memory_reserved=10000,
                    enforce_eager=True,
                    default_sampling_params=SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        max_tokens=1024,
                        kwargs={"frequency_penalty": 0.0},
                    ),
                    engine_args={
                        "trust_remote_code": True,
                    },
                ).model_dump(mode="json"),
            ),
        ),
        "<s><|user|>\n{query}<|end|>\n<|assistant|>\n",
    ),
]


@pytest.mark.parametrize(
    "setup_deployment, prompt_template", deployments, indirect=["setup_deployment"]
)
class TestTextGenerationDeployments:
    """Test text generation deployments."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", ["Who is Elon Musk?"])
    async def test_text_generation_methods(
        self, setup_deployment, prompt_template, query
    ):
        """Test text generation methods."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        query_hash = get_object_hash(query)
        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "text_generation"
            / f"{deployment_name}_{query_hash}.json"
        )

        prompt = prompt_template.format(query=query)

        # test generate method
        output = await handle.generate(
            prompt=prompt,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )
        verify_deployment_results(expected_output_path, output["text"])

        # test generate_stream method
        stream = handle.generate_stream(
            prompt=prompt,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )
        text = ""
        async for chunk in stream:
            text += chunk["text"]

        verify_deployment_results(expected_output_path, text)

        # test generate_batch method
        output = await handle.generate_batch(
            prompts=[prompt, prompt],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )
        texts = output["texts"]

        for text in texts:
            verify_deployment_results(expected_output_path, text)

        # test chat method
        dialog = ChatDialog(
            messages=[
                ChatMessage(
                    role="user",
                    content=query,
                )
            ]
        )
        output = await handle.chat(
            dialog=dialog,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )

        response_message = output["message"]
        assert response_message.role == "assistant"
        text = response_message.content

        verify_deployment_results(expected_output_path, text)

        # test chat_stream method
        stream = handle.chat_stream(
            dialog=dialog,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )

        text = ""
        async for chunk in stream:
            text += chunk["text"]

        verify_deployment_results(expected_output_path, text)

        # test generate method with too long prompt
        with pytest.raises(PromptTooLongException):
            output = await handle.generate(
                prompt=prompt * 1000,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
            )
