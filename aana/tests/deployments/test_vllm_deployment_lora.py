# ruff: noqa: S101
from importlib import resources

import pytest

from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.sampling import SamplingParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.vllm_deployment import (
    LoRAConfig,
    VLLMConfig,
    VLLMDeployment,
)
from aana.tests.utils import verify_deployment_results
from aana.utils.core import get_object_hash

# TODO: Add checks for invalid LoRA loading, e.g., non-existent LoRA, invalid structure, etc.
deployments = [
    (
        (
            "qwen1.5_1.8b_chat_vllm_deployment_with_lora",
            VLLMDeployment.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 1.0},
                user_config=VLLMConfig(
                    model_id="Qwen/Qwen1.5-1.8B-Chat",
                    gpu_memory_reserved=12000,
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
                    ),
                    max_model_len=2048,
                    loras=[
                        # LoRAConfig(name="test", model_id="vaibhavmeena/Phi-3.5-vision-instruct-amz-lora"),
                        # LoRAConfig(name="test", model_id="Tsumugii/Sunsimiao-Qwen1.5-1.8B-lora-0.5"),
                        LoRAConfig(
                            name="test", model_id="Speeeed/Qwen1.5-1.8B-Chat-wsc-lora"
                        )
                    ],
                ).model_dump(mode="json"),
            ),
        ),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
    ),
]


@pytest.mark.parametrize(
    "setup_deployment, prompt_template", deployments, indirect=["setup_deployment"]
)
class TestVLLMDeploymentLoRA:
    """Test VLLMDeployment with LoRA."""

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
            lora="test",
        )
        verify_deployment_results(
            expected_output_path, output["text"], allowed_error_rate=0.6
        )

        # test generate_stream method
        stream = handle.generate_stream(
            prompt=prompt,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
            lora="test",
        )
        text = ""
        async for chunk in stream:
            text += chunk["text"]

        verify_deployment_results(expected_output_path, text, allowed_error_rate=0.6)

        # test generate_batch method
        output = await handle.generate_batch(
            prompts=[prompt, prompt],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
            lora="test",
        )
        texts = output["texts"]

        for text in texts:
            verify_deployment_results(
                expected_output_path, text, allowed_error_rate=0.6
            )

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
            lora="test",
        )

        response_message = output["message"]
        assert response_message.role == "assistant"
        text = response_message.content

        verify_deployment_results(expected_output_path, text, allowed_error_rate=0.6)

        # test chat_stream method
        stream = handle.chat_stream(
            dialog=dialog,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
            lora="test",
        )

        text = ""
        async for chunk in stream:
            text += chunk["text"]

        verify_deployment_results(expected_output_path, text, allowed_error_rate=0.6)

        # test chat method with non-existent LoRA
        with pytest.raises(ValueError):
            await handle.chat(
                dialog=dialog,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
                lora="non_existent",
            )
