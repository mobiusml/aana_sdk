# ruff: noqa: S101
from importlib import resources

import pytest

from aana.core.models.chat import ChatMessage
from aana.core.models.image import Image
from aana.core.models.multimodal_chat import (
    FrameVideo,
    MultimodalChatDialog,
)
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.vllm_deployment import (
    GemliteMode,
    GemliteQuantizationConfig,
    VLLMConfig,
    VLLMDeployment,
)
from aana.tests.utils import verify_deployment_results
from aana.utils.core import get_object_hash

deployments = [
    (
        (
            "qwen2_vl_3b_gemlite_vllm_deployment",
            VLLMDeployment.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 1.0},
                user_config=VLLMConfig(
                    model_id="mobiuslabsgmbh/Qwen2.5-VL-3B-Instruct_4bitgs64_hqq_hf",
                    gpu_memory_reserved=40000,
                    gemlite_mode=GemliteMode.PREQUANTIZED,
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
                    ),
                    max_model_len=4096,
                    engine_args=dict(
                        limit_mm_per_prompt={"image": 3, "video": 1},
                        mm_processor_kwargs={
                            "min_pixels": 28 * 28,
                            "max_pixels": 640 * 640,
                        },
                    ),
                ).model_dump(mode="json"),
            ),
        ),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        (
            "qwen2_vl_3b_gemlite_onthefly_vllm_deployment",
            VLLMDeployment.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 1.0},
                user_config=VLLMConfig(
                    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                    gpu_memory_reserved=40000,
                    dtype=Dtype.FLOAT16,
                    gemlite_mode=GemliteMode.ONTHEFLY,
                    gemlite_config=GemliteQuantizationConfig(
                        weight_bits=8,
                        group_size=None,
                        quant_mode="dynamic_int8",
                        skip_modules=["lm_head", "visual", "vision"],
                    ),
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
                    ),
                    max_model_len=4096,
                    engine_args=dict(
                        limit_mm_per_prompt={"image": 3, "video": 1},
                        mm_processor_kwargs={
                            "min_pixels": 28 * 28,
                            "max_pixels": 640 * 640,
                        },
                    ),
                ).model_dump(mode="json"),
            ),
        ),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
    ),
]


@pytest.mark.parametrize(
    "setup_deployment, prompt_template", deployments, indirect=["setup_deployment"]
)
class TestMultimodalTextGenerationDeployments:
    """Test multimodal-text generation deployments."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prompt, image_name",
        [("Who is the painter of the image?", "Starry_Night.jpeg")],
    )
    async def test_multimodal_text_generation_methods(
        self, setup_deployment, prompt_template, prompt, image_name
    ):
        """Test mumtimodal-text generation methods."""
        deployment_name, handle_name, _ = setup_deployment
        handle = await AanaDeploymentHandle.create(handle_name)

        # test multimodal chat dialog with image

        prompt_hash = get_object_hash(prompt)
        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "image_text_generation"
            / f"{deployment_name}_{image_name}_{prompt_hash}.json"
        )

        image = Image(path=resources.files("aana.tests.files.images") / image_name)
        dialog = MultimodalChatDialog.from_prompt(prompt=prompt, images=[image])

        # test chat method with image
        output = await handle.chat(dialog=dialog)
        output_message = output["message"]

        assert isinstance(output_message, ChatMessage)
        assert output_message.role == "assistant"
        verify_deployment_results(expected_output_path, output_message.content)

        # test chat_stream method
        stream = handle.chat_stream(dialog=dialog)
        text = ""
        async for chunk in stream:
            text += chunk["text"]

        verify_deployment_results(expected_output_path, text)

        # test chat method with video
        frames = [
            Image(path=resources.files("aana.tests.files.images") / image_name)
        ] * 5
        video = FrameVideo(frames=frames)

        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "image_text_generation"
            / f"{deployment_name}_video_{image_name}_{prompt_hash}.json"
        )

        dialog = MultimodalChatDialog.from_prompt(prompt=prompt, videos=[video])

        # test chat method with video
        output = await handle.chat(dialog=dialog)
        output_message = output["message"]

        assert isinstance(output_message, ChatMessage)
        assert output_message.role == "assistant"
        verify_deployment_results(expected_output_path, output_message.content)

        # test chat_stream method with video
        stream = handle.chat_stream(dialog=dialog)
        text = ""
        async for chunk in stream:
            text += chunk["text"]

        verify_deployment_results(expected_output_path, text)
