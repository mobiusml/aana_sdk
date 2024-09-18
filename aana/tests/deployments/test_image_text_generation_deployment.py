# ruff: noqa: S101
from importlib import resources

import pytest

from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.image import Image
from aana.core.models.image_chat import ImageChatDialog
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment
from aana.exceptions.runtime import PromptTooLongException
from aana.tests.utils import verify_deployment_results
from aana.utils.core import get_object_hash

deployments = [
    (
        (
            "phi-3.5-vision-instruct_vllm_deployment",
            VLLMDeployment.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 1.0},
                user_config=VLLMConfig(
                    model_id="microsoft/Phi-3.5-vision-instruct",
                    dtype=Dtype.FLOAT16,
                    gpu_memory_reserved=12000,
                    enforce_eager=True,
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
                    ),
                    max_model_len=2048,
                    engine_args=dict(
                        trust_remote_code=True,
                        max_num_seqs=32,
                        limit_mm_per_prompt={"image": 3},
                    ),
                ).model_dump(mode="json"),
            ),
        ),
        "<s><|user|>\n{query}<|end|>\n<|assistant|>\n",
    ),
    # disabled because it requires unrealeased version of transformers
    # (
    #     (
    #         "qwen2_vl_7b_instruct_vllm_deployment",
    #         VLLMDeployment.options(
    #             num_replicas=1,
    #             ray_actor_options={"num_gpus": 1.0},
    #             user_config=VLLMConfig(
    #                 model_id="Qwen/Qwen2-VL-7B-Instruct",
    #                 gpu_memory_reserved=40000,
    #                 enforce_eager=True,
    #                 default_sampling_params=SamplingParams(
    #                     temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
    #                 ),
    #                 max_model_len=4096,
    #                 engine_args=dict(
    #                     limit_mm_per_prompt={"image": 3},
    #                 ),
    #             ).model_dump(mode="json"),
    #         ),
    #     ),
    #     "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
    # ),
    (
        (
            "pixtral_12b_2409_vllm_deployment",
            VLLMDeployment.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 1.0},
                user_config=VLLMConfig(
                    model_id="mistralai/Pixtral-12B-2409",
                    gpu_memory_reserved=40000,
                    enforce_eager=True,
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
                    ),
                    max_model_len=4096,
                    engine_args=dict(
                        tokenizer_mode="mistral",
                        limit_mm_per_prompt={"image": 3},
                    ),
                ).model_dump(mode="json"),
            ),
        ),
        "<s>[INST]{query}[/INST] \n",
    ),
]


@pytest.mark.parametrize(
    "setup_deployment, prompt_template", deployments, indirect=["setup_deployment"]
)
class TestImageTextGenerationDeployments:
    """Test image-text generation deployments."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prompt, image_name",
        [("Who is the painter of the image?", "Starry_Night.jpeg")],
    )
    async def test_image_text_generation_methods(
        self, setup_deployment, prompt_template, prompt, image_name
    ):
        """Test image-text generation methods."""
        deployment_name, handle_name, _ = setup_deployment
        handle = await AanaDeploymentHandle.create(handle_name)

        prompt_hash = get_object_hash(prompt)
        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "image_text_generation"
            / f"{deployment_name}_{image_name}_{prompt_hash}.json"
        )

        image = Image(path=resources.files("aana.tests.files.images") / image_name)
        dialog = ImageChatDialog.from_prompt(prompt=prompt, images=[image])

        # test chat method
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
        verify_deployment_results(
            expected_output_path, output["text"], allowed_error_rate=0.6
        )

        # test generate_stream method
        stream = handle.generate_stream(
            prompt=prompt,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )
        text = ""
        async for chunk in stream:
            text += chunk["text"]

        verify_deployment_results(expected_output_path, text, allowed_error_rate=0.6)

        # test generate_batch method
        output = await handle.generate_batch(
            prompts=[prompt, prompt],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
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
