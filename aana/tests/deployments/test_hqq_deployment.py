# ruff: noqa: S101
from importlib import resources

import pytest
from hqq.core.quantize import BaseQuantizeConfig

from aana.core.models.sampling import SamplingParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.hqq_text_generation_deployment import (
    HqqBackend,
    HqqTexGenerationConfig,
    HqqTextGenerationDeployment,
)
from aana.tests.utils import verify_deployment_results
from aana.utils.core import get_object_hash

deployments = [
    (
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            HqqTextGenerationDeployment.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 0.5},
                user_config=HqqTexGenerationConfig(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    backend=HqqBackend.TORCHAO_INT4,
                    compile=True,
                    quantize_on_fly=True,
                    quantization_config=BaseQuantizeConfig(
                        nbits=4, group_size=64, axis=1
                    ),
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512
                    ),
                    model_kwargs={"attn_implementation": "sdpa"},
                ).model_dump(mode="json"),
            ),
        ),
        "<s><|user|>\n{query}<|end|>\n<|assistant|>\n",
    ),
    (
        (
            "mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib",
            HqqTextGenerationDeployment.options(
                num_replicas=1,
                max_ongoing_requests=1000,
                ray_actor_options={"num_gpus": 0.5},
                user_config=HqqTexGenerationConfig(
                    model_id="mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib",
                    backend=HqqBackend.TORCHAO_INT4,
                    compile=False,
                    quantize_on_fly=False,
                    quantization_config=BaseQuantizeConfig(
                        nbits=4,
                        group_size=64,
                        quant_scale=False,
                        quant_zero=False,
                        axis=1,
                    ),
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512
                    ),
                ).model_dump(mode="json"),
            ),
        ),
        "<s><|user|>\n{query}<|end|>\n<|assistant|>\n",
    ),
    (
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            HqqTextGenerationDeployment.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 0.5},
                user_config=HqqTexGenerationConfig(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    backend=HqqBackend.BITBLAS,
                    compile=True,
                    quantize_on_fly=True,
                    quantization_config=BaseQuantizeConfig(
                        nbits=4, group_size=64, axis=1
                    ),
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512
                    ),
                    model_kwargs={"attn_implementation": "sdpa"},
                ).model_dump(mode="json"),
            ),
        ),
        "<s><|user|>\n{query}<|end|>\n<|assistant|>\n",
    ),
    (
        (
            "mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib",
            HqqTextGenerationDeployment.options(
                num_replicas=1,
                max_ongoing_requests=1000,
                ray_actor_options={"num_gpus": 0.5},
                user_config=HqqTexGenerationConfig(
                    model_id="mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib",
                    backend=HqqBackend.BITBLAS,
                    compile=False,
                    quantize_on_fly=False,
                    quantization_config=BaseQuantizeConfig(
                        nbits=4,
                        group_size=64,
                        quant_scale=False,
                        quant_zero=False,
                        axis=1,
                    ),
                    default_sampling_params=SamplingParams(
                        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512
                    ),
                ).model_dump(mode="json"),
            ),
        ),
        "<s><|user|>\n{query}<|end|>\n<|assistant|>\n",
    ),
]


@pytest.mark.parametrize(
    "setup_deployment, prompt_template", deployments, indirect=["setup_deployment"]
)
class TestHQQDeployments:
    """Test hqq deployments."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", ["Who is Elon Musk?"])
    async def test_hqq_text_generation_methods(
        self, setup_deployment, prompt_template, query
    ):
        """Test text generation methods."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        query_hash = get_object_hash(query)
        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "hqq_generation"
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
